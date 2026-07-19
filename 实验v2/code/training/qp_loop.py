"""
QP-Augmented SafePVC Main Loop

Orchestrates the full SafePVC + BarrierNet QP training + verification pipeline:
1. Load pretrained components (gen_net, state_net, PPO policy)
2. Initialize QP-augmented VTLearner
3. Alternate SBC training, verification, and controller+QP refinement
4. Log and save results

Supports both:
- 'qp' mode: Controller with CBF-QP safety filter (experimental)
- 'direct' mode: Original SafePVC controller (baseline)

Author: Experiment v2
"""

import os
import sys
import time
import math
import argparse
import json
import torch
import numpy as np
from datetime import datetime

# Add original project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artical-F122'))

from Aebs.system.env import Aebs
from Aebs.VT.verify import VTVerifier
from training.qp_trainer import QPVTLearner


class QPLoop:
    """
    Main training + verification loop for SafePVC + QP integration.

    Based on the original Loop class from VT/loop.py, extended to:
    - Support QP controller mode
    - Track CBF-specific metrics
    - Log QP feasibility and constraint violation rates
    """

    def __init__(self, learner, verifier, env, config):
        self.env = env
        self.learner = learner
        self.verifier = verifier
        self.config = config

        self.iter = 0
        self.info = {}
        self.history = {
            'iterations': [],
            'hard_violations': [],
            'prob_lower_bound': [],
            'cbf_metrics': [],
            'timing': [],
        }

        # Results directory
        self.results_dir = config.get('results_dir', './results')
        os.makedirs(self.results_dir, exist_ok=True)

    def train(self, model, num_epochs=10, violation_buffer=None):
        """Train SBC or controller."""
        train_ds, grid_lb, grid_ub = self.verifier.get_unfiltered_grid(
            self.env.train_space_split
        )
        current_delta = (
            self.env.observation_space.high - self.env.observation_space.low
        ) / self.env.train_space_split

        batch_size = self.config.get('batch_size', 256)
        lip_coeff = self.config.get('lip_coeff', 0.001)

        if model == 'p' and violation_buffer is not None:
            train_ds = violation_buffer

        start_time = time.time()

        epoch_metrics_list = self.learner.train_epoch(
            train_ds=train_ds,
            current_delta=current_delta,
            lip=lip_coeff,
            batch_size=batch_size,
            shuffle=True,
            num_epochs=num_epochs,
            train_fn=model,
        )

        elapsed = time.time() - start_time
        return epoch_metrics_list, elapsed

    def run(self, timeout=3600):
        """
        Main loop: alternate SBC training, verification, controller refinement.

        Args:
            timeout: Maximum runtime in seconds (default: 1 hour)

        Returns:
            results: Dict with final metrics
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"SafePVC + QP Experiment Starting")
        print(f"  Controller type: {self.learner.controller_type}")
        print(f"  Timeout: {timeout}s")
        print(f"  Results dir: {self.results_dir}")
        print(f"{'='*60}\n")

        # Step 1: Pre-fill training buffer with discretized grid
        self.verifier.prefill_train_buffer()

        max_reach_prob = 0
        prob_history = []
        hard_violation_history = []

        while True:
            runtime = time.time() - start_time

            if runtime > timeout:
                print(f"\nTimeout reached ({runtime:.0f}s). Stopping.")
                break

            if self.iter > self.config.get('max_iterations', 100):
                print(f"\nMax iterations ({self.config.get('max_iterations', 100)}) reached.")
                break

            print(f"\n{'─'*50}")
            print(f"Iteration {self.iter} ({runtime//60:.0f}m {runtime%60:.0f}s elapsed)")
            print(f"{'─'*50}")

            # Step 2: Train SBC (L-Net)
            print("[Train] SBC (L-Net), 10 epochs...")
            l_metrics, l_time = self.train('l', num_epochs=10)
            print(f"  SBC training: {l_time:.1f}s")
            if l_metrics:
                final = l_metrics[-1]
                print(f"  Loss: {final.get('loss_l', 'N/A'):.4f}, "
                      f"Violations: {final.get('train_violations_l', 'N/A'):.4f}")

            # Step 3: Estimate Lipschitz constant
            k_except_l = float(self.config.get('k_except_l', 1.2))
            self.info['K'] = k_except_l
            self.info['iter'] = self.iter
            self.info['runtime'] = runtime

            # Step 4: Verify decrease condition
            print("[Verify] Checking decrease condition...")
            sat, hard_violations, info, violation_buffer = self.verifier.check_dec_cond(k_except_l)

            for k, v in info.items():
                self.info[k] = v

            hard_violation_history.append(
                hard_violations.item() if isinstance(hard_violations, torch.Tensor) else hard_violations
            )

            print(f"  Satisfied: {sat}, Hard violations: {hard_violations}")

            # Step 5: If satisfied, compute probability bound
            if sat:
                print("[Verify] Computing probability bound...")
                _, ub_init = self.verifier.compute_bound_init(
                    self.config.get('jitter_grid', self.env.space_split)
                )
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(
                    self.config.get('jitter_grid', self.env.space_split)
                )
                domain_min, _ = self.verifier.compute_bound_domain(
                    self.config.get('jitter_grid', self.env.space_split)
                )

                self.info['ub_init'] = float(ub_init)
                self.info['lb_unsafe'] = float(lb_unsafe)
                self.info['domain_min'] = float(domain_min)

                if lb_unsafe > ub_init:
                    # Normalize and compute probability
                    ub_init_norm = ub_init - domain_min
                    lb_unsafe_norm = lb_unsafe - domain_min
                    ratio = lb_unsafe_norm / max(ub_init_norm, 1e-9)
                    reach_prob = 1.0 - 1.0 / max(ratio, 1e-9)

                    if reach_prob > max_reach_prob:
                        max_reach_prob = reach_prob
                        self._save_checkpoint()
                        print(f"  ★ New best probability bound: {reach_prob*100:.2f}%")
                else:
                    reach_prob = 0.0
                    print("  Warning: lb_unsafe <= ub_init, no valid probability bound")

                prob_history.append(max_reach_prob)
                self.info['reach_prob'] = float(max_reach_prob)
                print(f"  Current best prob: {max_reach_prob*100:.2f}%")

            # Step 6: Record history
            self.history['iterations'].append(self.iter)
            self.history['hard_violations'].append(
                hard_violations.item() if isinstance(hard_violations, torch.Tensor) else hard_violations
            )
            self.history['prob_lower_bound'].append(max_reach_prob)
            self.history['timing'].append(runtime)

            # Step 7: Train controller (with QP if enabled)
            train_fn = 'p_direct' if self.learner.controller_type == 'direct' else 'p'
            print(f"[Train] Controller ({train_fn}), 1 epoch...")
            p_metrics, p_time = self.train(train_fn, num_epochs=1, violation_buffer=violation_buffer)
            print(f"  Controller training: {p_time:.1f}s")
            if p_metrics:
                final = p_metrics[-1]
                cbf_info = f", CBF loss: {final.get('cbf_loss', 'N/A'):.4f}" if 'cbf_loss' in final else ""
                print(f"  Loss: {final.get('loss_p', 'N/A'):.4f}, "
                      f"MSE: {final.get('mse_loss_p', 'N/A'):.4f}{cbf_info}")

            # Step 8: Update iteration counter
            self.iter += 1

            # Periodic logging
            if self.iter % 10 == 0:
                self._save_history()

        # Final save
        self._save_history()
        self._save_results()

        return {
            'max_reach_prob': max_reach_prob,
            'final_iter': self.iter,
            'total_runtime': time.time() - start_time,
            'history': self.history,
        }

    def _save_checkpoint(self):
        """Save best model checkpoint."""
        ckpt_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(
            self.learner.l_model.state_dict(),
            os.path.join(ckpt_dir, 'l_model_best.pth')
        )
        torch.save(
            self.learner.p_net.state_dict(),
            os.path.join(ckpt_dir, 'p_net_best.pth')
        )

    def _save_history(self):
        """Save training history to disk."""
        hist_path = os.path.join(self.results_dir, 'training_history.npz')
        np.savez(
            hist_path,
            iterations=np.array(self.history['iterations']),
            hard_violations=np.array(self.history['hard_violations']),
            prob_lower_bound=np.array(self.history['prob_lower_bound']),
            timing=np.array(self.history['timing']),
        )

    def _save_results(self):
        """Save final results and info."""
        results_path = os.path.join(self.results_dir, 'final_results.json')
        serializable_info = {}
        for k, v in self.info.items():
            if isinstance(v, (int, float, str, bool)):
                serializable_info[k] = v
            elif isinstance(v, np.ndarray):
                serializable_info[k] = v.tolist()
            else:
                serializable_info[k] = str(v)

        results = {
            'controller_type': self.learner.controller_type,
            'config': self.config,
            'info': serializable_info,
            'max_reach_prob': float(self.history['prob_lower_bound'][-1])
                if self.history['prob_lower_bound'] else 0.0,
            'final_iter': self.iter,
        }

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {results_path}")


def run_experiment(config):
    """
    Run a single experiment with given configuration.

    Args:
        config: Dict with experiment parameters

    Returns:
        results: Dict with experiment results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    noise_factor = config.get('noise_factor', 0.05)
    env = Aebs(noise_factor)

    # Initialize QP-augmented learner
    vt_learner = QPVTLearner(
        l_model_config=config.get('l_model_config', [2, 16, 8, 1]),
        env=env,
        p_lip=config.get('p_lip', 2.0),
        l_lip=config.get('l_lip', 4.0),
        eps=config.get('eps', 0.1),
        gamma_decrease=config.get('gamma_decrease', 1.0),
        reach_prob=config.get('reach_prob', 0.95),
        controller_type=config.get('controller_type', 'qp'),
        controller_config=config.get('controller_config', {}),
        lambda_cbf=config.get('lambda_cbf', 1.0),
        lambda_mse=config.get('lambda_mse', 10.0),
        square_l_output=True,
        l_model_path=config.get('l_model_path'),
        p_model_path=config.get('p_model_path'),
    )

    # Initialize verifier
    l_ibp = vt_learner.create_bounded_module(vt_learner.l_model)
    vt_verifier = VTVerifier(
        vt_learner,
        env,
        l_ibp,
        batch_size=config.get('verify_batch_size', 2048),
        reach_prob=config.get('verify_reach_prob', 0.9),
        fail_check_fast=config.get('fail_check_fast', True),
    )

    # Create and run loop
    loop = QPLoop(vt_learner, vt_verifier, env, config)
    results = loop.run(timeout=config.get('timeout', 3600))

    return results


def get_default_config():
    """Get default experiment configuration."""
    return {
        # Environment
        'noise_factor': 0.05,
        # Controller
        'controller_type': 'qp',  # 'qp' or 'direct'
        'controller_config': {
            'input_dim': 2,
            'hidden_dims': [256, 256, 256],
            'control_dim': 1,
            'cbf_param_dim': 1,
            't_gap': 1.5,
            'dt': 0.05,
        },
        # SBC
        'l_model_config': [2, 16, 8, 1],
        'p_lip': 2.0,
        'l_lip': 4.0,
        'eps': 0.1,
        'gamma_decrease': 1.0,
        'reach_prob': 0.95,
        # Loss weights
        'lambda_cbf': 1.0,
        'lambda_mse': 10.0,
        # Training
        'batch_size': 256,
        'lip_coeff': 0.001,
        # Verification
        'verify_batch_size': 2048,
        'verify_reach_prob': 0.9,
        'fail_check_fast': True,
        'k_except_l': 1.2,
        'jitter_grid': None,  # Will use env.space_split
        # Loop
        'max_iterations': 100,
        'timeout': 3600,  # 1 hour
        # Paths
        'results_dir': './results/qp_integrated',
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SafePVC + QP Experiment')
    parser.add_argument('--mode', type=str, default='qp',
                        choices=['qp', 'baseline'],
                        help='Experiment mode: qp (with QP filter) or baseline (direct)')
    parser.add_argument('--noise-factor', type=float, default=0.05,
                        help='State perturbation factor')
    parser.add_argument('--lambda-cbf', type=float, default=1.0,
                        help='CBF loss weight')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Experiment timeout in seconds')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory')
    parser.add_argument('--tag', type=str, default=None,
                        help='Experiment tag for results naming')

    args = parser.parse_args()

    config = get_default_config()
    config['controller_type'] = 'direct' if args.mode == 'baseline' else 'qp'
    config['noise_factor'] = args.noise_factor
    config['lambda_cbf'] = args.lambda_cbf
    config['timeout'] = args.timeout

    if args.results_dir:
        config['results_dir'] = args.results_dir
    elif args.mode == 'baseline':
        config['results_dir'] = './results/baseline'
    else:
        tag_suffix = f"_{args.tag}" if args.tag else ""
        config['results_dir'] = f'./results/qp_integrated{tag_suffix}'

    # Make results dir absolute relative to 实验v2
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config['results_dir'] = os.path.join(base_dir, config['results_dir'])

    print(f"\n{'='*60}")
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    results = run_experiment(config)

    print(f"\n{'='*60}")
    print(f"Experiment Complete")
    print(f"  Mode: {args.mode}")
    print(f"  Max probability bound: {results['max_reach_prob']*100:.2f}%")
    print(f"  Iterations: {results['final_iter']}")
    print(f"  Runtime: {results['total_runtime']:.0f}s")
    print(f"{'='*60}")
