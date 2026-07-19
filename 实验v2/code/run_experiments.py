#!/usr/bin/env python3
"""
Self-contained experiment runner for 实验v2.

Runs baseline (SafePVC without QP) and QP-integrated (SafePVC + BarrierNet QP)
experiments, collecting metrics for comparison.

Usage:
    cd /root/paper-combination
    python 实验v2/code/run_experiments.py [--mode both] [--timeout 600] [--noise 0.05]
"""

import sys
import os
import time
import json
import math
import argparse
import traceback
from datetime import datetime

# === Path setup: MUST run from artical-F122 directory for imports ===
ORIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'artical-F122')
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, os.path.join(EXP_DIR, 'code'))
sys.path.insert(0, os.path.join(EXP_DIR, 'code', 'models'))
sys.path.insert(0, os.path.join(EXP_DIR, 'code', 'training'))
sys.path.insert(0, os.path.join(EXP_DIR, 'code', 'utils'))

os.chdir(ORIG_DIR)  # Work from artical-F122 directory

import torch
import numpy as np
import copy

from Aebs.system.env import Aebs, AebsEnv
from Aebs.VT.utils import triangular, martingale_loss, MLP
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier, TrainBuffer

from models.qp_controller import QPAebsController, DirectController
from models.cbf_constraints import AEBSCBFConstraints


class ExperimentRunner:
    """Runs a single SafePVC experiment and collects results."""

    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Runner] Device: {self.device}")
        print(f"[Runner] Mode: {config['mode']}")
        print(f"[Runner] Noise factor: {config['noise_factor']}")
        print(f"[Runner] Timeout: {config['timeout']}s")
        print(f"[Runner] Results: {results_dir}")

        # Results tracking
        self.iterations = []
        self.prob_bounds = []
        self.hard_violations = []
        self.soft_violations = []
        self.timing = []
        self.l_metrics = []
        self.p_metrics = []

        # Build environment
        self.env = Aebs(config['noise_factor'])

        # Build learner
        self.learner = self._build_learner()

        # Build verifier
        l_ibp = self.learner.create_bounded_module(self.learner.l_model)
        self.verifier = VTVerifier(
            self.learner, self.env, l_ibp,
            batch_size=config.get('verify_batch_size', 2048),
            reach_prob=config.get('verify_reach_prob', 0.9),
            fail_check_fast=config.get('fail_check_fast', True),
        )

        # Pre-fill grid
        self.verifier.prefill_train_buffer()

    def _build_learner(self):
        """Build the learner with appropriate controller type."""
        # Load gen_net
        from cGAN.taxi_models_and_data import AebsMLPGenerator
        gen_net = AebsMLPGenerator(4, 1)
        gen_net.load_state_dict(torch.load(
            "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth",
            map_location=self.device
        ))

        # Load PPO policy
        from stable_baselines3 import PPO
        from Combined_network.model import AebsEnd2EndNet, SubNet, CombinedPolicyNetwork

        model = PPO.load('./Aebs/controller/best_model/best_model.zip')
        policy = model.policy
        mlp_extractor = policy.mlp_extractor.policy_net
        action_net = policy.action_net

        state_layer_sizes = [1024, 256, 64, 1]

        # Build end-to-end net
        p_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)
        p_net.state_net.load_state_dict(torch.load(
            "./Aebs/controller/state_net_trained.pth",
            map_location=self.device
        ))
        p_net.to(self.device)

        # Create frozen teacher copy
        teacher_net = copy.deepcopy(p_net)
        teacher_net.eval()
        for param in teacher_net.parameters():
            param.requires_grad = False
        teacher_net.to(self.device)

        # Build SBC (l_model)
        l_model = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(self.device)

        # Replace controller if QP mode
        if self.config['mode'] == 'qp':
            qp_ctrl = QPAebsController(
                input_dim=2,
                hidden_dims=[256, 256, 256],
                control_dim=1,
                cbf_param_dim=1,
                t_gap=self.config.get('t_gap', 1.5),
                dt=0.05,
                device=self.device,
            ).to(self.device)
            p_net.controller_net = qp_ctrl
            print(f"[Runner] Using QP controller ({sum(p.numel() for p in qp_ctrl.parameters())} params)")
        else:
            print("[Runner] Using Direct controller (baseline)")

        # Store components
        self.p_net = p_net
        self.l_model = l_model
        self.teacher_net = teacher_net
        self.gen_net = gen_net

        # Create VTLearner-compatible wrapper
        # We manually set up the learner's attributes
        learner = VTLearner(
            l_model_config=[2, 16, 8, 1],
            env=self.env,
            p_lip=2.0,
            l_lip=4.0,
            eps=0.1,
            gamma_decrease=1.0,
            reach_prob=0.95,
            square_l_output=True,
        )
        # Override with our components
        learner.p_net = p_net
        learner.l_model = l_model
        learner.net = teacher_net
        learner.device = self.device
        learner.env = self.env
        learner.l_optimizer = torch.optim.Adam(l_model.parameters(), lr=3e-3)
        learner.p_optimizer = torch.optim.Adam(
            p_net.controller_net.parameters(), lr=5e-3
        ) if self.config['mode'] == 'qp' else torch.optim.Adam(
            [p for n, p in p_net.named_parameters() if 'controller_net' in n and 'gen_net' not in n and 'state_net' not in n],
            lr=5e-2
        )
        learner.eps = 0.1
        learner.gamma_decrease = 1.0
        learner.l_lip = 4.0
        learner.p_lip = 2.0
        learner.reach_prob = 0.95

        if self.config['mode'] == 'qp':
            learner.cbf_calculator = AEBSCBFConstraints(
                t_gap=self.config.get('t_gap', 1.5), dt=0.05
            )
            learner.lambda_cbf = self.config.get('lambda_cbf', 1.0)
            learner.lambda_mse = self.config.get('lambda_mse', 10.0)

        return learner

    def create_bounded_module(self, model):
        from auto_LiRPA import BoundedModule
        for p in model.parameters():
            p.requires_grad = False
        dummy = torch.randn(1, self.env.observation_space.shape[0]).to(self.device)
        return BoundedModule(model, dummy, device=self.device)

    def _train_sbc(self, train_ds, current_delta, num_epochs=10):
        """Train SBC (l_model)."""
        self.l_model.train()
        self.p_net.eval()

        N = train_ds.shape[0]
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=self.device)
        batch_size = self.config.get('batch_size', 256)
        current_delta = torch.tensor(current_delta, dtype=torch.float32, device=self.device)

        epoch_metrics = []
        for epoch in range(num_epochs):
            indices = np.random.permutation(N)
            all_y = all_y[indices]
            batch_losses = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y = all_y[start:end]
                B = y.shape[0]
                z = (torch.rand(B, 4, device=self.device) * 2.0 - 1.0).float()

                # Perturb states
                rng = torch.Generator(device=self.device)
                rng.manual_seed(19)
                delta_t = torch.tensor(current_delta, dtype=y.dtype, device=self.device)
                s_rand = torch.rand(y.shape, generator=rng, device=self.device) - 0.5
                y_pert = y + delta_t.unsqueeze(0) * s_rand

                self.learner.l_optimizer.zero_grad()

                # Get action from controller
                with torch.no_grad():
                    if self.config['mode'] == 'qp':
                        a = self.p_net.controller_net(y_pert.float(), state=y_pert.float(), mode='direct')
                    else:
                        a = self.p_net(z.float(), y_pert.float())

                s_next = self.env.v_next(y_pert, a).unsqueeze(1)
                noise = triangular((B, 16, y.shape[1]), device=self.device)
                noise_scale = torch.as_tensor(self.env.noise, device=self.device, dtype=torch.float32).view(1, 1, -1)
                s_next_rand = (s_next + noise * noise_scale).float()

                l_val = self.l_model(y_pert.float()).view(-1)
                l_next = self.l_model(s_next_rand.reshape(-1, y.shape[1]).float()).view(B, -1)
                exp_l_next = l_next.mean(dim=1)

                # Martingale loss
                dec_loss = martingale_loss(l_val, exp_l_next, eps=float(0.1))
                loss_l = dec_loss * 1000

                # Lipschitz regularization
                y_lip = y.detach().clone().requires_grad_(True)
                l_out = self.l_model(y_lip.float()).sum()
                l_grad = torch.autograd.grad(l_out, y_lip, create_graph=True)[0]
                lip_norm = l_grad.view(B, -1).norm(2, 1)
                lip_loss = torch.relu(lip_norm - 4.0).mean()
                loss_l = loss_l + 0.001 * lip_loss

                # Region constraints
                init_samples = self._sample_init(256)
                unsafe_samples = self._sample_unsafe(256)
                l_init = self.l_model(init_samples.float()).view(-1)
                l_unsafe = self.l_model(unsafe_samples.float()).view(-1)
                target = 1.0 / max(1e-6, 1.0 - 0.95)
                region_loss = (
                    torch.relu(torch.max(l_init) - 1.0) +
                    torch.relu(target - torch.min(l_unsafe))
                )
                loss_l = loss_l + region_loss

                loss_l.backward()
                torch.nn.utils.clip_grad_norm_(self.l_model.parameters(), 5.0)
                self.learner.l_optimizer.step()

                batch_losses.append({
                    'loss_l': loss_l.item(),
                    'dec_loss': dec_loss.item(),
                    'lip_loss': lip_loss.item(),
                    'region_loss': region_loss.item(),
                })

            epoch_losses = {k: np.mean([b[k] for b in batch_losses]) for k in batch_losses[0]}
            epoch_metrics.append(epoch_losses)

        return epoch_metrics

    def _train_controller(self, train_ds, current_delta, num_epochs=1):
        """Train controller."""
        self.l_model.eval()
        self.p_net.train()

        N = train_ds.shape[0]
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=self.device)
        batch_size = self.config.get('batch_size', 256)
        current_delta = torch.tensor(current_delta, dtype=torch.float32, device=self.device)

        epoch_metrics = []
        for epoch in range(num_epochs):
            indices = np.random.permutation(N)
            all_y = all_y[indices]
            batch_losses = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y = all_y[start:end]
                B = y.shape[0]
                z = (torch.rand(B, 4, device=self.device) * 2.0 - 1.0).float()

                rng = torch.Generator(device=self.device)
                rng.manual_seed(19)
                delta_t = torch.tensor(current_delta, dtype=y.dtype, device=self.device)
                s_rand = torch.rand(y.shape, generator=rng, device=self.device) - 0.5
                y_pert = y + delta_t.unsqueeze(0) * s_rand

                self.learner.p_optimizer.zero_grad()

                if self.config['mode'] == 'qp':
                    # QP controller forward
                    u_safe, q, p, debug = self.p_net.controller_net(
                        y_pert.float(), state=y_pert.float(), mode='train', return_debug=True
                    )
                    a_p = u_safe
                    cbf_violation = AEBSCBFConstraints(t_gap=1.5).cbf_violation_loss(
                        y_pert.float(), u_safe, p.squeeze(-1)
                    )
                else:
                    a_p = self.p_net(z.float(), y_pert.float())

                # SBC computation
                with torch.no_grad():
                    l_p = self.l_model(y_pert.float()).view(-1)

                s_next = self.env.v_next(y_pert.float(), a_p).unsqueeze(1)
                noise = triangular((B, 128, y.shape[1]), device=self.device)
                noise_scale = torch.as_tensor(self.env.noise, device=self.device, dtype=torch.float32).view(1, 1, -1)
                s_next_rand = (s_next + noise * noise_scale).float()

                with torch.no_grad():
                    l_next = self.l_model(s_next_rand.reshape(-1, y.shape[1]).float()).view(B, -1)

                exp_l_next = l_next.mean(dim=1)
                dec_loss = martingale_loss(l_p.detach(), exp_l_next, eps=float(0.1))
                loss_p = dec_loss * 10

                # MSE with teacher
                with torch.no_grad():
                    u_teacher = self.teacher_net(z.float(), y_pert.float())
                mse_loss = torch.nn.functional.mse_loss(a_p, u_teacher.view_as(a_p))
                loss_p = loss_p + 10.0 * mse_loss

                # CBF loss (QP mode only)
                cbf_loss_val = 0.0
                if self.config['mode'] == 'qp':
                    cbf_loss_val = cbf_violation
                    loss_p = loss_p + self.config.get('lambda_cbf', 1.0) * cbf_loss_val

                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.p_net.controller_net.parameters(), 1.0
                )
                self.learner.p_optimizer.step()

                metrics = {
                    'loss_p': loss_p.item(),
                    'dec_loss': dec_loss.item(),
                    'mse_loss': mse_loss.item(),
                }
                if self.config['mode'] == 'qp':
                    metrics['cbf_loss'] = cbf_loss_val.item() if isinstance(cbf_loss_val, torch.Tensor) else cbf_loss_val
                batch_losses.append(metrics)

            epoch_losses = {k: np.mean([b[k] for b in batch_losses]) for k in batch_losses[0]}
            epoch_metrics.append(epoch_losses)

        return epoch_metrics

    def _sample_init(self, n):
        num = len(self.env.init_spaces)
        per = n // num
        rng = torch.Generator(device="cpu")
        rng.manual_seed(13)
        batch = []
        for i in range(num):
            low = torch.tensor(self.env.init_spaces[i].low, dtype=torch.float32)
            high = torch.tensor(self.env.init_spaces[i].high, dtype=torch.float32)
            shape = (per, self.env.observation_space.shape[0])
            x = (high - low) * torch.rand(shape, generator=rng) + low
            batch.append(x)
        return torch.cat(batch, dim=0).to(self.device).float()

    def _sample_unsafe(self, n):
        num = len(self.env.unsafe_spaces)
        per = n // num
        rng = torch.Generator(device="cpu")
        rng.manual_seed(17)
        batch = []
        for i in range(num):
            low = torch.tensor(self.env.unsafe_spaces[i].low, dtype=torch.float32)
            high = torch.tensor(self.env.unsafe_spaces[i].high, dtype=torch.float32)
            shape = (per, self.env.observation_space.shape[0])
            x = (high - low) * torch.rand(shape, generator=rng) + low
            batch.append(x)
        return torch.cat(batch, dim=0).to(self.device).float()

    def run(self):
        """Run the full experiment loop."""
        start_time = time.time()
        max_iter = self.config.get('max_iterations', 50)
        timeout = self.config.get('timeout', 600)

        train_ds, _, _ = self.verifier.get_unfiltered_grid(self.env.train_space_split)
        current_delta = (self.env.observation_space.high - self.env.observation_space.low) / self.env.train_space_split

        max_prob = 0.0
        violation_buffer = None

        for iteration in range(max_iter):
            runtime = time.time() - start_time
            if runtime > timeout:
                print(f"\nTimeout at iteration {iteration}")
                break

            print(f"\n{'='*50}")
            print(f"Iteration {iteration+1}/{max_iter} ({runtime:.0f}s)")
            print(f"{'='*50}")

            # 1. Train SBC
            print("[SBC] Training...")
            l_epoch_metrics = self._train_sbc(train_ds, current_delta, num_epochs=10)
            final_l = l_epoch_metrics[-1]
            print(f"  loss_l={final_l['loss_l']:.4f}, dec={final_l['dec_loss']:.4f}")
            self.l_metrics.append(final_l)

            # 2. Verify
            print("[Verify] Checking decrease condition...")
            k_except_l = 1.2
            sat, hard_v, info, vb = self.verifier.check_dec_cond(k_except_l)

            hv = hard_v.item() if isinstance(hard_v, torch.Tensor) else hard_v
            print(f"  Satisfied: {sat}, Hard violations: {hv}")

            # 3. Compute probability bound
            prob = max_prob
            if sat:
                _, ub_init = self.verifier.compute_bound_init(self.env.space_split)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(self.env.space_split)
                domain_min, _ = self.verifier.compute_bound_domain(self.env.space_split)

                if lb_unsafe > ub_init:
                    ub_norm = ub_init - domain_min
                    lb_norm = lb_unsafe - domain_min
                    ratio = lb_norm / max(ub_norm, 1e-9)
                    prob = max(0.0, 1.0 - 1.0 / max(ratio, 1e-9))
                    if prob > max_prob:
                        max_prob = prob
                        print(f"  ★ New best probability: {prob*100:.2f}%")
                print(f"  Current prob bound: {prob*100:.2f}%")

            # 4. Record
            self.iterations.append(iteration)
            self.prob_bounds.append(max_prob)
            self.hard_violations.append(hv)
            self.timing.append(runtime)

            # 5. Train controller
            print("[Controller] Training...")
            p_epoch_metrics = self._train_controller(
                vb if vb is not None else train_ds,
                current_delta, num_epochs=1
            )
            final_p = p_epoch_metrics[-1]
            cbf_str = f", cbf={final_p.get('cbf_loss', 0):.4f}" if 'cbf_loss' in final_p else ""
            print(f"  loss_p={final_p['loss_p']:.4f}, mse={final_p['mse_loss']:.4f}{cbf_str}")
            self.p_metrics.append(final_p)

        # Save results
        total_time = time.time() - start_time
        results = {
            'mode': self.config['mode'],
            'noise_factor': self.config['noise_factor'],
            'config': self.config,
            'max_prob_bound': float(max_prob),
            'final_prob_bound': float(self.prob_bounds[-1]) if self.prob_bounds else 0.0,
            'iterations': self.iterations,
            'prob_bounds': [float(p) for p in self.prob_bounds],
            'hard_violations': [float(v) for v in self.hard_violations],
            'timing': self.timing,
            'total_runtime': total_time,
            'num_iterations': len(self.iterations),
            'final_l_metrics': {k: float(v) for k, v in self.l_metrics[-1].items()} if self.l_metrics else {},
            'final_p_metrics': {k: float(v) for k, v in self.p_metrics[-1].items()} if self.p_metrics else {},
        }

        results_path = os.path.join(self.results_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save numpy arrays
        np.savez(
            os.path.join(self.results_dir, 'history.npz'),
            iterations=np.array(self.iterations),
            prob_bounds=np.array(self.prob_bounds),
            hard_violations=np.array(self.hard_violations),
            timing=np.array(self.timing),
        )

        print(f"\n{'='*50}")
        print(f"Experiment Complete: {self.config['mode']}")
        print(f"  Max probability bound: {max_prob*100:.2f}%")
        print(f"  Iterations: {len(self.iterations)}")
        print(f"  Runtime: {total_time:.0f}s")
        print(f"  Results saved to: {results_path}")
        print(f"{'='*50}")

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both',
                       choices=['baseline', 'qp', 'both'])
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--lambda-cbf', type=float, default=1.0)
    parser.add_argument('--max-iters', type=int, default=30)
    args = parser.parse_args()

    results_dir_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )

    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    modes_to_run = ['baseline', 'qp'] if args.mode == 'both' else [args.mode]

    for mode in modes_to_run:
        print(f"\n{'#'*60}")
        print(f"# Running: {mode.upper()} experiment")
        print(f"{'#'*60}")

        results_dir = os.path.join(results_dir_base, f"{mode}_nf{str(args.noise).replace('.', '_')}_{timestamp}")
        config = {
            'mode': mode,
            'noise_factor': args.noise,
            'timeout': args.timeout,
            'max_iterations': args.max_iters,
            'lambda_cbf': args.lambda_cbf,
            'lambda_mse': 10.0,
            't_gap': 1.5,
            'batch_size': 256,
            'verify_batch_size': 2048,
            'verify_reach_prob': 0.9,
            'fail_check_fast': True,
        }

        try:
            runner = ExperimentRunner(config, results_dir)
            results = runner.run()
            all_results[mode] = results
        except Exception as e:
            print(f"ERROR in {mode}: {e}")
            traceback.print_exc()
            all_results[mode] = {'error': str(e)}

    # Save comparison summary
    summary_path = os.path.join(results_dir_base, f'comparison_{timestamp}.json')
    comparison = {}
    for mode, res in all_results.items():
        if 'error' in res:
            comparison[mode] = {'error': res['error']}
        else:
            comparison[mode] = {
                'max_prob_bound': res['max_prob_bound'],
                'num_iterations': res['num_iterations'],
                'total_runtime': res['total_runtime'],
                'final_hard_violations': res['hard_violations'][-1] if res['hard_violations'] else None,
            }

    with open(summary_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Baseline':<20} {'QP Integrated':<20}")
    print("-" * 75)

    b = comparison.get('baseline', {})
    q = comparison.get('qp', {})

    print(f"{'Max Probability Bound':<35} {b.get('max_prob_bound', 0)*100:>18.2f}%  {q.get('max_prob_bound', 0)*100:>18.2f}%")
    print(f"{'Iterations':<35} {b.get('num_iterations', 0):>20}  {q.get('num_iterations', 0):>20}")
    print(f"{'Runtime (s)':<35} {b.get('total_runtime', 0):>20.0f}  {q.get('total_runtime', 0):>20.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
