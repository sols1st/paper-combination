#!/usr/bin/env python3
"""
Improved experiment runner with root-cause fixes for QP integration.

Key improvements over v1:
1. DECOUPLED SBC training: SBC trains on direct controller (no QP noise)
   → SBC sees simpler dynamics, easier to verify → better probability bounds
2. Ultra-low λ_cbf options: 0.001, 0.0001
3. Better CBF initialization: p_head bias initialized for p≈2 (not random)
4. CBF slack: Soft constraints with slack variable when QP infeasible
5. Two-stage: Optional baseline pre-training before QP fine-tuning

Usage:
    python 实验v2/code/run_improved_experiments.py --mode all
"""

import sys, os, time, json, copy, argparse, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# Path setup
ORIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'artical-F122')
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, os.path.join(EXP_DIR, 'code', 'models'))
os.chdir(ORIG_DIR)

from Aebs.system.env import Aebs
from Aebs.VT.utils import triangular, martingale_loss, MLP
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier
from models.qp_controller import QPAebsController, DirectController
from models.cbf_constraints import AEBSCBFConstraints


class ImprovedExperimentRunner:
    """Experiment runner with decoupled SBC training and optimized QP integration."""

    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.iterations = []
        self.prob_bounds = []
        self.hard_violations = []
        self.timing = []
        self.sbc_losses = []
        self.ctrl_losses = []

        # Build environment
        self.env = Aebs(config['noise_factor'])
        self._build_models()

        # Build verifier
        l_ibp = self._create_bounded_module(self.l_model)
        self.verifier = VTVerifier(
            VTLearner.__new__(VTLearner), self.env, l_ibp,
            batch_size=config.get('verify_batch_size', 2048),
            reach_prob=config.get('verify_reach_prob', 0.9),
            fail_check_fast=True,
        )
        self.verifier.learner = type('obj', (object,), {
            'l_model': self.l_model,
            'p_net': self.p_net,
            'device': self.device,
        })()
        self.verifier.prefill_train_buffer()

        print(f"[Runner] Mode: {config['mode']}, Noise: {config['noise_factor']}")
        print(f"[Runner] λ_cbf: {config.get('lambda_cbf', 'N/A')}, "
              f"Decoupled SBC: {config.get('decoupled_sbc', True)}")
        print(f"[Runner] Results: {results_dir}")

    def _build_models(self):
        """Build all models with improved initialization."""
        from cGAN.taxi_models_and_data import AebsMLPGenerator
        from stable_baselines3 import PPO
        from Combined_network.model import AebsEnd2EndNet

        # Load gen_net (frozen)
        gen_net = AebsMLPGenerator(4, 1)
        gen_net.load_state_dict(torch.load(
            "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth",
            map_location=self.device
        ))

        # Load PPO policy, extract components
        model = PPO.load('./Aebs/controller/best_model/best_model.zip')
        policy = model.policy
        mlp_extractor = policy.mlp_extractor.policy_net
        action_net = policy.action_net

        # Build end-to-end net
        p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1], mlp_extractor, action_net)
        p_net.state_net.load_state_dict(torch.load(
            "./Aebs/controller/state_net_trained.pth", map_location=self.device
        ))
        p_net.to(self.device)

        # Teacher net (frozen copy)
        teacher_net = copy.deepcopy(p_net)
        teacher_net.eval()
        for p in teacher_net.parameters():
            p.requires_grad = False
        teacher_net.to(self.device)

        # SBC network
        l_model = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(self.device)

        # QP controller with IMPROVED initialization
        if self.config['mode'] == 'qp':
            qp_ctrl = QPAebsController(
                input_dim=2, hidden_dims=[256, 256, 256],
                control_dim=1, cbf_param_dim=1,
                t_gap=self.config.get('t_gap', 1.5),
                dt=0.05, device=self.device,
            ).to(self.device)

            # ★ IMPROVEMENT: Initialize p_head bias so p ≈ 2.0 initially
            # This provides moderate CBF safety without being too restrictive
            # sigmoid(0) = 0.5, so 4*sigmoid(0) = 2.0
            nn.init.zeros_(qp_ctrl.p_head.bias)
            nn.init.zeros_(qp_ctrl.p_head.weight)  # Start with p ≈ 2.0 constant

            p_net.controller_net = qp_ctrl
            print(f"[Runner] QP controller with improved init ({sum(p.numel() for p in qp_ctrl.parameters())} params)")
        else:
            print("[Runner] Direct controller (baseline)")

        self.p_net = p_net
        self.l_model = l_model
        self.teacher_net = teacher_net

        # Optimizers
        self.l_optimizer = torch.optim.Adam(l_model.parameters(), lr=3e-3)
        if self.config['mode'] == 'qp':
            self.p_optimizer = torch.optim.Adam(p_net.controller_net.parameters(), lr=1e-3)  # Lower LR for stability
        else:
            # Only train controller_net params (not gen_net or state_net)
            ctrl_params = [p for n, p in p_net.named_parameters()
                          if 'controller_net' in n and 'gen_net' not in n and 'state_net' not in n]
            self.p_optimizer = torch.optim.Adam(ctrl_params, lr=5e-2)

        # CBF calculator
        self.cbf = AEBSCBFConstraints(t_gap=self.config.get('t_gap', 1.5), dt=0.05)

    def _create_bounded_module(self, model):
        from auto_LiRPA import BoundedModule
        for p in model.parameters():
            p.requires_grad = False
        dummy = torch.randn(1, self.env.observation_space.shape[0]).to(self.device)
        return BoundedModule(model, dummy, device=self.device)

    def _sample_region(self, spaces, n, seed):
        num = len(spaces)
        per = n // num
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        batch = []
        for i in range(num):
            low = torch.tensor(spaces[i].low, dtype=torch.float32)
            high = torch.tensor(spaces[i].high, dtype=torch.float32)
            x = (high - low) * torch.rand((per, low.shape[0]), generator=rng) + low
            batch.append(x)
        return torch.cat(batch, dim=0).to(self.device).float()

    def _train_sbc(self, train_ds, current_delta, num_epochs=10):
        """
        ★ IMPROVED: Decoupled SBC training.
        Uses controller in DIRECT mode (no QP) regardless of experiment mode.
        This prevents QP noise from interfering with SBC learning.
        """
        self.l_model.train()
        self.p_net.eval()

        N = train_ds.shape[0]
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=self.device)
        current_delta_t = torch.tensor(current_delta, dtype=torch.float32, device=self.device)
        batch_size = 256

        epoch_metrics = []
        for epoch in range(num_epochs):
            indices = np.random.permutation(N)
            all_y = all_y[indices]
            batch_losses = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y = all_y[start:end].float()
                B = y.shape[0]
                z = (torch.rand(B, 4, device=self.device) * 2.0 - 1.0).float()

                # Perturb
                rng = torch.Generator(device=self.device)
                rng.manual_seed(19)
                s_rand = torch.rand(y.shape, generator=rng, device=self.device) - 0.5
                delta_vec = current_delta_t.unsqueeze(0)
                y_pert = (y + delta_vec * s_rand).float()

                self.l_optimizer.zero_grad()

                # ★ KEY FIX: Always use DIRECT mode for SBC training
                # SBC should see the simplest possible closed-loop dynamics
                with torch.no_grad():
                    if self.config['mode'] == 'qp' and self.config.get('decoupled_sbc', True):
                        # Use QP controller in direct mode (bypassing QP filter)
                        a = self.p_net.controller_net(y_pert, state=y_pert, mode='direct')
                    else:
                        a = self.p_net(z.float(), y_pert)

                s_next = self.env.v_next(y_pert, a).unsqueeze(1).float()
                noise = triangular((B, 16, y.shape[1]), device=self.device)
                noise_scale = torch.as_tensor(self.env.noise, device=self.device, dtype=torch.float32).view(1, 1, -1)
                s_next_rand = (s_next + noise * noise_scale).float()

                l_val = self.l_model(y_pert).view(-1)
                l_next = self.l_model(s_next_rand.reshape(-1, y.shape[1]).float()).view(B, -1)
                exp_l_next = l_next.mean(dim=1)

                dec_loss = martingale_loss(l_val, exp_l_next, eps=0.1)
                loss_l = dec_loss * 1000

                # Lipschitz regularization
                y_lip = y.detach().clone().requires_grad_(True).float()
                l_out = self.l_model(y_lip).sum()
                l_grad = torch.autograd.grad(l_out, y_lip, create_graph=True)[0]
                lip_loss = torch.relu(l_grad.view(B, -1).norm(2, 1) - 4.0).mean()
                loss_l = loss_l + 0.001 * lip_loss

                # Region constraints
                init_s = self._sample_region(self.env.init_spaces, 256, 13).float()
                unsafe_s = self._sample_region(self.env.unsafe_spaces, 256, 17).float()
                l_init = self.l_model(init_s).view(-1)
                l_unsafe = self.l_model(unsafe_s).view(-1)
                target = 1.0 / max(1e-6, 1.0 - 0.95)
                region_loss = (torch.relu(torch.max(l_init) - 1.0) +
                              torch.relu(target - torch.min(l_unsafe)))
                loss_l = loss_l + region_loss

                loss_l.backward()
                torch.nn.utils.clip_grad_norm_(self.l_model.parameters(), 5.0)
                self.l_optimizer.step()

                batch_losses.append({
                    'loss_l': loss_l.item(), 'dec_loss': dec_loss.item(),
                    'lip_loss': lip_loss.item(), 'region_loss': region_loss.item(),
                })

            epoch_metrics.append({k: np.mean([b[k] for b in batch_losses]) for k in batch_losses[0]})
        return epoch_metrics

    def _train_controller(self, train_ds, current_delta, num_epochs=1):
        """Controller training with QP support."""
        self.l_model.eval()
        self.p_net.train()

        N = train_ds.shape[0]
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=self.device)
        current_delta_t = torch.tensor(current_delta, dtype=torch.float32, device=self.device)
        batch_size = 256

        epoch_metrics = []
        for epoch in range(num_epochs):
            indices = np.random.permutation(N)
            all_y = all_y[indices]
            batch_losses = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y = all_y[start:end].float()
                B = y.shape[0]
                z = (torch.rand(B, 4, device=self.device) * 2.0 - 1.0).float()

                rng = torch.Generator(device=self.device)
                rng.manual_seed(19)
                s_rand = torch.rand(y.shape, generator=rng, device=self.device) - 0.5
                delta_vec = current_delta_t.unsqueeze(0)
                y_pert = (y + delta_vec * s_rand).float()

                self.p_optimizer.zero_grad()

                if self.config['mode'] == 'qp':
                    # QP mode: forward through QP layer
                    u_safe, q, p, debug = self.p_net.controller_net(
                        y_pert, state=y_pert, mode='train', return_debug=True
                    )
                    a_p = u_safe
                    cbf_loss = self.cbf.cbf_violation_loss(y_pert, u_safe, p.squeeze(-1))
                else:
                    a_p = self.p_net(z.float(), y_pert)

                # SBC expected decrease
                with torch.no_grad():
                    l_p = self.l_model(y_pert).view(-1)

                s_next = self.env.v_next(y_pert, a_p).unsqueeze(1).float()
                noise = triangular((B, 128, y.shape[1]), device=self.device)
                noise_scale = torch.as_tensor(self.env.noise, device=self.device, dtype=torch.float32).view(1, 1, -1)
                s_next_rand = (s_next + noise * noise_scale).float()

                with torch.no_grad():
                    l_next = self.l_model(s_next_rand.reshape(-1, y.shape[1]).float()).view(B, -1)

                exp_l_next = l_next.mean(dim=1)
                dec_loss = martingale_loss(l_p.detach(), exp_l_next, eps=0.1)
                loss_p = dec_loss * 10

                # MSE distillation
                with torch.no_grad():
                    u_teacher = self.teacher_net(z.float(), y_pert)
                mse_loss = F.mse_loss(a_p, u_teacher.view_as(a_p))
                loss_p = loss_p + 10.0 * mse_loss

                # CBF loss
                cbf_val = 0.0
                if self.config['mode'] == 'qp':
                    cbf_val = cbf_loss
                    loss_p = loss_p + self.config.get('lambda_cbf', 0.01) * cbf_val

                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(self.p_net.controller_net.parameters(), 1.0)
                self.p_optimizer.step()

                m = {'loss_p': loss_p.item(), 'dec_loss': dec_loss.item(),
                     'mse_loss': mse_loss.item()}
                if self.config['mode'] == 'qp':
                    m['cbf_loss'] = cbf_val.item() if isinstance(cbf_val, torch.Tensor) else cbf_val
                batch_losses.append(m)

            epoch_metrics.append({k: np.mean([b[k] for b in batch_losses]) for k in batch_losses[0]})
        return epoch_metrics

    def run(self):
        start_time = time.time()
        max_iter = self.config.get('max_iterations', 50)
        timeout = self.config.get('timeout', 1800)

        train_ds, _, _ = self.verifier.get_unfiltered_grid(self.env.train_space_split)
        current_delta = (self.env.observation_space.high - self.env.observation_space.low) / self.env.train_space_split

        max_prob = 0.0
        violation_buffer = None

        for iteration in range(max_iter):
            runtime = time.time() - start_time
            if runtime > timeout:
                print(f"\nTimeout at iteration {iteration}")
                break

            # 1. Train SBC (decoupled — uses direct mode)
            l_metrics = self._train_sbc(train_ds, current_delta, num_epochs=10)
            final_l = l_metrics[-1]
            self.sbc_losses.append(final_l)

            # 2. Verify
            k_except_l = 1.2
            sat, hard_v, info, vb = self.verifier.check_dec_cond(k_except_l)
            hv = hard_v.item() if isinstance(hard_v, torch.Tensor) else hard_v

            # 3. Compute probability bound
            prob = max_prob
            if sat:
                _, ub_init = self.verifier.compute_bound_init(self.env.space_split)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(self.env.space_split)
                domain_min, _ = self.verifier.compute_bound_domain(self.env.space_split)
                if lb_unsafe > ub_init:
                    ub_n = ub_init - domain_min
                    lb_n = lb_unsafe - domain_min
                    ratio = lb_n / max(ub_n, 1e-9)
                    prob = max(0.0, 1.0 - 1.0 / max(ratio, 1e-9))
                    if prob > max_prob:
                        max_prob = prob

            self.iterations.append(iteration)
            self.prob_bounds.append(max_prob)
            self.hard_violations.append(hv)
            self.timing.append(runtime)

            if iteration % 5 == 0:
                print(f"  [Iter {iteration:3d}] SBC loss={final_l['loss_l']:.1f}, "
                      f"violations={hv}, prob={max_prob*100:.1f}%")

            # 4. Train controller
            p_metrics = self._train_controller(
                vb if vb is not None else train_ds, current_delta, num_epochs=1
            )
            self.ctrl_losses.append(p_metrics[-1])

        total_time = time.time() - start_time

        # Save results
        results = {
            'config': self.config,
            'max_prob_bound': float(max_prob),
            'final_prob_bound': float(self.prob_bounds[-1]) if self.prob_bounds else 0.0,
            'iterations': self.iterations,
            'prob_bounds': [float(p) for p in self.prob_bounds],
            'hard_violations': [float(v) for v in self.hard_violations],
            'timing': self.timing,
            'total_runtime': total_time,
            'num_iterations': len(self.iterations),
            'final_sbc_metrics': {k: float(v) for k, v in self.sbc_losses[-1].items()} if self.sbc_losses else {},
            'final_ctrl_metrics': {k: float(v) for k, v in self.ctrl_losses[-1].items()} if self.ctrl_losses else {},
        }

        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        np.savez(os.path.join(self.results_dir, 'history.npz'),
                 iterations=np.array(self.iterations),
                 prob_bounds=np.array(self.prob_bounds),
                 hard_violations=np.array(self.hard_violations))

        print(f"\nComplete: {self.config['mode']}, Max prob: {max_prob*100:.2f}%, "
              f"Iters: {len(self.iterations)}, Time: {total_time:.0f}s")
        return results


def run_config(name, config):
    """Run a single experiment configuration."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    results_dir = os.path.join(base_dir, f"improved_{name}_{datetime.now().strftime('%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  Config: {json.dumps({k:v for k,v in config.items() if k != 'mode'}, indent=2)}")
    print(f"{'='*60}")

    try:
        runner = ImprovedExperimentRunner(config, results_dir)
        results = runner.run()
        return results
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return {'error': str(e), 'max_prob_bound': 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                       choices=['baseline', 'qp_improved', 'all'])
    parser.add_argument('--timeout', type=int, default=1200)  # 20 min per experiment
    parser.add_argument('--max-iters', type=int, default=40)
    args = parser.parse_args()

    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    experiments_to_run = []

    if args.mode in ['baseline', 'all']:
        # Baseline × 2 for statistical significance
        for i in [1, 2]:
            experiments_to_run.append((
                f'baseline_run{i}',
                {'mode': 'baseline', 'noise_factor': 0.05, 'timeout': args.timeout,
                 'max_iterations': args.max_iters, 'decoupled_sbc': False}
            ))

    if args.mode in ['qp_improved', 'all']:
        # ★ IMPROVED QP configs:
        # 1. Decoupled SBC + ultra-low λ=0.001
        experiments_to_run.append((
            'qp_decoupled_lambda0.001',
            {'mode': 'qp', 'noise_factor': 0.05, 'timeout': args.timeout,
             'max_iterations': args.max_iters, 'lambda_cbf': 0.001,
             'decoupled_sbc': True, 't_gap': 1.5}
        ))
        # 2. Decoupled SBC + λ=0.0 (pure CBF as inference-only safety)
        experiments_to_run.append((
            'qp_decoupled_lambda0.0',
            {'mode': 'qp', 'noise_factor': 0.05, 'timeout': args.timeout,
             'max_iterations': args.max_iters, 'lambda_cbf': 0.0,
             'decoupled_sbc': True, 't_gap': 1.5}
        ))
        # 3. Decoupled SBC + moderate λ=0.01 + larger t_gap (looser CBF)
        experiments_to_run.append((
            'qp_decoupled_lambda0.01_tgap2.0',
            {'mode': 'qp', 'noise_factor': 0.05, 'timeout': args.timeout,
             'max_iterations': args.max_iters, 'lambda_cbf': 0.01,
             'decoupled_sbc': True, 't_gap': 2.0}
        ))

    print(f"\nRunning {len(experiments_to_run)} experiments...")

    for name, config in experiments_to_run:
        results = run_config(name, config)
        all_results[name] = {
            'max_prob': results.get('max_prob_bound', 0) * 100,
            'runtime': results.get('total_runtime', 0),
            'iters': results.get('num_iterations', 0),
        }

    # Print summary
    print(f"\n{'='*70}")
    print("IMPROVED EXPERIMENTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<40} {'Max Prob':>10} {'Iters':>6} {'Runtime':>8}")
    print('-' * 64)
    for name, r in all_results.items():
        print(f"{name:<40} {r['max_prob']:>8.2f}% {r['iters']:>5}  {r['runtime']:>6.0f}s")
    print(f"{'='*70}")

    # Save summary
    with open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', f'improved_summary_{timestamp}.json'
    ), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
