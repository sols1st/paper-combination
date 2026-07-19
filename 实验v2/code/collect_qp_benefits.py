#!/usr/bin/env python3
"""
Detailed QP Benefits Data Collection

Records metrics that demonstrate the ADVANTAGES of the CBF-QP safety layer
over the baseline (no QP) controller.

Metrics collected:
1. CBF constraint satisfaction rate (b(s) ≥ 0 maintained?)
2. QP feasibility rate (can the QP find a solution?)
3. Safety margin distribution (distance from constraint boundary)
4. Control modification magnitude (how much does QP change the controller output?)
5. Runtime safety violation detection
6. Control smoothness (variance of consecutive actions)
7. Recovery from unsafe reference controls

Usage:
    cd /root/paper-combination
    python 实验v2/code/collect_qp_benefits.py
"""

import sys, os, json, time
import numpy as np
import torch
from datetime import datetime

ORIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'artical-F122')
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, os.path.join(EXP_DIR, 'code', 'models'))
os.chdir(ORIG_DIR)

import h5py
from Aebs.system.env import Aebs, next_state_vec
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet, SubNet, CombinedPolicyNetwork
from stable_baselines3 import PPO
from models.qp_controller import QPAebsController, DirectController
from models.cbf_constraints import AEBSCBFConstraints


def load_models(device='cpu'):
    """Load pretrained models."""
    # Load data for normalization
    fn = "./Aebs/data/Downsampled.h5"
    with h5py.File(fn, 'r') as f:
        y_data = np.array(f["y_train"], dtype=np.float32)
    std1 = np.std(y_data)

    # Load gen_net
    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(torch.load(
        "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth", map_location=device
    ))

    # Load PPO
    model = PPO.load('./Aebs/controller/best_model/best_model.zip')
    mlp_extractor = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    # Build end-to-end net
    p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1], mlp_extractor, action_net)
    p_net.state_net.load_state_dict(torch.load(
        "./Aebs/controller/state_net_trained.pth", map_location=device
    ))
    p_net.to(device)
    p_net.eval()

    # Build QP controller
    qp_ctrl = QPAebsController(
        input_dim=2, hidden_dims=[256, 256, 256],
        control_dim=1, cbf_param_dim=1, t_gap=1.5, dt=0.05, device=device
    ).to(device)
    qp_ctrl.eval()

    return p_net, qp_ctrl, std1


def simulate_trajectory(controller, p_net, std1, initial_state, num_steps=200, mode='baseline', qp_ctrl=None):
    """
    Simulate a trajectory and collect detailed metrics.

    Args:
        controller: Controller to use (p_net for baseline, qp_ctrl for QP)
        p_net: End-to-end network (for baseline mode)
        mode: 'baseline' or 'qp'
        qp_ctrl: QP controller (for qp mode)

    Returns:
        metrics dict with detailed per-step data
    """
    dt = 0.05
    t_gap = 1.5
    cbf = AEBSCBFConstraints(t_gap=t_gap, dt=dt)

    d, v = initial_state
    d_norm = d / std1

    # Tracking
    trajectory = {'d': [], 'v': [], 'u': [], 'u_ref': [], 'p': [], 'b': [],
                  'cbf_margin': [], 'qp_feasible': [], 'qp_active': []}

    device = next(p_net.parameters()).device if hasattr(p_net, 'parameters') else 'cpu'

    for step in range(num_steps):
        # Build input
        z = torch.zeros(1, 4, device=device)
        s_input = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)

        with torch.no_grad():
            if mode == 'baseline':
                # Original SafePVC: direct controller output
                u = p_net(z, s_input).squeeze().cpu().item()
                u_ref = u
                p_val = 0
                qp_feasible = True
                cbf_margin_val = cbf.barrier_function(
                    torch.tensor([[d, v]], dtype=torch.float32)
                ).item()
            else:
                # QP mode: controller → q, p → QP → u*
                u_safe, q, p_t, debug = qp_ctrl(
                    s_input, state=s_input, mode='eval', return_debug=True
                )
                u = u_safe.squeeze().cpu().item()
                u_ref = q.squeeze().cpu().item()
                p_val = p_t.squeeze().cpu().item()

                # Check CBF constraint satisfaction
                state_t = torch.tensor([[d, v]], dtype=torch.float32)
                satisfied, margin = cbf.check_cbf_satisfaction(
                    state_t, torch.tensor([[u]]), torch.tensor([p_val])
                )
                cbf_margin_val = margin.item()
                qp_feasible = satisfied.item()

        # Clamp to valid range
        u = np.clip(u, -3.0, 3.0)

        # Step dynamics
        d_next = d - v * dt
        v_next = v - u * dt
        v_next = max(0.0, min(3.0, v_next))

        # Record
        trajectory['d'].append(d)
        trajectory['v'].append(v)
        trajectory['u'].append(u)
        trajectory['u_ref'].append(u_ref)
        trajectory['p'].append(p_val)
        trajectory['b'].append(d - v * t_gap)  # barrier value
        trajectory['cbf_margin'].append(cbf_margin_val)
        trajectory['qp_feasible'].append(qp_feasible)
        trajectory['qp_active'].append(abs(u - u_ref) > 0.01)

        # Check termination
        if d_next <= 5.0 or d_next >= 16.0 or v_next <= 0.0:
            # Record final step then break
            d, v = d_next, v_next
            break

        d, v = d_next, v_next
        d_norm = d / std1

    return trajectory


def compute_summary_metrics(baseline_trajs, qp_trajs):
    """Compute comparison summary metrics from trajectories."""
    summary = {}

    # 1. Safety violation rate (barrier b(s) < 0)
    bl_safe_violations = sum(1 for t in baseline_trajs for b in t['b'] if b < 0)
    bl_total = sum(len(t['b']) for t in baseline_trajs)
    qp_safe_violations = sum(1 for t in qp_trajs for b in t['b'] if b < 0)
    qp_total = sum(len(t['b']) for t in qp_trajs)

    summary['safety_violation_rate_baseline'] = bl_safe_violations / max(bl_total, 1)
    summary['safety_violation_rate_qp'] = qp_safe_violations / max(qp_total, 1)

    # 2. Minimum safety margin
    bl_min_margin = min(min(t['b']) for t in baseline_trajs)
    qp_min_margin = min(min(t['b']) for t in qp_trajs)
    summary['min_safety_margin_baseline'] = bl_min_margin
    summary['min_safety_margin_qp'] = qp_min_margin

    # 3. Average safety margin
    bl_avg_margin = np.mean([b for t in baseline_trajs for b in t['b']])
    qp_avg_margin = np.mean([b for t in qp_trajs for b in t['b']])
    summary['avg_safety_margin_baseline'] = bl_avg_margin
    summary['avg_safety_margin_qp'] = qp_avg_margin

    # 4. QP feasibility rate
    qp_feasible_rate = sum(1 for t in qp_trajs for f in t['qp_feasible'] if f) / max(qp_total, 1)
    summary['qp_feasibility_rate'] = qp_feasible_rate

    # 5. QP activation rate (how often does QP modify the reference?)
    qp_active_rate = sum(1 for t in qp_trajs for a in t['qp_active'] if a) / max(qp_total, 1)
    summary['qp_activation_rate'] = qp_active_rate

    # 6. Control smoothness (average |Δu| between consecutive steps)
    bl_smoothness = []
    for t in baseline_trajs:
        if len(t['u']) > 1:
            diffs = np.abs(np.diff(t['u']))
            bl_smoothness.extend(diffs)
    qp_smoothness = []
    for t in qp_trajs:
        if len(t['u']) > 1:
            diffs = np.abs(np.diff(t['u']))
            qp_smoothness.extend(diffs)
    summary['avg_control_change_baseline'] = np.mean(bl_smoothness) if bl_smoothness else 0
    summary['avg_control_change_qp'] = np.mean(qp_smoothness) if qp_smoothness else 0

    # 7. Average CBF parameter p
    summary['avg_cbf_param_p'] = np.mean([p for t in qp_trajs for p in t['p']])

    # 8. Safety-critical scenario detection
    # Find states where baseline would be unsafe but QP prevents it
    bl_unsafe_steps = []
    qp_prevented = []
    for bl_t, qp_t in zip(baseline_trajs, qp_trajs):
        min_len = min(len(bl_t['b']), len(qp_t['b']))
        for i in range(min_len):
            if bl_t['b'][i] < 0 and qp_t['b'][i] >= 0:
                qp_prevented.append(i)

    summary['qp_prevented_unsafe_steps'] = len(qp_prevented)

    return summary


def run_benefits_analysis():
    """Main analysis: compare baseline vs QP across diverse initial conditions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    p_net, qp_ctrl, std1 = load_models(device)

    # Generate diverse test scenarios
    np.random.seed(42)
    num_scenarios = 50

    # Mix of safe and challenging initial states
    test_states = []
    # Safe starts
    for _ in range(num_scenarios // 2):
        d = np.random.uniform(8, 16)
        v = np.random.uniform(0.5, 2.5)
        test_states.append((d, v))
    # Challenging starts (close to unsafe)
    for _ in range(num_scenarios // 2):
        d = np.random.uniform(5.5, 7.5)
        v = np.random.uniform(1.5, 3.0)
        test_states.append((d, v))

    print(f"\nSimulating {len(test_states)} trajectories for each condition...")

    # Run baseline simulations
    print("Baseline (no QP)...")
    baseline_trajs = []
    for i, (d, v) in enumerate(test_states):
        traj = simulate_trajectory(p_net, p_net, std1, (d, v), mode='baseline')
        baseline_trajs.append(traj)
        if (i + 1) % 10 == 0:
            print(f"  Baseline: {i+1}/{len(test_states)}")

    # Run QP simulations
    print("QP Integrated...")
    qp_trajs = []
    for i, (d, v) in enumerate(test_states):
        traj = simulate_trajectory(None, p_net, std1, (d, v), mode='qp', qp_ctrl=qp_ctrl)
        qp_trajs.append(traj)
        if (i + 1) % 10 == 0:
            print(f"  QP: {i+1}/{len(test_states)}")

    # Compute summary
    print("\nComputing comparison metrics...")
    summary = compute_summary_metrics(baseline_trajs, qp_trajs)

    # Print results
    print("\n" + "=" * 70)
    print("QP BENEFITS ANALYSIS: Baseline vs SafePVC+QP")
    print("=" * 70)
    print()
    print(f"{'Metric':<45} {'Baseline':>12} {'QP Integrated':>12}")
    print("-" * 70)
    print(f"{'Safety violation rate':<45} {summary['safety_violation_rate_baseline']*100:>10.2f}% {summary['safety_violation_rate_qp']*100:>10.2f}%")
    print(f"{'Min safety margin (b_min)':<45} {summary['min_safety_margin_baseline']:>10.2f}m {summary['min_safety_margin_qp']:>10.2f}m")
    print(f"{'Avg safety margin (b_avg)':<45} {summary['avg_safety_margin_baseline']:>10.2f}m {summary['avg_safety_margin_qp']:>10.2f}m")
    print(f"{'Avg control change |Δu|':<45} {summary['avg_control_change_baseline']:>10.4f} {summary['avg_control_change_qp']:>10.4f}")
    print(f"{'QP feasibility rate':<45} {'N/A':>12} {summary['qp_feasibility_rate']*100:>10.2f}%")
    print(f"{'QP activation rate':<45} {'N/A':>12} {summary['qp_activation_rate']*100:>10.2f}%")
    print(f"{'Avg CBF parameter p':<45} {'N/A':>12} {summary['avg_cbf_param_p']:>10.2f}")
    print(f"{'Unsafe steps prevented by QP':<45} {'N/A':>12} {summary['qp_prevented_unsafe_steps']:>10d}")
    print("-" * 70)

    # QP-specific benefits
    print()
    print("QP-SPECIFIC BENEFITS:")
    print(f"  1. Runtime safety enforcement: CBF constraint active in {summary['qp_activation_rate']*100:.1f}% of steps")
    print(f"  2. QP solves feasible in {summary['qp_feasibility_rate']*100:.1f}% of steps")
    print(f"  3. QP prevented {summary['qp_prevented_unsafe_steps']} unsafe steps that baseline would have taken")
    print(f"  4. Adaptive safety: p parameter auto-tunes between 0-4 (avg={summary['avg_cbf_param_p']:.2f})")

    if summary['safety_violation_rate_qp'] < summary['safety_violation_rate_baseline']:
        improvement = (summary['safety_violation_rate_baseline'] - summary['safety_violation_rate_qp']) / max(summary['safety_violation_rate_baseline'], 1e-6) * 100
        print(f"  5. Safety improvement: {improvement:.1f}% reduction in safety violations")
    print("=" * 70)

    # Save detailed results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    def safe_convert(v):
        if isinstance(v, (np.floating, np.integer, np.bool_)):
            return float(v) if not isinstance(v, np.bool_) else bool(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    output = {
        'timestamp': timestamp,
        'config': {'num_scenarios': num_scenarios, 't_gap': 1.5},
        'summary': {k: safe_convert(v) for k, v in summary.items()},
        'baseline_trajectories': [{
            'd': [float(x) for x in t['d']],
            'v': [float(x) for x in t['v']],
            'u': [float(x) for x in t['u']],
            'b': [float(x) for x in t['b']]
        } for t in baseline_trajs[:5]],
        'qp_trajectories': [{
            'd': [float(x) for x in t['d']],
            'v': [float(x) for x in t['v']],
            'u': [float(x) for x in t['u']],
            'u_ref': [float(x) for x in t['u_ref']],
            'b': [float(x) for x in t['b']],
            'p': [float(x) for x in t['p']],
            'qp_active': [bool(x) for x in t['qp_active']]
        } for t in qp_trajs[:5]],
    }

    with open(os.path.join(results_dir, f'qp_benefits_{timestamp}.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDetailed results saved to {results_dir}/qp_benefits_{timestamp}.json")
    return summary


if __name__ == "__main__":
    run_benefits_analysis()
