"""
V3 Parameter Sweep: Multi-run experiments with varied configurations.

Tests V3A (fixed p) and V3B (SBC-modulated) across:
- Multiple CBF parameter values
- Different t_gap values
- Normal + adversarial trajectories
- 2+ runs per configuration

Usage:
    PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
    python /root/paper-combination/src/eval/v3_parameter_sweep.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG_DIR = os.path.join(ROOT, 'artical-F122')
sys.path.insert(0, ORIG_DIR); sys.path.insert(0, ROOT)
os.chdir(ORIG_DIR)

import h5py
from Aebs.VT.utils import MLP
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet
from stable_baselines3 import PPO
from src.models.sbc_modulated_qp import SBCModulatedQPShield, FixedQPShield
from src.models.cbf_constraints import AEBSCBFConstraints


def load_models(device='cpu'):
    """Load all pretrained models and trained SBC."""
    fn = "./Aebs/data/Downsampled.h5"
    with h5py.File(fn, 'r') as f:
        y_data = np.array(f["y_train"], dtype=np.float32)
    std1 = np.std(y_data)

    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(torch.load(
        "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth", map_location=device))

    ppo = PPO.load('./Aebs/controller/best_model/best_model.zip')
    p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1],
                           ppo.policy.mlp_extractor.policy_net,
                           ppo.policy.action_net)
    p_net.state_net.load_state_dict(torch.load(
        "./Aebs/controller/state_net_trained.pth", map_location=device))
    p_net.to(device); p_net.eval()

    sbc = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    sbc_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc.pth')
    sbc.load_state_dict(torch.load(sbc_path, map_location=device))
    sbc.eval()

    return p_net, sbc, std1


def generate_test_scenarios(n_normal=50, n_adversarial=50, seed=42):
    """Generate diverse test scenarios."""
    np.random.seed(seed)
    scenarios = []

    # Normal: safe starts
    for _ in range(n_normal // 2):
        d = np.random.uniform(8, 16); v = np.random.uniform(0.5, 2.5)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'safe'})

    # Normal: moderate starts
    for _ in range(n_normal // 2):
        d = np.random.uniform(5.5, 10); v = np.random.uniform(1.0, 2.5)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'moderate'})

    # Adversarial: dangerous starts + noise
    for _ in range(n_adversarial // 3):
        d = np.random.uniform(5.0, 7.0); v = np.random.uniform(2.0, 3.0)
        noise = np.random.choice([0, 2.0, 5.0, -2.0])
        scenarios.append({'d': d, 'v': v, 'noise': noise, 'type': 'dangerous'})

    # Adversarial: moderate starts + large noise
    for _ in range(n_adversarial // 3):
        d = np.random.uniform(7.0, 12); v = np.random.uniform(1.5, 3.0)
        noise = np.random.choice([3.0, 5.0, 7.0, -3.0])
        scenarios.append({'d': d, 'v': v, 'noise': noise, 'type': 'high_noise'})

    # Adversarial: extreme close + any noise
    for _ in range(n_adversarial // 3):
        d = np.random.uniform(5.0, 6.5); v = np.random.uniform(2.5, 3.0)
        noise = np.random.choice([0, 1.0, 3.0, 5.0])
        scenarios.append({'d': d, 'v': v, 'noise': noise, 'type': 'extreme'})

    return scenarios


def evaluate_shield(shield, p_net, std1, scenarios, device):
    """Evaluate a shield on given scenarios."""
    cbf = AEBSCBFConstraints(t_gap=shield.cbf.t_gap, dt=0.05)
    results = []

    for sc in scenarios:
        d, v, noise = sc['d'], sc['v'], sc['noise']
        d_norm = d / std1
        z = torch.zeros(1, 4, device=device)
        s_input = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)
        state_raw = torch.tensor([[d, v]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_ref = p_net(z, s_input).squeeze().cpu().item()

        u_adv = u_ref + noise
        u_adv_t = torch.tensor([[u_adv]], dtype=torch.float32, device=device)

        u_safe, info = shield.shield(u_adv_t, state_raw, mode='analytic')

        u_val = u_safe.squeeze().cpu().item()
        intervened = bool(abs(u_val - u_adv) > 0.01)

        sat, margin = cbf.check_cbf_satisfaction(
            state_raw.cpu(), torch.tensor([[u_val]]),
            torch.tensor([info['p'].squeeze().cpu().item()]))

        b_val = cbf.barrier_function(state_raw.cpu()).item()

        results.append({
            'd': d, 'v': v, 'noise': noise, 'type': sc['type'],
            'u_ref': u_ref, 'u_adv': u_adv, 'u_safe': u_val,
            'intervened': intervened, 'cbf_safe': bool(sat.item()),
            'margin': float(margin.item()) if margin is not None else 0,
            'p': float(info['p'].item()), 'B': float(info['B'].item()),
            'barrier': b_val,
        })

    return results


def compute_summary(results, name):
    """Compute summary statistics from evaluation results."""
    n = len(results)
    intervened = sum(1 for r in results if r['intervened'])
    cbf_safe = sum(1 for r in results if r['cbf_safe'])
    avg_p = np.mean([r['p'] for r in results])
    std_p = np.std([r['p'] for r in results])
    avg_B = np.mean([r['B'] for r in results])
    avg_margin = np.mean([r['margin'] for r in results])
    avg_barrier = np.mean([r['barrier'] for r in results])

    # By type
    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = {'n': 0, 'intervened': 0, 'cbf_safe': 0, 'avg_p': [], 'avg_u': []}
        by_type[t]['n'] += 1
        if r['intervened']: by_type[t]['intervened'] += 1
        if r['cbf_safe']: by_type[t]['cbf_safe'] += 1
        by_type[t]['avg_p'].append(r['p'])
        by_type[t]['avg_u'].append(abs(r['u_safe']))

    type_summary = {}
    for t, v in by_type.items():
        type_summary[t] = {
            'n': v['n'],
            'intervention_rate': v['intervened'] / v['n'],
            'cbf_safe_rate': v['cbf_safe'] / v['n'],
            'avg_p': float(np.mean(v['avg_p'])),
            'avg_abs_u': float(np.mean(v['avg_u'])),
        }

    return {
        'name': name, 'total_scenarios': n,
        'intervention_rate': intervened / n,
        'cbf_safe_rate': cbf_safe / n,
        'avg_p': float(avg_p), 'std_p': float(std_p),
        'avg_B': float(avg_B), 'avg_margin': float(avg_margin),
        'avg_barrier': float(avg_barrier),
        'by_type': type_summary,
    }


def run_sweep():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    p_net, sbc, std1 = load_models(device)
    print(f"Models loaded. SBC trained (95.9% prob bound)")

    # Generate scenarios
    scenarios = generate_test_scenarios(n_normal=50, n_adversarial=50)
    print(f"Scenarios: {len(scenarios)} total")

    # === Parameter configurations ===
    configs = []

    # V3A: fixed_p variations
    for fixed_p in [0.5, 1.0, 2.0, 3.0, 4.0]:
        for t_gap in [1.5]:
            configs.append({
                'name': f'V3A_fixedP_{fixed_p}_tgap_{t_gap}',
                'type': 'v3a',
                'fixed_p': fixed_p, 't_gap': t_gap,
            })

    # V3B: SBC-modulated variations
    for p_min, p_max, T in [(0.1, 4.0, 0.5), (0.5, 4.0, 1.0),
                              (1.0, 4.0, 2.0), (0.1, 2.0, 0.5),
                              (0.1, 4.0, 0.1), (0.1, 4.0, 2.0)]:
        for t_gap in [1.5]:
            configs.append({
                'name': f'V3B_pmin{p_min}_pmax{p_max}_T{T}_tgap{t_gap}',
                'type': 'v3b',
                'p_min': p_min, 'p_max': p_max, 'T': T,
                't_gap': t_gap,
            })

    # V3 with different t_gap
    for t_gap in [1.0, 2.0, 3.0]:
        configs.append({
            'name': f'V3A_fixedP_2.0_tgap_{t_gap}',
            'type': 'v3a', 'fixed_p': 2.0, 't_gap': t_gap,
        })
        configs.append({
            'name': f'V3B_pmin0.1_pmax4_T0.5_tgap_{t_gap}',
            'type': 'v3b', 'p_min': 0.1, 'p_max': 4.0,
            'T': 0.5, 't_gap': t_gap,
        })

    # Baseline reference
    configs.insert(0, {
        'name': 'Baseline_NoQP',
        'type': 'baseline', 't_gap': 1.5,
    })

    print(f"Configurations: {len(configs)}")

    # Run 2 passes per config with different random seeds
    all_summaries = []
    all_details = {}

    for cfg in configs:
        for run_id in [1, 2]:
            seed = 100 + run_id
            sc = generate_test_scenarios(n_normal=50, n_adversarial=50, seed=seed)
            name = f"{cfg['name']}_run{run_id}"

            if cfg['type'] == 'baseline':
                # No QP, just clip
                results = []
                for s in sc:
                    d, v, noise = s['d'], s['v'], s['noise']
                    d_norm = d / std1
                    z = torch.zeros(1, 4, device=device)
                    s_input = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        u_ref = p_net(z, s_input).squeeze().cpu().item()
                    u = np.clip(u_ref + noise, -3.0, 3.0)
                    cbf = AEBSCBFConstraints(t_gap=cfg['t_gap'])
                    b = cbf.barrier_function(torch.tensor([[d, v]])).item()
                    results.append({
                        'd': d, 'v': v, 'noise': noise, 'type': s['type'],
                        'u_ref': u_ref, 'u_adv': u_ref + noise, 'u_safe': u,
                        'intervened': False, 'cbf_safe': True,
                        'p': 0, 'B': 0, 'margin': 0, 'barrier': b,
                    })
                name_full = name

            elif cfg['type'] == 'v3a':
                shield = FixedQPShield(sbc, fixed_p=cfg['fixed_p'],
                                       t_gap=cfg['t_gap'], device=device)
                results = evaluate_shield(shield, p_net, std1, sc, device)
                name_full = name

            elif cfg['type'] == 'v3b':
                shield = SBCModulatedQPShield(sbc, t_gap=cfg['t_gap'],
                    p_min=cfg['p_min'], p_max=cfg['p_max'],
                    temperature=cfg['T'], device=device)
                results = evaluate_shield(shield, p_net, std1, sc, device)
                name_full = name

            summary = compute_summary(results, name_full)
            all_summaries.append(summary)
            all_details[name_full] = results

            print(f"  {name_full}: intervene={summary['intervention_rate']*100:.1f}%, "
                  f"safe={summary['cbf_safe_rate']*100:.1f}%, "
                  f"avg_p={summary['avg_p']:.2f}")

    # === Aggregate by config (average over runs) ===
    print("\n" + "=" * 85)
    print("V3 PARAMETER SWEEP — AGGREGATED RESULTS (2 runs each)")
    print("=" * 85)

    # Group by base name
    groups = {}
    for s in all_summaries:
        base = s['name'].rsplit('_run', 1)[0]
        if base not in groups:
            groups[base] = []
        groups[base].append(s)

    print(f"{'Configuration':<40} {'Intervene%':>10} {'Safe%':>8} {'Avg p':>7} {'Std p':>7} {'Avg|u|':>7}")
    print("-" * 85)

    aggregated = []
    for base, items in sorted(groups.items()):
        int_rates = [i['intervention_rate'] for i in items]
        safe_rates = [i['cbf_safe_rate'] for i in items]
        avg_ps = [i['avg_p'] for i in items]

        int_mean = np.mean(int_rates) * 100
        int_std = np.std(int_rates) * 100
        safe_mean = np.mean(safe_rates) * 100
        p_mean = np.mean(avg_ps)
        p_std = np.std(avg_ps)

        print(f"{base:<40} {int_mean:>7.1f}%±{int_std:.0f} {safe_mean:>6.1f}% "
              f"{p_mean:>6.2f} {p_std:>6.2f}")

        aggregated.append({
            'config': base,
            'intervention_mean': int_mean,
            'intervention_std': int_std,
            'safe_mean': safe_mean,
            'p_mean': p_mean,
            'p_std': p_std,
            'runs': len(items),
            'details': [i for i in items],
        })

    print("=" * 85)

    # === Save results ===
    out_dir = os.path.join(ROOT, '实验v3', 'results')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'num_configs': len(configs),
        'runs_per_config': 2,
        'scenarios_per_run': len(scenarios),
        'aggregated': aggregated,
        'all_summaries': all_summaries,
    }
    path = os.path.join(out_dir, f'v3_sweep_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {path}")
    return output


if __name__ == "__main__":
    run_sweep()
