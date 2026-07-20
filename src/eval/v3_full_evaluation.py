"""
V3 Full Evaluation: Comprehensive Shield Comparison

Compares 5 safety configurations on diverse trajectories with multiple parameter settings:
- B (Baseline): No QP, raw controller output
- V3A: Fixed-p QP shield (p = constant, various values)
- V3B: SBC-modulated QP shield (p = f(B(s)))
- V3C: Trained p-network QP shield (p = NN(state))
- V3D: Barrier-modulated QP shield (p = f(b(s))) — direct barrier modulation

Features:
- 5 configs × N param variations × 2+ seeds × 100 scenarios
- Normal + adversarial test scenarios
- Detailed per-trajectory metrics + aggregate summaries
- Comparison tables for documentation

Usage:
    PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
    python /root/paper-combination/src/eval/v3_full_evaluation.py
"""

import sys, os, json, time, copy
import numpy as np
import torch
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG_DIR = os.path.join(ROOT, 'artical-F122')
sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, ROOT)
os.chdir(ORIG_DIR)

import h5py
from Aebs.VT.utils import MLP
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet
from stable_baselines3 import PPO
from src.models.cbf_constraints import AEBSCBFConstraints
from src.models.sbc_modulated_qp import SBCModulatedQPShield, FixedQPShield
from src.models.qp_p_network import QPParameterNetwork, TrainedQPShield


# ============================================================
# Model Loading
# ============================================================

def load_models(device='cpu'):
    """Load all pretrained models."""
    fn = "./Aebs/data/Downsampled.h5"
    with h5py.File(fn, 'r') as f:
        y_data = np.array(f["y_train"], dtype=np.float32)
    std1 = np.std(y_data)

    # Generator
    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(torch.load(
        "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth", map_location=device))
    gen_net.eval()

    # PPO Controller
    ppo = PPO.load('./Aebs/controller/best_model/best_model.zip')
    p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1],
                           ppo.policy.mlp_extractor.policy_net,
                           ppo.policy.action_net)
    p_net.state_net.load_state_dict(torch.load(
        "./Aebs/controller/state_net_trained.pth", map_location=device))
    p_net.to(device); p_net.eval()

    # SBC — try improved first, then original
    sbc = MLP([2, 32, 16, 8, 1], activation="tanh", square_output=True).to(device)

    improved_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc_improved.pth')
    original_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc.pth')

    sbc_loaded = False
    if os.path.exists(improved_path):
        sbc.load_state_dict(torch.load(improved_path, map_location=device))
        print(f"  Loaded improved SBC from {improved_path}")
        sbc_loaded = True
    elif os.path.exists(original_path):
        sbc.load_state_dict(torch.load(original_path, map_location=device))
        print(f"  Loaded SBC from {original_path}")
        sbc_loaded = True
    else:
        print("  WARNING: No trained SBC found")
    sbc.eval()

    # P-network — try to load trained version
    p_network = QPParameterNetwork(input_dim=2, hidden_dims=[64, 32, 16],
                                    p_min=0.1, p_max=4.0).to(device)
    pnet_path = os.path.join(ROOT, '实验v3', 'results', 'trained_p_network.pth')
    pnet_loaded = False
    if os.path.exists(pnet_path):
        checkpoint = torch.load(pnet_path, map_location=device)
        p_network.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained p-network from {pnet_path}")
        pnet_loaded = True
    else:
        print("  WARNING: No trained p-network found, using random weights")
    p_network.eval()

    return p_net, sbc, p_network, std1, sbc_loaded, pnet_loaded


# ============================================================
# Test Scenario Generation
# ============================================================

def generate_test_scenarios(n_normal=50, n_adversarial=50, seed=42):
    """
    Generate diverse test scenarios with explicit categories.

    Normal scenarios: standard driving situations
    Adversarial scenarios: edge cases, dangerous starts, noise injection
    """
    np.random.seed(seed)
    scenarios = []

    # === Normal Scenarios ===
    # Safe starts (far from lead vehicle)
    for _ in range(n_normal // 4):
        d = np.random.uniform(10, 16)
        v = np.random.uniform(0.5, 2.0)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'safe'})

    # Moderate starts
    for _ in range(n_normal // 4):
        d = np.random.uniform(7, 12)
        v = np.random.uniform(1.0, 2.5)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'moderate'})

    # Approaching scenarios (decelerating toward lead vehicle)
    for _ in range(n_normal // 4):
        d = np.random.uniform(8, 14)
        v = np.random.uniform(0.5, 1.5)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'approaching'})

    # Steady following
    for _ in range(n_normal // 4):
        d = np.random.uniform(6, 10)
        v = np.random.uniform(0.3, 1.5)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'following'})

    # === Adversarial Scenarios ===
    # Dangerous: close distance + high speed
    for _ in range(n_adversarial // 4):
        d = np.random.uniform(5.0, 6.5)
        v = np.random.uniform(2.0, 3.0)
        scenarios.append({'d': d, 'v': v, 'noise': 0, 'type': 'dangerous'})

    # Dangerous + noise: controller error
    for _ in range(n_adversarial // 4):
        d = np.random.uniform(5.5, 7.5)
        v = np.random.uniform(2.0, 3.0)
        noise = np.random.choice([1.0, 2.0, 3.0, -0.5])
        scenarios.append({'d': d, 'v': v, 'noise': noise, 'type': 'dangerous_noisy'})

    # High noise: moderate start + large perturbation
    for _ in range(n_adversarial // 4):
        d = np.random.uniform(7, 12)
        v = np.random.uniform(1.5, 2.5)
        noise = np.random.choice([2.0, 4.0, 5.0, -1.5])
        scenarios.append({'d': d, 'v': v, 'noise': noise, 'type': 'high_noise'})

    # Extreme: very close + very fast
    for _ in range(n_adversarial // 4):
        d = np.random.uniform(5.0, 6.0)
        v = np.random.uniform(2.5, 3.0)
        noise = np.random.choice([0, 0.5, 1.5, 3.0])
        scenarios.append({'d': d, 'v': v, 'noise': noise, 'type': 'extreme'})

    np.random.shuffle(scenarios)
    return scenarios


# ============================================================
# Shield Factory
# ============================================================

class BarrierModulatedQPShield:
    """
    V3D: QP shield with CBF parameter modulated directly by barrier function b(s).

    Unlike V3B which uses SBC B(s), this uses the raw barrier value:
    p = p_min + (p_max - p_min) * sigmoid(-b(s) / margin_scale)

    When b(s) is small (near unsafe) → p is large (tight constraint)
    When b(s) is large (safe) → p is small (relaxed constraint)

    This is more direct and reliable than SBC modulation because
    the barrier function is deterministic and well-defined everywhere.
    """
    def __init__(self, t_gap=1.5, p_min=0.1, p_max=4.0, margin_scale=2.0, device='cpu'):
        self.cbf = AEBSCBFConstraints(t_gap=t_gap, dt=0.05)
        self.p_min = p_min
        self.p_max = p_max
        self.margin_scale = margin_scale
        self.device = device

        self.intervention_count = 0
        self.total_steps = 0
        self.p_history = []

    def shield(self, u_ref, state, mode='analytic'):
        from src.models.sbc_modulated_qp import SBCModulatedQPShield

        batch_size = state.shape[0]
        if u_ref.dim() == 1:
            u_ref = u_ref.unsqueeze(-1)
        state = state.to(self.device)

        # Compute barrier-based p
        d = state[:, 0]
        v = state[:, 1]
        b = d - v * self.cbf.t_gap

        # sigmoid(-b/scale): when b is negative (unsafe) → 1 → p large
        #                    when b is positive (safe) → 0 → p small
        ratio = torch.sigmoid(-b / self.margin_scale)
        p = self.p_min + (self.p_max - self.p_min) * ratio

        # Build and solve QP
        G_cbf, h_cbf = self.cbf.build_constraints(state, p)

        G_lower = -torch.ones(batch_size, 1, 1, device=self.device)
        h_lower = 3.0 * torch.ones(batch_size, 1, device=self.device)
        G_upper = torch.ones(batch_size, 1, 1, device=self.device)
        h_upper = 3.0 * torch.ones(batch_size, 1, device=self.device)

        G = torch.cat([G_cbf, G_lower, G_upper], dim=1)
        h = torch.cat([h_cbf, h_lower, h_upper], dim=1)

        u_safe = SBCModulatedQPShield._solve_qp_1d_analytic(
            None, u_ref.to(self.device), G, h)

        intervened = (u_safe - u_ref.to(self.device)).abs() > 0.01
        sat, margin = self.cbf.check_cbf_satisfaction(state, u_safe, p)

        self.total_steps += batch_size
        self.intervention_count += intervened.sum().item()
        self.p_history.extend(p.cpu().tolist())

        info = {
            'p': p,
            'B': torch.zeros(batch_size, device=self.device),
            'intervened': intervened,
            'margin': margin,
        }
        return u_safe, info


def create_shields(sbc, p_network, device):
    """Create all shield variants for evaluation."""
    shields = {}

    # V3A: Fixed-p shields with various p values
    for fixed_p in [0.5, 1.0, 2.0, 3.0, 4.0]:
        for t_gap in [1.0, 1.5]:
            key = f'V3A_fixedP{fixed_p}_tgap{t_gap}'
            shields[key] = {
                'shield': FixedQPShield(sbc, fixed_p=fixed_p, t_gap=t_gap, device=device),
                'description': f'Fixed p={fixed_p}, t_gap={t_gap}'
            }

    # V3B: SBC-modulated shields with various parameters
    for p_min, T in [(0.1, 0.5), (0.1, 1.0), (0.5, 0.5), (0.5, 1.0)]:
        for t_gap in [1.0, 1.5]:
            key = f'V3B_pmin{p_min}_T{T}_tgap{t_gap}'
            shields[key] = {
                'shield': SBCModulatedQPShield(sbc, t_gap=t_gap, p_min=p_min, p_max=4.0,
                                                temperature=T, device=device),
                'description': f'SBC-mod p_min={p_min}, T={T}, t_gap={t_gap}'
            }

    # V3C: Trained p-network shields
    for t_gap in [1.0, 1.5]:
        key = f'V3C_TrainedP_tgap{t_gap}'
        cbf = AEBSCBFConstraints(t_gap=t_gap, dt=0.05)
        shields[key] = {
            'shield': TrainedQPShield(p_network, cbf, device=device),
            'description': f'Trained p-network, t_gap={t_gap}'
        }

    # V3D: Barrier-modulated shields
    for p_min, margin_scale in [(0.1, 1.5), (0.1, 2.0), (0.5, 2.0)]:
        for t_gap in [1.0, 1.5]:
            key = f'V3D_BarrierMod_pmin{p_min}_m{margin_scale}_tgap{t_gap}'
            shields[key] = {
                'shield': BarrierModulatedQPShield(t_gap=t_gap, p_min=p_min, p_max=4.0,
                                                    margin_scale=margin_scale, device=device),
                'description': f'Barrier-mod p_min={p_min}, m={margin_scale}, t_gap={t_gap}'
            }

    return shields


# ============================================================
# Evaluation Logic
# ============================================================

def evaluate_baseline(p_net, scenarios, std1, t_gap, device):
    """Evaluate baseline (no QP) on scenarios."""
    cbf = AEBSCBFConstraints(t_gap=t_gap, dt=0.05)
    results = []

    for sc in scenarios:
        d, v, noise_val = sc['d'], sc['v'], sc['noise']
        d_norm = d / std1
        z = torch.zeros(1, 4, device=device)
        s_input = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_ref = p_net(z, s_input).squeeze().cpu().item()

        u_actual = np.clip(u_ref + noise_val, -3.0, 3.0)
        b = d - v * t_gap

        results.append({
            'd': d, 'v': v, 'noise': noise_val, 'type': sc['type'],
            'u_ref': u_ref, 'u_adv': u_ref + noise_val, 'u_safe': u_actual,
            'intervened': False, 'cbf_safe': b >= 0,
            'p': 0, 'B': 0, 'margin': b, 'barrier': b,
        })

    return results


def evaluate_shield(shield, p_net, scenarios, std1, device):
    """Evaluate a QP shield on scenarios."""
    results = []

    for sc in scenarios:
        d, v, noise_val = sc['d'], sc['v'], sc['noise']
        d_norm = d / std1
        z = torch.zeros(1, 4, device=device)
        s_input = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)
        state_raw = torch.tensor([[d, v]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_ref = p_net(z, s_input).squeeze().cpu().item()

        u_adv = u_ref + noise_val
        u_adv_t = torch.tensor([[u_adv]], dtype=torch.float32, device=device)

        u_safe, info = shield.shield(u_adv_t, state_raw, mode='analytic')

        u_val = u_safe.squeeze().cpu().item()
        intervened = bool(abs(u_val - u_adv) > 0.01)
        b_val = d - v * shield.cbf.t_gap

        results.append({
            'd': d, 'v': v, 'noise': noise_val, 'type': sc['type'],
            'u_ref': u_ref, 'u_adv': u_adv, 'u_safe': u_val,
            'intervened': intervened,
            'cbf_safe': info['margin'].squeeze().cpu().item() >= 0,
            'p': float(info['p'].squeeze().cpu().item()),
            'B': float(info['B'].squeeze().cpu().item()) if info['B'] is not None else 0,
            'margin': float(info['margin'].squeeze().cpu().item()),
            'barrier': b_val,
        })

    return results


def compute_summary(results, name):
    """Compute comprehensive summary statistics."""
    n = len(results)
    if n == 0:
        return {'name': name, 'total_scenarios': 0}

    intervened = sum(1 for r in results if r['intervened'])
    cbf_safe = sum(1 for r in results if r['cbf_safe'])
    p_vals = [r['p'] for r in results]
    u_dev = [abs(r['u_safe'] - r['u_ref']) for r in results]
    margins = [r['margin'] for r in results]
    barriers = [r['barrier'] for r in results]

    # Safety violation: barrier < 0 means collision risk
    barrier_violations = sum(1 for b in barriers if b < 0)

    # By scenario type
    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = {'n': 0, 'intervened': 0, 'unsafe': 0, 'p_vals': [], 'margins': []}
        by_type[t]['n'] += 1
        if r['intervened']: by_type[t]['intervened'] += 1
        if not r['cbf_safe']: by_type[t]['unsafe'] += 1
        by_type[t]['p_vals'].append(r['p'])
        by_type[t]['margins'].append(r['margin'])

    type_summary = {}
    for t, v in by_type.items():
        type_summary[t] = {
            'n': v['n'],
            'intervention_rate': v['intervened'] / v['n'],
            'unsafe_rate': v['unsafe'] / v['n'],
            'avg_p': float(np.mean(v['p_vals'])),
            'std_p': float(np.std(v['p_vals'])),
            'avg_margin': float(np.mean(v['margins'])),
        }

    return {
        'name': name, 'total_scenarios': n,
        'intervention_rate': intervened / n,
        'cbf_safe_rate': cbf_safe / n,
        'barrier_violation_rate': barrier_violations / n,
        'avg_p': float(np.mean(p_vals)),
        'std_p': float(np.std(p_vals)),
        'avg_control_deviation': float(np.mean(u_dev)),
        'max_control_deviation': float(np.max(u_dev)),
        'avg_margin': float(np.mean(margins)),
        'avg_barrier': float(np.mean(barriers)),
        'min_barrier': float(np.min(barriers)),
        'by_type': type_summary,
    }


# ============================================================
# Main Experiment Runner
# ============================================================

def run_full_evaluation(n_runs=3, n_normal=60, n_adversarial=60, output_dir=None):
    """
    Run comprehensive V3 evaluation.

    Args:
        n_runs: Number of independent runs (different random seeds) per config
        n_normal: Number of normal test scenarios per run
        n_adversarial: Number of adversarial test scenarios per run
        output_dir: Directory for saving results
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: {n_runs} runs × N shields × {n_normal+n_adversarial} scenarios")
    print("=" * 70)

    # Load models
    print("\n[1/4] Loading models...")
    p_net, sbc, p_network, std1, sbc_loaded, pnet_loaded = load_models(device)

    # Check SBC calibration
    print("\n  SBC Calibration Check:")
    l_model = sbc
    l_model.eval()
    with torch.no_grad():
        test_cases_raw = torch.tensor([
            [15.0/std1, 0.5], [12.0/std1, 1.5], [9.0/std1, 2.0],
            [7.0/std1, 2.5], [5.5/std1, 3.0],
        ], device=device)
        B_test = l_model(test_cases_raw.float()).squeeze()
        for i, (label, d_raw, v_raw) in enumerate(zip(
            ["Very Safe", "Safe", "Moderate", "Risky", "Dangerous"],
            [15.0, 12.0, 9.0, 7.0, 5.5], [0.5, 1.5, 2.0, 2.5, 3.0]
        )):
            b_val = d_raw - v_raw * 1.5
            print(f"    {label:12s}: d={d_raw:.1f}, v={v_raw:.1f}, b={b_val:.1f}, B={B_test[i].item():.3f}")

    # Create shields
    print("\n[2/4] Creating shield configurations...")
    all_shields = create_shields(sbc, p_network, device)
    print(f"  {len(all_shields)} shield configurations created")

    # Select a representative subset for readable results
    # Focus on key comparisons
    selected_shields = {
        # V3A: Best fixed-p configurations
        'V3A_fixedP0.5_tgap1.5': all_shields['V3A_fixedP0.5_tgap1.5'],
        'V3A_fixedP2.0_tgap1.5': all_shields['V3A_fixedP2.0_tgap1.5'],
        'V3A_fixedP4.0_tgap1.5': all_shields['V3A_fixedP4.0_tgap1.5'],

        # V3B: Best SBC-modulated
        'V3B_pmin0.1_T0.5_tgap1.5': all_shields['V3B_pmin0.1_T0.5_tgap1.5'],
        'V3B_pmin0.5_T0.5_tgap1.5': all_shields['V3B_pmin0.5_T0.5_tgap1.5'],

        # V3C: Trained p-network
        'V3C_TrainedP_tgap1.5': all_shields['V3C_TrainedP_tgap1.5'],

        # V3D: Barrier-modulated
        'V3D_BarrierMod_pmin0.1_m2.0_tgap1.5': all_shields['V3D_BarrierMod_pmin0.1_m2.0_tgap1.5'],
        'V3D_BarrierMod_pmin0.5_m2.0_tgap1.5': all_shields['V3D_BarrierMod_pmin0.5_m2.0_tgap1.5'],
    }

    # Also include t_gap=1.0 variants for V3D
    if 'V3D_BarrierMod_pmin0.1_m2.0_tgap1.0' in all_shields:
        selected_shields['V3D_BarrierMod_pmin0.1_m2.0_tgap1.0'] = all_shields['V3D_BarrierMod_pmin0.1_m2.0_tgap1.0']

    print(f"  {len(selected_shields)} shields selected for detailed comparison")

    # Run experiments
    print(f"\n[3/4] Running experiments...")
    all_results = {}
    all_summaries = []

    # Baseline first (shared across runs)
    print("\n  Baseline (no QP)...")
    for run_id in range(1, n_runs + 1):
        seed = 100 + run_id
        scenarios = generate_test_scenarios(n_normal, n_adversarial, seed=seed)
        baseline_results = evaluate_baseline(p_net, scenarios, std1, 1.5, device)
        baseline_summary = compute_summary(baseline_results, f'Baseline_run{run_id}')
        all_summaries.append(baseline_summary)
        all_results[f'Baseline_run{run_id}'] = baseline_results
        print(f"    Run {run_id}: barrier_viol={baseline_summary['barrier_violation_rate']*100:.1f}%")

    # Each shield
    for shield_key, shield_info in selected_shields.items():
        print(f"\n  {shield_key}...")
        shield = shield_info['shield']

        for run_id in range(1, n_runs + 1):
            seed = 100 + run_id
            scenarios = generate_test_scenarios(n_normal, n_adversarial, seed=seed)
            shield_results = evaluate_shield(shield, p_net, scenarios, std1, device)
            shield_summary = compute_summary(shield_results, f'{shield_key}_run{run_id}')
            all_summaries.append(shield_summary)
            all_results[f'{shield_key}_run{run_id}'] = shield_results

            print(f"    Run {run_id}: intervene={shield_summary['intervention_rate']*100:.1f}%, "
                  f"safe={shield_summary['cbf_safe_rate']*100:.1f}%, "
                  f"avg_p={shield_summary['avg_p']:.3f}")

    # ============================================================
    # [4/4] Aggregate and Analyze
    # ============================================================
    print("\n[4/4] Aggregating results...")

    # Group by config name
    config_groups = {}
    for s in all_summaries:
        # Strip _runN suffix
        base = '_'.join(s['name'].split('_')[:-1]) if '_run' in s['name'] else s['name']
        if base not in config_groups:
            config_groups[base] = []
        config_groups[base].append(s)

    # Compute aggregate metrics
    aggregated = {}
    for name, summaries in config_groups.items():
        int_rates = [s['intervention_rate'] for s in summaries]
        safe_rates = [s['cbf_safe_rate'] for s in summaries]
        viol_rates = [s['barrier_violation_rate'] for s in summaries]
        p_vals = [s['avg_p'] for s in summaries]
        margins = [s['avg_margin'] for s in summaries]
        barriers = [s['avg_barrier'] for s in summaries]
        devs = [s['avg_control_deviation'] for s in summaries]

        # Per-type aggregation
        all_types = set()
        for s in summaries:
            all_types.update(s.get('by_type', {}).keys())
        type_agg = {}
        for t in all_types:
            t_ints = [s['by_type'][t]['intervention_rate'] for s in summaries if t in s.get('by_type', {})]
            t_unsafe = [s['by_type'][t]['unsafe_rate'] for s in summaries if t in s.get('by_type', {})]
            type_agg[t] = {
                'intervention_mean': float(np.mean(t_ints)) if t_ints else 0,
                'intervention_std': float(np.std(t_ints)) if t_ints else 0,
                'unsafe_mean': float(np.mean(t_unsafe)) if t_unsafe else 0,
            }

        aggregated[name] = {
            'n_runs': len(summaries),
            'intervention_rate_mean': float(np.mean(int_rates)),
            'intervention_rate_std': float(np.std(int_rates)),
            'cbf_safe_rate_mean': float(np.mean(safe_rates)),
            'cbf_safe_rate_std': float(np.std(safe_rates)),
            'barrier_violation_rate_mean': float(np.mean(viol_rates)),
            'avg_p_mean': float(np.mean(p_vals)),
            'avg_p_std': float(np.std(p_vals)),
            'avg_margin_mean': float(np.mean(margins)),
            'avg_barrier_mean': float(np.mean(barriers)),
            'min_barrier_min': float(min(s['min_barrier'] for s in summaries)),
            'avg_control_deviation_mean': float(np.mean(devs)),
            'by_type': type_agg,
        }

    # ============================================================
    # Print Results
    # ============================================================
    print("\n" + "=" * 100)
    print("V3 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 100)

    # Main comparison table
    header = f"{'Configuration':<35} {'Intervene%':>9} {'CBF Safe%':>9} {'BarrierVio%':>10} {'Avg p':>6} {'AvgMargin':>9} {'CtrlDev':>7}"
    print(header)
    print("-" * 100)

    baseline_key = 'Baseline'
    for name in sorted(aggregated.keys()):
        a = aggregated[name]
        if name == baseline_key:
            print(f"{name:<35} {'N/A':>9} {'N/A':>9} "
                  f"{a['barrier_violation_rate_mean']*100:>9.1f}% {'N/A':>6} "
                  f"{a['avg_margin_mean']:>9.2f} {a['avg_control_deviation_mean']:>7.4f}")
        else:
            print(f"{name:<35} {a['intervention_rate_mean']*100:>8.1f}% "
                  f"{a['cbf_safe_rate_mean']*100:>8.1f}% "
                  f"{a['barrier_violation_rate_mean']*100:>9.1f}% "
                  f"{a['avg_p_mean']:>5.3f} {a['avg_margin_mean']:>9.2f} "
                  f"{a['avg_control_deviation_mean']:>7.4f}")

    print("=" * 100)

    # By scenario type
    print(f"\n{'─' * 70}")
    print("By Scenario Type (Intervention Rate):")
    print(f"{'─' * 70}")
    all_types_set = set()
    for a in aggregated.values():
        all_types_set.update(a['by_type'].keys())
    all_types_sorted = sorted(all_types_set)

    type_header = f"{'Configuration':<35}"
    for t in all_types_sorted:
        type_header += f" {t:>15}"
    print(type_header)
    print("-" * len(type_header))

    for name in sorted(aggregated.keys()):
        a = aggregated[name]
        if name == baseline_key: continue
        row = f"{name:<35}"
        for t in all_types_sorted:
            if t in a['by_type']:
                row += f" {a['by_type'][t]['intervention_mean']*100:>14.1f}%"
            else:
                row += f" {'N/A':>15}"
        print(row)

    # ============================================================
    # Key Findings
    # ============================================================
    print(f"\n{'=' * 100}")
    print("KEY FINDINGS")
    print(f"{'=' * 100}")

    baseline_agg = aggregated.get(baseline_key, {})
    baseline_viol = baseline_agg.get('barrier_violation_rate_mean', 0)

    # Find best configuration
    best_safety = None
    best_safety_rate = 0
    best_intervention = None
    lowest_intervention = 1.0

    for name, a in aggregated.items():
        if name == baseline_key: continue
        if a['cbf_safe_rate_mean'] > best_safety_rate:
            best_safety_rate = a['cbf_safe_rate_mean']
            best_safety = name
        if a['intervention_rate_mean'] < lowest_intervention and a['barrier_violation_rate_mean'] <= baseline_viol:
            lowest_intervention = a['intervention_rate_mean']
            best_intervention = name

    print(f"\n  1. Best CBF Safety: {best_safety} ({best_safety_rate*100:.1f}%)")
    print(f"  2. Lowest Intervention (safe): {best_intervention} ({lowest_intervention*100:.1f}%)")

    # Compare V3 variants
    v3a_keys = [k for k in aggregated if k.startswith('V3A')]
    v3b_keys = [k for k in aggregated if k.startswith('V3B')]
    v3c_keys = [k for k in aggregated if k.startswith('V3C')]
    v3d_keys = [k for k in aggregated if k.startswith('V3D')]

    for label, keys in [('V3A Fixed-p', v3a_keys), ('V3B SBC-mod', v3b_keys),
                         ('V3C Trained-p', v3c_keys), ('V3D Barrier-mod', v3d_keys)]:
        if keys:
            avg_int = np.mean([aggregated[k]['intervention_rate_mean'] for k in keys])
            avg_safe = np.mean([aggregated[k]['cbf_safe_rate_mean'] for k in keys])
            avg_p = np.mean([aggregated[k]['avg_p_mean'] for k in keys])
            print(f"  3. {label}: avg intervene={avg_int*100:.1f}%, safe={avg_safe*100:.1f}%, p={avg_p:.3f}")

    # ============================================================
    # Save Results
    # ============================================================
    if output_dir is None:
        output_dir = os.path.join(ROOT, '实验v3', 'results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'config': {
            'n_runs': n_runs,
            'n_normal': n_normal,
            'n_adversarial': n_adversarial,
            'n_shields': len(selected_shields),
        },
        'model_status': {
            'sbc_loaded': sbc_loaded,
            'p_network_loaded': pnet_loaded,
        },
        'aggregated': aggregated,
        'config_groups': {name: list(s['name'] for s in summaries)
                         for name, summaries in config_groups.items()},
    }

    path = os.path.join(output_dir, f'v3_full_evaluation_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    return output


if __name__ == "__main__":
    run_full_evaluation(n_runs=3, n_normal=60, n_adversarial=60)
