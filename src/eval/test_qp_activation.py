"""
Adversarial test: Demonstrate QP shield activation by injecting
deliberately unsafe control perturbations.

Tests V3A (fixed p) and V3B (SBC-modulated p) shields against
noisy/adversarial controller outputs.

Usage:
    PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
    python /root/paper-combination/src/eval/test_qp_activation.py
"""

import sys, os, json, time
import numpy as np
import torch

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


def adversarial_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load models
    print("Loading models...")
    fn = "./Aebs/data/Downsampled.h5"
    with h5py.File(fn, 'r') as f:
        y_data = np.array(f["y_train"], dtype=np.float32)
    std1 = np.std(y_data)

    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(torch.load(
        "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth", map_location=device))
    ppo = PPO.load('./Aebs/controller/best_model/best_model.zip')
    mlp_e = ppo.policy.mlp_extractor.policy_net
    act_n = ppo.policy.action_net
    p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1], mlp_e, act_n)
    p_net.state_net.load_state_dict(torch.load(
        "./Aebs/controller/state_net_trained.pth", map_location=device))
    p_net.to(device); p_net.eval()

    # Load trained SBC
    sbc = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    sbc_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc.pth')
    sbc.load_state_dict(torch.load(sbc_path, map_location=device))
    sbc.eval()
    print(f"  Trained SBC loaded (prob bound: ~95.9%)")

    # Build shields
    shield_v3a = FixedQPShield(sbc, fixed_p=2.0, t_gap=1.5, device=device)
    shield_v3b = SBCModulatedQPShield(sbc, t_gap=1.5, p_min=0.1, p_max=4.0,
                                       temperature=0.5, device=device)

    cbf = AEBSCBFConstraints(t_gap=1.5, dt=0.05)

    # Test scenarios: deliberately DANGEROUS states + noise injection
    np.random.seed(123)
    test_cases = []

    # Case 1: Normal states with ADVERSARIAL control noise
    # (simulating OOD controller behavior)
    for _ in range(40):
        d = np.random.uniform(6.0, 14.0)
        v = np.random.uniform(0.5, 3.0)
        # Inject noise: add -5 to +5 to the acceleration
        noise_level = np.random.choice([0, 1.0, 3.0, 5.0])
        test_cases.append({
            'd': d, 'v': v, 'noise': noise_level,
            'type': f'noise_{noise_level:.0f}'
        })

    # Case 2: Very dangerous states (close to obstacle, high speed)
    for _ in range(30):
        d = np.random.uniform(5.0, 6.5)
        v = np.random.uniform(2.0, 3.0)
        test_cases.append({
            'd': d, 'v': v, 'noise': 0,
            'type': 'dangerous'
        })

    # Case 3: Edge of safety boundary
    for _ in range(30):
        d = np.random.uniform(5.5, 8.0)
        v = np.random.uniform(2.0, 3.0)
        test_cases.append({
            'd': d, 'v': v, 'noise': np.random.uniform(-2.0, 2.0),
            'type': 'boundary'
        })

    print(f"\nTesting {len(test_cases)} adversarial scenarios...")

    results = {'v3a': [], 'v3b': [], 'no_shield': []}

    for i, tc in enumerate(test_cases):
        d, v, noise = tc['d'], tc['v'], tc['noise']
        d_norm = d / std1

        z = torch.zeros(1, 4, device=device)
        s_input = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)
        state_raw = torch.tensor([[d, v]], dtype=torch.float32, device=device)

        with torch.no_grad():
            u_ref = p_net(z, s_input).squeeze().cpu().item()

        # Inject adversarial noise
        u_adv = u_ref + noise
        u_adv_t = torch.tensor([[u_adv]], dtype=torch.float32, device=device)

        # Test V3A (fixed p=2.0)
        u_v3a, info_v3a = shield_v3a.shield(u_adv_t, state_raw, mode='cvxopt')
        u_v3a_val = u_v3a.squeeze().cpu().item()

        # Test V3B (SBC-modulated)
        u_v3b, info_v3b = shield_v3b.shield(u_adv_t, state_raw, mode='cvxopt')
        u_v3b_val = u_v3b.squeeze().cpu().item()

        # No shield: clip to [-3, 3] (worst case)
        u_noshield = np.clip(u_adv, -3.0, 3.0)

        # Check CBF satisfaction for each
        sat_no, _ = cbf.check_cbf_satisfaction(
            state_raw.cpu(), torch.tensor([[u_noshield]]), torch.tensor([2.0]))
        sat_v3a, _ = cbf.check_cbf_satisfaction(
            state_raw.cpu(), torch.tensor([[u_v3a_val]]), torch.tensor([2.0]))
        sat_v3b, _ = cbf.check_cbf_satisfaction(
            state_raw.cpu(), torch.tensor([[u_v3b_val]]),
            torch.tensor([info_v3b['p'].squeeze().cpu().item()]))

        results['no_shield'].append({
            'type': tc['type'], 'd': d, 'v': v, 'noise': noise,
            'u_ref': u_ref, 'u_adv': u_adv, 'u_out': u_noshield,
            'cbf_safe': bool(sat_no.item())
        })
        results['v3a'].append({
            'type': tc['type'], 'd': d, 'v': v, 'noise': noise,
            'u_ref': u_ref, 'u_adv': u_adv, 'u_out': u_v3a_val,
            'intervened': bool(info_v3a['intervened'].item()),
            'cbf_safe': bool(sat_v3a.item()),
            'p': 2.0, 'B': float(info_v3a['B'].item()),
        })
        results['v3b'].append({
            'type': tc['type'], 'd': d, 'v': v, 'noise': noise,
            'u_ref': u_ref, 'u_adv': u_adv, 'u_out': u_v3b_val,
            'intervened': bool(info_v3b['intervened'].item()),
            'cbf_safe': bool(sat_v3b.item()),
            'p': float(info_v3b['p'].item()),
            'B': float(info_v3b['B'].item()),
        })

    # === Summary ===
    print("\n" + "=" * 75)
    print("ADVERSARIAL TEST: Does QP Shield Activate Under Attack?")
    print("=" * 75)

    for label, data in [('No Shield (clipped)', results['no_shield']),
                         ('V3A Fixed p=2.0', results['v3a']),
                         ('V3B SBC-Modulated', results['v3b'])]:
        intervened = sum(1 for r in data if r.get('intervened', False))
        cbf_safe = sum(1 for r in data if r.get('cbf_safe', False))
        total = len(data)

        interventions_by_noise = {}
        for r in data:
            n = r.get('noise', 0)
            if n not in interventions_by_noise:
                interventions_by_noise[n] = {'total': 0, 'intervened': 0, 'cbf_safe': 0}
            interventions_by_noise[n]['total'] += 1
            if r.get('intervened'): interventions_by_noise[n]['intervened'] += 1
            if r.get('cbf_safe'): interventions_by_noise[n]['cbf_safe'] += 1

        avg_p = np.mean([r.get('p', 0) for r in data])

        print(f"\n{label}:")
        print(f"  Total: {total}, Intervened: {intervened} ({intervened/total*100:.1f}%), "
              f"CBF Safe: {cbf_safe} ({cbf_safe/total*100:.1f}%), Avg p: {avg_p:.2f}")
        for n in sorted(interventions_by_noise.keys()):
            nd = interventions_by_noise[n]
            print(f"  Noise={n:4.1f}: {nd['intervened']}/{nd['total']} intervened, "
                  f"{nd['cbf_safe']}/{nd['total']} CBF-safe")

    # === Detailed examples ===
    print("\n" + "-" * 75)
    print("DETAILED EXAMPLES (high noise / dangerous states):")
    print(f"{'d':>6} {'v':>5} {'noise':>6} {'u_ref':>7} {'u_adv':>7} "
          f"{'u_NS':>7} {'u_V3A':>7} {'u_V3B':>7} "
          f"{'V3A_int':>8} {'V3B_int':>8} {'V3B_p':>6}")
    print("-" * 85)

    shown = 0
    for i in range(len(results['v3a'])):
        ra, rb = results['v3a'][i], results['v3b'][i]
        rn = results['no_shield'][i]
        if ra['intervened'] or rb['intervened'] or abs(rn['noise']) >= 3.0:
            print(f"{ra['d']:>6.1f} {ra['v']:>5.1f} {ra['noise']:>6.1f} "
                  f"{ra['u_ref']:>7.2f} {ra['u_adv']:>7.2f} "
                  f"{rn['u_out']:>7.2f} {ra['u_out']:>7.2f} {rb['u_out']:>7.2f} "
                  f"{'YES' if ra['intervened'] else 'no':>8} "
                  f"{'YES' if rb['intervened'] else 'no':>8} "
                  f"{rb['p']:>6.1f}")
            shown += 1
            if shown >= 20: break

    print("=" * 75)

    # Save
    out_dir = os.path.join(ROOT, '实验v3', 'results')
    with open(os.path.join(out_dir, 'adversarial_test.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/adversarial_test.json")


if __name__ == "__main__":
    adversarial_test()
