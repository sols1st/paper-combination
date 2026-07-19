"""
Quick SBC training for V3 shield activation.

Trains a baseline controller + SBC for 30 iterations (~3 min),
saves the trained SBC model for use in V3 comparison.

Usage:
    PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
    python /root/paper-combination/src/eval/train_sbc_for_v3.py
"""

import sys, os, time, copy
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG_DIR = os.path.join(ROOT, 'artical-F122')
sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, ROOT)
os.chdir(ORIG_DIR)

from Aebs.system.env import Aebs
from Aebs.VT.utils import triangular, martingale_loss, MLP
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet
from stable_baselines3 import PPO
from auto_LiRPA import BoundedModule

def train_quick_sbc(num_iters=30, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Environment
    env = Aebs(0.05)

    # Load models
    print("Loading pretrained models...")
    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(torch.load(
        "./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth", map_location=device
    ))

    ppo_model = PPO.load('./Aebs/controller/best_model/best_model.zip')
    mlp_extractor = ppo_model.policy.mlp_extractor.policy_net
    action_net = ppo_model.policy.action_net

    p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1], mlp_extractor, action_net)
    p_net.state_net.load_state_dict(torch.load(
        "./Aebs/controller/state_net_trained.pth", map_location=device
    ))
    p_net.to(device)

    teacher_net = copy.deepcopy(p_net)
    teacher_net.eval()
    for p in teacher_net.parameters():
        p.requires_grad = False
    teacher_net.to(device)

    # SBC
    l_model = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    l_optimizer = torch.optim.Adam(l_model.parameters(), lr=3e-3)

    # Controller optimizer (only controller_net params)
    ctrl_params = [p for n, p in p_net.named_parameters()
                   if 'controller_net' in n]
    p_optimizer = torch.optim.Adam(ctrl_params, lr=5e-2)

    # Build verifier for grid
    l_ibp = BoundedModule(l_model, torch.randn(1, 2).to(device), device=device)

    learner_dummy = type('obj', (object,), {
        'l_model': l_model, 'p_net': p_net, 'device': device,
        'create_bounded_module': lambda m: BoundedModule(m, torch.randn(1, 2).to(device), device=device)
    })()

    verifier = VTVerifier(learner_dummy, env, l_ibp,
                         batch_size=2048, reach_prob=0.9, fail_check_fast=True)
    verifier.prefill_train_buffer()

    # Grid for training
    train_ds, _, _ = verifier.get_unfiltered_grid(env.train_space_split)
    current_delta = (env.observation_space.high - env.observation_space.low) / env.train_space_split
    current_delta_t = torch.tensor(current_delta, dtype=torch.float32, device=device)

    max_prob = 0.0
    print(f"\nTraining SBC + Controller for {num_iters} iterations...")

    for iteration in range(num_iters):
        # === Train SBC ===
        l_model.train(); p_net.eval()
        N = train_ds.shape[0]
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=device)

        for epoch in range(10):
            indices = np.random.permutation(N)
            all_y = all_y[indices]
            for start in range(0, N, 256):
                end = min(start + 256, N)
                y = all_y[start:end].float()
                B = y.shape[0]
                z = (torch.rand(B, 4, device=device) * 2.0 - 1.0).float()

                rng = torch.Generator(device=device); rng.manual_seed(19)
                s_rand = torch.rand(y.shape, generator=rng, device=device) - 0.5
                y_pert = (y + current_delta_t.unsqueeze(0) * s_rand).float()

                l_optimizer.zero_grad()
                with torch.no_grad():
                    a = p_net(z.float(), y_pert)
                s_next = env.v_next(y_pert, a).unsqueeze(1).float()
                noise = triangular((B, 16, y.shape[1]), device=device)
                ns = torch.as_tensor(env.noise, device=device, dtype=torch.float32).view(1,1,-1)
                s_next_r = (s_next + noise * ns).float()

                l_val = l_model(y_pert).view(-1)
                l_next = l_model(s_next_r.reshape(-1, y.shape[1]).float()).view(B, -1)
                exp_l = l_next.mean(dim=1)
                dec_loss = martingale_loss(l_val, exp_l, eps=0.1); loss_l = dec_loss * 1000

                # Region
                init_s = _sample_region(env.init_spaces, 256, 13, device).float()
                unsafe_s = _sample_region(env.unsafe_spaces, 256, 17, device).float()
                l_init = l_model(init_s).view(-1); l_unsafe = l_model(unsafe_s).view(-1)
                target = 1.0 / 0.05
                region_loss = torch.relu(torch.max(l_init)-1.0) + torch.relu(target-torch.min(l_unsafe))
                loss_l = loss_l + region_loss

                loss_l.backward()
                torch.nn.utils.clip_grad_norm_(l_model.parameters(), 5.0)
                l_optimizer.step()

        # === Verify ===
        k = 1.2
        sat, hv, info, vb = verifier.check_dec_cond(k)
        hv_val = hv.item() if isinstance(hv, torch.Tensor) else hv

        # Compute prob
        if sat:
            _, ub_init = verifier.compute_bound_init(env.space_split)
            lb_unsafe, _ = verifier.compute_bound_unsafe(env.space_split)
            domain_min, _ = verifier.compute_bound_domain(env.space_split)
            if lb_unsafe > ub_init:
                ub_n = ub_init - domain_min; lb_n = lb_unsafe - domain_min
                ratio = lb_n / max(ub_n, 1e-9)
                prob = max(0.0, 1.0 - 1.0 / max(ratio, 1e-9))
                if prob > max_prob: max_prob = prob

        # === Train Controller ===
        l_model.eval(); p_net.train()
        indices = np.random.permutation(N); all_y = all_y[indices]
        for start in range(0, min(N, 2048), 256):
            end = min(start + 256, min(N, 2048))
            y = all_y[start:end].float(); B = y.shape[0]
            z = (torch.rand(B, 4, device=device) * 2.0 - 1.0).float()
            rng = torch.Generator(device=device); rng.manual_seed(19)
            s_rand = torch.rand(y.shape, generator=rng, device=device) - 0.5
            y_pert = (y + current_delta_t.unsqueeze(0) * s_rand).float()

            p_optimizer.zero_grad()
            a_p = p_net(z.float(), y_pert)
            with torch.no_grad(): l_p = l_model(y_pert).view(-1)
            s_next = env.v_next(y_pert, a_p).unsqueeze(1).float()
            noise = triangular((B, 128, y.shape[1]), device=device)
            ns = torch.as_tensor(env.noise, device=device, dtype=torch.float32).view(1,1,-1)
            s_next_r = (s_next + noise * ns).float()
            with torch.no_grad():
                l_next = l_model(s_next_r.reshape(-1, y.shape[1]).float()).view(B, -1)
            exp_l = l_next.mean(dim=1)
            dec_loss = martingale_loss(l_p.detach(), exp_l, eps=0.1)
            loss_p = dec_loss * 10
            with torch.no_grad(): u_t = teacher_net(z.float(), y_pert)
            mse = torch.nn.functional.mse_loss(a_p, u_t.view_as(a_p))
            loss_p = loss_p + 10.0 * mse
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(ctrl_params, 1.0)
            p_optimizer.step()

        if iteration % 5 == 0:
            print(f"  Iter {iteration:3d}: violations={hv_val}, prob={max_prob*100:.1f}%")

    # Save SBC
    if save_path is None:
        save_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc.pth')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(l_model.state_dict(), save_path)
    print(f"\nSBC saved to {save_path}")
    print(f"Final probability bound: {max_prob*100:.2f}%")

    # Check SBC output range
    l_model.eval()
    with torch.no_grad():
        test_states = torch.rand(100, 2, device=device)
        test_states[:, 0] = test_states[:, 0] * 11 + 5   # d in [5,16]
        test_states[:, 1] = test_states[:, 1] * 3          # v in [0,3]
        B_vals = l_model(test_states.float()).squeeze()
    print(f"SBC B(s) range: [{B_vals.min().item():.3f}, {B_vals.max().item():.3f}], "
          f"mean={B_vals.mean().item():.3f}")

    return save_path, max_prob


def _sample_region(spaces, n, seed, device):
    num = len(spaces); per = n // num
    rng = torch.Generator(device="cpu"); rng.manual_seed(seed)
    batch = []
    for i in range(num):
        low = torch.tensor(spaces[i].low, dtype=torch.float32)
        high = torch.tensor(spaces[i].high, dtype=torch.float32)
        x = (high - low) * torch.rand((per, low.shape[0]), generator=rng) + low
        batch.append(x)
    return torch.cat(batch, dim=0).to(device)


if __name__ == "__main__":
    train_quick_sbc(num_iters=35)
