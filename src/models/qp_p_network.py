"""
QP P-Network: Neural Network for Optimal CBF Parameter Prediction

Instead of using a fixed formula p=f(B(s)) with sigmoid (which failed because
SBC B(s) values don't vary enough), this module trains a dedicated neural network
to predict the optimal CBF parameter p for each state.

Training signals:
1. SBC supervision: p should be high when B(s) is high (dangerous), low when safe
2. Barrier supervision: p should be high when b(s)=d-v*t_gap is small (near unsafe)
3. CBF constraint satisfaction: p should ensure CBF constraint -t*u <= -v + p*b holds
4. Control deviation penalty: p shouldn't be unnecessarily high (allows control freedom)

The trained p_network is then used at inference time to dynamically set CBF parameters.

Usage:
    PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
    python /root/paper-combination/src/models/qp_p_network.py
"""

import sys, os, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG_DIR = os.path.join(ROOT, 'artical-F122')
sys.path.insert(0, ORIG_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cbf_constraints import AEBSCBFConstraints


class QPParameterNetwork(nn.Module):
    """
    Neural network that predicts optimal CBF parameter p from state.

    Architecture: state [d, v] → MLP → p ∈ [p_min, p_max]
    """
    def __init__(self, input_dim=2, hidden_dims=[64, 32, 16], p_min=0.1, p_max=4.0):
        super().__init__()
        self.p_min = p_min
        self.p_max = p_max

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """
        Args:
            state: (batch, 2) tensor [d, v] in raw coordinates
        Returns:
            p: (batch,) tensor in [p_min, p_max]
        """
        raw = self.net(state).squeeze(-1)  # (batch,)
        p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(raw)
        return p


class QPParameterTrainer:
    """
    Trains a QPParameterNetwork using:
    1. SBC B(s) as a soft label
    2. Barrier function b(s) as direct safety signal
    3. CBF constraint satisfaction feedback
    4. Control deviation regularization
    """
    def __init__(self, p_network, sbc_model, p_net, cbf, device='cpu'):
        self.p_network = p_network
        self.sbc_model = sbc_model
        self.p_net = p_net  # The full perception+controller pipeline
        self.cbf = cbf
        self.device = device

        # Ensure SBC is frozen
        self.sbc_model.eval()
        for p in self.sbc_model.parameters():
            p.requires_grad = False

    def generate_training_data(self, env, std1, n_samples=5000):
        """
        Generate diverse state samples for training.
        Focuses on both safe and dangerous regions.
        """
        # Uniform grid over state space (normalized)
        d_norm_min, d_norm_max = 5.0/std1, 16.0/std1
        v_min, v_max = 0.0, 3.0

        states = []
        # Uniform random samples
        n_uniform = n_samples // 2
        d_uniform = torch.rand(n_uniform) * (d_norm_max - d_norm_min) + d_norm_min
        v_uniform = torch.rand(n_uniform) * (v_max - v_min) + v_min
        states.append(torch.stack([d_uniform, v_uniform], dim=-1))

        # Focus on dangerous region (d close to v*t_gap, v high)
        n_danger = n_samples // 4
        v_danger = torch.rand(n_danger) * 2.0 + 1.0  # v ∈ [1,3]
        t_gap = self.cbf.t_gap
        d_danger = v_danger * t_gap + torch.rand(n_danger) * 3.0  # d ∈ [v*t_gap, v*t_gap+3]
        states.append(torch.stack([d_danger / std1, v_danger], dim=-1))

        # Focus on safe region
        n_safe = n_samples // 4
        v_safe = torch.rand(n_safe) * 2.0 + 0.5  # v ∈ [0.5, 2.5]
        d_safe = v_safe * t_gap + torch.rand(n_safe) * 8.0 + 2.0  # d well above barrier
        states.append(torch.stack([d_safe / std1, v_safe], dim=-1))

        all_states = torch.cat(states, dim=0)  # (n, 2) normalized
        return all_states

    def compute_target_p(self, states_raw):
        """
        Compute target CBF parameter p based on SBC B(s) and barrier b(s).

        Two complementary signals:
        1. SBC-based: p_sbc = p_min + (p_max-p_min) * sigmoid((B(s)-B_thresh)/T)
        2. Barrier-based: p_bar = p_max - (p_max-p_min) * sigmoid(b(s)/margin_scale)
           (when b(s) is small/near zero, p should be high)

        The final target blends both signals.
        """
        batch_size = states_raw.shape[0]

        with torch.no_grad():
            # SBC-based
            B = self.sbc_model(states_raw.float().to(self.device)).squeeze(-1)
            B_thresh = min(1.0 / 0.05, 5.0)  # ~5.0 for p_target=0.95
            p_sbc = self.p_network.p_min + (self.p_network.p_max - self.p_network.p_min) * \
                    torch.sigmoid((B - B_thresh) / 0.5)

            # Barrier-based (more direct)
            d_raw = states_raw[:, 0].to(self.device)
            v_raw = states_raw[:, 1].to(self.device)
            b = d_raw - v_raw * self.cbf.t_gap
            # When b is large positive (safe) → sigmoid → 1 → p_bar ≈ p_min
            # When b is near 0 or negative (unsafe) → sigmoid → 0 → p_bar ≈ p_max
            p_bar = self.p_network.p_min + (self.p_network.p_max - self.p_network.p_min) * \
                    (1.0 - torch.sigmoid(b / 2.0))

            # Blend: 70% barrier-based (more reliable), 30% SBC-based
            target_p = 0.7 * p_bar + 0.3 * p_sbc

        return target_p, B, b

    def train_epoch(self, states_normalized, optimizer, env, std1, lambda_cbf=0.5, lambda_smooth=0.1):
        """
        Train p_network for one epoch.

        Loss = MSE(target_p, pred_p) + λ_cbf * L_CBF + λ_smooth * L_smooth

        Args:
            states_normalized: (N, 2) tensor of normalized states
            optimizer: torch optimizer
            env: Aebs environment
            std1: data std for denormalization
        """
        self.p_network.train()

        # Denormalize for CBF computation
        states_raw = states_normalized.clone()
        states_raw[:, 0] = states_raw[:, 0] * std1  # d in meters

        # Compute target p from SBC + barrier
        target_p, B_vals, b_vals = self.compute_target_p(states_raw)

        # Forward
        pred_p = self.p_network(states_raw.to(self.device))  # (batch,)

        # Loss 1: MSE against target
        loss_mse = F.mse_loss(pred_p, target_p.to(self.device))

        # Loss 2: CBF constraint satisfaction
        # For each state, generate a plausible control and check CBF
        with torch.no_grad():
            # Use PPO controller to get reference control
            z = torch.zeros(states_normalized.shape[0], 4, device=self.device)
            u_ref = self.p_net(z, states_normalized.to(self.device))
            # Add noise for exploration
            u_noisy = u_ref + 0.1 * torch.randn_like(u_ref)

        G, h = self.cbf.build_constraints(states_raw.to(self.device), pred_p)
        # Check: G(1,1,i) * u_noisy <= h(1,i)  →  margin = h - G*u >= 0
        margin = h.squeeze(-1) - (G.squeeze(1) * u_noisy.squeeze(-1))
        # If pred_p is too low, constraint is violated (margin < 0)
        loss_cbf = torch.relu(-margin).mean()

        # Loss 3: Smoothness regularization
        # Penalize sharp changes in p → encourage smooth control
        if states_normalized.shape[0] > 1:
            # Compute pairwise distances and p differences
            idx = torch.randperm(states_normalized.shape[0])[:min(512, states_normalized.shape[0])]
            states_sub = states_raw[idx].to(self.device)
            p_sub = pred_p[idx]

            dist = torch.cdist(states_sub, states_sub)
            p_diff = (p_sub.unsqueeze(0) - p_sub.unsqueeze(1)).abs()
            # Lipschitz-like: p_diff / dist should be bounded
            mask = (dist > 0.1) & (dist < 2.0)
            if mask.any():
                lip_ratio = (p_diff[mask] / dist[mask]).mean()
                loss_smooth = lip_ratio
            else:
                loss_smooth = torch.tensor(0.0, device=self.device)
        else:
            loss_smooth = torch.tensor(0.0, device=self.device)

        # Total loss
        loss = loss_mse + lambda_cbf * loss_cbf + lambda_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_network.parameters(), 1.0)
        optimizer.step()

        return {
            'loss': loss.item(),
            'mse': loss_mse.item(),
            'cbf': loss_cbf.item(),
            'smooth': loss_smooth.item() if isinstance(loss_smooth, float) else loss_smooth.item(),
            'pred_p_mean': pred_p.mean().item(),
            'pred_p_std': pred_p.std().item(),
            'target_p_mean': target_p.mean().item(),
            'B_mean': B_vals.mean().item(),
            'b_mean': b_vals.mean().item(),
        }


def train_p_network(
    p_network=None,
    sbc_model=None,
    p_net=None,
    num_epochs=200,
    batch_size=256,
    lr=1e-3,
    save_dir=None,
):
    """Main training function for QP p-network."""
    import h5py
    from Aebs.system.env import Aebs
    from Aebs.VT.utils import MLP

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ORIG_DIR = os.path.join(ROOT, 'artical-F122')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load std1
    fn = os.path.join(ORIG_DIR, "Aebs/data/Downsampled.h5")
    with h5py.File(fn, 'r') as f:
        y_data = np.array(f["y_train"], dtype=np.float32)
    std1 = np.std(y_data)

    env = Aebs(0.05)
    cbf = AEBSCBFConstraints(t_gap=1.5, dt=0.05)

    # Load SBC if not provided
    if sbc_model is None:
        from Aebs.VT.utils import MLP
        sbc_model = MLP([2, 32, 16, 8, 1], activation="tanh", square_output=True).to(device)
        sbc_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc_improved.pth')
        if os.path.exists(sbc_path):
            sbc_model.load_state_dict(torch.load(sbc_path, map_location=device))
            print(f"Loaded improved SBC from {sbc_path}")
        else:
            sbc_path = os.path.join(ROOT, '实验v3', 'results', 'trained_sbc.pth')
            if os.path.exists(sbc_path):
                sbc_model.load_state_dict(torch.load(sbc_path, map_location=device))
                print(f"Loaded SBC from {sbc_path}")
            else:
                print("WARNING: No trained SBC found, using random weights")
    sbc_model.to(device)
    sbc_model.eval()

    # Load perception+controller if not provided
    if p_net is None:
        from cGAN.taxi_models_and_data import AebsMLPGenerator
        from Combined_network.model import AebsEnd2EndNet
        from stable_baselines3 import PPO

        gen_net = AebsMLPGenerator(4, 1)
        gen_net.load_state_dict(torch.load(
            os.path.join(ORIG_DIR, "Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"),
            map_location=device))
        gen_net.eval()

        ppo = PPO.load(os.path.join(ORIG_DIR, 'Aebs/controller/best_model/best_model.zip'))
        p_net = AebsEnd2EndNet(gen_net, [1024, 256, 64, 1],
                               ppo.policy.mlp_extractor.policy_net,
                               ppo.policy.action_net)
        p_net.state_net.load_state_dict(torch.load(
            os.path.join(ORIG_DIR, "Aebs/controller/state_net_trained.pth"),
            map_location=device))
        p_net.to(device)
        p_net.eval()
        for param in p_net.parameters():
            param.requires_grad = False
        print("Loaded perception+controller")

    # Create p-network
    if p_network is None:
        p_network = QPParameterNetwork(input_dim=2, hidden_dims=[64, 32, 16],
                                        p_min=0.1, p_max=4.0).to(device)

    trainer = QPParameterTrainer(p_network, sbc_model, p_net, cbf, device=device)

    # Generate training data
    print(f"\nGenerating training data...")
    train_states = trainer.generate_training_data(env, std1, n_samples=8000)
    print(f"  {train_states.shape[0]} training states")

    # Split into train/val
    n_train = int(0.85 * train_states.shape[0])
    train_set = train_states[:n_train]
    val_set = train_states[n_train:]

    optimizer = torch.optim.Adam(p_network.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nTraining p-network for {num_epochs} epochs...")
    best_val_loss = float('inf')
    train_history = []

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(train_set.shape[0])
        train_set = train_set[perm]

        # Mini-batch training
        epoch_metrics = []
        for start in range(0, train_set.shape[0], batch_size):
            end = min(start + batch_size, train_set.shape[0])
            batch = train_set[start:end].to(device)
            metrics = trainer.train_epoch(batch, optimizer, env, std1)
            epoch_metrics.append(metrics)

        scheduler.step()

        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0]}

        # Validation
        p_network.eval()
        with torch.no_grad():
            val_states_raw = val_set.clone()
            val_states_raw[:, 0] = val_states_raw[:, 0] * std1
            val_target, _, _ = trainer.compute_target_p(val_states_raw)
            val_pred = p_network(val_states_raw.to(device))
            val_loss = F.mse_loss(val_pred, val_target.to(device)).item()

        train_history.append({**avg_metrics, 'val_loss': val_loss, 'epoch': epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(p_network.state_dict())

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={avg_metrics['loss']:.4f}, "
                  f"mse={avg_metrics['mse']:.4f}, cbf={avg_metrics['cbf']:.4f}, "
                  f"p_mean={avg_metrics['pred_p_mean']:.3f}, val_loss={val_loss:.4f}")

    # Restore best
    p_network.load_state_dict(best_state)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Analyze trained network
    print("\n=== Trained p-Network Analysis ===")
    p_network.eval()
    with torch.no_grad():
        test_cases_raw = torch.tensor([
            [15.0, 0.5],   # Very safe
            [12.0, 1.5],   # Safe
            [9.0, 2.0],    # Moderate
            [7.0, 2.5],    # Risky
            [5.5, 3.0],    # Dangerous
        ])
        test_p = p_network(test_cases_raw.to(device))

        for i, (label, (d, v)) in enumerate(zip(
            ["Very Safe", "Safe", "Moderate", "Risky", "Dangerous"],
            [(15.0, 0.5), (12.0, 1.5), (9.0, 2.0), (7.0, 2.5), (5.5, 3.0)]
        )):
            b_val = d - v * 1.5
            print(f"  {label:12s}: d={d:.1f}, v={v:.1f}, b={b_val:.1f}, p={test_p[i].item():.3f}")

    # Save
    if save_dir is None:
        save_dir = os.path.join(ROOT, '实验v3', 'results')
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'trained_p_network.pth')
    torch.save({
        'model_state_dict': p_network.state_dict(),
        'train_history': train_history,
        'final_val_loss': best_val_loss,
        'timestamp': datetime.now().isoformat(),
    }, model_path)
    print(f"\nSaved to {model_path}")

    return p_network, train_history


class TrainedQPShield:
    """
    QP safety shield using a trained p-network (V3C).

    This replaces the sigmoid formula with a learned mapping from state→p.
    The p-network is trained to output optimal CBF parameters using SBC
    supervision and CBF constraint satisfaction feedback.
    """
    def __init__(self, p_network, cbf, device='cpu'):
        self.p_network = p_network
        self.p_network.eval()
        self.cbf = cbf
        self.device = device

        self.intervention_count = 0
        self.total_steps = 0
        self.p_history = []

    def shield(self, u_ref, state, mode='analytic'):
        """Apply QP safety shield with trained p-network."""
        from src.models.sbc_modulated_qp import SBCModulatedQPShield

        batch_size = state.shape[0]

        # Get p from trained network
        with torch.no_grad():
            p = self.p_network(state.float().to(self.device))  # (batch,)
            B_dummy = torch.zeros(batch_size, device=self.device)

        # Build constraints
        G_cbf, h_cbf = self.cbf.build_constraints(state.float().to(self.device), p)

        G_lower = -torch.ones(batch_size, 1, 1, device=self.device)
        h_lower = 3.0 * torch.ones(batch_size, 1, device=self.device)
        G_upper = torch.ones(batch_size, 1, 1, device=self.device)
        h_upper = 3.0 * torch.ones(batch_size, 1, device=self.device)

        G = torch.cat([G_cbf, G_lower, G_upper], dim=1)
        h = torch.cat([h_cbf, h_lower, h_upper], dim=1)

        if u_ref.dim() == 1:
            u_ref = u_ref.unsqueeze(-1)

        # Solve QP analytically (1D)
        u_safe = SBCModulatedQPShield._solve_qp_1d_analytic(
            None, u_ref.to(self.device), G, h)

        intervened = (u_safe - u_ref.to(self.device)).abs() > 0.01

        sat, margin = self.cbf.check_cbf_satisfaction(
            state.float().to(self.device), u_safe, p)

        self.total_steps += batch_size
        self.intervention_count += intervened.sum().item()
        self.p_history.extend(p.cpu().tolist())

        info = {
            'p': p,
            'B': B_dummy,
            'intervened': intervened,
            'margin': margin,
        }
        return u_safe, info


if __name__ == "__main__":
    train_p_network(num_epochs=200)
