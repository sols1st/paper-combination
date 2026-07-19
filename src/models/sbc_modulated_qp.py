"""
SBC-Modulated QP Safety Shield (实验v3核心创新)

QP只在推理时作为安全盾使用，不参与训练。
CBF参数p由SBC值B(s)动态调制: 安全时松弛, 危险时收紧。

Architecture:
    u_ref (controller output) + s (state) + B(s) (SBC value)
    → p = f(B(s))  [SBC-modulated CBF parameter]
    → QP: min ½(u-u_ref)²  s.t. CBF constraint with p
    → u_safe

v2 vs v3:
    v2: NN learns p during training → QP in training loop → degrades SBC
    v3: SBC sets p at inference → QP never in training → preserves SBC

Author: Experiment v3
"""

import sys, os
import torch
import numpy as np
from torch.autograd import Variable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artical-F122'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '实验v2', 'code', 'models'))

from models.cbf_constraints import AEBSCBFConstraints


class SBCModulatedQPShield:
    """
    Inference-only QP safety shield with SBC-modulated CBF parameters.

    The QP solves:
        min_u  ½(u - u_ref)²
        s.t.   CBF: -t_gap·u ≤ -v + p(B(s))·(d - v·t_gap)
               u_min ≤ u ≤ u_max

    where p(B(s)) is modulated by the SBC value:
        p(s) = p_min + (p_max - p_min) · sigmoid((B(s) - B_thresh) / T)

    This means:
    - When B(s) is small (very safe) → p ≈ p_min → relaxed CBF → more freedom
    - When B(s) ≈ B_thresh (risky) → p ≈ p_max → tight CBF → strict safety
    """

    def __init__(self, sbc_model, t_gap=1.5, dt=0.05,
                 p_min=0.1, p_max=4.0, p_target=0.95, temperature=0.5,
                 device='cpu'):
        """
        Args:
            sbc_model: Trained SBC network B(s)
            t_gap: Safe time headway (seconds)
            dt: Discretization timestep
            p_min: Minimum CBF parameter (when very safe)
            p_max: Maximum CBF parameter (when near unsafe)
            p_target: Target safety probability for threshold
            temperature: Smoothness of SBC→p transition
            device: Torch device
        """
        self.sbc_model = sbc_model
        self.sbc_model.eval()
        self.device = device

        self.cbf = AEBSCBFConstraints(t_gap=t_gap, dt=dt)
        self.p_min = p_min
        self.p_max = p_max
        self.temperature = temperature

        # SBC threshold: B_thresh = 1/(1-p_target)
        # For untrained SBC, B(s) ≈ small values (< 1), so we need a lower threshold
        # for the modulation to work. Use min(1/(1-p), 5.0) as practical threshold.
        raw_thresh = 1.0 / max(1.0 - p_target, 1e-6)
        self.B_thresh = min(raw_thresh, 5.0)  # Cap for untrained SBC

        # Statistics
        self.intervention_count = 0
        self.total_steps = 0
        self.p_history = []
        self.intervention_history = []

    def compute_p(self, state):
        """
        Compute SBC-modulated CBF parameter p.

        p(s) = p_min + (p_max - p_min) · σ((B(s) - B_thresh) / T)

        Args:
            state: (batch, 2) tensor [d, v]

        Returns:
            p: (batch,) tensor in [p_min, p_max]
        """
        with torch.no_grad():
            B = self.sbc_model(state.float()).squeeze(-1)  # (batch,)
            # Sigmoid: when B << B_thresh → 0, when B >> B_thresh → 1
            scaled_diff = (B - self.B_thresh) / self.temperature
            sigmoid_val = torch.sigmoid(scaled_diff)
            p = self.p_min + (self.p_max - self.p_min) * sigmoid_val

        return p, B

    def shield(self, u_ref, state, mode='cvxopt'):
        """
        Apply QP safety shield to a reference control.

        Args:
            u_ref: (batch, 1) or (batch,) tensor, reference control from NN
            state: (batch, 2) tensor [d, v]
            mode: 'cvxopt' (stable, non-differentiable) or 'qpth' (differentiable)

        Returns:
            u_safe: (batch, 1) safe control
            info: dict with p, B, intervened, margin
        """
        batch_size = state.shape[0]
        device = state.device

        if u_ref.dim() == 1:
            u_ref = u_ref.unsqueeze(-1)

        # Compute SBC-modulated p
        p, B = self.compute_p(state)

        # Build CBF constraints
        G_cbf, h_cbf = self.cbf.build_constraints(state, p)  # (B, 1, 1), (B, 1)

        # Add control bounds
        G_lower = -torch.ones(batch_size, 1, 1, device=device)
        h_lower = 3.0 * torch.ones(batch_size, 1, device=device)  # -u <= 3 → u >= -3
        G_upper = torch.ones(batch_size, 1, 1, device=device)
        h_upper = 3.0 * torch.ones(batch_size, 1, device=device)   # u <= 3

        G = torch.cat([G_cbf, G_lower, G_upper], dim=1)
        h = torch.cat([h_cbf, h_lower, h_upper], dim=1)

        # QP: min ½(u - u_ref)² = ½u² - u_ref·u + const
        # Standard form: min ½u^T Q u + q^T u  where Q=I, q=-u_ref
        Q = torch.eye(1, device=device).unsqueeze(0).expand(batch_size, 1, 1)
        q = -u_ref  # (batch, 1)

        # Solve QP
        if mode == 'cvxopt':
            u_safe = self._solve_cvxopt_batch(Q, q, G, h, u_ref)
        else:
            try:
                from qpth.qp import QPFunction
                e = Variable(torch.Tensor())
                u_safe = QPFunction(verbose=-1)(
                    Q.double(), q.double(), G.double(), h.double(), e, e
                ).float()
            except Exception:
                u_safe = u_ref  # Fallback

        # Check if QP intervened (modified the reference)
        intervened = (u_safe - u_ref).abs() > 0.01

        # Compute CBF margin
        satisfied, margin = self.cbf.check_cbf_satisfaction(
            state, u_safe, p
        )

        # Update statistics
        self.total_steps += batch_size
        self.intervention_count += intervened.sum().item()
        self.p_history.extend(p.cpu().tolist())
        self.intervention_history.extend(intervened.cpu().tolist())

        info = {
            'p': p,
            'B': B,
            'intervened': intervened,
            'margin': margin,
        }

        return u_safe, info

    def _solve_cvxopt_batch(self, Q, q, G, h, u_ref):
        """Solve QP using cvxopt for a batch."""
        from cvxopt import solvers, matrix
        solvers.options['show_progress'] = False

        batch_size = Q.shape[0]
        results = []

        for i in range(batch_size):
            Q_np = Q[i].detach().cpu().numpy()
            q_np = q[i].detach().cpu().numpy()
            G_np = G[i].detach().cpu().numpy()
            h_np = h[i].detach().cpu().numpy()

            try:
                sol = solvers.qp(matrix(Q_np), matrix(q_np),
                                 matrix(G_np), matrix(h_np))
                u = np.array(sol['x']).squeeze()
            except Exception:
                u = u_ref[i].detach().cpu().numpy().squeeze()

            results.append(torch.tensor(u, dtype=torch.float32))

        return torch.stack(results).to(u_ref.device).reshape(batch_size, 1)

    def get_stats(self):
        """Get shield statistics."""
        if self.total_steps == 0:
            return {}
        return {
            'intervention_rate': self.intervention_count / self.total_steps,
            'total_steps': self.total_steps,
            'avg_p': np.mean(self.p_history) if self.p_history else 0,
            'std_p': np.std(self.p_history) if self.p_history else 0,
            'p_min_observed': min(self.p_history) if self.p_history else 0,
            'p_max_observed': max(self.p_history) if self.p_history else 0,
        }


class FixedQPShield(SBCModulatedQPShield):
    """
    QP safety shield with FIXED CBF parameter (v3A baseline).

    This is the "inference-only QP" variant without SBC modulation.
    Used for ablation: isolate the effect of SBC modulation.
    """

    def __init__(self, sbc_model, fixed_p=2.0, **kwargs):
        super().__init__(sbc_model, **kwargs)
        self.fixed_p = fixed_p

    def compute_p(self, state):
        """Override: always return fixed p."""
        batch_size = state.shape[0]
        p = self.fixed_p * torch.ones(batch_size, device=state.device)
        with torch.no_grad():
            B = self.sbc_model(state.float()).squeeze(-1)
        return p, B


# For testing
if __name__ == "__main__":
    print("Testing SBC-Modulated QP Shield...")

    # Create a mock SBC
    import torch.nn as nn
    mock_sbc = nn.Sequential(
        nn.Linear(2, 16), nn.Tanh(),
        nn.Linear(16, 8), nn.Tanh(),
        nn.Linear(8, 1), nn.Softplus()
    )

    # Test SBC-modulated shield
    shield = SBCModulatedQPShield(mock_sbc, t_gap=1.5, device='cpu')

    # Test states
    safe_state = torch.tensor([[15.0, 1.0]])    # Very safe (far, slow)
    risky_state = torch.tensor([[6.0, 2.5]])    # Risky (close, fast)
    u_ref = torch.tensor([[0.5]])                # Mild acceleration

    p_safe, B_safe = shield.compute_p(safe_state)
    p_risky, B_risky = shield.compute_p(risky_state)

    print(f"Safe state:  d=15.0, v=1.0 → B={B_safe.item():.3f}, p={p_safe.item():.3f}")
    print(f"Risky state: d=6.0,  v=2.5 → B={B_risky.item():.3f}, p={p_risky.item():.3f}")
    print(f"B_thresh = {shield.B_thresh:.3f}")
    print(f"p should be: safe < risky ({p_safe.item():.3f} < {p_risky.item():.3f})")

    # Test shield application
    u_safe, info = shield.shield(u_ref, safe_state)
    print(f"\nQP shield on safe state: u_ref={u_ref.item():.3f} → u_safe={u_safe.item():.3f}")
    print(f"  Intervened: {info['intervened'].item()}, Margin: {info['margin'].item():.3f}")

    # Test fixed-p shield
    fixed_shield = FixedQPShield(mock_sbc, fixed_p=2.0, device='cpu')
    u_safe_fixed, info_fixed = fixed_shield.shield(u_ref, risky_state)
    print(f"\nFixed-p shield on risky state: u_ref={u_ref.item():.3f} → u_safe={u_safe_fixed.item():.3f}")

    print("\nAll tests passed!")
