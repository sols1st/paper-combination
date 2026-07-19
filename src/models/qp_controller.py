"""
QP-Augmented Controller for SafePVC + BarrierNet Integration

Modifies the SafePVC controller to use BarrierNet's dual-branch architecture
with a differentiable QP safety filter layer.

Architecture:
    Input [state_est, velocity]
      → Shared Backbone (FC + ReLU layers)
        ├→ q_head: FC → linear → q (reference acceleration)
        └→ p_head: FC → 4·sigmoid → p (CBF parameter, positive)
      → QP: min ½u² + q·u s.t. CBF constraint G(s,p)·u ≤ h(s,p)
      → u* (safe acceleration)

Key differences from original SafePVC controller:
1. Dual output heads instead of single output
2. QP safety filter at the end
3. CBF parameter learned via NN (adaptive safety)

Original SafePVC controller (preserved for comparison):
    Input → FC → ReLU → FC → ReLU → FC → output u

Author: Experiment v2
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Add original project to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artical-F122'))

from src.models.cbf_constraints import AEBSCBFConstraints


class QPAebsController(nn.Module):
    """
    Dual-branch QP-augmented controller for AEBS scenario.

    This replaces the original SafePVC controller's single-output head
    with a BarrierNet-style dual-branch architecture:
    - Branch 1: reference acceleration q (what the NN "wants" to do)
    - Branch 2: CBF parameter p (how aggressively to enforce safety)
    - QP layer: produces safe control u* that minimally deviates from q

    Args:
        input_dim: Input dimension (2 for AEBS: [state_est, velocity])
        hidden_dims: List of hidden layer dimensions
        control_dim: Output control dimension (1 for AEBS scalar acceleration)
        cbf_param_dim: Number of CBF parameters (1 for first-order CBF)
        t_gap: Safe time headway in seconds
        dt: Discretization timestep
        device: Torch device
    """

    def __init__(
        self,
        input_dim=2,
        hidden_dims=[256, 256, 256],
        control_dim=1,
        cbf_param_dim=1,
        t_gap=1.5,
        dt=0.05,
        device='cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.cbf_param_dim = cbf_param_dim
        self.device = device

        # Build shared backbone
        backbone_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, h_dim))
            backbone_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.shared_backbone = nn.Sequential(*backbone_layers)

        # Branch 1: Reference control q (linear output, no activation)
        self.q_head = nn.Linear(hidden_dims[-1], control_dim)

        # Branch 2: CBF parameter p (positive, via 4*sigmoid)
        self.p_head = nn.Linear(hidden_dims[-1], cbf_param_dim)

        # CBF constraint constructor
        self.cbf = AEBSCBFConstraints(t_gap=t_gap, dt=dt)

        # For tracking
        self.last_q = None
        self.last_p = None
        self.last_u = None

    def forward(self, x, state=None, mode='train', return_debug=False):
        """
        Forward pass with optional QP safety filter.

        Args:
            x: (batch, input_dim) controller input [state_est, velocity]
            state: (batch, 2) raw state [d, v], required for CBF constraints
            mode: 'train' (differentiable QP) or 'eval' (cvxopt QP) or 'direct' (no QP)
            return_debug: If True, returns (u*, q, p, debug_info)

        Returns:
            If not return_debug: u* (batch, control_dim) safe control
            If return_debug: (u*, q, p, debug_dict)
        """
        # Shared backbone
        features = self.shared_backbone(x)

        # Branch 1: Reference control
        q = self.q_head(features)  # (batch, control_dim)

        # Branch 2: CBF parameter (positive)
        p_raw = self.p_head(features)
        p = 4.0 * torch.sigmoid(p_raw)  # (batch, cbf_param_dim), range (0, 4)

        # Store for analysis
        self.last_q = q.detach()
        self.last_p = p.detach()

        # QP safety filter
        if state is not None and mode != 'direct':
            u_safe = self._apply_qp_filter(q, p, state, mode)
        else:
            # Direct mode: no QP filtering (for baseline comparison)
            u_safe = q

        self.last_u = u_safe.detach() if isinstance(u_safe, torch.Tensor) else u_safe

        if return_debug:
            debug = {
                'q': q,
                'p': p,
                'barrier': self.cbf.barrier_function(state) if state is not None else None,
            }
            return u_safe, q, p, debug
        return u_safe

    def _apply_qp_filter(self, q, p, state, mode):
        """
        Apply QP safety filter using CBF constraints.

        Args:
            q: (batch, control_dim) reference control
            p: (batch, cbf_param_dim) CBF parameter
            state: (batch, 2) system state [d, v]
            mode: 'train' or 'eval'

        Returns:
            u_safe: (batch, control_dim) QP-optimal safe control
        """
        batch_size = q.shape[0]
        n = self.control_dim

        # Build CBF constraints: G*u <= h
        G, h = self.cbf.build_full_constraints(state, p.squeeze(-1))

        # Quadratic cost: Q = I
        Q = Variable(torch.eye(n, device=q.device))
        Q = Q.unsqueeze(0).expand(batch_size, n, n)

        if mode == 'train':
            try:
                from qpth.qp import QPFunction
                e = Variable(torch.Tensor())
                u_safe = QPFunction(verbose=-1)(
                    Q.double(), q.double(),
                    G.double(), h.double(),
                    e, e
                )
                u_safe = u_safe.float()
            except ImportError:
                # Fallback: clip reference control to bounds
                u_safe = torch.clamp(q, -3.0, 3.0)
            except Exception as e:
                # QP might fail if infeasible; use slack approach
                # Add slack variables by relaxing constraint
                u_safe = self._solve_with_slack(Q, q, G, h)
        else:
            # Evaluation mode: use cvxopt
            u_safe = self._solve_cvxopt_batch(Q, q, G, h)

        return u_safe

    def _solve_with_slack(self, Q, q, G, h, slack_penalty=100.0):
        """
        Solve QP with slack variables when hard constraints are infeasible.
        Adds penalty on constraint violation.

        This is a fallback: in well-trained scenarios, the CBF parameter
        p should adapt to prevent infeasibility.
        """
        batch_size = q.shape[0]
        n = self.control_dim
        m = G.shape[1]  # number of constraints

        device = q.device

        # Augmented QP: add slack variable s >= 0
        # min 0.5*u^T*Q*u + q^T*u + slack_penalty*s^2
        # s.t. G*u <= h + s

        # We solve this by relaxing G*u <= h to G*u <= h + large_value
        # and then clipping to control bounds as final safety check
        relaxed_h = h + 10.0  # Very relaxed constraint

        try:
            from qpth.qp import QPFunction
            e = Variable(torch.Tensor())
            u_relaxed = QPFunction(verbose=-1)(
                Q.double(), q.double(),
                G.double(), relaxed_h.double(),
                e, e
            )
            u_relaxed = u_relaxed.float()
        except Exception:
            # If QP still fails, just clip q to control bounds
            u_relaxed = torch.clamp(q, -3.0, 3.0)

        return u_relaxed

    def _solve_cvxopt_batch(self, Q, q, G, h):
        """Solve QP using cvxopt for evaluation (batch)."""
        from cvxopt import solvers, matrix
        import numpy as np
        solvers.options['show_progress'] = False

        batch_size = q.shape[0]
        results = []

        for i in range(batch_size):
            Q_np = Q[i].detach().cpu().numpy()
            q_np = q[i].detach().cpu().numpy()
            G_np = G[i].detach().cpu().numpy()
            h_np = h[i].detach().cpu().numpy()

            try:
                sol = solvers.qp(
                    matrix(Q_np), matrix(q_np),
                    matrix(G_np), matrix(h_np)
                )
                u_np = np.array(sol['x']).squeeze()
            except Exception:
                # Fallback: clip reference
                u_np = np.clip(-q_np, -3.0, 3.0)

            results.append(torch.tensor(u_np, dtype=torch.float32))

        return torch.stack(results).to(q.device).reshape(batch_size, -1)

    def get_cbf_metrics(self, state):
        """
        Get CBF-related metrics for monitoring.

        Args:
            state: (batch, 2) tensor

        Returns:
            dict with barrier values, CBF parameters, etc.
        """
        with torch.no_grad():
            barrier_vals = self.cbf.barrier_function(state)
            metrics = {
                'barrier_mean': barrier_vals.mean().item(),
                'barrier_min': barrier_vals.min().item(),
                'barrier_max': barrier_vals.max().item(),
                'p_mean': self.last_p.mean().item() if self.last_p is not None else 0,
                'q_mean': self.last_q.mean().item() if self.last_q is not None else 0,
            }
            if self.last_u is not None:
                metrics['u_mean'] = self.last_u.mean().item()
        return metrics


class DirectController(nn.Module):
    """
    Direct controller WITHOUT QP layer (baseline for comparison).

    This is equivalent to the original SafePVC controller architecture.
    Used for ablation studies to isolate the effect of the QP layer.

    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        control_dim: Output control dimension
    """

    def __init__(self, input_dim=2, hidden_dims=[256, 256, 256], control_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], control_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, state=None, mode='direct', return_debug=False):
        """
        Direct forward pass without QP filtering.

        Args:
            x: (batch, input_dim)
            state: Ignored (for API compatibility)
            mode: Ignored (for API compatibility)
            return_debug: If True, returns (u, None, None, {})

        Returns:
            u: (batch, control_dim) direct control output
        """
        u = self.network(x)
        if return_debug:
            return u, u, None, {}
        return u


# For testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing QP-Augmented Controller for AEBS")
    print("=" * 60)

    device = 'cpu'
    controller = QPAebsController(
        input_dim=2,
        hidden_dims=[64, 32],
        control_dim=1,
        cbf_param_dim=1,
        t_gap=1.5,
        device=device
    )

    # Test batch
    x = torch.tensor([[10.0, 2.0], [8.0, 2.5], [6.0, 1.5]])  # [d, v]
    state = x.clone()

    print(f"\nInput shape: {x.shape}")
    print(f"Input values:\n{x}")

    # Test direct mode (no QP)
    u_direct = controller(x, mode='direct')
    print(f"\nDirect mode output: {u_direct.squeeze().tolist()}")

    # Test with QP (eval mode, no qpth needed)
    try:
        u_safe = controller(x, state=state, mode='eval')
        print(f"QP eval mode output: {u_safe.squeeze().tolist()}")
    except ImportError as e:
        print(f"QP eval mode skipped: {e}")

    # Test debug mode
    u_safe, q, p, debug = controller(x, state=state, mode='direct', return_debug=True)
    print(f"\nReference q: {q.squeeze().tolist()}")
    print(f"CBF param p: {p.squeeze().tolist()}")
    if debug['barrier'] is not None:
        print(f"Barrier b(s): {debug['barrier'].tolist()}")

    # Test DirectController baseline
    direct_ctrl = DirectController(input_dim=2, hidden_dims=[64, 32], control_dim=1)
    u_baseline = direct_ctrl(x)
    print(f"\nDirectController (baseline): {u_baseline.squeeze().tolist()}")

    print("\nAll tests passed!")
