"""
QP Solver Wrapper for SafePVC + BarrierNet Integration

Provides both differentiable (qpth, for training) and non-differentiable
(cvxopt, for inference/evaluation) QP solvers.

The QP solves:
    min  0.5 * u^T Q u + q^T u
    s.t. G u <= h

For AEBS (1D control):
    Q = I (1x1 identity)
    q = reference acceleration from NN branch 1
    G, h = CBF constraints + control bounds

Author: Experiment v2
"""

import torch
import numpy as np
from torch.autograd import Variable


def solve_qp_qpth(Q, q, G, h, verbose=-1):
    """
    Solve QP using qpth (differentiable, for training).

    Args:
        Q: (batch, n, n) quadratic cost matrix
        q: (batch, n) linear cost vector
        G: (batch, m, n) inequality constraint matrix
        h: (batch, m) inequality constraint upper bound

    Returns:
        u: (batch, n) optimal solution
    """
    from qpth.qp import QPFunction

    e = Variable(torch.Tensor())  # No equality constraints

    # qpth needs double precision for numerical stability
    u = QPFunction(verbose=verbose)(
        Q.double(), q.double(),
        G.double(), h.double(),
        e, e
    )
    return u.float()


def solve_qp_cvxopt_batch(Q_batch, q_batch, G_batch, h_batch):
    """
    Solve QP using cvxopt (non-differentiable, batched for evaluation).

    Args:
        Q_batch: (batch, n, n)
        q_batch: (batch, n)
        G_batch: (batch, m, n)
        h_batch: (batch, m)

    Returns:
        u_batch: (batch, n) optimal solutions
    """
    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False

    batch_size = Q_batch.shape[0]
    results = []

    for i in range(batch_size):
        Q_np = Q_batch[i].detach().cpu().numpy()
        q_np = q_batch[i].detach().cpu().numpy()
        G_np = G_batch[i].detach().cpu().numpy()
        h_np = h_batch[i].detach().cpu().numpy()

        try:
            sol = solvers.qp(
                matrix(Q_np),
                matrix(q_np),
                matrix(G_np),
                matrix(h_np)
            )
            u = np.array(sol['x']).squeeze()
        except Exception:
            # If QP fails, return the reference control q
            # (this is a fallback - in practice, with proper constraints
            #  including slack variables, QP should rarely fail)
            u = -q_np  # q is the linear cost; u=-q minimizes ½u² + qu

        results.append(torch.tensor(u, dtype=torch.float32))

    return torch.stack(results)


def solve_qp_cvxopt_single(Q, q, G, h):
    """
    Solve a single QP using cvxopt (for single-sample inference).

    Args:
        Q: (n, n)
        q: (n,)
        G: (m, n)
        h: (m,)

    Returns:
        u: (n,) optimal solution, or fallback if QP fails
    """
    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False

    Q_np = Q.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()
    G_np = G.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()

    try:
        sol = solvers.qp(
            matrix(Q_np),
            matrix(q_np),
            matrix(G_np),
            matrix(h_np)
        )
        u = np.array(sol['x']).squeeze()
    except Exception:
        # Fallback: use reference control
        u = -q_np

    return torch.tensor(u, dtype=torch.float32)


class QPSafeFilter:
    """
    QP-based safety filter that wraps a controller's reference output
    with CBF constraints.

    Usage:
        qp_filter = QPSafeFilter(control_dim=1, device='cpu')
        u_safe = qp_filter(q_ref, G, h, mode='train')  # or mode='eval'
    """

    def __init__(self, control_dim=1, device='cpu'):
        self.control_dim = control_dim
        self.device = device

    def __call__(self, q_ref, G, h, mode='train'):
        """
        Apply QP safety filter.

        Args:
            q_ref: (batch, control_dim) reference control from NN
            G: (batch, m, control_dim) constraint matrix
            h: (batch, m) constraint upper bound
            mode: 'train' (qpth, differentiable) or 'eval' (cvxopt, stable)

        Returns:
            u_safe: (batch, control_dim) safe control
        """
        batch_size = q_ref.shape[0]
        n = self.control_dim

        # Quadratic cost: Q = I (identity)
        Q = Variable(torch.eye(n, device=self.device))
        Q = Q.unsqueeze(0).expand(batch_size, n, n)

        if mode == 'train':
            u_safe = solve_qp_qpth(Q, q_ref, G, h)
        else:
            u_safe = solve_qp_cvxopt_batch(Q, q_ref, G, h)
            u_safe = u_safe.to(self.device)

        return u_safe


# For testing
if __name__ == "__main__":
    print("Testing QP Solver...")

    # Simple 1D QP: min 0.5*u^2 + q*u s.t. G*u <= h
    # For q=-1, G=-1.5, h=2.0:
    #   min 0.5*u^2 - u  s.t. -1.5*u <= 2.0 → u >= -1.333
    # Solution: u = max(1.0, -1.333) = 1.0 (unconstrained min at u=1)

    Q = torch.eye(1).unsqueeze(0)  # (1, 1, 1)
    q = torch.tensor([[-1.0]])     # (1, 1)
    G = torch.tensor([[[-1.5]]])   # (1, 1, 1)
    h = torch.tensor([[2.0]])      # (1, 1)

    try:
        u_qpth = solve_qp_qpth(Q, q, G, h)
        print(f"qpth solution: u = {u_qpth.item():.4f}")
    except ImportError:
        print("qpth not installed, skipping qpth test")

    try:
        u_cvxopt = solve_qp_cvxopt_batch(Q, q, G, h)
        print(f"cvxopt solution: u = {u_cvxopt.item():.4f}")
    except ImportError:
        print("cvxopt not installed, skipping cvxopt test")

    print("Expected: u ≈ 1.0 (unconstrained minimum, constraint inactive)")
