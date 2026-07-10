"""
BarrierNet for CARLA AEBS (Emergency Braking System)

Implements a HOCBF-QP safety layer following the pattern from
BarrierNet/2D_Robot/models.py, adapted for the AEBS longitudinal
control problem.

System dynamics (continuous-time approximation):
    d_dot = -v
    v_dot = -acc

    x = [d, v]^T, u = acc in [-3, 3]
    f(x) = [-v, 0]^T, g(x) = [0, -1]^T

Barrier function: b(x) = d - d_safe  (relative degree r=2)
    Lf_b = -v,  Lf2b = 0,  LgLfb = 1

HOCBF constraint:
    u >= (p1+p2)*v - p1*p2*(d - d_safe)

QP:  min 0.5*u^2 + q*u
     s.t. G*u <= h  (HOCBF + box constraints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class BarrierNetAEBS(nn.Module):
    """
    BarrierNet for CARLA AEBS with HOCBF-QP safety layer.

    Architecture:
        Input [d_norm, v] -> Shared trunk -> p_head (ref control q)
                                          -> q_head (penalty p1, p2)
                                          -> HOCBF-QP -> safe control u*

    All layers use float64 (.double()) for QP numerical stability,
    matching the BarrierNet reference implementation.
    """

    def __init__(
        self,
        nFeatures=2,
        nHidden1=64,
        nHidden21=32,
        nHidden22=32,
        nCls=1,
        d_safe=6.0,
        u_min=-3.0,
        u_max=3.0,
        std1=1.0,
        robust_margin=0.0,
        device='cpu',
    ):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        self.d_safe = d_safe
        self.u_min = u_min
        self.u_max = u_max
        self.robust_margin = robust_margin
        self.device = device

        # Normalization: mean=0, std for d_norm dimension only
        # State is [d_norm, v]. d_norm is already normalized by std1.
        # We store std1 for denormalization in dCBF.
        self.std1 = std1
        mean_arr = np.array([0.0, 0.0])
        std_arr = np.array([1.0, 1.0])  # input is already normalized
        self.mean = torch.from_numpy(mean_arr).to(device)
        self.std = torch.from_numpy(std_arr).to(device)

        # Logging (set during inference)
        self.p1 = 0
        self.p2 = 0

        # ========== Network layers (all .double() for QP stability) ==========
        # Shared trunk
        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc1b = nn.Linear(nHidden1, nHidden1).double()

        # Head 1: reference control q (linear cost in QP)
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()

        # Head 2: HOCBF penalty parameters p1, p2 (must be positive)
        self.fc22 = nn.Linear(nHidden1, nHidden22).double()
        self.fc32 = nn.Linear(nHidden22, 2).double()  # always 2 outputs: p1, p2

    def forward(self, x, sgn=1):
        """
        Args:
            x: (B, 2) normalized state [d_norm, v]
            sgn: 1 = QPFunction (training, differentiable)
                 0 = cvxopt (inference, exact)
        Returns:
            (B, 1) safe control output u*
        """
        nBatch = x.size(0)
        x = x.view(nBatch, -1).double()  # convert to double for all layers

        # Denormalize for physics: d_norm needs * std1 to get real distance
        x0 = x * self.std + self.mean

        # Shared trunk with residual-style double layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1b(x))

        # Head 1: reference control (unbounded, can be negative)
        x21 = F.relu(self.fc21(x))
        q_ref = self.fc31(x21)  # (B, 1)

        # Head 2: penalty functions p1, p2 (must be positive)
        x22 = F.relu(self.fc22(x))
        p_params = self.fc32(x22)  # (B, 2)
        p_params = 4.0 * torch.sigmoid(p_params)  # ensures p in (0, 4)

        # HOCBF-QP safety layer
        u_safe = self.dCBF(x0, q_ref, p_params, sgn, nBatch)

        return u_safe

    def dCBF(self, x0, q_ref, p_params, sgn, nBatch):
        """
        Build and solve the HOCBF-QP.

        Args:
            x0:       (B, 2) denormalized state [d_norm, v] (same as input here)
            q_ref:    (B, 1) reference control from p_head
            p_params: (B, 2) penalty parameters [p1, p2] from q_head
            sgn:      1 = QPFunction, 0 = cvxopt
            nBatch:   batch size
        Returns:
            (B, 1) safe control
        """
        # Extract state (denormalize d from d_norm to real distance)
        d_norm = x0[:, 0]  # (B,)
        v = x0[:, 1]       # (B,)
        d = d_norm * self.std1  # real distance in meters

        p1 = p_params[:, 0]  # (B,)
        p2 = p_params[:, 1]  # (B,)

        # ========== QP cost: min 0.5 * u^T Q u + p^T u ==========
        # Q = I (1x1 identity), scalar for 1D control
        Q = Variable(torch.eye(self.nCls, dtype=torch.float64))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)

        # ========== HOCBF constraint ==========
        # Barrier: b(x) = d - d_safe
        # Lf_b = -v,  Lf2b = 0,  LgLfb = 1
        #
        # HOCBF condition: LgLfb*u + Lf2b + (p1+p2)*Lf_b + p1*p2*b >= 0
        # => u + 0 - (p1+p2)*v + p1*p2*(d-d_safe) >= 0
        # => u >= (p1+p2)*v - p1*p2*(d-d_safe)
        #
        # QP form: G*u <= h  =>  -u <= -[(p1+p2)*v - p1*p2*(d-d_safe)]
        G_hocbf = -torch.ones(nBatch, 1, dtype=torch.float64, device=self.device)
        h_hocbf = (-(p1 + p2) * v + p1 * p2 * (d - self.d_safe)
                   - self.robust_margin)
        h_hocbf = h_hocbf.view(nBatch, 1).double()

        # ========== Box constraints: u_min <= u <= u_max ==========
        # u <= u_max  =>  [1]*u <= u_max
        # u >= u_min  =>  [-1]*u <= -u_min
        G_ub = torch.ones(nBatch, 1, dtype=torch.float64, device=self.device)
        h_ub = torch.full((nBatch, 1), self.u_max,
                          dtype=torch.float64, device=self.device)

        G_lb = -torch.ones(nBatch, 1, dtype=torch.float64, device=self.device)
        h_lb = torch.full((nBatch, 1), -self.u_min,
                          dtype=torch.float64, device=self.device)

        # Stack all constraints: G (B, 3, 1), h (B, 3)
        G = torch.cat([G_hocbf, G_ub, G_lb], dim=1).unsqueeze(2)  # (B, 3, 1)
        h = torch.cat([h_hocbf, h_ub, h_lb], dim=1)               # (B, 3)

        # No equality constraints
        e = Variable(torch.Tensor()).to(self.device)

        # ========== Solve QP ==========
        if self.training or sgn == 1:
            # Training: differentiable batched QP via qpth
            from qpth.qp import QPFunction, QPSolvers
            u = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
                Q, q_ref.double(), G, h, e, e
            )
            # Handle potential infeasibility (NaN)
            if torch.isnan(u).any():
                u = torch.where(
                    torch.isnan(u),
                    torch.full_like(u, self.u_max),  # fallback: max braking
                    u
                )
        else:
            # Inference: cvxopt (exact, single sample)
            self.p1 = p1[0].item()
            self.p2 = p2[0].item()
            try:
                u = self._cvxopt_solve(Q[0], q_ref[0], G[0], h[0])
            except Exception:
                # QP infeasible -> fallback to maximum braking
                u = torch.tensor([[self.u_max]],
                                 dtype=torch.float64, device=self.device)

        return u.float()

    def _cvxopt_solve(self, Q, p, G, h):
        """
        Solve QP using cvxopt (for inference, single sample).

        Args:
            Q: (n, n) double tensor
            p: (n,) double tensor
            G: (m, n) double tensor
            h: (m,) double tensor
        Returns:
            (1, 1) float tensor
        """
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        mat_Q = matrix(Q.cpu().detach().numpy())
        mat_p = matrix(p.cpu().detach().numpy())
        mat_G = matrix(G.cpu().detach().numpy())
        mat_h = matrix(h.cpu().detach().numpy())

        sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)

        result = np.array(sol['x'], dtype=np.float64).flatten()
        return torch.tensor(result, dtype=torch.float64,
                            device=self.device).unsqueeze(0)  # (1, nCls)

    def get_barrier_value(self, state):
        """Compute barrier function value b(x) = d - d_safe."""
        d_norm = state[:, 0]
        d = d_norm * self.std1
        return d - self.d_safe

    def get_penalty_values(self, state):
        """Compute penalty parameters p1, p2 for the current state."""
        x = state.view(state.size(0), -1).double()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1b(x))
        x22 = F.relu(self.fc22(x))
        p = self.fc32(x22)
        p = 4.0 * torch.sigmoid(p)
        return p.float()
