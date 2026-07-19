"""
CBF Constraint Construction for AEBS (CARLA Emergency Braking)

Implements the Control Barrier Function constraints for the AEBS scenario:
- Safety rule: maintain safe time headway t_gap from lead vehicle
- Barrier function: b(s) = d - v * t_gap
- System: relative degree 1 (control u directly appears in ḃ)
- Standard CBF condition: ḃ + p·b ≥ 0

Dynamics:
    d_{k+1} = d_k - v_k * dt
    v_{k+1} = v_k - u_k * dt

Continuous approximation:
    ḋ = -v
    v̇ = -u

Author: Experiment v2
"""

import torch
import torch.nn as nn


class AEBSCBFConstraints:
    """
    Constructs CBF constraints for the AEBS emergency braking scenario.

    Barrier: b(s) = d - v * t_gap
    where:
        d = distance to obstacle/lead vehicle (m)
        v = ego vehicle velocity (m/s)
        t_gap = safe time headway (s)

    CBF condition (relative degree 1):
        L_f b + L_g b * u + p * b >= 0
        → -v + t_gap * u + p * (d - v * t_gap) >= 0
        → t_gap * u >= v - p * (d - v * t_gap)

    QP standard form (G*u <= h):
        G = -t_gap
        h = -v + p * (d - v * t_gap)
    """

    def __init__(self, t_gap=1.5, dt=0.05):
        """
        Args:
            t_gap: Safe time headway in seconds
            dt: Discretization timestep in seconds
        """
        self.t_gap = t_gap
        self.dt = dt

    def barrier_function(self, state):
        """
        Compute barrier function value b(s).

        Args:
            state: (batch, 2) tensor [d, v] where d is distance, v is velocity

        Returns:
            b: (batch,) tensor, barrier value. b >= 0 means safe.
        """
        d = state[:, 0]
        v = state[:, 1]
        b = d - v * self.t_gap
        return b

    def compute_lie_derivatives(self, state):
        """
        Compute Lie derivatives for the AEBS system.

        Continuous dynamics: ḋ = -v, v̇ = -u

        L_f b = ∂b/∂d * f_d + ∂b/∂v * f_v
              = 1 * (-v) + (-t_gap) * 0
              = -v

        L_g b = ∂b/∂d * g_d + ∂b/∂v * g_v
              = 1 * 0 + (-t_gap) * (-1)  [since v̇ = -u → g_v = -1]
              = t_gap

        Args:
            state: (batch, 2) tensor [d, v]

        Returns:
            Lf_b: (batch,) tensor, Lie derivative along drift
            Lg_b: (batch,) tensor, Lie derivative along control
        """
        v = state[:, 1]

        # L_f b = -v
        Lf_b = -v

        # L_g b = t_gap (constant w.r.t. state for this simple system)
        Lg_b = self.t_gap * torch.ones_like(v)

        return Lf_b, Lg_b

    def build_constraints(self, state, p):
        """
        Build QP constraint matrices G and h from state and CBF parameter p.

        The QP constraint is: G * u <= h
        where:
            G = -L_g b  (scalar for 1D control)
            h = L_f b + p * b(s)

        Args:
            state: (batch, 2) tensor [d, v]
            p: (batch, 1) or (batch,) tensor, CBF parameter (positive)

        Returns:
            G: (batch, 1, 1) tensor, inequality constraint matrix
            h: (batch, 1) tensor, inequality constraint upper bound
        """
        batch_size = state.shape[0]

        # Compute barrier and Lie derivatives
        b = self.barrier_function(state)       # (batch,)
        Lf_b, Lg_b = self.compute_lie_derivatives(state)  # (batch,)

        # Ensure p has correct shape
        if p.dim() == 2:
            p = p.squeeze(-1)  # (batch,)

        # CBF constraint: Lf_b + Lg_b * u + p * b >= 0
        # → -Lg_b * u <= Lf_b + p * b
        # → G * u <= h
        G = -Lg_b.reshape(batch_size, 1, 1)    # (batch, 1, 1)
        h = (Lf_b + p * b).reshape(batch_size, 1)  # (batch, 1)

        return G, h

    def build_full_constraints(self, state, p, u_min=-3.0, u_max=3.0):
        """
        Build QP constraints including both CBF and control bounds.

        Args:
            state: (batch, 2) tensor [d, v]
            p: (batch,) or (batch, 1) tensor, CBF parameter
            u_min: Minimum control (max deceleration), default -3.0 m/s²
            u_max: Maximum control (max acceleration), default 3.0 m/s²

        Returns:
            G_full: (batch, 3, 1) tensor, stacked constraints [CBF, lower, upper]
            h_full: (batch, 3) tensor, stacked bounds
        """
        batch_size = state.shape[0]
        device = state.device

        # CBF constraint
        G_cbf, h_cbf = self.build_constraints(state, p)  # (B, 1, 1), (B, 1)

        # Control bounds: u_min <= u <= u_max
        G_lower = -torch.ones(batch_size, 1, 1, device=device)
        h_lower = -u_min * torch.ones(batch_size, 1, device=device)

        G_upper = torch.ones(batch_size, 1, 1, device=device)
        h_upper = u_max * torch.ones(batch_size, 1, device=device)

        # Stack all constraints
        G_full = torch.cat([G_cbf, G_lower, G_upper], dim=1)  # (B, 3, 1)
        h_full = torch.cat([h_cbf, h_lower, h_upper], dim=1)  # (B, 3)

        return G_full, h_full

    def check_cbf_satisfaction(self, state, u, p):
        """
        Check if the CBF constraint is satisfied for given state, control, and parameter.

        Args:
            state: (batch, 2) tensor
            u: (batch, 1) tensor, control action
            p: (batch,) tensor, CBF parameter

        Returns:
            satisfied: (batch,) bool tensor
            margin: (batch,) tensor, constraint margin (positive = satisfied)
        """
        G, h = self.build_constraints(state, p)
        # G*u <= h means G*u - h <= 0
        # margin = h - G*u, positive means constraint satisfied
        margin = h.squeeze(-1) - (G.squeeze(1) * u.squeeze(-1)).sum(dim=-1)
        satisfied = margin >= 0
        return satisfied, margin

    def cbf_violation_loss(self, state, u, p):
        """
        Compute CBF constraint violation loss for training.
        Penalizes when CBF constraint is violated.

        Args:
            state: (batch, 2) tensor
            u: (batch, 1) tensor, control action
            p: (batch,) tensor, CBF parameter

        Returns:
            loss: scalar tensor, violation penalty
        """
        _, margin = self.check_cbf_satisfaction(state, u, p)
        # Penalize negative margin (constraint violation)
        violation = torch.relu(-margin)
        return violation.mean()


# For testing
if __name__ == "__main__":
    # Quick test
    cbf = AEBSCBFConstraints(t_gap=1.5, dt=0.05)

    # Test state: d=10m, v=2m/s
    state = torch.tensor([[10.0, 2.0]])
    p = torch.tensor([1.0])
    u = torch.tensor([[0.5]])  # mild acceleration

    b = cbf.barrier_function(state)
    print(f"Barrier b(s) = {b.item():.3f}")  # Should be 10 - 2*1.5 = 7.0

    G, h = cbf.build_constraints(state, p)
    print(f"G = {G}, h = {h}")

    satisfied, margin = cbf.check_cbf_satisfaction(state, u, p)
    print(f"Satisfied: {satisfied.item()}, Margin: {margin.item():.3f}")

    loss = cbf.cbf_violation_loss(state, u, p)
    print(f"Violation loss: {loss.item():.6f}")
