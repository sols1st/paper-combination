"""
QP-Augmented Trainer for SafePVC + BarrierNet Integration

Extends the original SafePVC VTLearner to support:
1. Dual-branch QP controller (q_head + p_head)
2. CBF constraint violation loss
3. Training through the differentiable QP layer
4. Backward-compatible with original SBC training

Key modifications from original VT/train.py:
- Controller forward pass now goes through QP layer
- New loss term: L_CBF (CBF constraint violation penalty)
- MSE loss compares QP output u* (not raw q) against teacher policy
- Supports both 'direct' and 'qp' controller modes for ablation

The original VTLearner training logic is preserved in the base class;
this class adds QP-specific functionality via inheritance.

Author: Experiment v2
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add original project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artical-F122'))

from Aebs.VT.utils import triangular, martingale_loss, MLP
from Aebs.VT.train import VTLearner
from models.qp_controller import QPAebsController, DirectController
from models.cbf_constraints import AEBSCBFConstraints


class QPVTLearner(VTLearner):
    """
    Extended VTLearner with QP safety filter support.

    Adds:
    - CBF constraint loss for controller training
    - QP-aware training step
    - Metrics tracking for CBF-related quantities

    Inherits all SBC (l_net) training from VTLearner.
    """

    def __init__(
        self,
        l_model_config,
        env,
        p_lip,
        l_lip,
        eps,
        gamma_decrease,
        reach_prob,
        controller_type='qp',       # 'qp' or 'direct'
        controller_config=None,     # Config for QPAebsController
        lambda_cbf=1.0,             # CBF loss weight
        lambda_mse=10.0,            # MSE loss weight
        square_l_output=True,
        l_model_path=None,
        p_model_path=None,
    ):
        # Call parent init (this loads gen_net, state_net, l_model, original p_net)
        super().__init__(
            l_model_config=l_model_config,
            env=env,
            p_lip=p_lip,
            l_lip=l_lip,
            eps=eps,
            gamma_decrease=gamma_decrease,
            reach_prob=reach_prob,
            square_l_output=square_l_output,
            l_model_path=l_model_path,
            p_model_path=p_model_path,
        )

        self.controller_type = controller_type
        self.lambda_cbf = lambda_cbf
        self.lambda_mse = lambda_mse

        # Override controller with QP version if specified
        if controller_type == 'qp':
            if controller_config is None:
                controller_config = {}
            self._replace_controller_with_qp(**controller_config)

        # CBF constraint calculator for loss computation
        self.cbf_calculator = AEBSCBFConstraints(t_gap=1.5, dt=0.05)

        # Store original p_net as teacher (for MSE distillation)
        self.teacher_net = copy.deepcopy(self.net)
        for param in self.teacher_net.parameters():
            param.requires_grad = False

    def _replace_controller_with_qp(self, **kwargs):
        """
        Replace the original controller_net inside p_net with QP controller.
        The gen_net and state_net remain frozen (unchanged).
        Only the controller part (controller_net) is replaced.
        """
        # Build QP controller
        qp_ctrl = QPAebsController(
            input_dim=kwargs.get('input_dim', 2),
            hidden_dims=kwargs.get('hidden_dims', [256, 256, 256]),
            control_dim=kwargs.get('control_dim', 1),
            cbf_param_dim=kwargs.get('cbf_param_dim', 1),
            t_gap=kwargs.get('t_gap', 1.5),
            dt=kwargs.get('dt', 0.05),
            device=self.device,
        )

        # Replace the controller_net inside p_net (AebsEnd2EndNet)
        self.p_net.controller_net = qp_ctrl

        # Re-register optimizer for new controller parameters
        self.p_optimizer = torch.optim.Adam(
            self.p_net.controller_net.parameters(),
            lr=5e-3  # Lower LR for QP controller (more stable)
        )

        print(f"[QPVTLearner] Replaced controller with QPAebsController")
        print(f"  Controller params: {sum(p.numel() for p in qp_ctrl.parameters()):,}")

    def train_step_p_qp(self, z, y, lip_coeff, current_delta, clip_grad=1.0):
        """
        QP-augmented controller training step.

        Key differences from original train_step_p:
        1. Controller forward goes through QP layer
        2. State is passed to controller for CBF constraint construction
        3. CBF violation loss added
        4. MSE compares QP output (u*) against teacher, not raw q

        Args:
            z: (batch, latent_dim) latent noise
            y: (batch, obs_dim) state observations [d_norm, v]
            lip_coeff: Lipschitz regularization coefficient
            current_delta: State perturbation magnitude
            clip_grad: Gradient clipping value
        """
        device = self.device
        B = y.shape[0]

        # Perturb states
        rng = torch.Generator(device=device)
        rng.manual_seed(19)
        s_random = torch.rand(y.shape, generator=rng, device=device) - 0.5
        current_delta_t = torch.tensor(current_delta, dtype=y.dtype, device=device).unsqueeze(0)
        y_pert = y + current_delta_t * s_random

        # ------------------------------------------------------------
        # 1. P-Net optimization: Fix L-Net (SBC frozen)
        # ------------------------------------------------------------
        self.l_model.eval()
        self.p_net.train()

        self.p_optimizer.zero_grad()

        # Get reference control + CBF params from QP controller
        # The QP controller needs state for building CBF constraints
        u_safe, q, p, debug = self.p_net.controller_net(
            y_pert,                      # Input: [state_est, velocity]
            state=y_pert,                # Raw state for CBF constraints
            mode='train',                # Differentiable QP
            return_debug=True
        )

        # Forward dynamics: s_{t+1} = f(s_t, u*)
        s_next_p = self.env.v_next(y_pert, u_safe).unsqueeze(1)

        # Add noise for SBC expectation computation
        noise_p = triangular((B, 128, y.shape[1]), device=device, dtype=y.dtype)
        noise_scale = torch.as_tensor(self.env.noise, device=device, dtype=y.dtype).view(1, 1, -1)
        noise_p = noise_p * noise_scale
        s_next_random_p = s_next_p + noise_p

        # Compute SBC values for next states (frozen L-Net)
        with torch.no_grad():
            l_p = self.l_model(y_pert).view(-1)
            l_next_p = self.l_model(
                s_next_random_p.reshape(-1, s_next_random_p.size(-1))
            ).view(B, s_next_random_p.size(1))

        exp_l_next_p = torch.mean(l_next_p, dim=1)
        violations_mean_p = (exp_l_next_p >= l_p).float().mean()

        # Martingale loss (same as original)
        if self.gamma_decrease < 1.0:
            dec_loss_p = martingale_loss(self.gamma_decrease * l_p.detach(), exp_l_next_p, eps=0.0)
        else:
            dec_loss_p = martingale_loss(l_p.detach(), exp_l_next_p, eps=float(self.eps))

        loss_p = dec_loss_p * 10

        # ------------------------------------------------------------
        # 2. Lipschitz regularization for controller
        # ------------------------------------------------------------
        # Compute Lipschitz of the full controller + QP pipeline
        # We compute on the shared backbone input for simplicity
        lip_loss_p = self._compute_controller_lipschitz(y_pert)
        loss_p = loss_p + lip_coeff * lip_loss_p

        # ------------------------------------------------------------
        # 3. MSE: QP output vs teacher (distillation)
        # ------------------------------------------------------------
        with torch.no_grad():
            u_teacher = self.teacher_net(z, y_pert)
        mse_loss_p = F.mse_loss(u_safe, u_teacher.view_as(u_safe))
        loss_p = loss_p + self.lambda_mse * mse_loss_p

        # ------------------------------------------------------------
        # 4. ★ NEW: CBF constraint violation loss
        # ------------------------------------------------------------
        cbf_loss = self.cbf_calculator.cbf_violation_loss(y_pert, u_safe, p.squeeze(-1))
        loss_p = loss_p + self.lambda_cbf * cbf_loss

        # ------------------------------------------------------------
        # 5. Optimize
        # ------------------------------------------------------------
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(
            self.p_net.controller_net.parameters(), clip_grad
        )
        self.p_optimizer.step()

        # ------------------------------------------------------------
        # 6. Metrics
        # ------------------------------------------------------------
        metrics = {
            "loss_p": loss_p.item(),
            "dec_loss_p": dec_loss_p.item(),
            "train_violations_p": violations_mean_p.item(),
            "lip_loss_p": lip_loss_p.item(),
            "mse_loss_p": mse_loss_p.item(),
            "cbf_loss": cbf_loss.item(),
            "p_mean": p.mean().item(),
            "q_mean": q.mean().item(),
            "u_safe_mean": u_safe.mean().item(),
        }
        return metrics

    def _compute_controller_lipschitz(self, y):
        """
        Compute Lipschitz constant of controller input→output mapping.
        This is a simplified version that computes the gradient norm
        of the controller output w.r.t. its input.

        Args:
            y: (batch, obs_dim) state tensor

        Returns:
            lip_loss: Lipschitz regularization loss
        """
        y_input = y.detach().clone().requires_grad_(True)

        # Forward through controller (direct mode to avoid QP gradient complexity)
        u = self.p_net.controller_net(y_input, state=y_input, mode='direct')
        u_sum = u.sum()

        grads = torch.autograd.grad(u_sum, y_input, create_graph=True, retain_graph=True)[0]
        local_lip = grads.view(grads.size(0), -1).norm(p=2, dim=1)
        lip_loss = torch.relu(local_lip - self.p_lip).mean()

        return lip_loss

    def train_step_direct(self, z, y, lip_coeff, current_delta, clip_grad=1.0):
        """
        Direct controller training step (no QP, for baseline comparison).

        This is identical to the original VTLearner.train_step_p but
        uses the DirectController interface for fair comparison.

        Args:
            z, y, lip_coeff, current_delta, clip_grad: Same as train_step_p
        """
        device = self.device
        B = y.shape[0]

        rng = torch.Generator(device=device)
        rng.manual_seed(19)
        s_random = torch.rand(y.shape, generator=rng, device=device) - 0.5
        current_delta_t = torch.tensor(current_delta, dtype=y.dtype, device=device).unsqueeze(0)
        y_pert = y + current_delta_t * s_random

        self.l_model.eval()
        self.p_net.train()

        self.p_optimizer.zero_grad()

        # Direct controller forward (no QP)
        a_p = self.p_net(z, y_pert)

        # SBC computation
        with torch.no_grad():
            l_p = self.l_model(y_pert).view(-1)

        s_next_p = self.env.v_next(y_pert, a_p).unsqueeze(1)
        noise_p = triangular((B, 128, y.shape[1]), device=device, dtype=y.dtype)
        noise_scale = torch.as_tensor(self.env.noise, device=device, dtype=y.dtype).view(1, 1, -1)
        noise_p = noise_p * noise_scale
        s_next_random_p = s_next_p + noise_p

        with torch.no_grad():
            l_next_p = self.l_model(
                s_next_random_p.reshape(-1, s_next_random_p.size(-1))
            ).view(B, s_next_random_p.size(1))

        exp_l_next_p = torch.mean(l_next_p, dim=1)
        violations_mean_p = (exp_l_next_p >= l_p).float().mean()

        if self.gamma_decrease < 1.0:
            dec_loss_p = martingale_loss(self.gamma_decrease * l_p.detach(), exp_l_next_p, eps=0.0)
        else:
            dec_loss_p = martingale_loss(l_p.detach(), exp_l_next_p, eps=float(self.eps))

        loss_p = dec_loss_p * 10

        # Lipschitz
        lip_loss_p = self._compute_controller_lipschitz(y_pert)
        loss_p = loss_p + lip_coeff * lip_loss_p

        # MSE
        with torch.no_grad():
            acc_labels = self.net(z, y_pert)
        mse_loss_p = F.mse_loss(a_p, acc_labels.view_as(a_p))
        loss_p = loss_p + self.lambda_mse * mse_loss_p

        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(self.p_net.controller_net.parameters(), clip_grad)
        self.p_optimizer.step()

        metrics = {
            "loss_p": loss_p.item(),
            "dec_loss_p": dec_loss_p.item(),
            "train_violations_p": violations_mean_p.item(),
            "lip_loss_p": lip_loss_p.item(),
            "mse_loss_p": mse_loss_p.item(),
        }
        return metrics

    def train_epoch(
        self,
        train_ds,
        current_delta,
        lip=0.001,
        batch_size=256,
        shuffle=True,
        num_epochs=10,
        train_fn='both'
    ):
        """
        Training epoch dispatcher. Overrides parent to support QP modes.

        Args:
            train_fn: 'l' (SBC only), 'p' (controller with QP),
                      'p_direct' (controller without QP, baseline),
                      'both' (joint SBC + controller)
        """
        N = train_ds.shape[0]
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=self.device)

        epoch_metrics_list = []

        for epoch in range(num_epochs):
            if shuffle:
                indices = np.random.permutation(N)
                all_y = all_y[indices]

            batch_metrics = []

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y_batch = all_y[start:end]
                z_batch = (torch.rand(y_batch.size(0), 4, device=self.device) * 2.0 - 1.0).float()

                if train_fn == 'l':
                    metrics = self.train_step_l(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                elif train_fn == 'p':
                    # Use QP training if controller_type is 'qp'
                    if self.controller_type == 'qp':
                        metrics = self.train_step_p_qp(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                    else:
                        metrics = self.train_step_direct(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                elif train_fn == 'p_direct':
                    # Force direct training (baseline)
                    metrics = self.train_step_direct(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                elif train_fn == 'both':
                    metrics = self.train_step_joint(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                else:
                    metrics_l = self.train_step_l(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                    if self.controller_type == 'qp':
                        metrics_p = self.train_step_p_qp(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                    else:
                        metrics_p = self.train_step_direct(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                    metrics = {**metrics_l, **metrics_p}

                batch_metrics.append(metrics)

            epoch_metrics = {k: np.mean([bm[k] for bm in batch_metrics]) for k in batch_metrics[0].keys()}
            epoch_metrics_list.append(epoch_metrics)

        return epoch_metrics_list


# For testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing QPVTLearner")
    print("=" * 60)

    # This test requires the original SafePVC environment setup
    # Skip in standalone test; run as part of full experiment
    print("QPVTLearner requires full SafePVC environment.")
    print("Run 'run_qp_experiment.py' for full integration test.")
