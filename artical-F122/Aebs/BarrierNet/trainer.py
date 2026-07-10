"""
BarrierNet Trainer for AEBS

Trains the BarrierNetAEBS model via behavioral cloning from a frozen
PPO teacher controller. The teacher uses the full VCLS pipeline
(gen_net -> state_net -> controller_net) to generate reference labels,
averaged over multiple z noise samples for stability.

Training loop:
    1. Sample states uniformly from observation space
    2. Generate PPO teacher labels (averaged over z)
    3. BarrierNet forward pass (differentiable QP)
    4. MSE loss: BarrierNet output vs teacher labels
    5. Backprop through QP layer -> update network weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from Combined_network.model import AebsEnd2EndNet
from cGAN.taxi_models_and_data import AebsMLPGenerator
from stable_baselines3 import PPO


def load_ppo_teacher(device):
    """
    Load the frozen PPO teacher network (VCLS pipeline).
    Exact pattern from VT/train.py lines 49-62.

    Returns:
        AebsEnd2EndNet (frozen, eval mode)
    """
    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(
        torch.load("./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth",
                   map_location=device)
    )

    state_layer_sizes = [1024, 256, 64, 1]
    model = PPO.load('./Aebs/controller/best_model/best_model.zip')
    policy = model.policy
    mlp_extractor = policy.mlp_extractor.policy_net
    action_net = policy.action_net

    p_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)
    p_net.state_net.load_state_dict(
        torch.load("./Aebs/controller/state_net_trained.pth",
                   map_location=device)
    )

    p_net.eval()
    p_net.to(device)

    # Freeze all parameters
    for param in p_net.parameters():
        param.requires_grad = False

    return p_net


class BarrierNetTrainer:
    """
    Trainer for BarrierNet-AEBS with behavioral cloning from PPO teacher.
    """

    def __init__(
        self,
        barrier_net,
        teacher_net,
        env,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        num_z=10,
        max_grad_norm=5.0,
    ):
        """
        Args:
            barrier_net: BarrierNetAEBS model
            teacher_net: Frozen AebsEnd2EndNet (PPO teacher)
            env: Aebs environment (provides observation_space, std1)
            lr: Learning rate
            weight_decay: L2 regularization
            batch_size: Training batch size
            num_z: Number of z samples for teacher label averaging
            max_grad_norm: Gradient clipping threshold
        """
        self.barrier_net = barrier_net
        self.teacher_net = teacher_net
        self.env = env
        self.batch_size = batch_size
        self.num_z = num_z
        self.max_grad_norm = max_grad_norm
        self.device = next(barrier_net.parameters()).device

        self.optimizer = torch.optim.Adam(
            barrier_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.MSELoss()

    @torch.no_grad()
    def generate_teacher_labels(self, states):
        """
        Generate PPO teacher action labels for given states.
        Averages over num_z random z vectors for stability.

        Args:
            states: (B, 2) tensor [d_norm, v]
        Returns:
            (B, 1) tensor of teacher actions (acceleration)
        """
        B = states.shape[0]
        num_z = self.num_z

        # Replicate each state for num_z different z samples
        states_rep = states.unsqueeze(1).expand(-1, num_z, -1)
        states_rep = states_rep.reshape(B * num_z, 2)

        # Random z ~ U(-1, 1)^4
        z = (torch.rand(B * num_z, 4, device=self.device) * 2.0 - 1.0).float()

        # Teacher forward (frozen, no gradients)
        acc = self.teacher_net(z, states_rep)  # (B*num_z, 1)
        acc = acc.view(B, num_z, 1)

        # Average over z samples
        acc_mean = acc.mean(dim=1)  # (B, 1)
        acc_mean = torch.clamp(acc_mean, -3.0, 3.0)

        return acc_mean.float()

    def sample_training_states(self, n):
        """
        Sample states uniformly from the observation space.

        Args:
            n: Number of samples
        Returns:
            (n, 2) tensor [d_norm, v]
        """
        low = torch.tensor(self.env.observation_space.low,
                           dtype=torch.float32, device=self.device)
        high = torch.tensor(self.env.observation_space.high,
                            dtype=torch.float32, device=self.device)
        states = low + (high - low) * torch.rand(n, 2, device=self.device)
        return states

    def train(self, num_epochs=50, samples_per_epoch=10000, verbose=True):
        """
        Full training loop.

        Args:
            num_epochs: Number of training epochs
            samples_per_epoch: Training samples per epoch
            verbose: Print progress
        Returns:
            dict of training history
        """
        history = {
            'train_loss': [],
            'mean_barrier': [],
            'constraint_active': [],
        }

        if verbose:
            print(f"Training BarrierNet-AEBS: {num_epochs} epochs, "
                  f"{samples_per_epoch} samples/epoch, "
                  f"batch_size={self.batch_size}")
            print(f"Device: {self.device}")
            print(f"Robust margin: {self.barrier_net.robust_margin}")
            print("-" * 60)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Sample training states for this epoch
            all_states = self.sample_training_states(samples_per_epoch)

            # Generate teacher labels (on-the-fly, batch by batch)
            self.barrier_net.train()
            epoch_loss = 0.0
            epoch_barrier = 0.0
            epoch_active = 0.0
            n_batches = 0

            # Shuffle
            perm = torch.randperm(all_states.size(0), device=self.device)
            all_states = all_states[perm]

            for start in range(0, all_states.size(0), self.batch_size):
                end = min(start + self.batch_size, all_states.size(0))
                batch_states = all_states[start:end]

                # Generate teacher labels for this batch
                self.teacher_net.eval()
                teacher_labels = self.generate_teacher_labels(batch_states)

                # BarrierNet forward pass (differentiable QP)
                u_safe = self.barrier_net(batch_states, sgn=1)  # (B, 1)

                # Ensure shape match
                if u_safe.dim() == 1:
                    u_safe = u_safe.unsqueeze(1)

                # MSE loss
                loss = self.loss_fn(u_safe, teacher_labels)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.barrier_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Metrics
                epoch_loss += loss.item()
                with torch.no_grad():
                    b_vals = self.barrier_net.get_barrier_value(batch_states)
                    epoch_barrier += b_vals.mean().item()

                    # Check how often HOCBF constraint is active
                    # (u_safe != teacher action => constraint modified the output)
                    diff = (u_safe - teacher_labels).abs()
                    active = (diff > 0.01).float().mean().item()
                    epoch_active += active

                n_batches += 1

            # Epoch averages
            avg_loss = epoch_loss / n_batches
            avg_barrier = epoch_barrier / n_batches
            avg_active = epoch_active / n_batches

            history['train_loss'].append(avg_loss)
            history['mean_barrier'].append(avg_barrier)
            history['constraint_active'].append(avg_active)

            if verbose and (epoch + 1) % 5 == 0:
                elapsed = time.time() - epoch_start
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Barrier: {avg_barrier:.3f} | "
                      f"Active: {avg_active*100:.1f}% | "
                      f"Time: {elapsed:.1f}s")

        if verbose:
            print("-" * 60)
            print(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")

        return history
