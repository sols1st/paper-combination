"""
Closed-loop simulation verifier for BarrierNet-AEBS.

Replaces SafePVC's IBP formal verification with simulation-based
safety assessment. BarrierNet provides safety by construction
(QP constraint ensures HOCBF condition at each step), so simulation
serves as empirical validation.

Workflow:
    1. Initialize state from initial set (d in [15,16], v in [2.5,3.0])
    2. At each step: BarrierNet(state) -> safe control -> dynamics update
    3. Check: trajectory never enters unsafe set (d in [5,6], v in [0.5,3.0])
    4. Report: safety rate, min barrier value, plots
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import os


class AEBSSimulator:
    """
    Closed-loop simulation verifier for BarrierNet-AEBS.
    """

    def __init__(self, barrier_net, env, dt=0.05, max_steps=500):
        """
        Args:
            barrier_net: Trained BarrierNetAEBS model
            env: Aebs environment
            dt: Time step (seconds)
            max_steps: Maximum simulation steps per episode
        """
        self.barrier_net = barrier_net
        self.env = env
        self.dt = dt
        self.max_steps = max_steps
        self.device = next(barrier_net.parameters()).device
        self.std1 = env.std1

    def simulate_episode(self, d0, v0, add_noise=False, noise_factor=0.05):
        """
        Simulate a single closed-loop trajectory.

        Args:
            d0: Initial real distance (meters)
            v0: Initial speed (m/s)
            add_noise: Whether to add state perturbation
            noise_factor: Noise magnitude factor
        Returns:
            dict with trajectory data and safety flag
        """
        self.barrier_net.eval()

        traj = {
            't': [], 'd': [], 'v': [], 'acc': [],
            'b': [], 'p1': [], 'p2': [],
        }

        d, v = d0, v0
        is_safe = True
        min_barrier = float('inf')

        with torch.no_grad():
            for step in range(self.max_steps):
                # Normalized state
                d_norm = d / self.std1
                state = torch.tensor(
                    [[d_norm, v]], dtype=torch.float32, device=self.device
                )

                # BarrierNet forward (inference: cvxopt)
                acc = self.barrier_net(state, sgn=0)  # (1, 1)
                acc_val = acc.item()

                # Record data
                b_val = d - self.barrier_net.d_safe
                traj['t'].append(step * self.dt)
                traj['d'].append(d)
                traj['v'].append(v)
                traj['acc'].append(acc_val)
                traj['b'].append(b_val)
                traj['p1'].append(self.barrier_net.p1)
                traj['p2'].append(self.barrier_net.p2)

                min_barrier = min(min_barrier, b_val)

                # Check unsafe condition: close distance AND high speed
                if d <= 6.0 and v > 0.5:
                    is_safe = False

                # Dynamics update: d_next = d - v*dt, v_next = v - acc*dt
                d_next = d - v * self.dt
                v_next = v - acc_val * self.dt

                # Add stochastic perturbation
                if add_noise:
                    d_range = self.env.observation_space.high[0] - self.env.observation_space.low[0]
                    v_range = self.env.observation_space.high[1] - self.env.observation_space.low[1]
                    noise_d = np.random.uniform(-1, 1) * d_range * noise_factor * self.std1
                    noise_v = np.random.uniform(-1, 1) * v_range * noise_factor
                    d_next += noise_d
                    v_next += noise_v

                # Clamp velocity
                v_next = np.clip(v_next, 0.0, 3.0)

                d, v = d_next, v_next

                # Termination conditions (matching AebsEnv)
                if d >= 16.0 or d <= 5.0 or v <= 0.0:
                    break

        traj['safe'] = is_safe
        traj['min_barrier'] = min_barrier
        traj['length'] = len(traj['t'])

        return traj

    def batch_evaluate(self, n_episodes=100, d_range=(15.0, 16.0),
                       v_range=(2.5, 3.0), add_noise=False,
                       noise_factor=0.05):
        """
        Batch evaluation over random initial conditions.

        Args:
            n_episodes: Number of episodes to simulate
            d_range: (min, max) initial distance range (meters)
            v_range: (min, max) initial speed range (m/s)
            add_noise: Whether to add stochastic perturbation
            noise_factor: Noise magnitude
        Returns:
            dict with aggregate metrics
        """
        results = {
            'safe_count': 0,
            'total': n_episodes,
            'min_barriers': [],
            'lengths': [],
            'min_distances': [],
            'trajectories': [],
        }

        for ep in range(n_episodes):
            d0 = np.random.uniform(d_range[0], d_range[1])
            v0 = np.random.uniform(v_range[0], v_range[1])

            traj = self.simulate_episode(
                d0, v0, add_noise=add_noise, noise_factor=noise_factor
            )

            if traj['safe']:
                results['safe_count'] += 1

            results['min_barriers'].append(traj['min_barrier'])
            results['lengths'].append(traj['length'])
            results['min_distances'].append(min(traj['d']))
            results['trajectories'].append(traj)

        results['safety_rate'] = results['safe_count'] / n_episodes * 100
        results['mean_min_barrier'] = np.mean(results['min_barriers'])
        results['min_min_barrier'] = np.min(results['min_barriers'])
        results['mean_length'] = np.mean(results['lengths'])
        results['mean_min_distance'] = np.mean(results['min_distances'])

        return results

    def print_results(self, results, label="BarrierNet"):
        """Pretty-print evaluation results."""
        print(f"\n{'='*60}")
        print(f"  {label} — Simulation Results")
        print(f"{'='*60}")
        print(f"  Episodes:           {results['total']}")
        print(f"  Safety Rate:        {results['safety_rate']:.1f}%"
              f" ({results['safe_count']}/{results['total']})")
        print(f"  Min Barrier (mean): {results['mean_min_barrier']:.4f}")
        print(f"  Min Barrier (worst):{results['min_min_barrier']:.4f}")
        print(f"  Mean Episode Len:   {results['mean_length']:.1f} steps")
        print(f"  Mean Min Distance:  {results['mean_min_distance']:.2f} m")
        print(f"{'='*60}\n")

    def plot_trajectory(self, traj, save_path=None):
        """Plot a single trajectory with 4 subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        t = traj['t']

        # (1) Distance over time
        ax = axes[0, 0]
        ax.plot(t, traj['d'], 'b-', linewidth=1.5)
        ax.axhline(y=6.0, color='r', linestyle='--', label='d_safe = 6.0m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.set_title('Distance Trajectory')
        ax.legend()

        # (2) Speed over time
        ax = axes[0, 1]
        ax.plot(t, traj['v'], 'g-', linewidth=1.5)
        ax.axhline(y=0.5, color='r', linestyle='--', label='v_safe = 0.5 m/s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Speed Trajectory')
        ax.legend()

        # (3) Barrier function value
        ax = axes[1, 0]
        ax.plot(t, traj['b'], 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', label='b=0 (safety boundary)')
        ax.fill_between(t, 0, traj['b'], where=[b >= 0 for b in traj['b']],
                         alpha=0.2, color='green', label='Safe (b >= 0)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('b(x) = d - d_safe')
        ax.set_title('Barrier Function Value')
        ax.legend()

        # (4) HOCBF penalty functions
        ax = axes[1, 1]
        ax.plot(t, traj['p1'], 'm-', linewidth=1.5, label='p1(z)')
        ax.plot(t, traj['p2'], 'c-', linewidth=1.5, label='p2(z)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Penalty Value')
        ax.set_title('HOCBF Penalty Functions')
        ax.legend()

        plt.suptitle(
            f"Trajectory: safe={'YES' if traj['safe'] else 'NO'}, "
            f"min_b={traj['min_barrier']:.3f}",
            fontsize=14,
        )
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to {save_path}")

        plt.close(fig)

    def plot_phase_portrait(self, results, save_path=None):
        """Plot d-v phase portrait with all trajectories."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all trajectories
        for traj in results['trajectories']:
            color = 'green' if traj['safe'] else 'red'
            alpha = 0.3 if traj['safe'] else 0.8
            ax.plot(traj['d'], traj['v'], '-', color=color,
                    alpha=alpha, linewidth=0.8)

        # Mark initial set
        ax.add_patch(plt.Rectangle(
            (15.0, 2.5), 1.0, 0.5,
            fill=False, edgecolor='blue', linewidth=2,
            label='Initial Set'
        ))

        # Mark unsafe set
        ax.add_patch(plt.Rectangle(
            (5.0, 0.5), 1.0, 2.5,
            fill=True, facecolor='red', alpha=0.2,
            edgecolor='red', linewidth=2,
            label='Unsafe Set'
        ))

        # Safety boundary
        ax.axvline(x=6.0, color='orange', linestyle='--',
                   linewidth=1.5, label='d_safe = 6.0m')

        ax.set_xlabel('Distance d (m)', fontsize=12)
        ax.set_ylabel('Speed v (m/s)', fontsize=12)
        ax.set_title('Phase Portrait (d vs v)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(4.5, 16.5)
        ax.set_ylim(0, 3.5)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Phase portrait saved to {save_path}")

        plt.close(fig)

    def plot_batch_barriers(self, results, save_path=None):
        """Plot barrier function values over time for multiple episodes."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for traj in results['trajectories'][:50]:  # plot first 50
            color = 'green' if traj['safe'] else 'red'
            alpha = 0.3 if traj['safe'] else 0.8
            ax.plot(traj['t'], traj['b'], '-', color=color,
                    alpha=alpha, linewidth=0.8)

        ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5,
                    label='Safety boundary (b=0)')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('b(x) = d - d_safe', fontsize=12)
        ax.set_title('Barrier Function Over Time (batch)', fontsize=14)
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Batch barrier plot saved to {save_path}")

        plt.close(fig)
