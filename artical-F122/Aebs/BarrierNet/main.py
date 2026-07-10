"""
SafePVC-BarrierNet Main Entry Point

Replaces the original Aebs/VT/loop.py CEGIS loop with BarrierNet
end-to-end training and simulation-based verification.

Pipeline:
    1. Load environment (Aebs class with std1, spaces, dynamics)
    2. Load frozen PPO teacher (VCLS: gen_net + state_net + controller_net)
    3. Create BarrierNet-AEBS (HOCBF-QP safety layer)
    4. Train via behavioral cloning (MSE vs teacher, gradients through QP)
    5. Verify via closed-loop simulation (batch evaluation)
    6. Save model + generate plots

Usage:
    python -m Aebs.BarrierNet.main
    python run_barriernet.py --epochs 50 --num-episodes 1000
"""

import argparse
import sys
import os
import time
import torch
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from Aebs.system.env import Aebs
from Aebs.BarrierNet.barrier_net_aebs import BarrierNetAEBS
from Aebs.BarrierNet.trainer import BarrierNetTrainer, load_ppo_teacher
from Aebs.BarrierNet.simulator import AEBSSimulator


def parse_args():
    parser = argparse.ArgumentParser(
        description='BarrierNet-AEBS: HOCBF-QP safe controller training'
    )

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Shared trunk hidden size (default: 64)')
    parser.add_argument('--p-hidden-dim', type=int, default=32,
                        help='p_head (ref control) hidden size (default: 32)')
    parser.add_argument('--q-hidden-dim', type=int, default=32,
                        help='q_head (penalty) hidden size (default: 32)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization (default: 1e-4)')
    parser.add_argument('--samples-per-epoch', type=int, default=10000,
                        help='Training samples per epoch (default: 10000)')
    parser.add_argument('--num-z', type=int, default=10,
                        help='Num z samples for teacher averaging (default: 10)')

    # Safety parameters
    parser.add_argument('--d-safe', type=float, default=6.0,
                        help='Safety distance threshold in meters (default: 6.0)')
    parser.add_argument('--robust-margin', type=float, default=0.0,
                        help='Robust margin delta for HOCBF (default: 0.0)')
    parser.add_argument('--noise-factor', type=float, default=0.05,
                        help='Environment noise factor (default: 0.05)')

    # Verification
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Verification simulation episodes (default: 1000)')
    parser.add_argument('--no-noise-eval', action='store_true',
                        help='Disable noise during evaluation')

    # I/O
    parser.add_argument('--save-dir', type=str, default='./Aebs/BarrierNet/results',
                        help='Directory for saving results (default: ./Aebs/BarrierNet/results)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to pre-trained BarrierNet model (skip training)')

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # ========== Setup ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Arguments: {vars(args)}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 1. Environment ==========
    env = Aebs(factor=args.noise_factor)
    print(f"Environment loaded: std1={env.std1:.4f}")
    print(f"  Observation space: [{env.observation_space.low}, {env.observation_space.high}]")
    print(f"  Init spaces: d in [15,16]/std1, v in [2.5,3.0]")
    print(f"  Unsafe spaces: d in [5,6]/std1, v in [0.5,3.0]")

    # ========== 2. PPO Teacher ==========
    teacher_net = load_ppo_teacher(device)
    print("PPO teacher loaded (frozen).")

    # Quick teacher sanity check
    with torch.no_grad():
        test_state = torch.tensor([[15.0 / float(env.std1), 2.75]],
                                  dtype=torch.float32, device=device)
        z = torch.zeros(1, 4, device=device)
        test_acc = teacher_net(z, test_state)
        print(f"  Teacher output for d=15, v=2.75: acc={test_acc.item():.4f}")

    # ========== 3. BarrierNet ==========
    barrier_net = BarrierNetAEBS(
        nFeatures=2,
        nHidden1=args.hidden_dim,
        nHidden21=args.p_hidden_dim,
        nHidden22=args.q_hidden_dim,
        nCls=1,
        d_safe=args.d_safe,
        u_min=-3.0,
        u_max=3.0,
        std1=env.std1,
        robust_margin=args.robust_margin,
        device=device,
    ).to(device)

    print(f"BarrierNet-AEBS created:")
    print(f"  Architecture: 2 -> {args.hidden_dim} -> [{args.p_hidden_dim}, {args.q_hidden_dim}] -> 1")
    print(f"  d_safe={args.d_safe}, robust_margin={args.robust_margin}")
    total_params = sum(p.numel() for p in barrier_net.parameters())
    print(f"  Total parameters: {total_params}")

    # ========== 4. Training ==========
    if args.load_path:
        print(f"\nLoading pre-trained model from {args.load_path}")
        barrier_net.load_state_dict(torch.load(args.load_path, map_location=device))
        print("Model loaded. Skipping training.")
    else:
        print(f"\n{'='*60}")
        print("  Phase 1: Training BarrierNet-AEBS")
        print(f"{'='*60}")

        trainer = BarrierNetTrainer(
            barrier_net=barrier_net,
            teacher_net=teacher_net,
            env=env,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_z=args.num_z,
        )

        history = trainer.train(
            num_epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
        )

        # Save model
        model_path = os.path.join(args.save_dir, 'barrier_net_aebs.pth')
        torch.save(barrier_net.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

        # Save training history
        history_path = os.path.join(args.save_dir, 'train_history.npy')
        np.save(history_path, history)

    # ========== 5. Verification ==========
    print(f"\n{'='*60}")
    print("  Phase 2: Closed-Loop Simulation Verification")
    print(f"{'='*60}")

    simulator = AEBSSimulator(barrier_net, env, dt=0.05, max_steps=500)

    # Nominal evaluation (no noise)
    print(f"\n--- Nominal Evaluation (no noise, {args.num_episodes} episodes) ---")
    results_nominal = simulator.batch_evaluate(
        n_episodes=args.num_episodes,
        d_range=(15.0, 16.0),
        v_range=(2.5, 3.0),
        add_noise=False,
    )
    simulator.print_results(results_nominal, label="BarrierNet (nominal)")

    # Stochastic evaluation (with noise)
    if not args.no_noise_eval:
        print(f"\n--- Stochastic Evaluation (noise_factor={args.noise_factor}, "
              f"{args.num_episodes} episodes) ---")
        results_noisy = simulator.batch_evaluate(
            n_episodes=args.num_episodes,
            d_range=(15.0, 16.0),
            v_range=(2.5, 3.0),
            add_noise=True,
            noise_factor=args.noise_factor,
        )
        simulator.print_results(results_noisy, label="BarrierNet (noisy)")

    # PPO baseline (no safety guarantee)
    print(f"\n--- PPO Baseline (no safety, {args.num_episodes} episodes) ---")
    ppo_results = _evaluate_ppo_baseline(
        teacher_net, env, args.num_episodes,
        add_noise=not args.no_noise_eval,
        noise_factor=args.noise_factor,
        device=device,
    )
    print(f"  PPO Safety Rate: {ppo_results['safety_rate']:.1f}%"
          f" ({ppo_results['safe_count']}/{ppo_results['total']})")

    # ========== 6. Plots ==========
    print(f"\n{'='*60}")
    print("  Phase 3: Generating Plots")
    print(f"{'='*60}")

    # Plot nominal results
    simulator.plot_phase_portrait(
        results_nominal,
        save_path=os.path.join(args.save_dir, 'phase_portrait_nominal.png')
    )
    simulator.plot_batch_barriers(
        results_nominal,
        save_path=os.path.join(args.save_dir, 'barrier_values_nominal.png')
    )

    # Plot a typical trajectory
    d0 = 15.5
    v0 = 2.75
    traj = simulator.simulate_episode(d0, v0, add_noise=False)
    simulator.plot_trajectory(
        traj,
        save_path=os.path.join(args.save_dir, f'trajectory_d{d0}_v{v0}.png')
    )

    if not args.no_noise_eval:
        simulator.plot_phase_portrait(
            results_noisy,
            save_path=os.path.join(args.save_dir, 'phase_portrait_noisy.png')
        )

    # ========== Summary ==========
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time:              {elapsed/60:.1f} minutes")
    print(f"  BarrierNet (nominal):    {results_nominal['safety_rate']:.1f}% safe")
    if not args.no_noise_eval:
        print(f"  BarrierNet (noisy):      {results_noisy['safety_rate']:.1f}% safe")
    print(f"  PPO baseline:            {ppo_results['safety_rate']:.1f}% safe")
    print(f"  Results saved to:        {args.save_dir}")
    print(f"{'='*60}")


def _evaluate_ppo_baseline(teacher_net, env, n_episodes,
                            add_noise=False, noise_factor=0.05, device='cpu'):
    """Evaluate PPO teacher without safety layer (baseline comparison)."""
    safe_count = 0
    dt = 0.05

    for _ in range(n_episodes):
        d = np.random.uniform(15.0, 16.0)
        v = np.random.uniform(2.5, 3.0)
        is_safe = True

        for step in range(500):
            # Check unsafe
            if d <= 6.0 and v > 0.5:
                is_safe = False
                break

            # PPO teacher action (z=0 for deterministic)
            d_norm = d / env.std1
            state = torch.tensor([[d_norm, v]], dtype=torch.float32, device=device)
            z = torch.zeros(1, 4, device=device)

            with torch.no_grad():
                acc = teacher_net(z, state).item()

            # Dynamics
            d_next = d - v * dt
            v_next = v - acc * dt

            if add_noise:
                d_range = env.observation_space.high[0] - env.observation_space.low[0]
                v_range = env.observation_space.high[1] - env.observation_space.low[1]
                d_next += np.random.uniform(-1, 1) * d_range * noise_factor * env.std1
                v_next += np.random.uniform(-1, 1) * v_range * noise_factor

            v_next = np.clip(v_next, 0.0, 3.0)
            d, v = d_next, v_next

            if d >= 16.0 or d <= 5.0 or v <= 0.0:
                break

        if is_safe:
            safe_count += 1

    return {
        'safety_rate': safe_count / n_episodes * 100,
        'safe_count': safe_count,
        'total': n_episodes,
    }


if __name__ == '__main__':
    main()
