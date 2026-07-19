#!/usr/bin/env python3
"""
Run QP-Integrated Experiment: SafePVC + BarrierNet QP Safety Filter

This runs the modified SafePVC pipeline with the dual-branch controller
and CBF-QP safety filter layer.

Usage:
    cd /root/paper-combination/实验v2/code
    python run_qp_experiment.py [--noise-factor 0.05] [--lambda-cbf 1.0] [--timeout 3600]

The original artical-F122 codebase is NOT modified.
This script imports from it and configures the experiment with QP augmentation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from training.qp_loop import run_experiment, get_default_config


def main():
    import argparse
    parser = argparse.ArgumentParser(description='SafePVC + QP Experiment')
    parser.add_argument('--noise-factor', type=float, default=0.05,
                        help='State perturbation factor (0.01, 0.05, 0.10)')
    parser.add_argument('--lambda-cbf', type=float, default=1.0,
                        help='CBF loss weight')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Experiment timeout in seconds')
    parser.add_argument('--t-gap', type=float, default=1.5,
                        help='Safe time headway (seconds)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Experiment tag for results naming')
    args = parser.parse_args()

    config = get_default_config()
    config['controller_type'] = 'qp'  # QP-augmented controller
    config['noise_factor'] = args.noise_factor
    config['lambda_cbf'] = args.lambda_cbf
    config['timeout'] = args.timeout
    config['controller_config']['t_gap'] = args.t_gap

    # Results directory
    tag = f"_{args.tag}" if args.tag else ""
    config['results_dir'] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', f'qp_integrated{tag}'
    )

    print("=" * 60)
    print("QP-INTEGRATED EXPERIMENT: SafePVC + BarrierNet QP")
    print(f"  Noise factor: {config['noise_factor']}")
    print(f"  CBF loss weight: {config['lambda_cbf']}")
    print(f"  Safe time gap: {args.t_gap}s")
    print(f"  Timeout: {config['timeout']}s")
    print(f"  Results: {config['results_dir']}")
    print("=" * 60)

    results = run_experiment(config)

    print(f"\nQP experiment complete. Max prob: {results['max_reach_prob']*100:.2f}%")
    return results


if __name__ == "__main__":
    main()
