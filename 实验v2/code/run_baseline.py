#!/usr/bin/env python3
"""
Run Baseline Experiment: Original SafePVC (without QP)

This runs the original SafePVC pipeline with the direct controller
(no CBF-QP safety filter) for comparison against the QP-integrated version.

Usage:
    cd /root/paper-combination/实验v2/code
    python run_baseline.py [--noise-factor 0.05] [--timeout 3600]

The original artical-F122 codebase is NOT modified.
This script imports from it and configures the experiment.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from training.qp_loop import run_experiment, get_default_config


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Baseline SafePVC Experiment')
    parser.add_argument('--noise-factor', type=float, default=0.05)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    config = get_default_config()
    config['controller_type'] = 'direct'  # No QP - baseline
    config['noise_factor'] = args.noise_factor
    config['timeout'] = args.timeout

    # Results directory
    tag = f"_{args.tag}" if args.tag else ""
    config['results_dir'] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', f'baseline{tag}'
    )

    print("=" * 60)
    print("BASELINE EXPERIMENT: Original SafePVC (no QP)")
    print(f"  Noise factor: {config['noise_factor']}")
    print(f"  Timeout: {config['timeout']}s")
    print(f"  Results: {config['results_dir']}")
    print("=" * 60)

    results = run_experiment(config)

    print(f"\nBaseline complete. Max prob: {results['max_reach_prob']*100:.2f}%")
    return results


if __name__ == "__main__":
    main()
