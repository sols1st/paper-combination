#!/usr/bin/env python3
"""
Run All Comparison Experiments

Runs the full experiment matrix:
- Baseline (SafePVC without QP) × 3 noise factors [0.01, 0.05, 0.10]
- QP Integrated (SafePVC + BarrierNet QP) × 3 noise factors [0.01, 0.05, 0.10]

Each experiment runs for the configured timeout (default: 1 hour).
Results are saved to 实验v2/results/.

Usage:
    cd /root/paper-combination/实验v2/code
    python run_all_experiments.py [--quick] [--timeout 600]
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def run_experiment_suite(configs, results_dir_base):
    """
    Run a suite of experiments.

    Args:
        configs: List of (name, config_dict) tuples
        results_dir_base: Base directory for results

    Returns:
        summary: Dict with summary of all experiments
    """
    from training.qp_loop import run_experiment, get_default_config

    summary = {}
    total = len(configs)

    for i, (name, overrides) in enumerate(configs):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i+1}/{total}: {name}")
        print(f"{'#'*60}")

        config = get_default_config()
        config.update(overrides)
        config['results_dir'] = os.path.join(results_dir_base, name)

        try:
            start = time.time()
            results = run_experiment(config)
            elapsed = time.time() - start

            summary[name] = {
                'status': 'completed',
                'max_prob': results['max_reach_prob'],
                'iterations': results['final_iter'],
                'runtime': elapsed,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            summary[name] = {
                'status': 'failed',
                'error': str(e),
            }

        # Save incremental summary
        summary_path = os.path.join(results_dir_base, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run All Comparison Experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: shorter timeout (10 min per experiment)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per experiment in seconds')
    parser.add_argument('--noise-factors', type=str, default='0.01,0.05,0.10',
                        help='Comma-separated noise factors')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline experiments')
    parser.add_argument('--skip-qp', action='store_true',
                        help='Skip QP experiments')
    args = parser.parse_args()

    if args.quick:
        args.timeout = 600  # 10 minutes per experiment
        print("Quick mode: 10 min per experiment")

    noise_factors = [float(x) for x in args.noise_factors.split(',')]

    # Results base directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', f'suite_{timestamp}'
    )
    os.makedirs(results_base, exist_ok=True)

    # Build experiment list
    experiments = []

    if not args.skip_baseline:
        for nf in noise_factors:
            experiments.append((
                f'baseline_nf{nf:.2f}'.replace('.', '_'),
                {
                    'controller_type': 'direct',
                    'noise_factor': nf,
                    'timeout': args.timeout,
                }
            ))

    if not args.skip_qp:
        for nf in noise_factors:
            experiments.append((
                f'qp_nf{nf:.2f}_cbf1.0'.replace('.', '_'),
                {
                    'controller_type': 'qp',
                    'noise_factor': nf,
                    'lambda_cbf': 1.0,
                    'timeout': args.timeout,
                }
            ))

    print(f"\nExperiment Suite: {len(experiments)} experiments")
    for name, cfg in experiments:
        print(f"  - {name}: controller={cfg['controller_type']}, "
              f"noise={cfg['noise_factor']}, timeout={cfg['timeout']}s")
    print(f"Results: {results_base}")

    # Save experiment config
    config_path = os.path.join(results_base, 'suite_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'experiments': [(n, c) for n, c in experiments],
            'timestamp': timestamp,
            'quick_mode': args.quick,
        }, f, indent=2, default=str)

    # Run all experiments
    summary = run_experiment_suite(experiments, results_base)

    # Print final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'Status':<12} {'Max Prob':<12} {'Iters':<8}")
    print("-" * 67)
    for name, info in summary.items():
        status = info.get('status', 'unknown')
        prob = info.get('max_prob', 0) * 100
        iters = info.get('iterations', '-')
        print(f"{name:<35} {status:<12} {prob:>8.2f}%   {iters:>6}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
