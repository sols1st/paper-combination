#!/usr/bin/env python3
"""
Analyze and Visualize Experiment Results

Reads results from baseline and QP-integrated experiments,
produces comparison plots and summary tables.

Usage:
    cd /root/paper-combination/实验v2/code
    python analyze_results.py --results-dir ../results/
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def load_experiment_results(results_dir):
    """Load all experiment results from a directory tree."""
    experiments = {}

    for root, dirs, files in os.walk(results_dir):
        results_file = os.path.join(root, 'final_results.json')
        history_file = os.path.join(root, 'training_history.npz')

        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)

            exp_name = os.path.relpath(root, results_dir)
            experiments[exp_name] = {'results': results}

            if os.path.exists(history_file):
                history = np.load(history_file)
                experiments[exp_name]['history'] = {
                    k: history[k] for k in history.files
                }

    return experiments


def plot_probability_comparison(experiments, save_path):
    """Plot probability lower bound comparison across experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Parse experiments by type and noise factor
    baseline_exps = {}
    qp_exps = {}

    for name, exp in experiments.items():
        results = exp['results']
        ctype = results.get('controller_type', 'unknown')
        config = results.get('config', {})
        nf = config.get('noise_factor', 0.05)

        if 'history' in exp:
            if ctype == 'direct':
                baseline_exps[nf] = exp
            elif ctype == 'qp':
                qp_exps[nf] = exp

    # Plot 1: Probability vs iterations for all experiments
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    for (nf, exp), c in zip(sorted(baseline_exps.items()), colors[:3]):
        hist = exp['history']
        label = f'Baseline (nf={nf:.2f})'
        if len(hist['prob_lower_bound']) > 0:
            ax.plot(hist['iterations'], hist['prob_lower_bound'] * 100,
                   'o-', color=c, label=label, markersize=3, alpha=0.8)

    for (nf, exp), c in zip(sorted(qp_exps.items()), colors[3:]):
        hist = exp['history']
        label = f'QP Integrated (nf={nf:.2f})'
        if len(hist['prob_lower_bound']) > 0:
            ax.plot(hist['iterations'], hist['prob_lower_bound'] * 100,
                   's-', color=c, label=label, markersize=3, alpha=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability Lower Bound (%)')
    ax.set_title('Safety Probability Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Plot 2: Bar chart - max probability by noise factor
    ax = axes[1]
    noise_factors = sorted(set(
        list(baseline_exps.keys()) + list(qp_exps.keys())
    ))
    x = np.arange(len(noise_factors))
    width = 0.35

    baseline_probs = []
    qp_probs = []
    for nf in noise_factors:
        b_prob = max(baseline_exps[nf]['history']['prob_lower_bound']) * 100 \
            if nf in baseline_exps and len(baseline_exps[nf]['history']['prob_lower_bound']) > 0 else 0
        q_prob = max(qp_exps[nf]['history']['prob_lower_bound']) * 100 \
            if nf in qp_exps and len(qp_exps[nf]['history']['prob_lower_bound']) > 0 else 0
        baseline_probs.append(b_prob)
        qp_probs.append(q_prob)

    ax.bar(x - width/2, baseline_probs, width, label='Baseline (SafePVC)', color='#3498db')
    ax.bar(x + width/2, qp_probs, width, label='SafePVC + QP', color='#e74c3c')
    ax.set_xlabel('Noise Factor')
    ax.set_ylabel('Max Probability Lower Bound (%)')
    ax.set_title('Safety Probability vs Perturbation Strength')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nf:.2f}' for nf in noise_factors])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Probability comparison saved to {save_path}")


def plot_violations_comparison(experiments, save_path):
    """Plot hard violation count comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, exp in experiments.items():
        if 'history' not in exp:
            continue
        hist = exp['history']
        results = exp['results']
        ctype = results.get('controller_type', 'unknown')
        config = results.get('config', {})
        nf = config.get('noise_factor', 0.05)

        style = 'o-' if ctype == 'direct' else 's-'
        label = f"{'Baseline' if ctype == 'direct' else 'QP'} (nf={nf:.2f})"

        if len(hist['hard_violations']) > 0:
            ax.semilogy(hist['iterations'], hist['hard_violations'] + 1,
                       style, label=label, markersize=3, alpha=0.8)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Hard Violations + 1')
    ax.set_title('SBC Verification: Hard Violations over Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Violations comparison saved to {save_path}")


def generate_summary_table(experiments):
    """Generate a summary table of all experiments."""
    rows = []
    for name, exp in sorted(experiments.items()):
        results = exp['results']
        config = results.get('config', {})
        ctype = results.get('controller_type', 'unknown')
        info = results.get('info', {})

        max_prob = 0
        if 'history' in exp and len(exp['history']['prob_lower_bound']) > 0:
            max_prob = max(exp['history']['prob_lower_bound']) * 100

        row = {
            'Experiment': name,
            'Type': 'QP' if ctype == 'qp' else 'Baseline',
            'Noise Factor': config.get('noise_factor', 'N/A'),
            'Max Prob (%)': f'{max_prob:.1f}',
            'Final Iter': results.get('final_iter', 'N/A'),
            'Runtime (s)': info.get('runtime', 'N/A'),
            'λ_CBF': config.get('lambda_cbf', 'N/A') if ctype == 'qp' else 'N/A',
        }
        rows.append(row)

    return rows


def print_summary_table(rows):
    """Print summary table in markdown format."""
    if not rows:
        print("No results found.")
        return

    headers = list(rows[0].keys())
    col_widths = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]

    # Header
    header_line = '| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths)) + ' |'
    sep_line = '|-' + '-|-'.join('-' * w for w in col_widths) + '-|'

    print('\n' + header_line)
    print(sep_line)
    for row in rows:
        line = '| ' + ' | '.join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths)) + ' |'
        print(line)
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze Experiment Results')
    parser.add_argument('--results-dir', type=str, default='../results/',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='../figures/',
                        help='Directory for output figures')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    experiments = load_experiment_results(results_dir)
    print(f"Found {len(experiments)} experiment(s)")

    if not experiments:
        print("No results found. Run experiments first with run_all_experiments.py")
        return

    # Generate plots
    plot_probability_comparison(
        experiments,
        os.path.join(output_dir, 'probability_comparison.png')
    )
    plot_violations_comparison(
        experiments,
        os.path.join(output_dir, 'violations_comparison.png')
    )

    # Print summary table
    rows = generate_summary_table(experiments)
    print_summary_table(rows)

    # Save summary to file
    summary_path = os.path.join(output_dir, 'summary_table.md')
    with open(summary_path, 'w') as f:
        f.write('# Experiment Results Summary\n\n')
        if rows:
            headers = list(rows[0].keys())
            f.write('| ' + ' | '.join(headers) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(headers)) + '|\n')
            for row in rows:
                f.write('| ' + ' | '.join(str(row[h]) for h in headers) + ' |\n')

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
