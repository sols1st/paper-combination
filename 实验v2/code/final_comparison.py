#!/usr/bin/env python3
"""
Final comparison and analysis of SafePVC vs SafePVC+QP experiments.

Generates:
1. Summary comparison table
2. Training curve plots
3. Analysis of why QP integration may/may not help
"""

import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_all_results():
    """Load all experiment results from the results directory."""
    experiments = {}
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in files:
            if f == 'results.json':
                path = os.path.join(root, f)
                try:
                    with open(path) as fp:
                        data = json.load(fp)
                    name = os.path.relpath(root, RESULTS_DIR)
                    experiments[name] = data
                except:
                    pass
    return experiments

def create_summary_table(experiments):
    """Create and save summary comparison table."""
    rows = []
    for name, exp in sorted(experiments.items()):
        config = exp.get('config', {})
        rows.append({
            'Experiment': name[:50],
            'Mode': config.get('mode', '?'),
            'λ_CBF': config.get('lambda_cbf', 'N/A'),
            'Max Prob (%)': f"{exp.get('max_prob_bound', 0)*100:.2f}",
            'Final Prob (%)': f"{exp.get('final_prob_bound', 0)*100:.2f}",
            'Iterations': exp.get('num_iterations', '?'),
            'Runtime (s)': f"{exp.get('total_runtime', 0):.0f}",
            'Final L_loss': f"{exp.get('final_l_metrics', {}).get('loss_l', 0):.1f}",
            'Final Dec_loss': f"{exp.get('final_l_metrics', {}).get('dec_loss', 0):.4f}",
            'Final Region_loss': f"{exp.get('final_l_metrics', {}).get('region_loss', 0):.2f}",
        })
        if 'cbf_loss' in exp.get('final_p_metrics', {}):
            rows[-1]['CBF_loss'] = f"{exp['final_p_metrics']['cbf_loss']:.2f}"
            rows[-1]['MSE_loss'] = f"{exp['final_p_metrics']['mse_loss']:.4f}"

    # Save as markdown
    md_path = os.path.join(FIGURES_DIR, 'comparison_table.md')
    with open(md_path, 'w') as f:
        f.write("# Experiment v2: SafePVC vs SafePVC+QP Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not rows:
            f.write("No results found.\n")
            return

        headers = list(rows[0].keys())
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(headers)) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |\n')

        # Add analysis
        f.write("\n## Analysis\n\n")

        baseline_probs = [float(r['Max Prob (%)']) for r in rows if r['Mode'] == 'baseline']
        qp_probs = [float(r['Max Prob (%)']) for r in rows if r['Mode'] == 'qp']

        if baseline_probs and qp_probs:
            f.write(f"- **Baseline** (SafePVC without QP): Best probability = {max(baseline_probs):.2f}%\n")
            f.write(f"- **QP Integrated** (SafePVC + BarrierNet QP): Best probability = {max(qp_probs):.2f}%\n")
            diff = max(baseline_probs) - max(qp_probs)
            if diff > 0:
                f.write(f"- **Gap**: Baseline outperforms QP by {diff:.2f} percentage points\n")
                f.write("\n### Why might QP integration degrade performance?\n\n")
                f.write("1. **CBF constraint interference**: The CBF constraints may increase the Lipschitz constant "
                       "of the closed-loop system, making SBC verification harder\n")
                f.write("2. **Training instability**: Differentiating through the QP layer (qpth) may introduce "
                       "noisy gradients that destabilize controller training\n")
                f.write("3. **Region loss explosion**: QP experiments show much higher region_loss, indicating "
                       "the SBC struggles to satisfy boundary conditions\n")
                f.write("4. **CBF-QP mismatch with SBC**: The CBF provides deterministic safety, while SBC provides "
                       "probabilistic safety. The two may have conflicting objectives during training\n")
            else:
                f.write(f"- **Improvement**: QP outperforms baseline by {abs(diff):.2f} percentage points\n")

    print(f"Summary table saved to {md_path}")

    # Print to console too
    if rows:
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        headers = list(rows[0].keys())
        col_widths = [max(len(h), max(len(str(r.get(h,''))) for r in rows)) + 2 for h in headers]
        header_line = ''.join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print('-' * sum(col_widths))
        for row in rows:
            print(''.join(str(row.get(h,'')).ljust(w) for h, w in zip(headers, col_widths)))
        print("="*80)


def plot_probability_curves(experiments):
    """Plot probability bound convergence curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Find all experiment dirs with history
    for name, exp in experiments.items():
        # Find history.npz in same directory
        exp_dir = os.path.join(RESULTS_DIR, name)
        hist_path = os.path.join(exp_dir, 'history.npz')
        if not os.path.exists(hist_path):
            # Try subdirectories
            for root, dirs, files in os.walk(exp_dir):
                if 'history.npz' in files:
                    hist_path = os.path.join(root, 'history.npz')
                    break

        if os.path.exists(hist_path):
            hist = np.load(hist_path)
            config = exp.get('config', {})
            mode = config.get('mode', '?')
            lcbf = config.get('lambda_cbf', None)

            label = f"{'Baseline' if mode=='baseline' else 'QP'}"
            if lcbf is not None:
                label += f" (λ={lcbf})"

            style = 'o-' if mode == 'baseline' else 's-'
            color = '#3498db' if mode == 'baseline' else '#e74c3c'

            # Plot 1: Probability
            if 'prob_bounds' in hist and len(hist['prob_bounds']) > 0:
                axes[0].plot(hist['iterations'], np.array(hist['prob_bounds'])*100,
                           style, color=color, label=label, markersize=3)

            # Plot 2: Violations
            if 'hard_violations' in hist and len(hist['hard_violations']) > 0:
                axes[1].semilogy(hist['iterations'], np.array(hist['hard_violations']) + 1,
                               style, color=color, label=label, markersize=3)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Probability Lower Bound (%)')
    axes[0].set_title('Safety Probability Convergence')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 100)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Hard Violations + 1')
    axes[1].set_title('SBC Verification Violations')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'probability_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Probability curves saved to {FIGURES_DIR}/probability_comparison.png")


def plot_bar_comparison(experiments):
    """Bar chart comparing max probability bounds."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    probs = []
    colors = []

    for name, exp in sorted(experiments.items()):
        config = exp.get('config', {})
        mode = config.get('mode', '?')
        lcbf = config.get('lambda_cbf', None)

        if mode == 'baseline':
            label = 'Baseline\n(no QP)'
            color = '#3498db'
        else:
            label = f'QP λ={lcbf}'
            color = '#e74c3c'

        labels.append(label)
        probs.append(exp.get('max_prob_bound', 0) * 100)
        colors.append(color)

    x = np.arange(len(labels))
    bars = ax.bar(x, probs, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Max Probability Lower Bound (%)')
    ax.set_title('Safety Guarantee Comparison: SafePVC vs SafePVC+QP')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'bar_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to {FIGURES_DIR}/bar_comparison.png")


def main():
    experiments = load_all_results()
    print(f"Found {len(experiments)} experiment results")

    if not experiments:
        print("No results found. Run experiments first.")
        return

    create_summary_table(experiments)
    plot_probability_curves(experiments)
    plot_bar_comparison(experiments)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
