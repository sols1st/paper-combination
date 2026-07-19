#!/usr/bin/env python3
"""
Extended experiments with different λ_cbf values and longer runs.
Tests the hypothesis that CBF loss weight affects SBC training quality.
"""

import sys, os, json, time, subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment configurations
configs = [
    # (name, mode, noise_factor, lambda_cbf, max_iters, timeout)
    ("baseline_long", "baseline", 0.05, None, 30, 900),
    ("qp_cbf0.1", "qp", 0.05, 0.1, 30, 900),
    ("qp_cbf0.01", "qp", 0.05, 0.01, 30, 900),
    ("qp_cbf10.0", "qp", 0.05, 10.0, 30, 900),
]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
summary = {}

for name, mode, nf, lcbf, iters, timeout in configs:
    print(f"\n{'#'*60}")
    print(f"# Extended Experiment: {name}")
    print(f"# Mode: {mode}, Noise: {nf}, λ_cbf: {lcbf}, Iters: {iters}")
    print(f"{'#'*60}")

    results_dir = os.path.join(RESULTS_DIR, f"extended_{name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Build command
    script = os.path.join(BASE_DIR, 'code', 'run_experiments.py')
    cmd = [
        sys.executable, script,
        '--mode', mode,
        '--noise', str(nf),
        '--timeout', str(timeout),
        '--max-iters', str(iters),
    ]
    if lcbf is not None:
        cmd.extend(['--lambda-cbf', str(lcbf)])

    print(f"Command: {' '.join(cmd)}")

    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 120,
                              cwd=os.path.dirname(BASE_DIR))
        elapsed = time.time() - start

        # Parse output for key metrics
        output = result.stdout + result.stderr
        max_prob = 0.0
        for line in output.split('\n'):
            if 'Max probability bound:' in line:
                try:
                    pct_str = line.split(':')[1].strip().replace('%', '')
                    max_prob = float(pct_str) / 100.0
                except:
                    pass

        summary[name] = {
            'status': 'completed' if result.returncode == 0 else 'error',
            'max_prob': max_prob,
            'runtime': elapsed,
            'returncode': result.returncode,
        }

        # Save output
        with open(os.path.join(results_dir, 'output.log'), 'w') as f:
            f.write(output)

        # Copy results if any were saved
        artical_results = os.path.join(
            os.path.dirname(BASE_DIR), 'artical-F122', '实验v2', 'results'
        )
        if os.path.exists(artical_results):
            for d in os.listdir(artical_results):
                src = os.path.join(artical_results, d)
                dst = os.path.join(results_dir, d)
                if os.path.isdir(src) and name in d.lower():
                    import shutil
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

        print(f"  Completed in {elapsed:.0f}s, Max prob: {max_prob*100:.2f}%")

    except subprocess.TimeoutExpired:
        summary[name] = {'status': 'timeout', 'max_prob': 0.0, 'runtime': timeout}
        print(f"  TIMEOUT after {timeout}s")
    except Exception as e:
        summary[name] = {'status': 'error', 'error': str(e)}
        print(f"  ERROR: {e}")

    # Save incremental summary
    with open(os.path.join(RESULTS_DIR, f'extended_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

# Print final summary
print(f"\n{'='*60}")
print("EXTENDED EXPERIMENT SUMMARY")
print(f"{'='*60}")
print(f"{'Experiment':<25} {'Max Prob':<12} {'Runtime':<10} {'Status'}")
print("-" * 57)
for name, info in summary.items():
    prob = info.get('max_prob', 0) * 100
    runtime = info.get('runtime', 0)
    status = info.get('status', 'unknown')
    print(f"{name:<25} {prob:>8.2f}%   {runtime:>6.0f}s   {status}")
print(f"{'='*60}")

# Also print comparison with original baseline
print(f"\n{'='*60}")
print("FULL COMPARISON (including earlier results)")
print(f"{'='*60}")
all_results = {
    'Baseline (20 iters)': 0.9064,      # from first run
    'QP λ=1.0 (20 iters)': 0.7639,      # from second run
}
for name, info in summary.items():
    if info['status'] == 'completed':
        all_results[name] = info['max_prob']

print(f"{'Experiment':<30} {'Max Prob':<12}")
print("-" * 42)
for name, prob in sorted(all_results.items(), key=lambda x: -x[1]):
    print(f"{name:<30} {prob*100:>8.2f}%")
print(f"{'='*60}")
