# Experiment v2: SafePVC vs SafePVC+QP Comparison

Generated: 2026-07-19 13:52:49

| Experiment | Mode | λ_CBF | Max Prob (%) | Final Prob (%) | Iterations | Runtime (s) | Final L_loss | Final Dec_loss | Final Region_loss |
|---|---|---|---|---|---|---|---|---|---|
| baseline_nf005/baseline_nf0_05_20260719_133625 | baseline | 1.0 | 90.64 | 90.64 | 20 | 126 | 15.7 | 0.0148 | 0.87 |
| qp_integrated_nf005 | qp | 1.0 | 76.39 | 76.39 | 20 | 147 | 27.5 | 0.0089 | 18.65 |

## Analysis

- **Baseline** (SafePVC without QP): Best probability = 90.64%
- **QP Integrated** (SafePVC + BarrierNet QP): Best probability = 76.39%
- **Gap**: Baseline outperforms QP by 14.25 percentage points

### Why might QP integration degrade performance?

1. **CBF constraint interference**: The CBF constraints may increase the Lipschitz constant of the closed-loop system, making SBC verification harder
2. **Training instability**: Differentiating through the QP layer (qpth) may introduce noisy gradients that destabilize controller training
3. **Region loss explosion**: QP experiments show much higher region_loss, indicating the SBC struggles to satisfy boundary conditions
4. **CBF-QP mismatch with SBC**: The CBF provides deterministic safety, while SBC provides probabilistic safety. The two may have conflicting objectives during training
