# Paper Combination Project

Combining SafePVC (probabilistic safety verification via SBC) with BarrierNet (differentiable CBF-QP safety layer) for dual safety guarantees in vision-based neural network control systems.

## Project Structure

- `artical-F122/` — SafePVC codebase (cGAN perception + PPO controller + SBC verification)
- `BarrierNet/` — BarrierNet codebase (differentiable CBF-QP safety layer)
- `实验v2/` — Experiment v2: integrated SafePVC + BarrierNet QP (current work)

## Key Papers

- `Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems.md` — SafePVC paper
- `BarrierNet_A_Safety-Guaranteed_Layer_for_Neural_Networks.md` — BarrierNet paper

## Key Concepts

- **SBC (Stochastic Barrier Certificate)**: Lyapunov-like function providing probabilistic safety bounds over infinite horizons. Four conditions: non-negativity, initial bound ≤ 1, unsafe bound ≥ 1/(1-p), expected decrease (supermartingale).
- **CBF-QP (Control Barrier Function Quadratic Program)**: Runtime safety filter. min ½u² + q·u s.t. CBF constraints Gu ≤ h. Differentiable via KKT implicit differentiation (qpth).
- **HOCBF**: High-order CBF for constraints with relative degree > 1. ψ_i = ψ̇_{i-1} + p_i·α_i(ψ_{i-1}).
- **VCLS (Verifiable Closed-Loop System)**: gen_net ∘ state_net ∘ controller_net, verified via IBP.
- **IBP (Interval Bound Propagation)**: Sound over-approximation of NN output bounds over input intervals.

## Memory

Project memory is stored in `/root/.claude/projects/-root-paper-combination/memory/`. See MEMORY.md for the index of preserved knowledge about papers and codebases.

## Current Work: Experiment v2

Integrating BarrierNet's differentiable QP layer into SafePVC's controller, creating a dual safety mechanism:
- **Online**: CBF-QP enforces deterministic safety constraints at each timestep
- **Offline**: SBC provides probabilistic safety verification over infinite horizons

All experiment code and results are in `实验v2/`.

## Experiment v2 Status (2026-07-19)

**Completed:**
- Detailed experimental plan with architecture diagrams, data flow, CBF derivation
- All code modules: CBF constraints, QP controller, QP trainer, main loop, analysis
- Unit tests pass for all standalone modules (CBF, QP solver, QP controller)
- All dependencies verified (auto_LiRPA, gymnasium, SB3, qpth, cvxopt)
- Original artical-F122/ code preserved (no modifications)

**To run experiments:**
```bash
cd 实验v2/code
python run_all_experiments.py --quick   # 10-min quick test
python run_all_experiments.py           # full 6-experiment suite
python analyze_results.py               # generate figures and tables
```

**Key files to read first:**
- `实验v2/docs/experimental_plan.md` — Complete experimental design
- `实验v2/README.md` — Quick start guide
- `实验v2/code/models/qp_controller.py` — Core QP controller implementation
