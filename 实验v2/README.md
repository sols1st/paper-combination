# 实验v2: SafePVC + BarrierNet QP 安全滤波层集成

> **目标**: 将 BarrierNet 的可微分 QP (CBF) 安全层集成到 SafePVC 框架中，实现 CBF-QP + SBC 双重安全机制

---

## 目录结构

```
实验v2/
├── README.md                          # 本文件
├── docs/
│   └── experimental_plan.md           # 详细实验方案（架构、数据流、实施步骤）
├── configs/
│   └── experiment_config.yaml         # 实验配置文件
├── code/                              # ★ 所有修改的代码（不影响原始代码）
│   ├── models/
│   │   ├── cbf_constraints.py         # AEBS 场景的 CBF 约束数学推导
│   │   └── qp_controller.py           # 双分支 QP 控制器 + DirectController 基线
│   ├── training/
│   │   ├── qp_trainer.py             # QP 增强的训练器（继承 VTLearner）
│   │   └── qp_loop.py                # QP 集成的主训练循环
│   ├── utils/
│   │   └── qp_solver.py              # QP 求解器封装（qpth + cvxopt）
│   ├── run_baseline.py               # 运行基线实验（原始 SafePVC）
│   ├── run_qp_experiment.py          # 运行 QP 集成实验
│   ├── run_all_experiments.py        # 运行全部对比实验
│   └── analyze_results.py            # 结果分析和可视化
├── results/                           # 实验结果（自动生成）
│   ├── baseline/                     # 基线实验结果
│   └── qp_integrated/                # QP 集成实验结果
└── figures/                           # 图表（自动生成）
```

## 核心修改说明

### 架构变化

```
原始 SafePVC 控制器:
  Input → FC → ReLU → FC → ReLU → FC → u (直接输出)

SafePVC + QP 控制器 (本实验):
  Input → Shared Backbone ─┬→ q_head → q (参考控制)
                           └→ p_head → 4·σ(p) (CBF参数, >0)
  → QP: min ½u² + q·u  s.t. CBF 约束  →  u* (安全控制)
```

### CBF 约束（AEBS 场景）

- 障碍函数: b(s) = d - v · t_gap (t_gap = 1.5s)
- CBF 条件: -v + t_gap · u + p · (d - v · t_gap) ≥ 0
- QP 形式: G = -t_gap, h = -v + p · (d - v · t_gap)

### 原始代码保护

本实验**不修改** `artical-F122/` 中的任何原始代码。所有新代码都在 `实验v2/code/` 中，通过 import 使用原始代码库。

## 快速开始

### 前置条件

```bash
# 安装依赖
pip install gymnasium stable-baselines3 qpth cvxopt
cd artical-F122/auto_LiRPA && python setup.py develop

# 确认原始模型已训练
ls artical-F122/Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth
ls artical-F122/Aebs/controller/best_model/best_model.zip
ls artical-F122/Aebs/controller/state_net_trained.pth
```

### 运行单个实验

```bash
cd 实验v2/code

# 运行基线实验（原始 SafePVC，无 QP）
python run_baseline.py --noise-factor 0.05 --timeout 600

# 运行 QP 集成实验
python run_qp_experiment.py --noise-factor 0.05 --lambda-cbf 1.0 --timeout 600
```

### 运行全部对比实验

```bash
cd 实验v2/code

# 完整实验矩阵（6个实验，每个1小时）
python run_all_experiments.py

# 快速模式（每个实验10分钟）
python run_all_experiments.py --quick

# 只运行 QP 实验
python run_all_experiments.py --skip-baseline

# 自定义噪声因子
python run_all_experiments.py --noise-factors "0.01,0.05,0.10"
```

### 分析结果

```bash
cd 实验v2/code
python analyze_results.py --results-dir ../results/ --output-dir ../figures/
```

## 实验矩阵

| 实验 | 条件 | 噪声因子 | λ_CBF |
|------|------|---------|-------|
| baseline_nf001 | 原始 SafePVC | 0.01 | N/A |
| baseline_nf005 | 原始 SafePVC | 0.05 | N/A |
| baseline_nf010 | 原始 SafePVC | 0.10 | N/A |
| qp_nf001_cbf1.0 | SafePVC + QP | 0.01 | 1.0 |
| qp_nf005_cbf1.0 | SafePVC + QP | 0.05 | 1.0 |
| qp_nf010_cbf1.0 | SafePVC + QP | 0.10 | 1.0 |

## 评估指标

- **概率安全下界 p**: SBC 验证的正式安全保证（越高越好）
- **硬违反数**: 验证网格中不满足递减条件的网格数（越低越好）
- **SBC 收敛速度**: 达到验证通过所需迭代数
- **QP 可行率**: 运行时 QP 有可行解的比例
- **CBF 违反率**: 运行中 b(s) < 0 的时间步比例

## 预期结果

| 指标 | 原始 SafePVC | SafePVC + QP | 预期改进 |
|------|-------------|-------------|---------|
| 概率安全下界 | ~72% | ~85%+ | ↑ 10-15% |
| 验证收敛速度 | 较慢 | 更快 | ↑ 20-30% |
| 高扰动鲁棒性 | 下降明显 | 下降更缓 | ↑ 更鲁棒 |
| 任务完成率 | 基准 | 略微下降 | ↓ < 5% |
