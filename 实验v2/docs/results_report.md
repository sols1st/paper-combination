# 实验v2 结果报告

> **日期**: 2026-07-19
> **状态**: 代码实现完成，待运行完整实验

---

## 1. 实验准备完成情况

### ✅ 已完成

| 项目 | 状态 |
|------|------|
| 实验方案文档 | ✅ `docs/experimental_plan.md` |
| 代码实现 | ✅ `code/` 目录下所有模块 |
| 单元测试 | ✅ CBF约束、QP求解器、QP控制器均通过测试 |
| 依赖安装 | ✅ auto_LiRPA, gymnasium, SB3, qpth, cvxopt |
| 原始代码保护 | ✅ `artical-F122/` 未修改任何文件 |
| 记忆持久化 | ✅ CLAUDE.md + memory/ 维护上下文 |

### 代码文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `code/models/cbf_constraints.py` | ~200 | AEBS CBF约束构造 + Lie导数 |
| `code/models/qp_controller.py` | ~350 | 双分支QP控制器 + DirectController基线 |
| `code/training/qp_trainer.py` | ~350 | QP增强训练器 (继承VTLearner) |
| `code/training/qp_loop.py` | ~300 | 主训练+验证循环 |
| `code/utils/qp_solver.py` | ~200 | QP求解器封装 (qpth + cvxopt) |
| `code/run_baseline.py` | ~50 | 基线实验运行脚本 |
| `code/run_qp_experiment.py` | ~60 | QP实验运行脚本 |
| `code/run_all_experiments.py` | ~140 | 全实验矩阵运行脚本 |
| `code/analyze_results.py` | ~180 | 结果分析与可视化 |

## 2. 实现的核心功能

### 2.1 CBF约束模块 (`cbf_constraints.py`)

- **障碍函数**: `b(s) = d - v * t_gap`
- **Lie导数计算**: `L_f b = -v`, `L_g b = t_gap`
- **QP约束构造**: `G * u <= h`
- **约束违反检测**: `check_cbf_satisfaction()`
- **训练损失**: `cbf_violation_loss()` (ReLU惩罚)

### 2.2 QP控制器 (`qp_controller.py`)

- **QPAebsController**: 双分支架构 (q_head + p_head + QP层)
- **DirectController**: 无QP的基线控制器
- **训练模式**: qpth.QPFunction (可微分)
- **推理模式**: cvxopt.solvers.qp (稳定)
- **容错机制**: 约束松弛 + 控制裁剪

### 2.3 QP训练器 (`qp_trainer.py`)

- **QPVTLearner**: 继承原始VTLearner
- **新增损失**: `L_CBF` (CBF约束违反惩罚)
- **MSE蒸馏**: QP输出与教师策略的距离
- **支持模式**: 'qp' / 'direct' 切换

### 2.4 主循环 (`qp_loop.py`)

- **QPLoop**: 完整训练+验证流程
- **历史记录**: 概率界、违反数、CBF指标
- **检查点**: 自动保存最佳模型
- **支持**: 命令行参数配置

## 3. 验证测试结果

### 单元测试

```
CBF Constraints Test:
  Barrier b(s) = 7.000            ✓
  G = [[-1.5]], h = [[5.0]]       ✓
  Constraint satisfied: True      ✓

QP Solver Test:
  qpth solution: u = 1.0000       ✓
  cvxopt solution: u = 1.0000     ✓

QP Controller Test:
  Input:  [10m, 2m/s] → q=0.23   ✓
  CBF param p range: (0, 4)      ✓
  Barrier values correct          ✓
```

### 依赖检查

```
auto_LiRPA:   ✓
gymnasium:    ✓
stable-baselines3: ✓
qpth:         ✓
cvxopt:       ✓
torch:        1.10.1
```

## 4. 运行实验

### 快速启动

```bash
cd /root/paper-combination/实验v2/code

# 快速测试 (10分钟)
python run_baseline.py --noise-factor 0.05 --timeout 600
python run_qp_experiment.py --noise-factor 0.05 --timeout 600

# 完整实验矩阵 (6个实验 × 1小时)
python run_all_experiments.py

# 分析结果
python analyze_results.py --results-dir ../results/
```

## 5. 预期结果（假设）

基于论文结果和方法分析：

| 指标 | 原始 SafePVC | SafePVC + QP (预期) |
|------|-------------|-------------------|
| 概率安全下界 (nf=0.05) | ~72% | ~85%+ |
| 验证收敛迭代数 | ~100 | ~70-80 |
| 高扰动 (nf=0.10) 安全下界 | ~60% | ~78%+ |
| 控制器输出平滑性 | 基准 | 更平滑 (QP正则化) |

## 6. 关键设计决策记录

1. **代码隔离**: 所有新代码在 `实验v2/code/`，不修改原始 `artical-F122/`
2. **API兼容**: QPAebsController 和 DirectController 共享相同的 forward 签名
3. **训练容错**: QP无解时使用松弛变量或回退到控制裁剪
4. **CBF参数范围**: p ∈ (0, 4) 通过 4*sigmoid 保证正性和有界性
5. **双层安全**: CBF-QP (在线确定性) + SBC (离线概率性) 互补
