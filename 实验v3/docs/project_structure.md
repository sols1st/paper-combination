# Paper Combination 项目结构与代码文档

> 完整项目结构、代码文件说明、数据流、实验体系

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构总览](#2-目录结构总览)
3. [原始代码 (artical-F122)](#3-原始代码-artical-f122)
4. [共享实验代码 (src/)](#4-共享实验代码-src)
5. [实验v2](#5-实验v2)
6. [实验v3](#6-实验v3)
7. [完整数据流](#7-完整数据流)
8. [文件依赖关系图](#8-文件依赖关系图)

---

## 1. 项目概述

### 1.1 研究目标

将 **SafePVC** (随机屏障证书 SBC, 概率安全验证) 与 **BarrierNet** (可微分 CBF-QP 安全层) 结合, 构建**双重安全保证**的神经网络控制系统:

- **SBC (Stochastic Barrier Certificate)**: 离线形式化验证, 提供"系统整体以 X% 概率安全"的保证
- **CBF-QP (Control Barrier Function Quadratic Program)**: 在线运行时安全盾, 每步确定性地检查和修正控制

### 1.2 核心论文

| 论文 | 内容 |
|------|------|
| `Provably Probabilistic Safe Controller Synthesis...` | SafePVC: cGAN感知 + PPO控制器 + SBC验证 |
| `BarrierNet_A_Safety-Guaranteed_Layer...` | BarrierNet: 可微分 CBF-QP 安全层 |

### 1.3 实验演进

| 版本 | 核心思路 | SBC概率界 | QP位置 | 状态 |
|------|---------|----------|--------|------|
| **v1** | 原始 SafePVC | 95.8% | 无 QP | ✅ 基准 |
| **v2** | QP 嵌入训练循环 | 87.9% (-8pp) | 训练+推理 | ❌ QP干扰SBC |
| **v3** | QP 仅推理时使用 | 96.6% | 仅推理 | ✅ 当前 |

---

## 2. 目录结构总览

```
paper-combination/
│
├── 📄 CLAUDE.md                          # 项目说明 (AI 助手指令)
├── 📄 README.md                          # 项目 Readme
├── 📄 means.md                           # 核心概念笔记
│
├── 📁 artical-F122/                      # ★ 原始 SafePVC 代码 (未修改)
│   ├── Aebs/                             #   AEBS 系统模块
│   │   ├── system/                       #   环境 + 动力学
│   │   │   ├── env.py                    #   Aebs 环境类 (状态空间/动力学)
│   │   │   ├── combined.py              #   端到端模型加载脚本
│   │   │   └── estimate.py              #   状态估计网络训练
│   │   ├── VT/                           #   SBC 验证+训练 (核心)
│   │   │   ├── loop.py                  #   主训练循环
│   │   │   ├── train.py                 #   VTLearner (SBC + Controller训练)
│   │   │   ├── verify.py                #   VTVerifier (IBP 形式化验证)
│   │   │   └── utils.py                 #   MLP / martingale_loss / triangular 噪声
│   │   └── controller/                   #   PPO 控制器
│   │       ├── Controller_train.py      #   PPO 训练脚本
│   │       ├── best_model/              #   训练好的 PPO 模型
│   │       └── state_net_trained.pth    #   训练好的状态估计网络
│   │
│   ├── cGAN/                             # cGAN 生成器 (图像生成)
│   │   └── taxi_models_and_data.py      # AebsMLPGenerator 定义
│   │
│   ├── Combined_network/                 # 端到端网络
│   │   └── model.py                     # AebsEnd2EndNet 定义
│   │
│   ├── auto_LiRPA/                       # IBP 验证库 (第三方)
│   │
│   └── run_barriernet.py                 # BarrierNet 实验脚本 (原始)
│
├── 📁 BarrierNet/                        # BarrierNet 原始代码 (参考)
│   ├── Driving/                          #   自动驾驶场景 CBF-QP
│   ├── 2D_Robot/                         #   2D 机器人避障
│   ├── 3D_Robot/                         #   3D 无人机
│   └── Merging/                          #   车辆并道场景
│
├── 📁 src/                               # ★ 共享实验代码 (v2 + v3)
│   ├── __init__.py
│   ├── models/                           #   模型定义
│   │   ├── cbf_constraints.py           #   CBF 约束数学实现
│   │   ├── qp_controller.py             #   v2 QP 增强控制器
│   │   ├── sbc_modulated_qp.py          #   v3 SBC 调制 QP 盾 (V3A/V3B)
│   │   └── qp_p_network.py              #   v3 p-network + V3C/V3D 盾
│   ├── training/                         #   训练逻辑
│   │   ├── qp_trainer.py                #   v2 QP 增强 VTLearner
│   │   └── qp_loop.py                   #   v2 主训练循环
│   ├── eval/                             #   评估脚本
│   │   ├── compare_shields.py           #   v3 四配置对比 (早期)
│   │   ├── train_sbc_for_v3.py          #   v3 SBC 快速训练 (35轮)
│   │   ├── train_improved_sbc.py        #   v3 SBC 改进训练 (120轮, 96.6%)
│   │   ├── test_qp_activation.py        #   v3 QP 激活验证
│   │   ├── v3_parameter_sweep.py        #   v3 参数扫描 (26配置)
│   │   ├── v3_full_evaluation.py        #   v3 完整评估 (9配置)
│   │   ├── qp_benefit_experiments.py    #   QP 价值证明实验 (4项)
│   │   └── collect_qp_benefits.py       #   QP 收益数据收集
│   └── utils/
│       └── qp_solver.py                 #   QP 求解器封装 (qpth/cvxopt)
│
├── 📁 实验v2/                            # 实验 v2 文档 + 结果
│   ├── README.md
│   ├── docs/                             #   实验方案 + 结果报告
│   └── results/                          #   实验数据 (JSON)
│
├── 📁 实验v3/                            # ★ 实验 v3 文档 + 结果 (当前)
│   ├── README.md
│   ├── configs/                          #   实验配置
│   ├── docs/                             #   详细文档
│   │   ├── experimental_plan.md         #   原始计划
│   │   ├── experiment_design.md         #   详细设计 (架构/数据流)
│   │   ├── complete_results.md          #   首轮完整结果
│   │   ├── sweep_results.md             #   参数扫描结果
│   │   ├── v3_improved_results.md       #   改进版实验全结果
│   │   └── qp_shield_benefit_report.md  #   QP 价值验证报告
│   └── results/                          #   实验数据 + 模型
│       ├── trained_sbc.pth              #   原始 SBC 模型
│       ├── trained_sbc_improved.pth     #   改进 SBC (96.58%)
│       ├── sbc_metadata.json            #   SBC 训练元数据
│       ├── trained_p_network.pth        #   训练 p-network
│       ├── v3_comparison_*.json         #   对比评估结果
│       ├── v3_sweep_*.json              #   参数扫描结果
│       ├── v3_full_evaluation_*.json    #   完整评估结果
│       └── qp_benefit_experiments_*.json #  QP 价值实验结果
│
├── 📁 实验结果v1/                        # 实验 v1 文档
├── 📁 ABNet/                            # ABNet 论文笔记
│
└── 📄 *.md                              # 论文原文 + 笔记
    ├── Provably Probabilistic Safe...   # SafePVC 论文
    ├── BarrierNet_A_Safety...           # BarrierNet 论文
    └── Differentiable_Control...        # 可微分 CBF 论文
```

---

## 3. 原始代码 (artical-F122)

> ⚠️ **这些文件未做任何修改**, 保持原始 SafePVC 代码完整。

### 3.1 环境与动力学 (`Aebs/system/`)

#### `env.py` — AEBS 系统定义

```
类: Aebs (用于 SBC 训练/验证)
  - 状态空间: d ∈ [5, 16]m, v ∈ [0, 3]m/s (归一化)
  - 控制空间: u ∈ [-3, 3] m/s²
  - 初始区域: d∈[15,16], v∈[2.5,3.0]
  - 不安全区域: d∈[5,6], v∈[0.5,3.0]
  - 噪声: triangular 分布, factor=0.01
  - 方法: v_next(s, a) — 一步动力学前向

类: AebsEnv (用于 PPO 训练)
  - Gymnasium 环境接口
  - step/ reset/ reward

函数: next_state_vec(d, v, acc, dt)
  - d_next = d - v*dt
  - v_next = v - acc*dt  (注意: 正 u = v 减小)

关键约定: v_next = v - u*dt
  → u>0 表示"加速/减速"使 v 减小
  → u<0 表示速度增加
```

#### `combined.py` — 端到端模型加载

加载完整的 SafePVC 流水线: gen_net → state_net → PPO controller

#### `estimate.py` — 状态估计网络

StateEstimate_train: 训练 state_net (从图像到状态)

### 3.2 SBC 核心模块 (`Aebs/VT/`)

#### `utils.py` — 基础工具

```
MLP(features, activation, square_output):
  - 多层感知机 (用于 SBC B(s) 网络)
  - square_output=True: 最终输出 = x² (保证非负)
  - 默认架构: [2, 16, 8, 1]

triangular(shape):
  - 生成三角分布噪声 (模拟更真实的不确定性)

martingale_loss(l, l_next, eps):
  - SBC 上鞅损失: mean(max(l_next - l + eps, 0))
  - 确保 E[B(s_{t+1})] ≤ B(s_t) (期望递减)
```

#### `train.py` — VTLearner (控制器 + SBC 联合训练)

```
VTLearner:
  - 加载 gen_net (frozen) + state_net (frozen) + PPO controller
  - 构造 VCLS = gen_net ∘ state_net ∘ controller (端到端)
  - 方法:
    - train('l'): 训练 SBC B(s) 网络
      → 损失 = martingale_loss + region_loss (初始≤1, 不安全≥1/(1-p))
    - train('p'): 训练控制器
      → 损失 = martingale_loss + MSE(teacher)
    - create_bounded_module(): 创建 IBP 可验证模块
```

**数据流 (训练阶段)**:
```
采样状态 s_t ∈ 100×100 网格
  → 加三角扰动
  → VCLS: gen_net → state_net → controller → u
  → 动力学: s_{t+1} = f(s_t, u) + 噪声
  → SBC: B(s_t), B(s_{t+1})
  → 损失: martingale + region
```

#### `verify.py` — VTVerifier (IBP 形式化验证)

```
VTVerifier:
  - 对 100×100 状态网格进行 IBP 验证
  - 方法:
    - check_dec_cond(k): 检查 B(s_{t+1}) ≤ k·B(s_t) 是否全局成立
    - compute_bound_init(): 计算初始区域 B(s) 上界
    - compute_bound_unsafe(): 计算不安全区域 B(s) 下界
    - 概率下界: p ≥ 1 - ub_init/lb_unsafe
```

#### `loop.py` — 主训练循环

```
Loop:
  - 交替执行:
    1. train('l'): 训练 SBC
    2. verify: IBP 验证 → 计算概率下界
    3. train('p'): 用反例训练控制器
  - 记录: violations, prob_bound, timing
```

### 3.3 感知模块

#### `cGAN/taxi_models_and_data.py` — 生成器

```
AebsMLPGenerator:
  - 输入: z (隐向量, dim=4) + d (距离, dim=1)
  - 输出: 32×32 灰度图像 (模拟相机)
  - 已预训练, frozen
```

#### `Combined_network/model.py` — 端到端网络

```
AebsEnd2EndNet:
  - gen_net: 隐向量 → 图像 (frozen)
  - state_net: 图像 → 状态估计 [d_est, v_est] (frozen)
  - controller_net: [state_est, v] → u (PPO 预训练, 可微调)
```

### 3.4 PPO 控制器 (`Aebs/controller/`)

```
- Controller_train.py: PPO 强化学习训练
- best_model/best_model.zip: 训练好的 PPO 模型权重
- state_net_trained.pth: 训练好的状态估计网络权重
```

---

## 4. 共享实验代码 (src/)

> 所有 v2 和 v3 的新增代码都在这里。通过 `import` 使用原始 artical-F122 的模块。

### 4.1 模型层 (`src/models/`)

#### `cbf_constraints.py` — CBF 约束数学 (v2 起)

```
AEBSCBFConstraints(t_gap=1.5, dt=0.05):
  - barrier_function(state): b(s) = d - v·t_gap
  - compute_lie_derivatives(state): Lf_b = -v, Lg_b = t_gap
  - build_constraints(state, p): G=-t_gap, h=-v+p·(d-v·t_gap)
    对应 QP 约束: -t_gap·u ≤ -v + p·(d - v·t_gap)
  - build_full_constraints(state, p): CBF + u∈[-3,3]
  - check_cbf_satisfaction(state, u, p): 检查约束满足 + margin
  - cbf_violation_loss(state, u, p): 训练用约束违反惩罚

数学:
  CBF条件: ḃ + p·b ≥ 0
  离散化: -t_gap·u ≤ -v + p·(d - v·t_gap)
  QP: min ½(u-u_ref)²  s.t. -t_gap·u ≤ -v + p·(d-v·t_gap), -3≤u≤3
```

#### `qp_controller.py` — v2 QP 增强控制器

```
QPAebsController (v2 训练时 QP):
  - 双分支架构: shared_backbone → q_head (u_ref) + p_head (CBF参数)
  - p = 4·sigmoid(p_raw) ∈ (0, 4) — NN 学习最优 CBF 参数
  - QP 层: min ½u² + q·u  s.t. CBF 约束
  - 模式:
    - 'train': 可微分 QP (qpth)
    - 'eval': 数值 QP (cvxopt)
    - 'direct': 跳过 QP (等同于 baseline)

DirectController:
  - 无 QP 的原始控制器, 用于 ablation

问题 (v2→v3的原因):
  - QP 在训练循环中增加了闭环非线性
  - Lipschitz 常数增大 → SBC 验证更保守
  - 概率界从 95.8% 降至 87.9%
```

#### `sbc_modulated_qp.py` — v3 推理时 QP 盾 (V3A/V3B)

```
SBCModulatedQPShield (V3B):
  - QP 只在推理时使用, 不参与训练
  - p(s) = p_min + (p_max-p_min)·σ((B(s)-B_thresh)/T)
    → SBC B(s) 调制 CBF 参数
    → B(s) 小 (安全) → p≈p_min (松弛约束)
    → B(s) 大 (危险) → p≈p_max (收紧约束)
  - shield(u_ref, state): QP 求解 → u_safe
  - _solve_qp_1d_analytic(): 1D QP 解析解 (高效)

FixedQPShield (V3A):
  - 继承 SBCModulatedQPShield
  - 覆写 compute_p(): 固定 p 值, 不调制
  - 用于 ablation: 对比固定 vs SBC 调制的效果

关键设计:
  - 训练时 controller 使用 DIRECT 模式 (无 QP)
  - 推理时插入 QP 盾: controller → u_ref → QP → u_safe
```

#### `qp_p_network.py` — v3 p-network + V3C/V3D

```
QPParameterNetwork (V3C):
  - 神经网络学习 state → 最优 p
  - 架构: [d,v] → MLP[64,32,16] → p ∈ [p_min, p_max]
  - 训练信号: SBC B(s) 软标签 + Barrier b(s) 直接信号

QPParameterTrainer:
  - generate_training_data(): 生成 8000 样本
  - compute_target_p(): 混合 SBC + Barrier 目标
  - train_epoch(): MSE + CBF约束 + 平滑正则化

TrainedQPShield (V3C):
  - 使用训练好的 p-network 推理

BarrierModulatedQPShield (V3D):
  - 直接用 barrier b(s) 调制 p:
    p = p_min + (p_max-p_min)·σ(-b(s)/margin_scale)
  - b(s) 小 (近不安全) → p 大 → 约束紧
  - b(s) 大 (安全) → p 小 → 约束松
  - 优势: 不需要 SBC, 直接可靠
```

### 4.2 训练层 (`src/training/`)

#### `qp_trainer.py` — v2 QP 增强训练器

```
QPVTLearner(VTLearner):
  - 继承原始 VTLearner, 添加 QP 支持
  - controller_type: 'qp' | 'direct'
  - 新增损失: L_CBF (CBF 约束违反惩罚)
  - λ_cbf: CBF 损失权重 (关键超参, 需 ≤0.01)
  - λ_mse: MSE vs teacher 权重
```

#### `qp_loop.py` — v2 主训练循环

```
QPLoop:
  - 替代原始 VT/loop.py
  - 支持 QP 模式 + CBF 指标追踪
  - 记录: violations, prob_bound, CBF_metrics
```

### 4.3 评估层 (`src/eval/`)

| 文件 | 用途 | 实验版本 |
|------|------|---------|
| `compare_shields.py` | 四种配置对比: B/V2/V3A/V3B | v3 早期 |
| `train_sbc_for_v3.py` | 训练 SBC (35轮, 快速) | v3 早期 |
| `train_improved_sbc.py` | 训练 SBC (120轮, 96.58%) | v3 改进 |
| `test_qp_activation.py` | 验证 QP 盾是否激活 | v3 早期 |
| `v3_parameter_sweep.py` | 参数扫描: 26配置×2运行 | v3 中期 |
| `v3_full_evaluation.py` | 完整评估: 9配置×3运行×120场景 | v3 后期 |
| `qp_benefit_experiments.py` | QP 价值证明: 4项系统实验 | v3 最终 |
| `collect_qp_benefits.py` | QP 收益详细指标收集 | v2/v3 |

### 4.4 工具层 (`src/utils/`)

#### `qp_solver.py` — QP 求解器封装

```
solve_qp_qpth(Q, q, G, h): 可微分 QP (训练用)
solve_qp_cvxopt(Q, q, G, h): 数值 QP (推理用)
solve_qp_batch(...): 批量求解 + 失败回退
```

---

## 5. 实验v2

> **QP 嵌入训练循环** — 否定的结果, 导致 v3 设计

### 5.1 目录结构

```
实验v2/
├── README.md
├── docs/
│   ├── experimental_plan.md          # 实验计划
│   ├── complete_experiment_report.md # 完整报告
│   ├── final_results.md              # 最终结果
│   └── results_report.md             # 结果总结
├── results/                          # 实验数据 (JSON)
│   ├── baseline_nf005/               # Baseline 实验
│   ├── qp_integrated_nf005/          # QP 集成实验
│   ├── qp_cbf001_50iters/            # CBF 消融实验
│   └── qp_benefits_*.json            # QP 收益分析
└── figures/
    └── comparison_table.md           # 对比表
```

### 5.2 关键发现

- Baseline SBC 概率界: **95.8%**
- QP 嵌入训练: **87.9%** (-8pp)
- QP 提升了运行时指标 (控制平滑 64×, 安全裕度 +12%)
- 但形式化保证下降是不可接受的
- **结论**: QP 必须与 SBC 训练解耦 → v3

---

## 6. 实验v3

> **QP 仅推理时使用, 与 SBC 训练解耦** — 当前工作

### 6.1 五种对比配置

| 配置 | 代码位置 | QP训练 | QP推理 | p参数来源 | 状态 |
|------|---------|:-----:|:-----:|----------|:----:|
| **B** (Baseline) | 直接使用 PPO controller | ✗ | ✗ | N/A | 基准 |
| **V2** | `qp_controller.py` | ✓ | ✓ | NN 学习 | ❌ SBC 降8% |
| **V3A** | `sbc_modulated_qp.FixedQPShield` | ✗ | ✓ | 固定 p=2.0 | ✅ **最优** |
| **V3B** | `sbc_modulated_qp.SBCModulatedQPShield` | ✗ | ✓ | SBC 调制 | ❌ SBC 不校准 |
| **V3C** | `qp_p_network.TrainedQPShield` | ✗ | ✓ | 训练 p-network | △ 待改进 |
| **V3D** | `qp_p_network.BarrierModulatedQPShield` | ✗ | ✓ | Barrier 调制 | ✅ **自适应** |

### 6.2 文档体系

| 文档 | 内容 |
|------|------|
| `README.md` | 快速入门 + 结果摘要 |
| `docs/experimental_plan.md` | 原始实验计划 (v3 设计) |
| `docs/experiment_design.md` | 详细架构设计 + 数据流 + v2 根因分析 |
| `docs/complete_results.md` | 首轮实验: v2 vs v3 对比 |
| `docs/sweep_results.md` | 参数扫描: 26配置×2运行 详细分析 |
| `docs/v3_improved_results.md` | 改进版完整结果: SBC 96.6%, 9配置评估, 场景分析, p参数指南 |
| `docs/qp_shield_benefit_report.md` | QP 价值验证: 4项实验, SBC+QP vs 单独QP, 受控验证 |

### 6.3 实验结果文件

| 文件 | 内容 |
|------|------|
| `results/trained_sbc.pth` | 原始 SBC 模型 (35轮, 95.9%) |
| `results/trained_sbc_improved.pth` | 改进 SBC (120轮, 96.58%) |
| `results/sbc_metadata.json` | SBC 训练元数据 |
| `results/trained_p_network.pth` | p-network 权重 + 训练历史 |
| `results/v3_comparison_*.json` | 四配置对比结果 |
| `results/v3_sweep_*.json` | 26配置参数扫描原始数据 |
| `results/v3_full_evaluation_*.json` | 完整评估原始数据 |
| `results/qp_benefit_experiments_*.json` | QP 价值验证原始数据 |
| `results/adversarial_test.json` | 对抗测试结果 |

---

## 7. 完整数据流

### 7.1 训练阶段数据流 (v3)

```
┌─────────────────────────────────────────────────────────────────┐
│                      V3 训练阶段 (无 QP!)                         │
│                                                                 │
│  1. 状态采样                                                      │
│     s_t 从 100×100 均匀网格采样                                    │
│     [d ∈ (5,16), v ∈ (0,3)] (归一化)                             │
│                                                                 │
│  2. 扰动                                                         │
│     s_t_pert = s_t + δ·random([-0.5, 0.5])                       │
│                                                                 │
│  3. 感知 (frozen)                                                │
│     z ~ N(0,1) ──→ gen_net(z, d) ──→ img(32×32)                 │
│                                        │                        │
│                                   state_net(img)                 │
│                                        │                        │
│                                    ŝ_t = [d_est, v_est]          │
│                                                                 │
│  4. ★ 控制器 (DIRECT 模式, 无 QP!)                                │
│     u_t = controller_net([ŝ_t, v_t])                             │
│     ↑ PPO 预训练权重, 通过 martingale + MSE 微调                  │
│                                                                 │
│  5. 动力学                                                       │
│     d_{t+1} = d_t - v_t·dt                                      │
│     v_{t+1} = v_t - u_t·dt                                      │
│     v_{t+1} = clip(v_{t+1}, 0, 3)                                │
│     + triangular 噪声                                            │
│                                                                 │
│  6. SBC 训练                                                     │
│     B(s_t) = MLP[2,32,16,8,1](tanh+square)(s_t)                 │
│     损失:                                                        │
│       L_martingale = mean(max(B(s_next)_mean - B(s) + 0.1, 0))   │
│       L_region = relu(max(B_init)-1) + relu(20-min(B_unsafe))    │
│     → 保证: B(s) 在初始区 ≤ 1, 在不安全区 ≥ 1/(1-0.95)=20        │
│                                                                 │
│  7. IBP 形式化验证                                                │
│     VTVerifier: 对 100×100 网格验证 B(s_{t+1}) ≤ 1.2·B(s_t)     │
│     概率下界: p ≥ 1 - UB(B_init)/LB(B_unsafe) = 96.58%          │
│                                                                 │
│  8. 控制器微调                                                    │
│     L_ctrl = martingale_loss + MSE(teacher)                      │
│     → 策略精化: 让 SBC 的期望递减条件更容易满足                   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 推理阶段数据流 (v3)

```
┌─────────────────────────────────────────────────────────────────┐
│                    V3 推理阶段 (QP 安全盾)                         │
│                                                                 │
│  对每个时间步:                                                    │
│                                                                 │
│  1. 感知 (同训练)                                                │
│     gen_net(z, d) → img → state_net(img) → ŝ_t                  │
│                                                                 │
│  2. 控制器                                                       │
│     u_ref = controller_net([ŝ_t, v_t])   ← 参考控制              │
│                                                                 │
│  3. ★ QP 安全盾 (★ v3 新增)                                      │
│     ┌──────────────────────────────────────┐                     │
│     │  计算 p 参数:                         │                     │
│     │    V3A: p = 2.0 (固定)                │                     │
│     │    V3B: p = f(B(s)) (SBC调制)         │                     │
│     │    V3C: p = NN(s) (训练p-network)      │                     │
│     │    V3D: p = f(b(s)) (Barrier调制)     │                     │
│     │                                       │                     │
│     │  CBF约束: -t_gap·u ≤ -v + p·b(s)      │                     │
│     │  控制限: -3 ≤ u ≤ 3                    │                     │
│     │                                       │                     │
│     │  QP求解: min ½(u-u_ref)²              │                     │
│     │          s.t. CBF + 控制限             │                     │
│     │                                       │                     │
│     │  IF |u_safe - u_ref| > 0.01:          │                     │
│     │    干预! u = u_safe                    │                     │
│     │  ELSE:                                 │                     │
│     │    u = u_ref (不干预)                  │                     │
│     └──────────────────────────────────────┘                     │
│                                                                 │
│  4. 执行                                                         │
│     s_{t+1} = f(s_t, u_safe)                                     │
│     d_{t+1} = d_t - v_t·dt                                      │
│     v_{t+1} = v_t - u_safe·dt                                   │
│                                                                 │
│  5. 记录                                                         │
│     u_safe, u_ref, p, B(s), intervened, margin, b(s)            │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 v2 vs v3 关键区别

```
v2 (QP 嵌入训练):
  ┌──────────────────────────────────────┐
  │  训练循环包含 QP:                     │
  │  controller → shared_backbone        │
  │           ├→ q_head (u_ref)           │
  │           └→ p_head (CBF参数)         │
  │  u* = QP(u_ref, p, constraints)      │
  │  → u* 进入动力学 → SBC 验证           │
  │                                      │
  │  问题: QP 层增加 Lipschitz 常数       │
  │  → SBC 验证 K 值增大                  │
  │  → 概率界 95.8% → 87.9% (-8pp)       │
  └──────────────────────────────────────┘

v3 (QP 仅推理):
  ┌──────────────────────────────────────┐
  │  训练循环无 QP:                       │
  │  controller → u (直接输出)            │
  │  → u 进入动力学 → SBC 验证            │
  │  → 概率界保持 96.58%                  │
  │                                      │
  │  推理时插入 QP 盾:                    │
  │  controller → u_ref                  │
  │  QP 盾 → u_safe (安全修正)            │
  │  → 运行时安全保证                      │
  │  → 不影响 SBC 质量                    │
  └──────────────────────────────────────┘
```

---

## 8. 文件依赖关系图

### 8.1 模块依赖

```
artical-F122/ (原始代码, 不修改)
│
├── Aebs/system/env.py          ← 环境定义
├── Aebs/VT/utils.py            ← MLP, triangular, martingale_loss
├── Aebs/VT/train.py            ← VTLearner (SBC+Controller训练)
├── Aebs/VT/verify.py           ← VTVerifier (IBP验证)
├── Aebs/VT/loop.py             ← 主训练循环
├── cGAN/taxi_models_and_data   ← AebsMLPGenerator
├── Combined_network/model.py   ← AebsEnd2EndNet
└── Aebs/controller/            ← PPO 模型

src/ (新增代码, import 原始模块)
│
├── models/
│   ├── cbf_constraints.py      ← 独立模块 (CBF数学)
│   │   └── 被引用: qp_controller, sbc_modulated_qp, qp_p_network
│   │
│   ├── qp_controller.py        ← v2 QP控制器
│   │   ├── 依赖: cbf_constraints, qpth, cvxopt
│   │   └── 被引用: qp_trainer, compare_shields
│   │
│   ├── sbc_modulated_qp.py     ← v3 V3A/V3B QP盾
│   │   ├── 依赖: cbf_constraints, Aebs.VT.utils (MLP)
│   │   └── 被引用: 所有 eval/ 脚本
│   │
│   └── qp_p_network.py         ← v3 V3C/V3D QP盾 + p-network训练
│       ├── 依赖: cbf_constraints, sbc_modulated_qp
│       └── 被引用: v3_full_evaluation, qp_benefit_experiments
│
├── training/
│   ├── qp_trainer.py           ← v2 QP VTLearner
│   │   ├── 依赖: Aebs.VT.train, qp_controller
│   │   └── 被引用: qp_loop
│   └── qp_loop.py              ← v2 主循环
│       └── 依赖: qp_trainer, Aebs.VT.verify
│
├── eval/
│   ├── train_sbc_for_v3.py     ← SBC 快速训练
│   ├── train_improved_sbc.py   ← SBC 改进训练
│   ├── compare_shields.py      ← 四配置对比
│   ├── test_qp_activation.py   ← QP 激活测试
│   ├── v3_parameter_sweep.py   ← 参数扫描
│   ├── v3_full_evaluation.py   ← 完整评估
│   ├── qp_benefit_experiments.py ← QP价值实验
│   └── collect_qp_benefits.py  ← QP 收益收集
│   (所有 eval/ 脚本依赖: sbc_modulated_qp, cbf_constraints)
│
└── utils/
    └── qp_solver.py            ← QP 求解器
        └── 被引用: qp_controller
```

### 8.2 实验脚本演进链

```
v2 实验 (训练时 QP):
  qp_trainer.py → qp_loop.py
  → 结果: SBC 概率界下降 → v3 设计

v3 早期 (验证概念):
  train_sbc_for_v3.py (35轮训练)
  → compare_shields.py (四配置对比)
  → test_qp_activation.py (QP激活测试)
  → 发现: QP 未激活 (SBC 不校准)

v3 中期 (参数扫描):
  train_improved_sbc.py (120轮, 96.6%)
  → v3_parameter_sweep.py (26配置×2)
  → 发现: 干预率由有效 p 值决定

v3 后期 (完整评估):
  qp_p_network.py (训练 p-network)
  → v3_full_evaluation.py (9配置×3运行)
  → 发现: V3A p=2.0 最优, V3D p_min=0.5 自适应

v3 最终 (价值证明):
  qp_benefit_experiments.py (4项实验)
  → 受控验证: QP 阻止故障控制器
  → 报告: qp_shield_benefit_report.md
```

---

## 9. 关键概念速查

| 概念 | 代码位置 | 说明 |
|------|---------|------|
| **SBC B(s)** | `Aebs/VT/utils.py:MLP` | 随机屏障证书, [2,32,16,8,1], tanh + square |
| **Martingale Loss** | `Aebs/VT/utils.py:martingale_loss` | max(B(s_next) - B(s) + ε, 0) |
| **Region Loss** | `train_improved_sbc.py` | B(init)≤1, B(unsafe)≥1/(1-p) |
| **IBP 验证** | `Aebs/VT/verify.py` | auto_LiRPA 包围分析, 100×100 网格 |
| **CBF 约束** | `src/models/cbf_constraints.py` | -t_gap·u ≤ -v + p·(d-v·t_gap) |
| **QP 求解** | `src/models/sbc_modulated_qp.py` | 1D 解析解 (高效) |
| **QP 求解 (v2)** | `src/utils/qp_solver.py` | qpth (训练) / cvxopt (推理) |
| **p-network** | `src/models/qp_p_network.py` | NN 学习 state → optimal p |
| **Barrier 调制** | `src/models/qp_p_network.py:BarrierModulatedQPShield` | p = f(b(s)) = f(d - v·t_gap) |
| **干预率** | 所有 eval/ 脚本 | |u_safe - u_ref| > 0.01 的步骤比例 |

---

## 10. 运行命令速查

```bash
# 所有命令从 artical-F122/ 目录运行
cd /root/paper-combination/artical-F122

# === 训练 ===
# 改进 SBC (120轮, 96.58% 概率界)
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/train_improved_sbc.py

# 训练 p-network (V3C)
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/models/qp_p_network.py

# === 评估 ===
# 完整评估 (9配置×3运行)
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/v3_full_evaluation.py

# QP 价值证明实验
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/qp_benefit_experiments.py

# 参数扫描 (26配置)
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/v3_parameter_sweep.py

# === 查看结果 ===
ls /root/paper-combination/实验v3/results/
cat /root/paper-combination/实验v3/docs/qp_shield_benefit_report.md
```
