# 实验方案：SafePVC + BarrierNet QP 安全滤波层集成

> **实验编号**: 实验v2
> **日期**: 2026-07-19
> **目标**: 在 SafePVC（Provably Probabilistic Safe Controller Synthesis for Vision-Based NNCS）框架中集成 BarrierNet 的可微分 QP 安全滤波层，构建 CBF-QP + SBC 双重安全机制

---

## 目录

1. [研究动机与核心思想](#1-研究动机与核心思想)
2. [两篇论文方法回顾](#2-两篇论文方法回顾)
3. [集成架构设计](#3-集成架构设计)
4. [详细数据流](#4-详细数据流)
5. [实施步骤](#5-实施步骤)
6. [代码修改清单](#6-代码修改清单)
7. [对比实验设计](#7-对比实验设计)
8. [评估指标](#8-评估指标)
9. [预期结果](#9-预期结果)

---

## 1. 研究动机与核心思想

### 1.1 研究动机

SafePVC（artical-F122）使用 SBC（Stochastic Barrier Certificate）提供离线概率安全验证，但运行时没有显式的安全约束——控制器直接输出动作，安全性完全依赖网络权重的隐式编码。BarrierNet 使用可微分 QP 层在运行时强制执行 CBF（Control Barrier Function）约束，提供确定性安全保证，但缺乏离线形式化验证和概率安全界。

**核心思路**：将 BarrierNet 的可微分 QP 安全滤波层嵌入 SafePVC 的控制器末端，实现：
- **在线确定性安全**（CBF-QP 逐步约束）
- **离线概率安全验证**（SBC 无限时域验证）
- **双重安全机制互补**

### 1.2 科学问题

1. CBF-QP 层能否提升 SafePVC 的概率安全下界？
2. QP 层的引入是否会影响 SBC 验证的收敛速度？
3. 双重安全机制下，控制器性能（任务完成率、控制平滑性）是否保持？
4. 在扰动增强条件下，QP 层是否提供额外的鲁棒性？

---

## 2. 两篇论文方法回顾

### 2.1 SafePVC (artical-F122) 核心方法

**系统模型**（CARLA 紧急制动场景）：
- 状态: $s = (d, v)$，d = 距离 (m), v = 速度 (m/s)
- 动力学: $d_{k+1} = d_k - v_k \Delta t$, $v_{k+1} = v_k - a_k \Delta t$
- 控制: $a \in [-3, 3]$ m/s² (加速度)

**流水线**：
1. **感知模型**: cGAN/MLP 将低维状态映射为视觉观测 $o_t = g(s_t, z_t)$
2. **RL 预训练**: PPO 在视觉观测上预训练控制器 $\pi_0$
3. **VCLS 构建**: 连接感知模型 + 控制器形成端到端系统
4. **扰动估计**: 数据驱动估计状态扰动分布 $\Delta s \sim \mu$
5. **SBC 训练**: 神经网络 $B(s)$ 学习随机障碍证书
6. **验证与精化**: 使用 IBP (Interval Bound Propagation) 验证 SBC 条件，反例驱动交替优化

**SBC 四个条件** (Theorem 2.2):
- (i) $B(s) \geq 0$ for all $s \in S$
- (ii) $B(s) \leq 1$ for all $s \in S_0$
- (iii) $B(s) \geq \frac{1}{1-p}$ for all $s \in X_u$
- (iv) $B(s) \geq \mathbb{E}_{\Delta s \sim \mu}[B(\tilde{F}(s, z_0, \Delta s))] + \epsilon$

### 2.2 BarrierNet 核心方法

**HOCBF 条件**:
$$\psi_i(x, z) := \dot{\psi}_{i-1}(x, z) + p_i(z) \alpha_i(\psi_{i-1}(x, z))$$

**QP 形式**:
$$u^* = \arg\min_u \frac{1}{2}u^T H(z|\theta_h) u + F^T(z|\theta_f) u$$
$$\text{s.t. } L_f^m b(x) + [L_g L_f^{m-1} b(x)]u + O(b(x), z) + p_m(z)\alpha_m(\psi_{m-1}(x,z)) \geq 0$$

**双分支架构**:
- 分支 1: 参考控制 $q$ (线性输出)
- 分支 2: CBF 参数 $p = 4 \cdot \sigma(\text{NN}(z))$ (sigmoid 保证正性)
- QP 层: 接收 $q, p, s$，输出安全控制 $u^*$

---

## 3. 集成架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SafePVC + BarrierNet QP 集成架构                           │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  阶段 1: 感知模型 (不变)                                              │   │
│  │  s, z ──► cGAN/MLP g(s,z) ──► 图像 o                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  阶段 2: RL 预训练 (不变)                                             │   │
│  │  o ──► PPO π₀(o) ──► u ──► 环境                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  阶段 3: 双分支控制器 + QP (★ 核心修改)                               │   │
│  │                                                                      │   │
│  │  o_t ──► 共享主干 (FC+ReLU)                                         │   │
│  │           │                                                          │   │
│  │           ├──► 分支 1 (q_head): FC → q_t ∈ ℝ¹ (参考加速度)          │   │
│  │           │                                                          │   │
│  │           └──► 分支 2 (p_head): FC → 4·σ(·) ∈ ℝ¹ (CBF 参数)        │   │
│  │                      │             │                                 │   │
│  │                      ▼             ▼                                 │   │
│  │  ┌──────────────────────────────────────────────────────┐            │   │
│  │  │  QP 安全滤波层                                        │            │   │
│  │  │  min  ½‖u‖² + q_t·u                                   │            │   │
│  │  │  s.t. CBF: G(s_t, p_t)·u ≤ h(s_t, p_t)               │            │   │
│  │  │       u_min ≤ u ≤ u_max                               │            │   │
│  │  │  → u_t* (安全加速度)                                   │            │   │
│  │  └──────────────────────────────────────────────────────┘            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  阶段 4: SBC 验证与交替训练 (修改)                                    │   │
│  │  - SBC 验证具 QP 层的闭环系统                                        │   │
│  │  - 控制器损失加入 CBF 约束损失                                       │   │
│  │  - 反例驱动的交替优化                                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 网络结构对比

**原始 SafePVC 控制器**:
```
输入: o_t (图像/状态估计)
  → FC(1024→256) → ReLU → FC(256→64) → ReLU → FC(64→1)
  → 输出: a ∈ ℝ (加速度)
```

**修改后 SafePVC+QP 控制器**:
```
输入: [state_estimate, velocity] ∈ ℝ²
  → FC(2→256) → ReLU → FC(256→256) → ReLU → FC(256→256) → ReLU
  → ┌─ FC(256→1) → q ∈ ℝ (参考加速度)
    └─ FC(256→1) → 4·σ(·) → p ∈ (0,4) (CBF 参数)
  → QP: min ½u² + q·u s.t. CBF 约束 + 控制边界
  → 输出: u* ∈ [-3, 3] (安全加速度)
```

### 3.3 AEBS 场景的 CBF 约束推导

**障碍函数**（安全距离约束）:
$$b(s) = d - v \cdot t_{\text{gap}}$$

其中 $t_{\text{gap}} = 1.5$ s 是安全时距。

**系统动力学**（离散时间）:
$$d_{k+1} = d_k - v_k \Delta t$$
$$v_{k+1} = v_k - u_k \Delta t$$

其中 $\Delta t = 0.05$ s.

**相对阶分析**: 控制 u 直接出现在 $\dot{v}$ 中，进而出现在 $\dot{b}$ 中 → 相对阶 = 1，使用标准 CBF。

**CBF 条件推导**:

首先将离散动力学近似为连续形式：
$$\dot{d} = -v, \quad \dot{v} = -u$$

计算 $\dot{b}(s)$:
$$\dot{b} = \frac{\partial b}{\partial d}\dot{d} + \frac{\partial b}{\partial v}\dot{v} = 1 \cdot (-v) + (-t_{\text{gap}}) \cdot (-u) = -v + t_{\text{gap}} \cdot u$$

CBF 条件:
$$\dot{b} + p \cdot b \geq 0$$
$$-v + t_{\text{gap}} \cdot u + p \cdot (d - v \cdot t_{\text{gap}}) \geq 0$$
$$t_{\text{gap}} \cdot u \geq v - p \cdot (d - v \cdot t_{\text{gap}})$$

**QP 标准形式** ($Gu \leq h$):
$$-t_{\text{gap}} \cdot u \leq -v + p \cdot (d - v \cdot t_{\text{gap}})$$

即:
$$G = -t_{\text{gap}} = -1.5$$
$$h = -v + p \cdot (d - 1.5v)$$

---

## 4. 详细数据流

### 4.1 训练阶段数据流

```
每个训练迭代:

1. 采样状态 s_t ∈ S (从离散网格或反例缓冲区)
2. 生成扰动状态: s_t_pert = s_t + δ·(random[-0.5, 0.5])
3. 感知: o_t = g(s_t_pert, z₀)  [gen_net, frozen]
4. 状态估计: ŝ_t = state_net(o_t)  [state_net, frozen]
5. 控制器前向:
   a. features = shared_backbone([ŝ_t, v_t])
   b. q_t = q_head(features)       [参考控制]
   c. p_t = 4·sigmoid(p_head(features))  [CBF参数]
6. QP求解:
   a. 构造 G, h (基于 s_t 和 p_t)
   b. u_t* = QPFunction(Q, q_t, G, h)
7. 动力学: s_{t+1} = f(s_t, u_t*) + noise
8. SBC评估: B(s_t), B(s_{t+1})
9. 损失计算:
   - L_dec (SBC 期望递减)
   - L_lip (Lipschitz 正则化)
   - L_mse (QP输出 vs RL策略)
   - L_CBF (CBF约束违反惩罚)
10. 梯度反传: 通过 QPFunction 的隐式微分
11. 参数更新: controller_net (q_head + p_head + shared_backbone)
```

### 4.2 推理阶段数据流

```
每个时间步:

1. 摄像头捕捉 → 图像 o_t
2. state_net(o_t) → 状态估计 ŝ_t
3. shared_backbone([ŝ_t, v_t]) → features
4. q_head(features) → q_t, p_head(features) → p_t = 4·σ(·)
5. cvxopt/cvxpy 求解 QP:
   min ½u² + q_t·u
   s.t. -t_gap·u ≤ -v + p_t·(d - t_gap·v)
        -3 ≤ u ≤ 3
6. u_t* → 执行器
```

---

## 5. 实施步骤

### 步骤 1: 修改控制器网络 (Combined_network/model.py)

**新建**: `QPAebsController` 类
- 共享主干: 3层 FC + ReLU
- 分支 1 (q_head): FC → 线性输出 (参考加速度)
- 分支 2 (p_head): FC → 4×sigmoid (CBF 参数)
- QP 层: build_cbf_constraints + solve_qp

### 步骤 2: 实现 CBF 约束构建

**新建**: `utils/cbf_constraints.py`
- `barrier_function(state)`: 计算 b(s) = d - v * t_gap
- `compute_lie_derivatives(state)`: 计算 Lf_b, Lg_b
- `build_cbf_constraints(state, p)`: 构造 G, h 矩阵

### 步骤 3: 修改 VCLS 构建 (Aebs/system/combined.py)

- 使用新的 `QPAebsController` 替换原始控制器
- 闭环系统: $s_{t+1} = f(s_t, \text{QP}(\pi_\theta(\text{VCLS}(s_t)), s_t)) + \Delta s$

### 步骤 4: 修改训练器 (Aebs/VT/train.py)

- `train_step_p`: 
  - 前向传播包含 QP 层
  - 损失增加 CBF 约束违反项
  - MSE 损失比较 QP 输出与 RL 策略
- `train_step_joint`:
  - 支持 QP 层的梯度反传

### 步骤 5: 修改主循环 (Aebs/VT/loop.py)

- 加载新的 QP 控制器
- 调整训练超参数
- 添加 CBF 相关的日志记录

### 步骤 6: 运行对比实验

详见第 7 节。

---

## 6. 代码修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `实验v2/code/models/qp_controller.py` | **新建** | 双分支 QP 控制器 + CBF 约束构造 |
| `实验v2/code/models/cbf_constraints.py` | **新建** | AEBS 场景的 CBF 约束数学实现 |
| `实验v2/code/training/qp_trainer.py` | **新建** | 支持 QP 层的训练器 |
| `实验v2/code/training/qp_loop.py` | **新建** | 集成 QP 的主训练循环 |
| `实验v2/code/utils/qp_solver.py` | **新建** | QP 求解器封装 (qpth + cvxopt) |
| `实验v2/code/run_baseline.py` | **新建** | 基线实验运行脚本 |
| `实验v2/code/run_qp_experiment.py` | **新建** | QP 集成实验运行脚本 |
| `实验v2/configs/experiment_config.yaml` | **新建** | 实验配置文件 |

---

## 7. 对比实验设计

### 7.1 实验条件

| 条件 | 代号 | 运行时安全 | 离线验证 | 说明 |
|------|------|-----------|---------|------|
| **基线 (SafePVC 原始)** | `baseline` | ❌ 无 | ✅ SBC | 原始 SafePVC，控制器直接输出 |
| **SafePVC + QP (本方案)** | `qp_integrated` | ✅ CBF-QP | ✅ SBC | 双分支控制器 + QP 层 |
| **Pure BarrierNet** | `barriernet_only` | ✅ CBF-QP | ❌ 无 | 纯 BarrierNet，无 SBC 验证 |

### 7.2 实验场景

所有实验在 **CARLA 紧急制动 (AEBS)** 场景上进行：
- 状态空间: $d \in [5, 16]$ m, $v \in [0, 3]$ m/s
- 初始集: $d \in [15, 16]$ m, $v \in [2.5, 3]$ m/s
- 不安全集: $d \in [5, 6]$ m, $v \in [0.5, 3]$ m/s
- 控制空间: $a \in [-3, 3]$ m/s²

### 7.3 实验矩阵

| 实验编号 | 条件 | 扰动因子 | 训练轮数 | 重复次数 |
|---------|------|---------|---------|---------|
| E1 | baseline | 0.01 | 100 | 3 |
| E2 | baseline | 0.05 | 100 | 3 |
| E3 | baseline | 0.10 | 100 | 3 |
| E4 | qp_integrated | 0.01 | 100 | 3 |
| E5 | qp_integrated | 0.05 | 100 | 3 |
| E6 | qp_integrated | 0.10 | 100 | 3 |

### 7.4 消融实验

| 消融实验 | 说明 |
|---------|------|
| **A1: λ_CBF 权重** | 测试 λ_CBF ∈ {0.1, 1.0, 10.0} 对性能的影响 |
| **A2: CBF 参数范围** | 测试 p ∈ (0,2) vs (0,4) vs (0,10) 的影响 |
| **A3: QP 求解器** | 对比 qpth vs cvxopt 的训练稳定性 |
| **A4: 训练策略** | 对比交替训练 vs 联合训练 |

### 7.5 鲁棒性测试

- 不同初始条件下的安全概率
- 不同扰动强度下的安全概率
- QP 无可行解频率统计

---

## 8. 评估指标

### 8.1 安全性指标

| 指标 | 定义 | 期望方向 |
|------|------|---------|
| **概率安全下界 $p$** | SBC 验证得到的 $\mathbb{P}[\text{Safe}] \geq p$ | ↑ 越高越好 |
| **硬违反数** | 验证网格中不满足递减条件的网格数 | ↓ 越低越好 |
| **SBC 收敛轮数** | 达到验证通过所需迭代轮数 | ↓ 越快越好 |
| **QP 可行率** | QP 有可行解的时间步比例 | ↑ 越接近 1 越好 |
| **CBF 违反率** | 运行中 $b(s) < 0$ 的时间步比例 | ↓ 应为 0 |

### 8.2 性能指标

| 指标 | 定义 | 期望 |
|------|------|------|
| **任务完成率** | 成功进入安全距离且速度低于阈值的比例 | ↑ |
| **平均制动距离** | 从初始状态到安全状态的平均距离 | 适中 |
| **控制平滑性** | 相邻时间步控制变化的方差 | ↓ 越小越平滑 |
| **与 RL 策略偏差** | $\|u^* - u_{RL}\|_2$ 的平均值 | 适中 |

### 8.3 训练指标

| 指标 | 定义 |
|------|------|
| **L_dec 收敛曲线** | SBC 期望递减损失随迭代变化 |
| **L_mse 收敛曲线** | MSE 损失随迭代变化 |
| **L_CBF 收敛曲线** | CBF 约束损失随迭代变化 |
| **梯度范数** | QP 层梯度范数，监控训练稳定性 |

---

## 9. 预期结果

### 9.1 主要假设

1. **H1 (安全性提升)**: SafePVC+QP 的概率安全下界 $p$ 显著高于基线 SafePVC
   - 预期: baseline $p \approx 72\%$, qp_integrated $p \approx 85\%+$
   
2. **H2 (验证加速)**: SafePVC+QP 的 SBC 验证收敛更快
   - 预期: qp_integrated 需要更少的迭代轮数

3. **H3 (鲁棒性增强)**: 在高扰动条件下，SafePVC+QP 的安全概率下降更慢
   - 预期: qp_integrated 在 factor=0.10 时仍能保持较高安全概率

4. **H4 (性能保持)**: 集成 QP 不会显著降低任务完成率
   - 预期: 任务完成率下降 < 5%

### 9.2 潜在风险

1. **QP 梯度不稳定**: qpth 的隐式微分可能导致梯度爆炸
   - 缓解: 梯度裁剪 + double 精度
   
2. **QP 频繁无解**: CBF 约束与控制边界冲突
   - 缓解: 引入松弛变量或调整 CBF 参数范围

3. **SBC 验证保守性增加**: QP 层引入额外非线性
   - 缓解: 增加 SBC 网络容量

---

## 附录 A: 依赖关系

```
实验v2/
├── code/                          # 修改后的代码
│   ├── models/
│   │   ├── qp_controller.py       # ★ 核心: 双分支 QP 控制器
│   │   └── cbf_constraints.py     # ★ 核心: CBF 约束数学
│   ├── training/
│   │   ├── qp_trainer.py          # ★ 核心: QP 训练器
│   │   └── qp_loop.py             # ★ 核心: 主循环
│   ├── utils/
│   │   └── qp_solver.py           # QP 求解器封装
│   ├── run_baseline.py            # 基线运行脚本
│   └── run_qp_experiment.py       # QP 实验运行脚本
├── configs/
│   └── experiment_config.yaml     # 实验配置
├── results/                       # 实验结果
│   ├── baseline/                  # 基线结果
│   └── qp_integrated/             # QP 集成结果
├── figures/                       # 图表
└── docs/                          # 文档
    ├── experimental_plan.md       # 本文档
    └── results_report.md          # 结果报告
```

## 附录 B: 关键超参数

| 超参数 | 值 | 说明 |
|--------|-----|------|
| 学习率 (SBC) | 3×10⁻³ | L-Net Adam |
| 学习率 (Controller) | 5×10⁻² | P-Net Adam |
| λ_CBF | 1.0 | CBF 损失权重 |
| λ_M | 10.0 | MSE 损失权重 |
| λ_lip | 0.001 | Lipschitz 正则化权重 |
| γ_decrease | 1.0 | 期望递减因子 |
| ε | 0.1 | 递减裕度 |
| t_gap | 1.5 | 安全时距 (s) |
| batch_size | 256 | 训练批次大小 |
| grid_size | [100, 100] | 验证网格 |
| 扰动因子 | 0.01/0.05/0.10 | 状态扰动幅度 |
