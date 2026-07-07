# SafePVC × BarrierNet 合成方案：用 BarrierNet 替换 SBC 后端安全保障

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 两套系统深度对比](#2-两套系统深度对比)
- [3. 可行性分析](#3-可行性分析)
- [4. 总体合成架构](#4-总体合成架构)
- [5. 数学理论适配](#5-数学理论适配)
- [5½. 随机扰动的处理：三种方案详解](#5½-随机扰动的处理三种方案详解)
- [6. 代码修改详细方案](#6-代码修改详细方案)
- [7. 新训练流程](#7-新训练流程)
- [8. 实验方案](#8-实验方案)
- [9. 风险与缓解策略](#9-风险与缓解策略)
- [10. 实施路线图](#10-实施路线图)
- [11. 文件修改清单](#11-文件修改清单)

---

## 1. 项目概述

### 1.1 目标

将论文 SafePVC（artical-F122）的**后端安全保障模块**——随机障碍证书（Stochastic Barrier Certificate, SBC）及其配套的鞅理论验证、IBP形式化验证、CEGIS交替训练循环——**整体替换**为 BarrierNet 的**高阶控制障碍函数（HOCBF）可微QP层**方案。

### 1.2 核心动机

| 维度 | SafePVC (SBC) | BarrierNet (HOCBF-QP) |
|------|--------------|----------------------|
| 安全保证类型 | 概率性（P_safe ≥ 1 - 1/ratio） | 确定性（b(x(t)) ≥ 0, ∀t） |
| 验证方式 | 需要 IBP 形式化验证 + 网格搜索 | 安全-by-construction，无需后验验证 |
| 运行时机制 | 控制器直接输出动作，离线验证 | 每步求解QP，在线实时安全过滤 |
| 保守性 | 较高（超鞅上界） | 可调（通过学习 p_i(z) 软化） |
| 训练复杂度 | 高（CEGIS循环100+次迭代） | 低（标准行为克隆 + 端到端梯度） |
| 代码复杂度 | 高（auto_LiRPA + IBP + 噪声网格积分） | 中（OptNet QP层 + 正向传播） |

### 1.3 结论：**可行，但需要重大架构改造**

替换在技术上是可行的，但并非简单的"热替换"。需要对系统动力学建模、安全约束定义、训练流程和论文理论部分进行**全面重构**。

---

## 2. 两套系统深度对比

### 2.1 SafePVC 架构 (artical-F122)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SafePVC 完整流水线                                │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  阶段一：可验证闭环系统 (VCLS) 构建                                 │  │
│  │                                                                   │  │
│  │  z (潜在变量) ─┐                                                  │  │
│  │                ├─→ gen_net(z, d) → 图像(32×32) → state_net → d_est │  │
│  │  d (距离)   ───┘       [冻结]              [冻结]                  │  │
│  │                                                                    │  │
│  │  v (速度) ──────────────────────────────→ [d_est, v] → controller → acc │  │
│  │                                                         [可训练]    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  阶段二：SBC 验证 + CEGIS 交替优化                                 │  │
│  │                                                                   │  │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐  │  │
│  │  │ 训练 B(s)    │ ──→ │ IBP 验证     │ ──→ │ 提取反例          │  │  │
│  │  │ (10 epochs)  │     │ 网格搜索     │     │ (违反状态)        │  │  │
│  │  │ 鞅损失+区域  │     │ E[B(s')]≤B(s)│     │                  │  │  │
│  │  └──────────────┘     └──────────────┘     └────────┬─────────┘  │  │
│  │       ▲                                              │            │  │
│  │       │              ┌──────────────────┐            │            │  │
│  │       └──────────────│ 训练控制器 π     │◄───────────┘            │  │
│  │                      │ (1 epoch)        │                         │  │
│  │                      │ 鞅损失+MSE蒸馏   │                         │  │
│  │                      └──────────────────┘                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  输出：P_safe ≥ 1 - max B(s₀) / min B(s_unsafe)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键组件：**
- `Aebs/VT/loop.py` — CEGIS 主循环（100次迭代）
- `Aebs/VT/train.py` — VTLearner（SBC训练 + 控制器微调）
- `Aebs/VT/verify.py` — VTVerifier（IBP验证 + 反例生成）
- `Aebs/VT/utils.py` — MLP障碍证书 + 鞅损失函数
- `auto_LiRPA/` — 神经网络区间传播验证库

**系统动力学 (CARLA AEBS):**
```
d_{t+1} = d_t - v_t × dt          (dt = 0.05s)
v_{t+1} = v_t - acc_t × dt

状态空间: s = [d_norm, v]  ∈ R²
控制输入: u = acc           ∈ R¹
d_norm ∈ [~0.31, ~1.0]  (归一化距离, std1 ≈ 16)
v ∈ [0, 3] m/s
acc ∈ [-3, 3] m/s²
```

**安全约束:**
```
初始集 S₀:  d ∈ [15, 16]/std1,  v ∈ [2.5, 3.0]   (远距离高速)
不安全集 X_u: d ∈ [5, 6]/std1,   v ∈ [0.5, 3.0]   (近距离高速 = 碰撞)
```

### 2.2 BarrierNet 架构

```
┌──────────────────────────────────────────────────────────────┐
│                    BarrierNet 架构                            │
│                                                              │
│  观测 z ──→ ┌─────────────────────┐                          │
│             │ 上游神经网络         │                          │
│             │ (FC / CNN+LSTM)     │                          │
│             │                     │                          │
│             │ 输出:               │                          │
│             │  • p₁(z), p₂(z)   ─┼──→ 惩罚函数 (Sigmoid×4)  │
│             │  • q(z)            ─┼──→ 参考控制              │
│             └─────────────────────┘                          │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────┐              │
│  │  BarrierNet 可微QP层                       │              │
│  │                                            │              │
│  │  min  ½uᵀQu + qᵀu                         │              │
│  │  s.t. Gu ≤ h   (HOCBF约束)                │              │
│  │       u_min ≤ u ≤ u_max                    │              │
│  │                                            │              │
│  │  G = -[Lg_Lf^{m-1}_b(x)]                  │              │
│  │  h = Lf^m_b + (p₁+p₂)ḃ + p₁·p₂·b       │              │
│  │                                            │              │
│  │  训练: QPFunction (可微, GPU批量)          │              │
│  │  推理: cvxopt (精确, CPU)                  │              │
│  └────────────────────────────────────────────┘              │
│                      │                                       │
│                      ▼                                       │
│               u* (安全控制输出)                               │
│                                                              │
│  安全保证: b(x(t)) ≥ 0, ∀t ≥ 0  (by construction)          │
└──────────────────────────────────────────────────────────────┘
```

**关键实现模式（以2D Robot为例）：**
```python
# 障碍函数
b(x) = (px - obs_x)² + (py - obs_y)² - R²

# HOCBF (相对度 r=2)
ḃ(x) = Lf_b = 2(px-obs_x)·v·cos(θ) + 2(py-obs_y)·v·sin(θ)
Lf²b = 2v²
LgLfb = [-2(px-obs_x)·v·sin(θ) + 2(py-obs_y)·v·cos(θ),
          2(px-obs_x)·cos(θ) + 2(py-obs_y)·sin(θ)]

# QP 约束
G = -[LgLfb]    # shape: (B, 1, 2)
h = Lf²b + (p₁+p₂)·ḃ + p₁·p₂·b    # shape: (B, 1)

# 求解
u* = QPFunction()(Q=I, q=reference, G, h)
```

---

## 3. 可行性分析

### 3.1 核心兼容性检查

#### ✅ 条件1：系统是否为控制仿射形式

SafePVC 的 CARLA AEBS 动力学：
```
d_{t+1} = d_t - v_t · dt              ← 不含 acc, 仿射 (g₁=0)
v_{t+1} = v_t - acc_t · dt            ← 关于 acc 仿射 (g₂=-dt)
```

写成标准形式 `x' = f(x) + g(x)·u`:
```
f(x) = [d - v·dt,  v]ᵀ       (漂移项)
g(x) = [0,  -dt]ᵀ             (控制矩阵, 常数)
```

**结论：✅ 完全满足控制仿射条件**，且 g(x) 为常数矩阵，这是最有利的情况。

#### ✅ 条件2：安全约束是否连续可微

不安全集为矩形区域 `d ∈ [5/std1, 6/std1] × v ∈ [0.5, 3.0]`。

需要构造一个连续可微的障碍函数 `b(x) ≥ 0` 来定义安全集。有几种选择：

**选项A：距离型障碍函数（推荐）**
```
b(d, v) = d_norm - d_safe_norm
```
其中 `d_safe_norm = 6.0/std1` 为安全距离阈值。这要求车辆始终保持在安全距离之外。

**选项B：椭圆型障碍函数**
```
b(d, v) = ((d_norm - d_center) / a)² + ((v - v_center) / b)² - 1
```
用椭圆包围不安全集，外部为安全区域。

**选项C：组合型障碍函数（最精确）**
```
b₁(d) = d_norm - d_min_norm      (距离下界)
b₂(v) = v_max - v                (速度上界)
```
多约束通过 QP 中多个不等式处理。

**结论：✅ 可以构造连续可微的障碍函数。**

#### ✅ 条件3：障碍函数的相对度

对于 `b(d,v) = d_norm - d_safe_norm`：
```
b(x) = d/std1 - d_safe/std1

ḃ = ∂b/∂x · ẋ
  = (1/std1) · ḋ
  = (1/std1) · (-v)          ← 不含 u (acc)

b̈ = (1/std1) · (-v̇)
  = (1/std1) · (-(v - acc·dt - v)/dt)  ← 包含 acc
  ≈ (1/std1) · acc            ← 包含 u
```

**相对度 r = 2**（需要两次求导才出现控制输入），与 BarrierNet 2D_Robot 示例完全一致。

#### ⚠️ 条件4：随机扰动的处理

这是**最大的理论冲突**：

- **SBC**：显式建模随机扰动 `Δs ~ μ`，提供概率性安全保证
- **BarrierNet**：确定性框架，无扰动建模

**解决方案（详细推导见第 5½ 节）：**

1. **方案A（推荐）：鲁棒HOCBF** — 将扰动视为有界不确定性，在 HOCBF 约束中减去最坏情况裕量 $\kappa(\delta_{\max}) = \delta_{\max}(p_1 + p_2 + p_1 p_2)$。确定性 100% 安全保证，但保守性较高。

2. **方案B：机会约束HOCBF** — 假设扰动服从高斯分布，将 HOCBF 约束转化为 chance constraint：$\mathbb{P}[\text{safe}] \geq 1-\epsilon$。保守性较低，但提供的是概率性保证。

3. **方案C：自适应数据驱动裕量** — 用网络学习状态依赖的裕量 $\kappa(z)$。最灵活但安全保证最弱。

> 📌 **完整的扰动传播分析、数学推导、代码实现和方案对比，见第 5½ 节「随机扰动的处理：三种方案详解」。**

#### ✅ 条件5：实时性要求

BarrierNet 的 QP 求解开销：
- 训练时：QPFunction (GPU批量, ~0.01s/batch)
- 推理时：cvxopt (CPU, ~0.01s/step)
- SafePVC 的 dt = 0.05s

**结论：✅ QP求解时间 << 控制周期，满足实时性。**

### 3.2 可行性总结

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 控制仿射形式 | ✅ 满足 | 线性系统，g(x) 为常数 |
| 安全约束可微 | ✅ 满足 | 可构造距离型/椭圆型障碍函数 |
| 相对度 | ✅ r=2 | 与 BarrierNet 2D 示例一致 |
| 随机扰动 | ⚠️ 需处理 | 需要鲁棒化或概率化扩展 |
| 实时性 | ✅ 满足 | QP ~10ms << 控制周期 50ms |
| 代码兼容性 | ✅ 可行 | 均为 PyTorch，可集成 |

**总体判断：技术可行，但需要对 BarrierNet 做鲁棒化扩展以处理扰动。**

---

## 4. 总体合成架构

### 4.1 新系统架构 (SafePVC-BarrierNet)

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    SafePVC-BarrierNet 合成架构                                │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  阶段一：感知与预训练 (保持不变)                                        │   │
│  │                                                                        │   │
│  │  ① cGAN 感知模型训练 → gen_net (冻结)                                  │   │
│  │  ② MLP 蒸馏 + Lipschitz 正则化 → state_net (冻结)                     │   │
│  │  ③ PPO 控制器预训练 → π₀ (预训练权重)                                  │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  阶段二：BarrierNet 安全层集成 (新)                                     │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────────────────────────────┐    │   │
│  │  │  BarrierNet-AEBS 模型                                         │    │   │
│  │  │                                                                │    │   │
│  │  │  ┌────────────────────┐                                        │    │   │
│  │  │  │ 上游网络           │                                        │    │   │
│  │  │  │ (从 PPO 策略初始化)│                                        │    │   │
│  │  │  │                    │                                        │    │   │
│  │  │  │ 输入: [d_est, v]   │                                        │    │   │
│  │  │  │ 输出:              │                                        │    │   │
│  │  │  │  • p₁, p₂ (惩罚)  │──→ Sigmoid × 4 → 正值                 │    │   │
│  │  │  │  • q (参考控制)    │──→ 期望加速度                          │    │   │
│  │  │  └────────────────────┘                                        │    │   │
│  │  │           │                                                    │    │   │
│  │  │           ▼                                                    │    │   │
│  │  │  ┌─────────────────────────────────────────┐                   │    │   │
│  │  │  │  HOCBF-QP 安全层                        │                   │    │   │
│  │  │  │                                         │                   │    │   │
│  │  │  │  b(d,v) = d_norm - d_safe_norm          │                   │    │   │
│  │  │  │  相对度 r = 2                            │                   │    │   │
│  │  │  │                                         │                   │    │   │
│  │  │  │  min  ½u² + q·u                         │                   │    │   │
│  │  │  │  s.t. Gu ≤ h  (HOCBF + 鲁棒裕量)       │                   │    │   │
│  │  │  │       -3 ≤ u ≤ 3                        │                   │    │   │
│  │  │  │                                         │                   │    │   │
│  │  │  │  G = -LgLfb = dt                        │                   │    │   │
│  │  │  │  h = Lf²b + (p₁+p₂)ḃ + p₁p₂b          │                   │    │   │
│  │  │  │      - κ(δ_max)    ← 鲁棒裕量          │                   │    │   │
│  │  │  │  κ = δ_max·(p₁+p₂+p₁p₂)               │                   │    │   │
│  │  │  └─────────────────────────────────────────┘                   │    │   │
│  │  │           │                                                    │    │   │
│  │  │           ▼                                                    │    │   │
│  │  │      u* = 安全加速度                                           │    │   │
│  │  └────────────────────────────────────────────────────────────────┘    │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│                                    ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  阶段三：训练与验证 (简化)                                             │   │
│  │                                                                        │   │
│  │  训练: MSE(u*, π₀(s)) 行为克隆 + 端到端梯度                           │   │
│  │  验证: 闭环仿真 b(x(t)) ≥ 0 ? (无需 IBP/CEGIS)                       │   │
│  │                                                                        │   │
│  │  输出: 安全-by-construction 的控制器                                   │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 被删除的模块

| 模块 | 文件 | 删除原因 |
|------|------|----------|
| SBC 神经网络 `MLP` | `utils.py:27-57` | 被 HOCBF-QP 替代 |
| 鞅损失函数 `martingale_loss` | `utils.py:15-21` | 鞅理论不再使用 |
| 三角噪声采样 `triangular` | `utils.py:5-13` | 不再需要蒙特卡罗噪声采样 |
| IBP 验证器 `VTVerifier` | `verify.py` (全部) | BarrierNet 安全-by-construction |
| CEGIS 交替训练循环 | `loop.py:83-171` | 被标准训练循环替代 |
| L-net 训练 `train_step_l` | `train.py:103-203` | SBC 不再存在 |
| P-net 交替训练 `train_step_p` | `train.py:206-299` | 被端到端训练替代 |
| auto_LiRPA 依赖 | `auto_LiRPA/` 目录 | 不再需要形式化验证 |
| 噪声网格积分 | `verify.py:214-349` | 不再需要扰动网格搜索 |
| 概率界计算 | `loop.py:129-159` | 不再计算概率安全界 |

### 4.3 被保留的模块

| 模块 | 文件 | 保留原因 |
|------|------|----------|
| cGAN 感知模型 | `cGAN/` + `Aebs/cGAN/` | 视觉感知不变 |
| 状态估计器 `state_net` | `Combined_network/model.py:43-56` | 图像→状态映射不变 |
| PPO 预训练 | `Aebs/controller/` | 提供 BarrierNet 的参考控制标签 |
| 系统动力学 | `Aebs/system/env.py:11-15` | HOCBF 需要动力学模型 |
| VCLS 端到端结构 | `Combined_network/model.py:70-89` | 整体流水线不变 |
| 数据收集 | `Aebs/connect/` | 数据集生成不变 |

### 4.4 新增模块

| 新模块 | 文件名（建议） | 功能 |
|--------|---------------|------|
| BarrierNet-AEBS 层 | `Aebs/BarrierNet/barrier_net_aebs.py` | HOCBF-QP 安全层 |
| 上游网络 | `Aebs/BarrierNet/upstream_net.py` | 双分支FC网络 (p头+q头) |
| 鲁棒裕量计算器 | `Aebs/BarrierNet/robust_margin.py` | 基于扰动估计的鲁棒裕量 |
| 行为克隆训练器 | `Aebs/BarrierNet/trainer.py` | 端到端训练循环 |
| 闭环验证器 | `Aebs/BarrierNet/simulator.py` | 仿真验证 b(x(t)) ≥ 0 |

---

## 5. 数学理论适配

### 5.1 原始 SBC 理论（被替换）

SafePVC 使用鞅理论下的随机障碍证书：

```
定理 2.2 (SBC 条件):
(i)   B(s) ≥ 0,                    ∀s ∈ S
(ii)  B(s) ≤ 1,                    ∀s ∈ S₀
(iii) B(s) ≥ 1/(1-p),              ∀s ∈ X_u
(iv)  B(s) ≥ E[B(s')] + ε,        ∀s ∈ S\X_u  (超鞅递减)

安全概率保证:
P_safe ≥ 1 - max_{s∈S₀} B(s) / min_{s∈X_u} B(s)
```

### 5.2 新 HOCBF 理论（替换后）

#### 系统建模

连续时间 AEBS 动力学：
```
ḋ = -v
v̇ = -acc

即 ẋ = f(x) + g(x)u, 其中:
  f(x) = [-v, 0]ᵀ
  g(x) = [0, -1]ᵀ
  u = acc
```

#### 安全约束

定义安全集 `C = {x : b(x) ≥ 0}`：

**主约束（碰撞避免）：**
```
b(d, v) = d - d_safe

其中 d_safe = 6.0 m (原不安全集的距离上界)
```

**相对度计算：**
```
b(x) = d - d_safe
L_f b(x) = ∂b/∂x · f(x) = [1, 0] · [-v, 0]ᵀ = -v
L_g b(x) = ∂b/∂x · g(x) = [1, 0] · [0, -1]ᵀ = 0   ← 相对度 ≠ 1

L_f² b(x) = ∂(L_f b)/∂x · f(x) = [0, -1] · [-v, 0]ᵀ = 0
L_g L_f b(x) = ∂(L_f b)/∂x · g(x) = [0, -1] · [0, -1]ᵀ = 1  ← 相对度 r = 2
```

#### HOCBF 约束（相对度 r=2）

```
ψ₀(x) = b(x) = d - d_safe
ψ₁(x) = ḃ(x) + α₁(b(x)) = -v + p₁·(d - d_safe)

HOCBF 条件:
sup_u [ L_f² b + (L_g L_f b)·u + (p₁+p₂)·ψ₀̇ + p₁·p₂·ψ₀ ] ≥ 0

展开:
sup_u [ 0 + 1·u + (p₁+p₂)·(-v) + p₁·p₂·(d - d_safe) ] ≥ 0

即:
u ≥ (p₁+p₂)·v - p₁·p₂·(d - d_safe)
```

#### QP 公式

```
u* = argmin_u  ½u² + q·u                    (q = 参考控制)

s.t.  u ≥ (p₁+p₂)·v - p₁·p₂·(d - d_safe)   (HOCBF约束)
      -3 ≤ u ≤ 3                              (控制约束)

等价于标准QP:
min  ½u·(1)·u + q·u
s.t. (-1)·u ≤ -(p₁+p₂)·v + p₁·p₂·(d - d_safe)    → Gu ≤ h
      u ≤ 3                                        → 上界
     -u ≤ 3                                        → 下界

其中:
  Q = [1]                    (标量)
  G = [-1]                   (1×1 矩阵)
  h = -(p₁+p₂)·v + p₁·p₂·(d - d_safe)   (标量)
```

#### 鲁棒化扩展（处理随机扰动）

> 📌 **此处为简要概述。完整的扰动传播分析、三种方案的详细推导和对比，见第 5½ 节。**

设扰动 Δs = [Δd, Δv] 满足 ||Δs||_∞ ≤ δ，障碍函数 Lipschitz 常数为 L_b：

```
鲁棒 HOCBF 约束 (方案 A 简版):
L_f² b + (L_g L_f b)·u + (p₁+p₂)·ḃ + p₁·p₂·b - κ(δ) ≥ 0

其中:
  κ(δ) = δ · (p₁ + p₂ + p₁·p₂)    ← 精确的扰动传播裕量 (见 5½.2)
  
  注意: 这里比简单用 L_b·δ 更精确，
  因为扰动不仅影响 b，还通过 ḃ = -v 影响约束的 (p₁+p₂) 项。

鲁棒化 h:
h_robust = h_nominal - δ · (p₁ + p₂ + p₁·p₂)
```

其他两种方案（机会约束 HOCBF、自适应数据驱动裕量）见 5½.3 和 5½.4 节。

#### 安全定理

```
定理 (BarrierNet-AEBS 安全保证):

若无扰动 (δ=0)，且 p₁(z), p₂(z) > 0 且 Lipschitz 连续，
则在控制律 u*(x) = QP 解 下，
闭环系统满足 b(x(t)) ≥ 0, ∀t ≥ 0，
即 d(t) ≥ d_safe, ∀t ≥ 0。

定理 (鲁棒化扩展, 方案 A):
若扰动 ||ε||_∞ ≤ δ_max，使用鲁棒裕量 κ = δ_max·(p₁+p₂+p₁p₂)，
则闭环系统仍满足 b(x(t)) ≥ 0, ∀t ≥ 0。
(证明概要见 5½.2 节)

定理 (机会约束扩展, 方案 B):
若扰动 ε ~ N(μ, Σ)，使用分位数裕量 Φ⁻¹(1-ε)·σ_W，
则单步安全概率 P[b(x_{t+1}) ≥ 0 | x_t] ≥ 1-ε。
(推导见 5½.3 节)
```

### 5.3 与原论文理论的对比

| 理论维度 | 原 SBC | 新 HOCBF-QP |
|----------|--------|-------------|
| 核心定理 | 超鞅递减 + 可选停止定理 | HOCBF 前向不变性 |
| 安全类型 | 概率保证 P_safe ≥ p | 确定性保证 b(x) ≥ 0 |
| 时间范围 | 无限时域（鞅的可选停止） | 无限时域（前向不变性） |
| 扰动处理 | 显式建模 Δs ~ μ | 三种方案: 鲁棒裕量 κ / 机会约束 / 自适应 (见 5½ 节) |
| 证书构造 | 学习神经网络 B(s) | 安全-by-construction (QP) |
| 验证方式 | IBP + 网格搜索 | 闭环仿真 + 解析证明 |

---

## 5½. 随机扰动的处理：三种方案详解

> **这是整个合成方案中最关键的理论难点。** SafePVC 的核心场景是有随机环境扰动的视觉控制系统，而 BarrierNet 原生是确定性框架。本节详细分析扰动如何传播、如何将其融入 HOCBF-QP 框架，并给出三种从保守到激进的解决方案。

### 5½.1 扰动传播机制分析

#### SafePVC 中的扰动模型（原始设定）

SafePVC 的系统存在不可观测的环境扰动 $z_t$，其传播链条为：

```
环境扰动 z_t ∈ Z (不可观测)
     │
     ▼
观测模型: o_t = g(s_t, z_t)      ← cGAN/MLP 近似，z 影响生成的图像
     │
     ▼
策略网络: u_t = π(o_t)            ← 视觉控制器基于（受扰的）图像做决策
     │
     ▼
系统动力学: s_{t+1} = f(s_t, u_t) ← 受扰的控制输入导致偏离标称轨迹
```

SafePVC 将这个链条整体建模为一个**等价状态扰动**：

$$
s_{t+1} = F(s_t, z_0) + \Delta s, \quad \Delta s \sim \mu
$$

其中 $F(s, z_0)$ 是参考环境条件下的标称闭环动力学，$\Delta s = F(s, z) - F(s, z_0)$ 是由环境变化引起的状态偏差，被建模为**与当前状态独立的随机变量**（Assumption 1），分布 $\mu$ 通过数据驱动方式估计。

#### BarrierNet 需要面对的问题

在合成系统中，扰动的传播路径变为：

```
                    SafePVC 前端 (保留)                    BarrierNet 后端 (新)
                    ─────────────────────                  ────────────────────

真实状态 s_t ───→ cGAN g(s_t, z_t) ──→ 图像 o_t ──→ state_net ──→ ŝ_t = s_t + ε_state
                                                              │
                                                              ▼
                                              u_t = BarrierNet(ŝ_t)
                                                              │
                                                              ▼
                                              s_{t+1} = f(s_t, u_t) + δ_dynamics
```

扰动通过**两条路径**影响系统安全：

**路径 ①：感知误差（Perception Error）**
$$
\varepsilon_{\text{state}} = \hat{s}_t - s_t
$$
由于 $z_t$ 影响生成的图像，state_net 的状态估计 $\hat{s}_t$ 会偏离真实状态 $s_t$。BarrierNet 基于 $\hat{s}_t$ 而非 $s_t$ 做决策，导致 HOCBF 约束被错误地评估。

**路径 ②：动力学扰动（Dynamics Disturbance）**
$$
\delta_{\text{dyn}} \in \mathbb{R}^2
$$
即使控制输入 $u_t$ 是正确的，执行后系统也可能因为外部因素（风、路面摩擦变化等）偏离预期轨迹。SafePVC 的 $\Delta s$ 实际上包含了这条路径和路径①的综合效果。

#### 量化扰动对 HOCBF 约束的影响

设真实状态为 $x = [d, v]^\top$，BarrierNet 使用的估计状态为 $\hat{x} = x + \varepsilon$，其中 $\varepsilon = [\varepsilon_d, \varepsilon_v]^\top$。

对于 AEBS 系统的障碍函数 $b(x) = d - d_{\text{safe}}$：

$$
b(\hat{x}) = \hat{d} - d_{\text{safe}} = (d + \varepsilon_d) - d_{\text{safe}} = b(x) + \varepsilon_d
$$

对 HOCBF 约束各分项的影响：

$$
\begin{aligned}
b(\hat{x}) &= b(x) + \varepsilon_d & \text{误差: } \Delta_b &= \varepsilon_d \\
\dot{b}(\hat{x}) &= -\hat{v} = -(v + \varepsilon_v) & \text{误差: } \Delta_{\dot{b}} &= -\varepsilon_v \\
L_f^2 b(\hat{x}) &= 0 & \text{误差: } \Delta_{L_f^2 b} &= 0 \\
L_g L_f b(\hat{x}) &= 1 & \text{误差: } \Delta_{L_g L_f b} &= 0
\end{aligned}
$$

因此，BarrierNet 基于估计状态 $\hat{x}$ 构建的 HOCBF 约束为：

$$
\underbrace{0}_{L_f^2 b} + \underbrace{1}_{L_g L_f b} \cdot u + (p_1 + p_2) \cdot \underbrace{(-v - \varepsilon_v)}_{\dot{b}(\hat{x})} + p_1 p_2 \cdot \underbrace{(d + \varepsilon_d - d_{\text{safe}})}_{b(\hat{x})} \geq 0
$$

而**我们真正需要满足的**是基于真实状态 $x$ 的约束。两者之差为：

$$
\Delta_{\text{HOCBF}} = (p_1 + p_2)(-\varepsilon_v) + p_1 p_2 \cdot \varepsilon_d
$$

这意味着：**当 $\varepsilon_d > 0$（高估距离）且 $\varepsilon_v < 0$（低估速度）时，HOCBF 约束被虚假放松，系统可能在不安全的情况下"以为"自己安全。** 这正是扰动带来的危险。

---

### 5½.2 方案 A：鲁棒 HOCBF（Worst-case Bound）— 推荐

#### 核心思想

将扰动视为**有界不确定性**，在 HOCBF 约束中减去**最坏情况下的扰动影响**，确保即使扰动取到极端值，安全约束仍然满足。

#### 扰动假设

$$
\|\varepsilon\|_\infty = \max(|\varepsilon_d|, |\varepsilon_v|) \leq \delta_{\max}
$$

其中 $\delta_{\max}$ 可以从 SafePVC 的数据驱动扰动估计中获得（见 5½.5 节）。

#### 鲁棒化推导

基于真实状态的 HOCBF 约束为：

$$
L_f^2 b(x) + [L_g L_f b(x)] u + (p_1 + p_2) \dot{b}(x) + p_1 p_2 b(x) \geq 0
$$

但 BarrierNet 只能基于估计状态 $\hat{x}$ 构建约束。我们需要确保：**即使 $\hat{x}$ 和 $x$ 之间存在误差，基于 $\hat{x}$ 构建的约束仍能保护真实状态 $x$ 的安全。**

基于 $\hat{x}$ 构建的约束值 $h(\hat{x})$ 与基于真实 $x$ 的约束值 $h(x)$ 之间的关系：

$$
h(\hat{x}) = h(x) + \Delta_{\text{HOCBF}}
$$

其中：

$$
\Delta_{\text{HOCBF}} = (p_1 + p_2)(-\varepsilon_v) + p_1 p_2 \cdot \varepsilon_d
$$

在最坏情况下（$\varepsilon_v = +\delta_{\max}$, $\varepsilon_d = -\delta_{\max}$，即**高估速度、低估距离**——对安全最不利）：

$$
\Delta_{\text{HOCBF}}^{\text{worst}} = -(p_1 + p_2) \delta_{\max} - p_1 p_2 \cdot \delta_{\max} = -\delta_{\max}(p_1 + p_2 + p_1 p_2)
$$

因此，鲁棒化的 HOCBF 约束为：

$$
\boxed{
L_f^2 b(\hat{x}) + [L_g L_f b(\hat{x})] u + (p_1 + p_2) \dot{b}(\hat{x}) + p_1 p_2 b(\hat{x}) - \delta_{\max}(p_1 + p_2 + p_1 p_2) \geq 0
}
$$

#### 对应的 QP 修改

在 AEBS 系统中展开：

$$
u \geq (p_1 + p_2) \hat{v} - p_1 p_2 (\hat{d} - d_{\text{safe}}) + \delta_{\max}(p_1 + p_2 + p_1 p_2)
$$

对应标准 QP 的 $h$ 向量变为：

$$
h_{\text{robust}} = \underbrace{-(p_1 + p_2) \hat{v} + p_1 p_2 (\hat{d} - d_{\text{safe}})}_{h_{\text{nominal}}} - \underbrace{\delta_{\max}(p_1 + p_2 + p_1 p_2)}_{\text{robust margin } \kappa(\delta_{\max})}
$$

注意 $G = [-1]$ 不变，鲁棒化只影响 $h$。

#### 安全性定理

> **定理（鲁棒 BarrierNet-AEBS 安全保证）**
> 
> 设 $p_1(z), p_2(z) > 0$ 且 Lipschitz 连续。若扰动满足 $\|\varepsilon\|_\infty \leq \delta_{\max}$，则在鲁棒 HOCBF-QP 控制律下，闭环系统满足：
> 
> $$b(x(t)) \geq 0, \quad \forall t \geq 0$$
> 
> 即 $d(t) \geq d_{\text{safe}}, \forall t \geq 0$，**即使存在有界扰动**。

**证明概要：** 在每个时刻 $t$，鲁棒化约束确保：

$$
h(\hat{x}_t) - \kappa(\delta_{\max}) \geq 0 \implies h(x_t) \geq h(\hat{x}_t) - |\Delta_{\text{HOCBF}}| \geq h(\hat{x}_t) - \kappa(\delta_{\max}) \geq 0
$$

由 HOCBF 定理（BarrierNet 论文 Theorem 8），$h(x_t) \geq 0$ 递推保证 $\psi_0(x_t) = b(x_t) \geq 0$。$\blacksquare$

#### 保守性分析

鲁棒裕量 $\kappa(\delta_{\max}) = \delta_{\max}(p_1 + p_2 + p_1 p_2)$。

当 $p_1, p_2 \in (0, 4)$（由 Sigmoid × 4 保证）时：
- 最小裕量：$p_1 \to 0, p_2 \to 0$ 时 $\kappa \to 0$
- 最大裕量：$p_1 = p_2 = 4$ 时 $\kappa = \delta_{\max}(4 + 4 + 16) = 24 \delta_{\max}$

**问题**：当 $p_1, p_2$ 较大时，鲁棒裕量会很大，可能导致 QP 约束过于保守（要求很大的加速度）甚至无可行解。

**缓解方法**：训练过程中，$\mathcal{L}_{\text{smooth}} = \lambda_{\text{smooth}} \cdot \text{mean}(p_1 + p_2)$ 会自然惩罚过大的 $p$ 值，使网络学习到在安全裕量和保守性之间的平衡。

#### 代码实现要点

```python
# 在 _hocbf_qp 方法中，h_hocbf 的计算修改为：

# 鲁棒裕量
p1, p2 = p[:, 0], p[:, 1]
kappa = self.delta_max * (p1 + p2 + p1 * p2)  # [B]

# 鲁棒化 h
h_hocbf = (Lf2b 
           + (p1 + p2) * b_dot 
           + p1 * p2 * b 
           - kappa).view(nBatch, 1)  # 减去鲁棒裕量
```

---

### 5½.3 方案 B：机会约束 HOCBF（Chance-Constrained）

#### 核心思想

不要求**所有**扰动下都安全（太保守），而是要求**以高概率**安全：

$$
\mathbb{P}_{\varepsilon}\big[\text{HOCBF 约束满足}\big] \geq 1 - \epsilon
$$

其中 $\epsilon \in (0, 1)$ 是允许的风险水平（例如 $\epsilon = 0.05$）。

#### 扰动分布假设

假设通过 SafePVC 的数据驱动方法（见 5½.5 节），我们已经估计出：

$$
\varepsilon_d \sim \mathcal{N}(\mu_d, \sigma_d^2), \quad \varepsilon_v \sim \mathcal{N}(\mu_v, \sigma_v^2)
$$

两者独立（或已知协方差矩阵 $\Sigma$）。

#### 推导

基于估计状态的 HOCBF 约束为：

$$
C(\hat{x}, u) := L_f^2 b(\hat{x}) + [L_g L_f b(\hat{x})] u + (p_1 + p_2) \dot{b}(\hat{x}) + p_1 p_2 b(\hat{x}) \geq 0
$$

对于 AEBS 系统，将真实状态 $x = \hat{x} - \varepsilon$ 代入：

$$
C(x, u) = u - (p_1 + p_2) v + p_1 p_2 (d - d_{\text{safe}})
$$

$$
C(\hat{x} - \varepsilon, u) = u - (p_1 + p_2)(\hat{v} - \varepsilon_v) + p_1 p_2 (\hat{d} - \varepsilon_d - d_{\text{safe}})
$$

$$
= \underbrace{C(\hat{x}, u)}_{\text{nominal}} + \underbrace{(p_1 + p_2) \varepsilon_v - p_1 p_2 \varepsilon_d}_{\text{stochastic term } W}
$$

随机项 $W$ 的分布：

$$
W = (p_1 + p_2) \varepsilon_v - p_1 p_2 \varepsilon_d
$$

$$
W \sim \mathcal{N}\Big((p_1 + p_2) \mu_v - p_1 p_2 \mu_d, \quad (p_1 + p_2)^2 \sigma_v^2 + (p_1 p_2)^2 \sigma_d^2\Big)
$$

记 $W \sim \mathcal{N}(\mu_W, \sigma_W^2)$，其中：

$$
\mu_W = (p_1 + p_2) \mu_v - p_1 p_2 \mu_d
$$

$$
\sigma_W = \sqrt{(p_1 + p_2)^2 \sigma_v^2 + (p_1 p_2)^2 \sigma_d^2}
$$

机会约束 $\mathbb{P}[C(\hat{x}, u) + W \geq 0] \geq 1 - \epsilon$ 等价于：

$$
C(\hat{x}, u) + \mu_W \geq \Phi^{-1}(1 - \epsilon) \cdot \sigma_W
$$

其中 $\Phi^{-1}$ 是标准正态分布的逆 CDF（分位函数）。

整理得到**确定性等价的 HOCBF 约束**：

$$
\boxed{
C(\hat{x}, u) \geq -\mu_W + \Phi^{-1}(1 - \epsilon) \cdot \sigma_W
}
$$

#### 具体展开

$$
u \geq (p_1 + p_2) \hat{v} - p_1 p_2 (\hat{d} - d_{\text{safe}}) - \mu_W + \Phi^{-1}(1 - \epsilon) \cdot \sigma_W
$$

对应 QP 的 $h$ 向量：

$$
h_{\text{chance}} = h_{\text{nominal}} - \mu_W + \Phi^{-1}(1 - \epsilon) \cdot \sigma_W
$$

**注意**：$\mu_W$ 和 $\sigma_W$ 都依赖于 $p_1, p_2$，而 $p_1, p_2$ 是网络的输出，所以 $h$ 仍然是 $p$ 的非线性函数。这在端到端训练中是可微的（$\Phi^{-1}$ 是常数，$\sigma_W$ 是关于 $p$ 的平滑函数）。

#### 数值例子

设 $\mu_d = \mu_v = 0$（零均值扰动），$\sigma_d = 0.1$ m，$\sigma_v = 0.05$ m/s，$\epsilon = 0.05$：

- $\Phi^{-1}(0.95) = 1.645$
- $\mu_W = 0$
- $\sigma_W = \sqrt{(p_1 + p_2)^2 \times 0.0025 + (p_1 p_2)^2 \times 0.01}$

当 $p_1 = p_2 = 2$ 时：
- $\sigma_W = \sqrt{16 \times 0.0025 + 16 \times 0.01} = \sqrt{0.04 + 0.16} = \sqrt{0.2} \approx 0.447$
- 裕量 $= 1.645 \times 0.447 \approx 0.735$

对比方案 A（$\delta_{\max} = 3\sigma = 0.3$）：
- $\kappa = 0.3 \times (2 + 2 + 4) = 2.4$

**Chance-constrained 的裕量 (0.735) 远小于 worst-case 的裕量 (2.4)**，保守性大幅降低，代价是允许 5% 的概率违反约束。

#### 安全性讨论

> **定理（机会约束 BarrierNet 安全保证）**
> 
> 在扰动 $\varepsilon \sim \mathcal{N}(\mu, \Sigma)$ 的假设下，Chance-Constrained HOCBF-QP 控制律保证：
> 
> $$\mathbb{P}[b(x(t)) \geq 0] \geq (1 - \epsilon)^t, \quad \forall t \geq 0$$
> 
> 即单步安全概率 $\geq 1 - \epsilon$，多步安全概率随时间指数衰减。
> 
> **注意**：这比 SafePVC 的 SBC 弱（SBC 保证无限时域的概率下界），但比纯确定性方法在有扰动时更实际。

#### 代码实现要点

```python
from scipy.stats import norm

class ChanceConstrainedBarrierNet(BarrierNetAEBS):
    def __init__(self, ..., epsilon=0.05, mu_d=0.0, mu_v=0.0, sigma_d=0.1, sigma_v=0.05):
        super().__init__(...)
        self.epsilon = epsilon
        self.mu_d, self.mu_v = mu_d, mu_v
        self.sigma_d, self.sigma_v = sigma_d, sigma_v
        self.z_score = norm.ppf(1 - epsilon)  # Φ⁻¹(1-ε), 常数
    
    def _hocbf_qp(self, state, q, p, nBatch, sgn):
        # ... (前面和原来一样) ...
        
        p1, p2 = p[:, 0], p[:, 1]
        
        # 计算 μ_W 和 σ_W
        mu_W = (p1 + p2) * self.mu_v - p1 * p2 * self.mu_d  # [B]
        sigma_W = torch.sqrt(
            (p1 + p2)**2 * self.sigma_v**2 + 
            (p1 * p2)**2 * self.sigma_d**2
        )  # [B]
        
        # Chance-constrained 裕量
        cc_margin = -mu_W + self.z_score * sigma_W  # [B]
        
        h_hocbf = (Lf2b + (p1 + p2) * b_dot + p1 * p2 * b - cc_margin).view(nBatch, 1)
        
        # ... (后面和原来一样) ...
```

---

### 5½.4 方案 C：自适应数据驱动裕量（Adaptive Learned Margin）

#### 核心思想

方案 A 和 B 都需要手动设定扰动上界 $\delta_{\max}$ 或分布参数 $(\mu, \Sigma)$。方案 C 更进一步：**用一个小型神经网络从数据中自适应地学习鲁棒裕量**，让网络自己决定在每个状态下需要多大的安全裕量。

#### 架构设计

在上游网络中新增一个**裕量输出头**（Margin Head）：

```
上游网络
├── 主干 FC(d_est, v)
│   ├── q 头 → 参考控制 q(z)
│   ├── p 头 → 惩罚参数 p₁(z), p₂(z)
│   └── κ 头 → 鲁棒裕量 κ(z) ← 新增！
```

HOCBF 约束变为：

$$
L_f^2 b + [L_g L_f b] u + (p_1 + p_2) \dot{b} + p_1 p_2 b - \kappa(z) \geq 0
$$

其中 $\kappa(z) \geq 0$ 是网络学习到的状态依赖裕量。

#### 训练策略：两阶段法

**阶段一：扰动数据收集（复用 SafePVC 的方法）**

利用 SafePVC 的 `Distribution_Estimation` 模块（Algorithm 1 Line 7），收集扰动数据：

```python
# 对每个状态 s_i，在参考环境 z_0 和扰动环境 z 下分别仿真
for s_i in state_grid:
    s_nominal = step(s_i, pi(s_i, z_0), z_0)  # 标称下一状态
    for z_j in perturbation_samples:
        s_perturbed = step(s_i, pi(s_i, z_j), z_j)  # 扰动下一状态
        delta_s = s_perturbed - s_nominal
        disturbance_dataset.append((s_i, delta_s))
```

**阶段二：裕量网络训练**

$\kappa$ 头的训练目标：学习一个映射 $\kappa: \hat{x} \to \mathbb{R}_{\geq 0}$，使得：

$$
\mathcal{L}_\kappa = \mathbb{E}_{(\hat{x}, \Delta s) \sim \mathcal{D}}\Big[\max\big(0, \Delta_{\text{HOCBF}}(\hat{x}, \Delta s) - \kappa(\hat{x})\big)^2\Big] + \lambda_\kappa \cdot \mathbb{E}[\kappa(\hat{x})]
$$

第一项确保 $\kappa(\hat{x})$ 足够大以覆盖实际扰动影响，第二项惩罚过大的裕量（避免保守）。

#### 安全保证的弱化

方案 C 不提供严格的确定性保证（因为 $\kappa$ 是从有限数据学习的），但提供了**经验性的安全保证**：

> **经验安全声明**
> 
> 若 $\kappa$ 网络在测试集上覆盖了 $\alpha\%$ 的扰动样本（即 $\kappa(\hat{x}) \geq |\Delta_{\text{HOCBF}}|$ 对 $\alpha\%$ 的样本成立），则闭环系统的安全率**经验性地** $\geq \alpha\%$。

这类似于 SafePVC 的概率安全界，但通过仿真而非鞅理论来获得。

#### 代码实现要点

```python
class AdaptiveBarrierNet(BarrierNetAEBS):
    def __init__(self, ...):
        super().__init__(...)
        # 新增 κ 头
        self.fc23 = nn.Linear(self.hidden1, 32)
        self.bn23 = nn.BatchNorm1d(32)
        self.fc33 = nn.Linear(32, 1)
    
    def forward(self, state, sgn=1):
        # ... 主干 ...
        x = torch.relu(self.fc1(state))
        x = self.bn1(x)
        
        # p 头
        x22 = torch.relu(self.fc22(x))
        x22 = self.bn22(x22)
        p = 4.0 * torch.sigmoid(self.fc32(x22))
        
        # κ 头 (自适应裕量, Softplus 确保非负)
        x23 = torch.relu(self.fc23(x))
        x23 = self.bn23(x23)
        kappa = F.softplus(self.fc33(x23))  # [B, 1], κ ≥ 0
        
        # ... q 头 + QP (使用 kappa 替代固定 robust_margin) ...
```

---

### 5½.5 与 SafePVC 扰动估计模块的对接

SafePVC 已经有一套完整的数据驱动扰动估计流程（Remark 1 + Algorithm 1 Line 7），我们可以**直接复用**其输出来为方案 A/B/C 提供参数。

#### SafePVC 的 Distribution_Estimation 输出

```python
def Distribution_Estimation(VCLS, f, z_0, z_samples, S):
    """
    SafePVC 原有的扰动估计模块。
    
    输出: Δs 的经验分布，用于 SBC 的 Monte Carlo 采样验证。
    我们将其复用为 BarrierNet 鲁棒裕量的输入。
    """
    deltas = []
    for s in S:  # 离散状态网格
        s_nom = f(s, VCLS.pi(s, z_0))       # 标称转移
        for z in z_samples:                   # 扰动采样
            s_pert = f(s, VCLS.pi(s, z))      # 扰动转移
            deltas.append(s_pert - s_nom)
    
    deltas = np.array(deltas)  # shape: (N_states × N_samples, 2)
    return deltas
```

#### 从扰动数据提取各方案参数

```python
def extract_disturbance_params(deltas, method='robust'):
    """
    从 SafePVC 的扰动数据中提取 BarrierNet 需要的参数。
    
    deltas: shape (N, 2), 列 = [Δd, Δv]
    """
    delta_d = deltas[:, 0]
    delta_v = deltas[:, 1]
    
    if method == 'robust':
        # 方案 A: 最坏情况 bound
        delta_max = max(np.abs(delta_d).max(), np.abs(delta_v).max())
        return {'delta_max': delta_max}
    
    elif method == 'chance':
        # 方案 B: 高斯拟合
        mu_d, sigma_d = np.mean(delta_d), np.std(delta_d)
        mu_v, sigma_v = np.mean(delta_v), np.std(delta_v)
        return {'mu_d': mu_d, 'sigma_d': sigma_d, 
                'mu_v': mu_v, 'sigma_v': sigma_v}
    
    elif method == 'adaptive':
        # 方案 C: 返回完整数据集用于训练 κ 网络
        return {'deltas': deltas}
```

#### 典型数值（基于 SafePVC 论文的 CARLA AEBS 实验）

SafePVC 论文中使用噪声因子为状态空间跨度的 1%~10%。以 5% 为例：

$$
\begin{aligned}
d_{\text{range}} &= 16 - 5 = 11 \text{ m} \implies \delta_d^{\max} = 0.05 \times 11 = 0.55 \text{ m} \\
v_{\text{range}} &= 3 - 0 = 3 \text{ m/s} \implies \delta_v^{\max} = 0.05 \times 3 = 0.15 \text{ m/s}
\end{aligned}
$$

| 参数 | 方案A (Robust) | 方案B (Chance, ε=0.05) |
|------|--------------|---------------------|
| 扰动上界/标准差 | $\delta_{\max} = 0.55$ | $\sigma_d \approx 0.18, \sigma_v \approx 0.05$ |
| 当 $p_1 = p_2 = 2$ 时的裕量 | $0.55 \times 8 = 4.4$ | $1.645 \times \sqrt{16 \times 0.0025 + 16 \times 0.032} \approx 1.22$ |
| 保守程度 | 高 | 中 |
| 安全保证 | 确定性 (100%) | 概率性 (95%/步) |

---

### 5½.6 三种方案对比与选择建议

| 维度 | 方案 A: 鲁棒 HOCBF | 方案 B: 机会约束 | 方案 C: 自适应裕量 |
|------|-------------------|-----------------|-------------------|
| **扰动假设** | 有界 $\|\varepsilon\| \leq \delta_{\max}$ | 已知分布 $\varepsilon \sim \mathcal{N}(\mu, \Sigma)$ | 从数据学习，无需显式假设 |
| **安全保证** | 确定性 (100%) | 概率性 ($1 - \epsilon$ /步) | 经验性 (依赖训练数据覆盖) |
| **保守性** | 🔴 高（最坏情况） | 🟡 中（分位数） | 🟢 低（数据驱动自适应） |
| **QP 可行性** | 🔴 裕量大时易无解 | 🟡 偶尔无解 | 🟢 网络可学习避免 |
| **实现复杂度** | 🟢 低（改一行 $h$） | 🟡 中（加入 $\sigma_W$ 计算） | 🟡 中（新增网络头 + 预训练） |
| **可微性** | ✅ 完全可微 | ✅ 完全可微 | ✅ 完全可微 |
| **理论严谨性** | 🟢 强（解析证明） | 🟡 中（依赖分布假设） | 🔴 弱（仅经验保证） |
| **与 SafePVC 对比** | 比 SBC 更保守但更强 | 最接近 SBC 的概率保证 | 最灵活但最不安全 |

#### 推荐策略

**首选方案 A（鲁棒 HOCBF）**，理由：
1. **实现最简单**：只需在 $h$ 中减去一个常数项，改动最小
2. **理论最严谨**：有解析的安全证明，论文贡献更扎实
3. **与 BarrierNet 原论文一致**：BarrierNet 本身就是确定性框架，鲁棒扩展是最自然的
4. **安全保证最强**：在论文中可以和 SafePVC 的概率保证形成鲜明对比

**在实验中同时测试方案 B**，理由：
1. 方案 B 的保守性更低，控制性能更好
2. 可以与方案 A 做 ablation 对比，展示 trade-off
3. 概率保证与 SafePVC 的 SBC 保证可以定量比较

**方案 C 作为 Future Work**，理由：
1. 理论上最有趣，但安全保证最弱
2. 适合作为"可扩展方向"在论文 Discussion 中提及
3. 实验验证需要更多工程工作

---

## 6. 代码修改详细方案

### 6.1 新增核心文件：`Aebs/BarrierNet/barrier_net_aebs.py`

```python
import torch
import torch.nn as nn
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import numpy as np


class BarrierNetAEBS(nn.Module):
    """
    BarrierNet for CARLA AEBS (Emergency Braking)
    
    系统动力学:
        d_{t+1} = d_t - v_t * dt
        v_{t+1} = v_t - acc_t * dt
        即 ẋ = f(x) + g(x)u, f=[-v, 0], g=[0, -1]
    
    安全约束:
        b(d,v) = d - d_safe ≥ 0  (相对度 r=2)
    
    HOCBF QP:
        min  ½u² + q·u
        s.t. u ≥ (p₁+p₂)·v - p₁·p₂·(d - d_safe) + κ(δ_max)
             u_min ≤ u ≤ u_max
             
        鲁棒裕量: κ(δ_max) = δ_max · (p₁ + p₂ + p₁·p₂)
        详见 5½.2 节推导
    """
    
    def __init__(
        self,
        state_dim=2,          # [d_norm, v]
        hidden1=64,           # 主干隐藏层
        hidden21=32,          # q头隐藏层
        hidden22=32,          # p头隐藏层
        d_safe=6.0,           # 安全距离 (m)
        std1=1.0,             # 距离归一化系数
        u_min=-3.0,           # 最小加速度
        u_max=3.0,            # 最大加速度
        dt=0.05,              # 时间步长
        noise_factor=0.05,    # 扰动因子
        device='cuda'
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_safe = d_safe
        self.std1 = std1
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.noise_factor = noise_factor
        self.device = device
        
        # 计算扰动上界 δ_max (见 5½.2 节)
        # δ_max = noise_factor × max(state_range)
        d_range = 16.0 - 5.0  # 距离范围 (m)
        v_range = 3.0          # 速度范围 (m/s)
        self.delta_max = noise_factor * max(d_range, v_range)
        # 注意: 鲁棒裕量 κ = δ_max · (p₁ + p₂ + p₁·p₂) 是动态的,
        # 依赖于网络输出的 p₁, p₂, 在 _hocbf_qp 中实时计算
        
        # ========== 上游网络 ==========
        # 主干
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        # q头 (参考控制)
        self.fc21 = nn.Linear(hidden1, hidden21)
        self.bn21 = nn.BatchNorm1d(hidden21)
        self.fc31 = nn.Linear(hidden21, 1)
        
        # p头 (HOCBF 惩罚函数)
        self.fc22 = nn.Linear(hidden1, hidden22)
        self.bn22 = nn.BatchNorm1d(hidden22)
        self.fc32 = nn.Linear(hidden22, 2)  # p₁, p₂
        
    def forward(self, state, sgn=1):
        """
        state: [B, 2] = [d_norm, v]
        sgn: 1=QPFunction(训练), 0=cvxopt(推理)
        """
        nBatch = state.size(0)
        
        # ===== 上游网络前向传播 =====
        x = state
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        
        # q头: 参考控制
        x21 = torch.relu(self.fc21(x))
        x21 = self.bn21(x21)
        q = self.fc31(x21)  # [B, 1] 参考加速度
        
        # p头: HOCBF 惩罚参数 (必须为正)
        x22 = torch.relu(self.fc22(x))
        x22 = self.bn22(x22)
        p = self.fc32(x22)  # [B, 2]
        p = 4.0 * torch.sigmoid(p)  # 确保 p₁, p₂ > 0
        
        # ===== HOCBF-QP 层 =====
        u_safe = self._hocbf_qp(state, q, p, nBatch, sgn)
        
        return u_safe
    
    def _hocbf_qp(self, state, q, p, nBatch, sgn):
        """
        构建并求解 HOCBF-QP
        
        障碍函数: b(x) = d - d_safe  (d 需反归一化)
        Lf_b = -v
        Lf²b = 0
        LgLf_b = 1  (因为 g_v = -1, ∂Lf_b/∂v = -1, LgLf = (-1)(-1) = 1)
        
        注意：这里用的是连续时间近似:
        ẋ = f(x) + g(x)u
        f = [-v, 0]ᵀ, g = [0, -1]ᵀ
        """
        d_norm = state[:, 0]  # 归一化距离
        v = state[:, 1]       # 速度
        
        # 反归一化
        d = d_norm * self.std1  # 实际距离 (m)
        
        # 障碍函数值
        b = d - self.d_safe     # [B]
        
        # 一阶 Lie 导数
        b_dot = -v              # Lf_b = -v  [B]
        
        # 二阶 Lie 导数
        Lf2b = torch.zeros(nBatch, device=self.device)  # Lf²b = 0
        LgLfb = torch.ones(nBatch, device=self.device)   # LgLfb = 1
        
        # ===== 构建 QP =====
        # QP cost: ½u² + q·u  (Q=1, q=reference)
        Q = Variable(torch.ones(nBatch, 1, 1, device=self.device))
        
        # 注意 qpth 的 QP 形式: min ½xᵀQx + pᵀx
        # 所以 p_vector = q (参考控制的负方向，因为 QP 最小化)
        p_vector = q.view(nBatch, 1)
        
        # HOCBF 约束: Gu ≤ h
        # 约束: u ≥ (p₁+p₂)·v - p₁·p₂·b + κ(δ_max)
        # 等价: -u ≤ -(p₁+p₂)·v + p₁·p₂·b - κ(δ_max)
        # 即 G = [-1], h = -(p₁+p₂)·v + p₁·p₂·b - κ
        #
        # 鲁棒裕量 κ = δ_max · (p₁ + p₂ + p₁·p₂)
        # 推导见 5½.2 节: 扰动对 HOCBF 约束的最坏影响为
        # Δ_HOCBF = (p₁+p₂)(-ε_v) + p₁p₂·ε_d
        # 最坏情况 |ε_d|,|ε_v| ≤ δ_max → |Δ| ≤ δ_max·(p₁+p₂+p₁p₂)
        
        p1, p2 = p[:, 0], p[:, 1]
        kappa = self.delta_max * (p1 + p2 + p1 * p2)  # [B], 动态鲁棒裕量
        
        G_hocbf = -LgLfb.view(nBatch, 1)  # [B, 1] = [-1]
        h_hocbf = (Lf2b 
                   + (p1 + p2) * b_dot 
                   + p1 * p2 * b 
                   - kappa).view(nBatch, 1)
        
        # 控制约束: u_min ≤ u ≤ u_max
        # u ≤ u_max  →  [1]·u ≤ u_max
        # u ≥ u_min  →  [-1]·u ≤ -u_min
        G_ub = torch.ones(nBatch, 1, device=self.device)    # u ≤ u_max
        h_ub = torch.full((nBatch, 1), self.u_max, device=self.device)
        G_lb = -torch.ones(nBatch, 1, device=self.device)   # -u ≤ -u_min → u ≥ u_min
        h_lb = torch.full((nBatch, 1), -self.u_min, device=self.device)
        
        # 合并所有约束
        G = torch.cat([G_hocbf, G_ub, G_lb], dim=1)   # [B, 3, 1]
        h = torch.cat([h_hocbf, h_ub, h_lb], dim=1)   # [B, 3]
        
        # 没有等式约束
        e = Variable(torch.Tensor().to(self.device))
        
        # 求解 QP
        if self.training or sgn == 1:
            u_safe = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
                Q.double(), p_vector.double(), 
                G.double(), h.double(), e, e
            )
        else:
            # 推理时使用 cvxopt
            from cvxopt import solvers, matrix
            solvers.options['show_progress'] = False
            results = []
            for i in range(nBatch):
                sol = solvers.qp(
                    matrix(Q[i].cpu().numpy()),
                    matrix(p_vector[i].cpu().numpy()),
                    matrix(G[i].cpu().numpy()),
                    matrix(h[i].cpu().numpy())
                )
                results.append(float(sol['x'][0]))
            u_safe = torch.tensor(results, device=self.device).unsqueeze(1)
        
        return u_safe.float()
    
    def get_barrier_value(self, state):
        """返回当前状态的障碍函数值 b(x)"""
        d = state[:, 0] * self.std1
        return d - self.d_safe
    
    def get_penalty_values(self, state):
        """返回当前状态的惩罚函数值 p₁, p₂"""
        x = state
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x22 = torch.relu(self.fc22(x))
        x22 = self.bn22(x22)
        p = self.fc32(x22)
        p = 4.0 * torch.sigmoid(p)
        return p
```

### 6.2 新增核心文件：`Aebs/BarrierNet/upstream_from_ppo.py`

```python
import torch
import torch.nn as nn
from Combined_network.model import AebsEnd2EndNet


class UpstreamFromPPO(nn.Module):
    """
    从预训练的 PPO 控制器初始化 BarrierNet 的上游网络。
    
    策略: 将 PPO 的 controller_net 作为 BarrierNet 上游的 q头，
    然后新增一个 p头 用于学习 HOCBF 惩罚参数。
    """
    
    def __init__(self, ppo_p_net: AebsEnd2EndNet, hidden_p=32):
        super().__init__()
        
        # 复用 PPO 的 gen_net + state_net (冻结)
        self.gen_net = ppo_p_net.gen_net
        self.state_net = ppo_p_net.state_net
        
        # 复用 PPO 的 controller_net 作为特征提取器
        self.ppo_controller = ppo_p_net.controller_net
        
        # 获取 PPO controller 的输出维度
        # PPO: mlp_extractor(obs) → latent → action_net → action
        # 我们需要在 mlp_extractor 输出后分叉出 p头
        
        # 新增 p头 (HOCBF 惩罚参数)
        # PPO mlp_extractor 输出维度 = 64 (默认 stable_baselines3)
        self.p_head = nn.Sequential(
            nn.Linear(64, hidden_p),
            nn.ReLU(),
            nn.Linear(hidden_p, 2),  # p₁, p₂
            nn.Sigmoid()
        )
        # 初始化 p头 使 p₁≈p₂≈2.0 (Sigmoid×4 → 4×0.5=2)
        for m in self.p_head:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
    
    def extract_features(self, z, s):
        """
        复用 VCLS 流水线提取特征
        """
        with torch.no_grad():
            d = s[:, 0].unsqueeze(1)
            v = s[:, 1].unsqueeze(1)
            img = self.gen_net(z, d)
            img_flat = img.view(img.size(0), -1)
            state_est = self.state_net(img_flat)
        x = torch.cat([state_est, v], dim=1)
        return x
    
    def forward(self, z, s):
        """
        返回: (q, p)
        q: [B, 1] 参考控制 (来自 PPO)
        p: [B, 2] HOCBF 惩罚参数
        """
        obs = self.extract_features(z, s)
        
        # PPO 特征提取
        latent = self.ppo_controller.mlp_extractor(obs)
        
        # q头: PPO 原始动作
        q = self.ppo_controller.action_net(latent)  # [B, 1]
        
        # p头: HOCBF 惩罚参数
        p = self.p_head(latent) * 4.0  # [B, 2], 确保 > 0
        
        return q, p
```

### 6.3 新增训练器：`Aebs/BarrierNet/trainer.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import copy


class BarrierNetTrainer:
    """
    BarrierNet-AEBS 训练器
    
    训练目标:
    1. 行为克隆: BarrierNet 输出 u* ≈ PPO 参考控制 π₀(s)
    2. 惩罚参数学习: p₁, p₂ 自动调整以平衡安全性和性能
    3. QP 层保证每次输出都满足 HOCBF 安全约束
    
    训练策略:
    - 不需要 CEGIS 循环
    - 不需要 IBP 验证
    - 标准端到端梯度下降
    """
    
    def __init__(
        self,
        barrier_net,        # BarrierNetAEBS 模型
        ppo_teacher,         # PPO 预训练教师模型 (AebsEnd2EndNet, 冻结)
        env,                 # Aebs 环境
        lr=1e-3,             # 学习率
        lambda_mse=1.0,      # 行为克隆损失权重
        lambda_barrier=0.1,  # 障碍函数正则化权重
        lambda_smooth=0.01,  # 平滑性正则化权重
    ):
        self.barrier_net = barrier_net
        self.ppo_teacher = ppo_teacher
        self.env = env
        self.lambda_mse = lambda_mse
        self.lambda_barrier = lambda_barrier
        self.lambda_smooth = lambda_smooth
        
        # 优化器 (只优化 BarrierNet 的参数)
        self.optimizer = torch.optim.Adam(
            barrier_net.parameters(), lr=lr
        )
        
        self.device = next(barrier_net.parameters()).device
        
        # 冻结教师模型
        for param in self.ppo_teacher.parameters():
            param.requires_grad = False
    
    def generate_training_data(self, n_samples=10000):
        """
        在状态空间中均匀采样训练点
        """
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        
        states = np.random.uniform(
            obs_low, obs_high, size=(n_samples, 2)
        ).astype(np.float32)
        
        return torch.tensor(states, device=self.device)
    
    def compute_loss(self, states, z_batch):
        """
        复合损失函数
        
        L = λ_mse · L_mse + λ_barrier · L_barrier + λ_smooth · L_smooth
        
        其中:
        - L_mse: 行为克隆损失 (与 PPO 教师模型的偏差)
        - L_barrier: 障碍函数正则化 (鼓励 b(x) 远离 0)
        - L_smooth: 控制平滑性 (惩罚 p 的剧烈变化)
        """
        # BarrierNet 前向传播
        u_safe = self.barrier_net(states, sgn=1)  # [B, 1]
        
        # PPO 教师标签
        with torch.no_grad():
            u_teacher = self.ppo_teacher(z_batch, states)  # [B, 1]
        
        # 1. 行为克隆 MSE 损失
        loss_mse = F.mse_loss(u_safe, u_teacher)
        
        # 2. 障碍函数正则化 (鼓励安全裕量)
        b_values = self.barrier_net.get_barrier_value(states)
        # 对于接近不安全区域的状态，惩罚 b 接近 0
        loss_barrier = torch.mean(F.relu(1.0 - b_values))
        
        # 3. 惩罚参数平滑性正则化
        p_values = self.barrier_net.get_penalty_values(states)
        # 惩罚 p 值过大 (过度保守)
        loss_smooth = torch.mean(p_values)
        
        # 复合损失
        total_loss = (
            self.lambda_mse * loss_mse +
            self.lambda_barrier * loss_barrier +
            self.lambda_smooth * loss_smooth
        )
        
        return total_loss, {
            'loss_mse': loss_mse.item(),
            'loss_barrier': loss_barrier.item(),
            'loss_smooth': loss_smooth.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, states, batch_size=256):
        """训练一个 epoch"""
        self.barrier_net.train()
        
        # 打乱数据
        perm = torch.randperm(states.size(0))
        states = states[perm]
        
        total_metrics = {}
        n_batches = 0
        
        for start in range(0, states.size(0), batch_size):
            end = min(start + batch_size, states.size(0))
            batch_states = states[start:end]
            batch_z = (torch.rand(batch_states.size(0), 4, device=self.device) * 2 - 1)
            
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch_states, batch_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.barrier_net.parameters(), 5.0)
            self.optimizer.step()
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            n_batches += 1
        
        return {k: v / n_batches for k, v in total_metrics.items()}
    
    def train(self, n_epochs=100, n_samples=10000, batch_size=256):
        """完整训练循环"""
        states = self.generate_training_data(n_samples)
        
        for epoch in range(n_epochs):
            metrics = self.train_epoch(states, batch_size)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
                print("-" * 40)
        
        return metrics
```

### 6.4 新增仿真验证器：`Aebs/BarrierNet/simulator.py`

```python
import torch
import numpy as np
import matplotlib.pyplot as plt


class SafetySimulator:
    """
    闭环仿真验证器
    
    替代 SafePVC 的 IBP 形式化验证，通过仿真验证 b(x(t)) ≥ 0。
    虽然仿真不能提供形式化保证，但 BarrierNet 的安全性是
    by-construction 的（QP 约束保证每步都满足 HOCBF）。
    """
    
    def __init__(self, barrier_net, env, dt=0.05):
        self.barrier_net = barrier_net
        self.env = env
        self.dt = dt
        self.device = next(barrier_net.parameters()).device
    
    def simulate_trajectory(self, s0, n_steps=200, add_noise=True):
        """
        仿真一条轨迹，记录每步的状态、控制、障碍函数值
        """
        self.barrier_net.eval()
        
        trajectory = {
            'd': [], 'v': [], 'acc': [],
            'b': [], 'p1': [], 'p2': [],
            'safe': []
        }
        
        state = torch.tensor(s0, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            for t in range(n_steps):
                # 获取安全控制
                z = torch.zeros(1, 4, device=self.device)
                acc = self.barrier_net(state, sgn=0)
                
                # 记录数据
                d_norm = state[0, 0].item()
                v = state[0, 1].item()
                b_val = self.barrier_net.get_barrier_value(state).item()
                p_vals = self.barrier_net.get_penalty_values(state)
                
                trajectory['d'].append(d_norm * self.env.std1)
                trajectory['v'].append(v)
                trajectory['acc'].append(acc[0, 0].item())
                trajectory['b'].append(b_val)
                trajectory['p1'].append(p_vals[0, 0].item())
                trajectory['p2'].append(p_vals[0, 1].item())
                trajectory['safe'].append(b_val >= 0)
                
                # 动力学更新
                d = d_norm * self.env.std1
                d_next = d - v * self.dt
                v_next = v - acc[0, 0].item() * self.dt
                
                # 添加扰动 (模拟环境不确定性)
                if add_noise:
                    noise_d = np.random.uniform(
                        self.env.noise_bounds[0][0],
                        self.env.noise_bounds[1][0]
                    )
                    noise_v = np.random.uniform(
                        self.env.noise_bounds[0][1],
                        self.env.noise_bounds[1][1]
                    )
                    d_next += noise_d
                    v_next += noise_v
                
                v_next = np.clip(v_next, 0.0, 3.0)
                d_next_norm = d_next / self.env.std1
                
                state = torch.tensor(
                    [[d_next_norm, v_next]],
                    dtype=torch.float32, device=self.device
                )
                
                # 终止条件
                if d_next <= 5.0 or d_next >= 16.0 or v_next <= 0.0:
                    break
        
        return trajectory
    
    def batch_evaluate(self, n_episodes=100):
        """
        批量评估安全率
        """
        safe_count = 0
        min_b_values = []
        
        for ep in range(n_episodes):
            # 随机初始状态
            d0 = np.random.uniform(15.0, 16.0)
            v0 = np.random.uniform(2.5, 3.0)
            s0 = [d0 / self.env.std1, v0]
            
            traj = self.simulate_trajectory(s0, n_steps=500)
            min_b = min(traj['b'])
            min_b_values.append(min_b)
            
            if min_b >= -0.01:  # 允许微小数值误差
                safe_count += 1
        
        safety_rate = safe_count / n_episodes * 100
        
        print(f"Safety Rate: {safety_rate:.1f}% ({safe_count}/{n_episodes})")
        print(f"Min barrier value: {min(min_b_values):.4f}")
        print(f"Mean min barrier: {np.mean(min_b_values):.4f}")
        
        return {
            'safety_rate': safety_rate,
            'min_b_values': min_b_values,
            'mean_min_b': np.mean(min_b_values),
        }
    
    def plot_trajectory(self, trajectory, save_path=None):
        """可视化单条轨迹"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        t = range(len(trajectory['d']))
        
        # (1) 距离轨迹
        axes[0, 0].plot(t, trajectory['d'], 'b-')
        axes[0, 0].axhline(y=6.0, color='r', linestyle='--', label='d_safe')
        axes[0, 0].set_xlabel('Time step')
        axes[0, 0].set_ylabel('Distance (m)')
        axes[0, 0].set_title('Distance Trajectory')
        axes[0, 0].legend()
        
        # (2) 速度轨迹
        axes[0, 1].plot(t, trajectory['v'], 'g-')
        axes[0, 1].set_xlabel('Time step')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Trajectory')
        
        # (3) 障碍函数值
        axes[1, 0].plot(t, trajectory['b'], 'r-')
        axes[1, 0].axhline(y=0, color='k', linestyle='--', label='b=0 (safety boundary)')
        axes[1, 0].set_xlabel('Time step')
        axes[1, 0].set_ylabel('b(x)')
        axes[1, 0].set_title('Barrier Function Value')
        axes[1, 0].legend()
        
        # (4) 惩罚函数值
        axes[1, 1].plot(t, trajectory['p1'], 'm-', label='p₁')
        axes[1, 1].plot(t, trajectory['p2'], 'c-', label='p₂')
        axes[1, 1].set_xlabel('Time step')
        axes[1, 1].set_ylabel('Penalty Value')
        axes[1, 1].set_title('HOCBF Penalty Functions')
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
```

### 6.5 修改后的主入口：`Aebs/BarrierNet/main.py`

```python
"""
SafePVC-BarrierNet 主入口

替代原有的 Aebs/VT/loop.py CEGIS 循环。
新流程:
1. 加载预训练的 VCLS (cGAN + state_net + PPO controller)
2. 构建 BarrierNet-AEBS 模型
3. 端到端训练 (行为克隆 + QP安全约束)
4. 闭环仿真验证
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from Aebs.system.env import Aebs
from Aebs.BarrierNet.barrier_net_aebs import BarrierNetAEBS
from Aebs.BarrierNet.trainer import BarrierNetTrainer
from Aebs.BarrierNet.simulator import SafetySimulator
from Combined_network.model import AebsEnd2EndNet
from cGAN.taxi_models_and_data import AebsMLPGenerator
from stable_baselines3 import PPO


def load_ppo_teacher():
    """加载预训练的 PPO 教师模型 (VCLS)"""
    gen_net = AebsMLPGenerator(4, 1)
    gen_net.load_state_dict(torch.load("./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"))
    
    state_layer_sizes = [1024, 256, 64, 1]
    model = PPO.load('./Aebs/controller/best_model/best_model.zip')
    policy = model.policy
    mlp_extractor = policy.mlp_extractor.policy_net
    action_net = policy.action_net
    
    p_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)
    p_net.state_net.load_state_dict(torch.load("./Aebs/controller/state_net_trained.pth"))
    p_net.eval()
    
    return p_net


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== 1. 环境初始化 ==========
    env = Aebs(0.05)  # noise_factor = 0.05
    
    # ========== 2. 加载 PPO 教师模型 ==========
    ppo_teacher = load_ppo_teacher().to(device)
    print("PPO teacher loaded.")
    
    # ========== 3. 构建 BarrierNet-AEBS ==========
    barrier_net = BarrierNetAEBS(
        state_dim=2,
        hidden1=64,
        hidden21=32,
        hidden22=32,
        d_safe=6.0,
        std1=env.std1,
        u_min=-3.0,
        u_max=3.0,
        dt=0.05,
        noise_factor=0.05,
        device=device
    ).to(device)
    print(f"BarrierNet-AEBS created. delta_max: {barrier_net.delta_max:.4f}")
    
    # ========== 4. 训练 ==========
    trainer = BarrierNetTrainer(
        barrier_net=barrier_net,
        ppo_teacher=ppo_teacher,
        env=env,
        lr=1e-3,
        lambda_mse=1.0,
        lambda_barrier=0.1,
        lambda_smooth=0.01,
    )
    
    print("\n=== Training BarrierNet-AEBS ===")
    metrics = trainer.train(n_epochs=100, n_samples=10000, batch_size=256)
    
    # 保存模型
    torch.save(barrier_net.state_dict(), "./Aebs/BarrierNet/model_bn_aebs.pth")
    print("Model saved.")
    
    # ========== 5. 仿真验证 ==========
    print("\n=== Safety Simulation ===")
    simulator = SafetySimulator(barrier_net, env)
    results = simulator.batch_evaluate(n_episodes=100)
    
    # 可视化一条典型轨迹
    d0, v0 = 15.5 / env.std1, 2.75
    traj = simulator.simulate_trajectory([d0, v0], n_steps=200, add_noise=True)
    simulator.plot_trajectory(traj, save_path="./Aebs/BarrierNet/trajectory.png")
    
    print(f"\nFinal Safety Rate: {results['safety_rate']:.1f}%")


if __name__ == "__main__":
    main()
```

### 6.6 X-Plane11 适配（第二个 Benchmark）

X-Plane11 的动力学：
```
p_{k+1} = p_k + v·dt·sin(θ_k)
θ_{k+1} = θ_k + (v/L)·dt·tan(φ_k)

状态: [p_norm, θ_norm]  (横向偏差 + 航向角偏差)
控制: φ (前轮转角)
```

这同样可以建模为控制仿射系统，但障碍函数和 Lie 导数需要重新推导。

**X-Plane11 的 BarrierNet 需要单独实现一个 `BarrierNetXPlane` 类**，结构与 AEBS 类似，但 Lie 导数计算不同。这里不再赘述，但核心模式完全一致。

---

## 7. 新训练流程

### 7.1 完整训练流水线

```
阶段 1: 预训练 (与 SafePVC 完全一致)
│
├── 1a. 训练 cGAN 感知模型
│       python Aebs/cGAN/train_mlp.py
│       → mlp_supervised.pth
│
├── 1b. 训练状态估计器
│       python Aebs/controller/StateEstimate_train.py
│       → state_net_trained.pth
│
└── 1c. PPO 控制器预训练
        python Aebs/controller/Controller_train.py
        → best_model.zip

阶段 2: BarrierNet 集成训练 (新)
│
├── 2a. 构建 BarrierNet-AEBS 模型
│       (定义 HOCBF + 鲁棒裕量)
│
├── 2b. 端到端训练
│       python -m Aebs.BarrierNet.main
│       → model_bn_aebs.pth
│
│   训练循环:
│   ┌──────────────────────────────────────────────────────────┐
│   │  FOR epoch = 1 TO 100:                                  │
│   │    FOR batch IN state_samples:                          │
│   │      u_safe = BarrierNet(state)    # QP 求解            │
│   │      u_teacher = PPO(state)         # 教师标签           │
│   │      loss = MSE(u_safe, u_teacher)  # 行为克隆           │
│   │             + barrier_reg + smooth_reg                   │
│   │      loss.backward()                # 梯度穿过 QP 层     │
│   │      optimizer.step()                                   │
│   └──────────────────────────────────────────────────────────┘
│
└── 2c. 闭环仿真验证
        simulator.batch_evaluate(100 episodes)
        → safety_rate, trajectory plots

(不再需要 CEGIS 循环 / IBP 验证 / 反例生成)
```

### 7.2 训练时间对比

| 阶段 | SafePVC (SBC) | BarrierNet |
|------|--------------|------------|
| 感知模型训练 | 数小时 | 数小时（不变） |
| PPO 预训练 | ~200K steps | ~200K steps（不变） |
| 安全验证/训练 | **100次CEGIS迭代 × (10 epoch L训练 + 验证 + 1 epoch P训练)** | **100 epoch 标准训练** |
| 单次迭代/epoch | L训练: ~2min, 验证: ~5min | ~10s/epoch (含QP求解) |
| 总后端时间 | **~10小时** | **~15分钟** |

### 7.3 损失函数对比

| 损失组件 | SafePVC L-Net | SafePVC P-Net | BarrierNet |
|---------|--------------|--------------|------------|
| 核心损失 | 鞅损失 (E[B(s')]≤B(s)) | 鞅损失 (反向传播到控制器) | MSE (u* vs u_teacher) |
| Lipschitz 正则 | ✅ 梯度范数约束 | ✅ 梯度范数约束 | ❌ (p_i 由 Sigmoid 保证 Lipschitz) |
| 区域约束 | ✅ B(s₀)≤1, B(s_u)≥1/(1-p) | ❌ | ❌ (QP 约束替代) |
| 蒸馏损失 | ❌ | ✅ MSE vs 原始 PPO | ✅ MSE vs PPO |
| 安全保证来源 | 损失 + 验证 | 损失 + 验证 | **QP 硬约束** |

---

## 8. 实验方案

### 8.1 核心实验

#### 实验 1：安全率对比

| 方法 | 指标 | 测试条件 |
|------|------|---------|
| PPO (无安全) | 碰撞率 | 100 episodes, 有扰动 |
| SafePVC (SBC, Fixed) | 概率安全下界 | 原论文结果 |
| SafePVC (SBC, Alternating) | 概率安全下界 | 原论文结果 |
| **BarrierNet (无鲁棒裕量)** | **仿真安全率** | 100 episodes, 有扰动 |
| **BarrierNet (有鲁棒裕量)** | **仿真安全率** | 100 episodes, 有扰动 |

#### 实验 2：控制性能对比

| 指标 | 说明 |
|------|------|
| 到达时间 | 车辆到达目标距离所需时间 |
| 控制平滑度 | Σ|u_t - u_{t-1}|² |
| 安全干预率 | QP 约束激活的比例（u* ≠ q 的比例） |
| 最小距离 | 轨迹中距目标的最近距离 |

#### 实验 3：不同扰动强度下的鲁棒性

| 扰动因子 | SafePVC 结果 | BarrierNet 结果 |
|---------|-------------|----------------|
| 1% | 原论文数据 | 仿真安全率 |
| 3% | 原论文数据 | 仿真安全率 |
| 5% | 原论文数据 | 仿真安全率 |
| 10% | 原论文数据 | 仿真安全率 |

### 8.2 可视化

1. **障碍函数值时序图**：b(x(t)) 随时间变化，始终 ≥ 0
2. **惩罚函数自适应图**：p₁(t), p₂(t) 在接近障碍时的变化
3. **相空间图**：(d, v) 相空间中的轨迹 + 安全边界
4. **安全干预热力图**：QP 约束在哪些状态区域被激活

---

## 9. 风险与缓解策略

### 9.1 理论风险

| 风险 | 严重程度 | 缓解策略 |
|------|---------|---------|
| 确定性安全 vs 概率安全：BarrierNet 无法提供概率安全界 | 🔴 高 | 在论文中重新定位贡献为"确定性安全保证"而非"概率安全保证"；补充鲁棒裕量的理论分析 |
| 离散时间 QP 的采样间效应：HOCBF 仅在采样时刻保证 b(x) ≥ 0 | 🟡 中 | 减小 dt 或使用连续时间 HOCBF 理论；加入额外的安全裕量 |
| 鲁棒裕量过大导致过度保守：δ 太大会使 QP 无可行解 | 🟡 中 | 自适应调整 δ；使用 chance-constrained QP 替代最坏情况分析 |

### 9.2 工程风险

| 风险 | 严重程度 | 缓解策略 |
|------|---------|---------|
| QP 无可行解：在极端状态下 HOCBF 约束与控制约束矛盾 | 🔴 高 | 加入 slack variable + 惩罚项；设计 fallback 控制器（最大制动） |
| 训练不收敛：QP 层梯度在某些区域不稳定 | 🟡 中 | 梯度裁剪 + warmup 学习率 + 从简单问题开始 |
| qpth 依赖的兼容性问题 | 🟢 低 | 备选方案：cvxpy_layers 或自定义 QP 求解器 |

### 9.3 论文风险

| 风险 | 严重程度 | 缓解策略 |
|------|---------|---------|
| 失去"概率安全"作为核心贡献 | 🔴 高 | 强调"确定性安全 + 可学习自适应"作为新贡献；与 SBC 做定量对比实验 |
| 审稿人质疑"为什么不用 SBC" | 🟡 中 | 在 Related Work 中对比两种方法的优劣；做 ablation 实验 |
| 安全保证的强度变化 | 🟡 中 | 明确讨论确定性保证 vs 概率保证的 trade-off |

### 9.4 QP 无可行解的应急方案

```python
class BarrierNetAEBS_Fallback(BarrierNetAEBS):
    """
    带 fallback 的 BarrierNet
    当 QP 无可行解时，使用最大制动作为安全控制
    """
    
    def forward(self, state, sgn=1):
        try:
            u_safe = super().forward(state, sgn)
            
            # 检查 QP 是否成功 (NaN 表示失败)
            if torch.isnan(u_safe).any():
                u_safe = self._fallback(state)
            
            return u_safe
        except Exception:
            return self._fallback(state)
    
    def _fallback(self, state):
        """最大制动: acc = -3.0 (最保守的安全动作)"""
        return torch.full(
            (state.size(0), 1), 
            self.u_min, 
            device=self.device
        )
```

---

## 10. 实施路线图

### Phase 1: 基础搭建 (1-2 天)

- [ ] 创建 `Aebs/BarrierNet/` 目录结构
- [ ] 实现 `BarrierNetAEBS` 核心模型
- [ ] 实现 HOCBF Lie 导数计算（AEBS 动力学）
- [ ] 安装 qpth 依赖并验证 QP 求解
- [ ] 单元测试：验证 QP 输出的安全约束

### Phase 2: 训练集成 (2-3 天)

- [ ] 实现 `BarrierNetTrainer` 训练器
- [ ] 加载 PPO 教师模型并生成训练标签
- [ ] 端到端训练循环
- [ ] 超参数调优 (lr, λ_mse, λ_barrier, λ_smooth)
- [ ] 训练收敛性验证

### Phase 3: 验证与分析 (2-3 天)

- [ ] 实现 `SafetySimulator` 闭环仿真
- [ ] 批量评估安全率 (100 episodes)
- [ ] 可视化：轨迹、障碍函数值、惩罚函数
- [ ] 不同扰动强度下的鲁棒性测试
- [ ] 与 SafePVC 原方法的安全率对比

### Phase 4: X-Plane11 适配 (2-3 天)

- [ ] 推导 X-Plane11 动力学的 Lie 导数
- [ ] 实现 `BarrierNetXPlane` 模型
- [ ] 训练与验证
- [ ] 与 SafePVC X-Plane11 结果对比

### Phase 5: 论文撰写 (3-5 天)

- [ ] 修改 Introduction：强调确定性安全 + 可学习自适应
- [ ] 修改 Preliminaries：HOCBF 理论替代 SBC 鞅理论
- [ ] 修改 Method：BarrierNet-QP 层描述
- [ ] 修改 Experiments：新实验结果
- [ ] 新增 Discussion：确定性 vs 概率安全的 trade-off

**总计预估时间: 10-16 天**

---

## 11. 文件修改清单

### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `Aebs/BarrierNet/__init__.py` | 包初始化 |
| `Aebs/BarrierNet/barrier_net_aebs.py` | BarrierNet-AEBS 核心模型 |
| `Aebs/BarrierNet/upstream_from_ppo.py` | PPO 初始化上游网络 |
| `Aebs/BarrierNet/trainer.py` | 端到端训练器 |
| `Aebs/BarrierNet/simulator.py` | 闭环仿真验证器 |
| `Aebs/BarrierNet/main.py` | 主入口 |
| `Aebs/BarrierNet/barrier_net_xplane.py` | X-Plane11 适配模型 |
| `Aebs/BarrierNet/plot_utils.py` | 可视化工具 |

### 保留不变的文件

| 文件路径 | 说明 |
|---------|------|
| `Aebs/cGAN/*` | cGAN 感知模型 |
| `Aebs/controller/*` | PPO 预训练 |
| `Aebs/connect/*` | 数据收集 |
| `Aebs/system/env.py` | 环境定义与动力学 |
| `Aebs/data/*` | 训练数据 |
| `Combined_network/model.py` | VCLS 端到端模型 |
| `cGAN/*` | cGAN 通用库 |

### 不再需要的文件（保留但不使用）

| 文件路径 | 原功能 |
|---------|--------|
| `Aebs/VT/loop.py` | CEGIS 主循环 |
| `Aebs/VT/train.py` | SBC + 控制器交替训练 |
| `Aebs/VT/verify.py` | IBP 验证 + 反例生成 |
| `Aebs/VT/utils.py` | MLP 障碍证书 + 鞅损失 |
| `auto_LiRPA/*` | 神经网络验证库 |

---

## 附录 A: Lie 导数推导详解

### AEBS 系统

连续时间动力学：
```
ẋ = f(x) + g(x)u

x = [d, v]ᵀ
f(x) = [-v, 0]ᵀ
g(x) = [0, -1]ᵀ
u = acc
```

障碍函数：`b(x) = d - d_safe`

**第零阶：**
```
b(x) = d - d_safe
```

**一阶 Lie 导数：**
```
∂b/∂x = [1, 0]

L_f b(x) = [1, 0] · [-v, 0]ᵀ = -v
L_g b(x) = [1, 0] · [0, -1]ᵀ = 0    ← 相对度 ≠ 1
```

**二阶 Lie 导数：**
```
∂(L_f b)/∂x = [0, -1]

L_f² b(x) = [0, -1] · [-v, 0]ᵀ = 0
L_g L_f b(x) = [0, -1] · [0, -1]ᵀ = 1    ← 相对度 = 2
```

**HOCBF 约束 (r=2, 软化)：**
```
L_f² b + (L_g L_f b)·u + (p₁+p₂)·L_f b + p₁·p₂·b ≥ 0
0 + 1·u + (p₁+p₂)·(-v) + p₁·p₂·(d - d_safe) ≥ 0
u ≥ (p₁+p₂)·v - p₁·p₂·(d - d_safe)
```

### X-Plane11 系统

连续时间近似：
```
ṗ = v·sin(θ)
θ̇ = (v/L)·tan(φ)

x = [p, θ]ᵀ
u = φ
f(x) = [v·sin(θ), 0]ᵀ
g(x) = [0, v/L · sec²(φ)]ᵀ  ← 依赖于 u，需要特殊处理
```

这个系统的控制出现在 tan(φ) 中，不是标准仿射形式。需要使用**输入变换**：
```
设 w = tan(φ) 作为新的控制输入

则 θ̇ = (v/L)·w
f(x) = [v·sin(θ), 0]ᵀ
g(x) = [0, v/L]ᵀ

u_actual = arctan(w)
```

障碍函数：`b(x) = p_max² - p²` (保持在跑道中心线附近)

```
∂b/∂x = [-2p, 0]

L_f b = [-2p, 0] · [v·sin(θ), 0]ᵀ = -2p·v·sin(θ)
L_g b = [-2p, 0] · [0, v/L]ᵀ = 0    ← 相对度 ≠ 1

∂(L_f b)/∂x = [-2v·sin(θ), -2p·v·cos(θ)]

L_f² b = [-2v·sin(θ), -2p·v·cos(θ)] · [v·sin(θ), 0]ᵀ
       = -2v²·sin²(θ)

L_g L_f b = [-2v·sin(θ), -2p·v·cos(θ)] · [0, v/L]ᵀ
          = -2p·v²·cos(θ)/L    ← 相对度 = 2
```

**X-Plane11 的 HOCBF QP：**
```
min  ½w² + q·w
s.t. -2v²·sin²(θ) + (-2p·v²·cos(θ)/L)·w 
     + (p₁+p₂)·(-2p·v·sin(θ)) + p₁·p₂·(p_max² - p²) ≥ 0
     w_min ≤ w ≤ w_max
```

---

## 附录 B: 论文修改建议

### 标题建议

原标题: *"Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems"*

新标题候选:
1. *"Provably Safe Controller Synthesis with Differentiable Barrier Functions for Vision-Based Neural Network Control Systems"*
2. *"Safety-Guaranteed Vision-Based Control via Learnable Control Barrier Functions"*
3. *"BarrierNet-based Safe Controller Synthesis for Vision-Based Neural Network Control Systems"*

### 贡献重新表述

原贡献:
1. 概率安全控制器合成框架
2. 可学习的随机障碍证书 (SBC + 鞅理论)
3. 反例引导的交替优化

新贡献:
1. **确定性安全控制器合成框架**：将 HOCBF-QP 层首次引入视觉 NNCS 的安全综合
2. **可学习的安全裕度**：通过环境自适应的惩罚函数 p_i(z) 降低 HOCBF 保守性
3. **简化训练流程**：端到端训练替代 CEGIS 循环，训练效率提升 ~40×
4. **鲁棒化扩展**：将扰动建模为鲁棒裕量，在确定性框架下处理环境不确定性

---

## 附录 C: 环境配置

```yaml
# 新增依赖 (在 environment.yml 基础上)
dependencies:
  - pytorch>=1.12
  - numpy
  - scipy
  - matplotlib
  - pip:
    - qpth>=0.0.15        # 可微 QP 求解器 (BarrierNet 核心)
    - cvxopt>=1.3.0       # 推理时 QP 求解器
    - stable-baselines3   # PPO 预训练 (保留)
    - gymnasium           # 环境定义 (保留)
    - h5py                # 数据加载 (保留)
    # 注意: 不再需要 auto_LiRPA
```
