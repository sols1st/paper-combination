# BarrierNet 论文深度解析: CBF-QP原理、架构、数据流

> 论文: *BarrierNet: A Safety-Guaranteed Layer for Neural Networks*
> 基于 BarrierNet/ 代码库的完整分析

---

## 目录

1. [核心问题: NN控制器的安全保障](#1-核心问题)
2. [CBF 数学基础](#2-cbf-数学基础)
3. [HOCBF: 高阶控制屏障函数](#3-hocbf)
4. [BarrierNet 架构: 双分支 + QP](#4-barriernet-架构)
5. [可微分 QP: qpth 原理](#5-可微分-qp)
6. [2D 机器人示例 (最简实现)](#6-2d-机器人)
7. [自动驾驶场景 (完整实现)](#7-自动驾驶场景)
8. [完整数据流](#8-完整数据流)
9. [与 SafePVC/SBC 的关系](#9-与-safepvcsbc-的关系)
10. [关键超参数](#10-关键超参数)

---

## 1. 核心问题: NN 控制器的安全保障

### 1.1 问题

神经网络控制器 $\pi_{NN}(s)$ 可以从数据中学习复杂的控制策略, 但**无法提供安全保证**——你不知道它什么时候会输出危险动作。

**BarrierNet 的解决方案**: 在 NN 输出后面加一个**数学上保证安全**的 QP 层。

### 1.2 基本思路

```
传统 NN 控制器:
  state → NN → u (直接输出, 无安全保证)

BarrierNet:
  state → NN → u_ref (参考控制)
             → p (CBF参数, 控制"保守程度")
             → QP: min ½(u-u_ref)²  s.t. CBF安全约束
             → u_safe (保证安全的控制)
```

**核心**: QP 是一个凸优化问题, 有解析的KKT条件。通过隐式微分, 梯度可以通过 QP 反向传播到 NN —— NN 学习输出**容易被 QP 修正为安全的**参考控制。

---

## 2. CBF 数学基础

### 2.1 控制屏障函数 (CBF)

对于一个控制系统 $\dot{x} = f(x) + g(x)u$, 定义一个**安全集**:

$$\mathcal{C} = \{x \mid b(x) \geq 0\}$$

其中 $b(x)$ 是**屏障函数**。$b(x) \geq 0$ 表示安全。

**CBF 条件**: $b(x)$ 是 CBF 当且仅当存在 $\alpha > 0$:

$$\sup_{u} [L_f b(x) + L_g b(x) \cdot u + \alpha \cdot b(x)] \geq 0$$

其中 $L_f b = \nabla b \cdot f$, $L_g b = \nabla b \cdot g$ 是**李导数**。

**含义**: 存在某个控制 u, 使得屏障值在变负之前被"推回来"。满足此条件 → 系统永远不离开安全集。

### 2.2 离散形式的 CBF-QP

实际推理时, CBF 条件转化为 QP 约束:

```
QP:  min  ½(u - u_ref)²          ← 尽可能接近参考控制
     s.t. L_f b + L_g b · u + p · b ≥ 0   ← CBF 约束
          u_min ≤ u ≤ u_max               ← 控制限幅
```

其中 p 是**CBF 参数** (即 $\alpha$), 控制安全约束的"紧度":
- p 大 → 约束紧 → 更保守 → 安全性高
- p 小 → 约束松 → 更自由 → 可能不够安全

### 2.4 CBF 约束是怎么给的? (★ 关键理解)

**CBF 约束是手工推导、硬编码的, 不是 NN 学的。**

NN 只学两个东西: **参考控制 q** 和 **CBF 参数 p**。约束的形式和结构完全由人预先推导。

#### 整个流程三步走

```
第1步 (人): 定义屏障函数 b(s)
  例: b(s) = (px-40)² + (py-15)² - 6²  ← 人根据场景设计

第2步 (人): 手工推导李导数 (纸笔算好, 硬编码到代码)
  ḃ    = ∂b/∂s · f(s)      ← 对漂移动力学求导
  b̈    = Lf²b + LgLf_b·u   ← 推到 u 出现为止 (相对度决定推几阶)

第3步 (NN + QP): 
  NN 输出 q (参考控制) 和 p (CBF 参数)
  QP: min ½||u-q||²  s.t. G·u ≤ h  ← G, h 由第2步的推导结果构造
```

#### 具体例子: 2D 机器人避障

**第1步 — 人定义屏障函数**:
```python
# 障碍物位置 (40, 15), 安全半径 R=6
barrier = (px - 40)² + (py - 15)² - 6²
#         └────距离平方─────┘   └半径²┘
#  b > 0 → 机器人到障碍物距离 > 6 → 安全
#  b < 0 → 碰撞!
```

**第2步 — 人推导李导数 (纸笔):**

动力学: ẋ = v·cos(θ), ẏ = v·sin(θ), θ̇ = ω, v̇ = a

```
一阶导数 ḃ (对时间求导):
  ḃ = 2(px-40)·ẋ + 2(py-15)·ẏ
    = 2(px-40)·v·cos(θ) + 2(py-15)·v·sin(θ)
    ← 注意: u=[ω,a] 不出现! 相对度=2

二阶导数 b̈ (再对时间求导):
  b̈ = ∂ḃ/∂px·ẋ + ∂ḃ/∂py·ẏ + ∂ḃ/∂θ·θ̇ + ∂ḃ/∂v·v̇
     = [不含u的项] + [含ω的项] + [含a的项]
     = Lf²b + LgLf_u1·ω + LgLf_u2·a
     └──漂移──┘ └─────控制项──────┘
```

然后把这些推导结果直接硬编码:

```python
# 这些是人在纸上推导好的, 直接写进代码!
barrier_dot = 2*(px-40)*v*cos(θ) + 2*(py-15)*v*sin(θ)

Lf2b    = 2*v²                                    # 漂移贡献 (不含u)
LgLf_u1 = -2*(px-40)*v*sin(θ) + 2*(py-15)*v*cos(θ)  # ∂b̈/∂ω
LgLf_u2 =  2*(px-40)*cos(θ) + 2*(py-15)*sin(θ)       # ∂b̈/∂a
```

**第3步 — NN 只输出两个值, QP 负责满足约束**:

```python
# NN 输出 (网络学出来的):
q = q_mlp(features)  = [ω_ref, a_ref]    ← "我想这么走"
p = p_mlp(features)  = [p1, p2]          ← "约束多紧"

# QP (硬编码的G, h + NN输出的q, p):
HOCBF: b̈ + (p1+p2)·ḃ + p1·p2·b ≥ 0
  → 代入 b̈ = Lf2b + LgLf_u1·ω + LgLf_u2·a
  → -(LgLf_u1·ω + LgLf_u2·a) ≤ Lf2b + (p1+p2)·barrier_dot + p1·p2·barrier

G = [[-LgLf_u1, -LgLf_u2]]          ← 人推导的, 硬编码
h = Lf2b + (p1+p2)*barrier_dot + p1·p2*barrier  ← 人推导的, p由NN提供

QP: min ½||u - q||²  s.t. G·u ≤ h   ← q来自NN, G/h来自人
```

#### NN 学什么 vs 人给什么

```
         ┌─────────────────────────────────────────────┐
         │              人预先设计的                      │
         │  • 屏障函数 b(s) 的形式 (如到障碍物的距离)      │
         │  • 李导数 Lf, Lg 的表达式 (纸笔推导)           │
         │  • HOCBF 的结构 (几阶, 怎么组合)               │
         │  • G, h 的构造逻辑                             │
         ├─────────────────────────────────────────────┤
         │              NN 学习的                        │
         │  • q: 参考控制 (从图像特征 → 想要的运动)        │
         │  • p: CBF 参数 (从图像特征 → 应该多保守)        │
         └─────────────────────────────────────────────┘
```

#### 这也是 BarrierNet 的一个局限

**每换一个场景, 需要重新推导 CBF 的李导数**:

| 场景 | b(s) 形式 | 动力学 | 李导数推导 |
|------|---------|------|:---:|
| AEBS | d - v·t_gap | 线性 | 简单 (相对度1) |
| 2D 机器人避障 | (px-ox)²+(py-oy)²-R² | 非线性 | 复杂 (相对度2) |
| CARLA 驾驶 | 避障 + 车道保持 | 非线性 | 很复杂 (相对度2, 多个CBF) |

如果要换到新场景, 需要:
1. 定义新的 b(s)
2. 对新的动力学推导李导数
3. 修改代码中的 G, h 构建逻辑
4. NN 部分 (CNN+LSTM+q_mlp+p_mlp) 可以复用

### 2.5 相对度 (Relative Degree)

**相对度**: 控制 u 出现在 b 的第几阶导数中。

```
相对度 1:  ḃ 中包含 u → 标准 CBF 可直接约束
相对度 2:  ḃ 中不含 u, b̈ 中才含 u → 需要 HOCBF
```

**AEBS 示例** (相对度 1):
$$b(s) = d - v \cdot t_{gap}$$
$$\dot{b} = -v + t_{gap} \cdot u \quad \text{(u 直接出现)}$$

**2D 机器人示例** (相对度 2):
$$b(x) = (p_x - obs_x)^2 + (p_y - obs_y)^2 - R^2$$
$$\dot{b} = 2(p_x - obs_x)v\cos\theta + 2(p_y - obs_y)v\sin\theta \quad \text{(u 不出现!)}$$
$$\ddot{b} = \cdots + L_g L_f b \cdot u \quad \text{(u 出现在二阶导数)}$$

---

## 3. HOCBF: 高阶控制屏障函数

### 3.1 为什么需要 HOCBF

当相对度 > 1 时, 标准的单步 CBF 不够——需要构造**高阶** CBF。

### 3.2 HOCBF 构造

对相对度为 2 的系统, 定义两个屏障函数:

$$\psi_0(x) = b(x) \quad \text{(原始屏障)}$$
$$\psi_1(x) = \dot{\psi}_0(x) + p_1 \cdot \alpha_1(\psi_0(x))$$

其中 $\alpha_1$ 是 class-$\mathcal{K}$ 函数 (通常 $\alpha_1(r) = r$)。

**HOCBF 约束** (相对度 2):

$$\dot{\psi}_1(x) + p_2 \cdot \psi_1(x) \geq 0$$

展开:
$$L_f^2 b + L_g L_f b \cdot u + (p_1 + p_2)\dot{b} + p_1 p_2 \cdot b \geq 0$$

### 3.3 代码中的体现 (2D 机器人)

```python
# 原始屏障: 到障碍物的距离平方 - 安全半径平方
barrier = (px - obs_x)² + (py - obs_y)² - R²

# 一阶导数 (相对度1, u不出现)
barrier_dot = 2*(px-obs_x)*v*cos(θ) + 2*(py-obs_y)*v*sin(θ)

# 二阶李导数
Lf2b = 2*v²   # L_f² b (漂移项的贡献)
LgLfbu1 = -2*(px-obs_x)*v*sin(θ) + 2*(py-obs_y)*v*cos(θ)  # 对 ω (角速度) 的系数
LgLfbu2 = 2*(px-obs_x)*cos(θ) + 2*(py-obs_y)*sin(θ)       # 对 a (加速度) 的系数

# HOCBF 约束 (相对度2):
G = [-LgLfbu1, -LgLfbu2]   # QP 约束矩阵
h = Lf2b + (p1+p2)*barrier_dot + p1*p2*barrier  # QP 约束上界
```

### 3.4 一般 HOCBF 公式

对相对度 m:

$$\psi_0 = b(x)$$
$$\psi_i = \dot{\psi}_{i-1} + p_i \cdot \psi_{i-1}, \quad i=1,\ldots,m-1$$

约束: $\dot{\psi}_{m-1} + p_m \cdot \psi_{m-1} \geq 0$

---

## 4. BarrierNet 架构: 双分支 + QP

### 4.1 网络结构

```
输入: 图像序列 (T帧) + 车辆状态

  ┌─ CNN (5层卷积) ──────────────────────┐
  │  输入: (B, T, 3, H, W) 图像序列        │
  │  架构: [3→24→36→48→64→64] 卷积        │
  │  输出: (B×T, 64) 图像特征              │
  └──────────────────────────────────────┘
                    ↓
  ┌─ LSTM ──────────────────────────────┐
  │  输入: (B, T, 64) 时序特征            │
  │  隐藏: 64维                          │
  │  输出: (B×T, 64) 时序感知特征          │
  └──────────────────────────────────────┘
                    ↓
         ┌─────────┴─────────┐
         ↓                   ↓
  ┌─ q_mlp ─┐         ┌─ p_mlp ─┐
  │ [64→32  │         │ [64→32  │
  │  →32→2] │         │  →32→2] │
  │ 输出: q  │         │ 输出: p  │
  │ (参考控制)│        │ (CBF参数)│
  └─────────┘         └─────────┘
         ↓                   ↓
         └─────────┬─────────┘
                   ↓
  ┌─ QP 安全层 ──────────────────────────┐
  │  min  ½u² + q·u                       │
  │  s.t.  G(s, p)·u ≤ h(s, p)           │
  │        u_min ≤ u ≤ u_max              │
  │                                       │
  │  输入: q (参考控制), p (CBF参数),       │
  │        s (车辆状态), obs (障碍物)       │
  │  输出: u* (安全控制)                   │
  └──────────────────────────────────────┘
```

### 4.2 双分支的意义

| 分支 | 输出 | 物理含义 | 谁决定 |
|------|------|---------|--------|
| **q_head** | q | "NN 想做什么" (参考加速度, 参考角速度) | 模仿学习 |
| **p_head** | p | "安全约束应该多紧" (CBF 参数) | 端到端学习 |
| **QP 层** | u* | "在满足安全约束下, 最接近 q 的控制" | 凸优化 |

**关键洞察**: p 不是手工调的, 而是**NN 自己学的**! 网络学会了在不同场景下选择不同的保守程度。

### 4.3 训练信号

```python
# BarrierNet 使用模仿学习 (Imitation Learning)
loss = MSE(u_safe, u_expert)  # 让 QP 输出接近专家驾驶行为

# 不需要额外的安全损失!
# 因为 QP 层本身就保证输出满足 CBF 约束
# NN 学习输出一个"容易被 QP 接受"的 q
```

---

## 5. 可微分 QP: qpth 原理

### 5.1 为什么需要可微分 QP

```
训练时需要梯度反向传播:
  loss → u* → QP求解器 → q (参考控制) → NN权重

如果 QP 是不可微分的 "黑箱" → 梯度在此中断 → 无法训练 NN
```

### 5.2 KKT 隐式微分

QP 的标准形式:

$$\min_u \frac{1}{2}u^T Q u + q^T u \quad \text{s.t.} \quad Gu \leq h$$

**KKT 条件** (最优性的充要条件):

$$Qu^* + q + G^T\lambda = 0 \quad \text{(稳定性)}$$
$$\lambda \odot (Gu^* - h) = 0 \quad \text{(互补松弛)}$$
$$\lambda \geq 0, \quad Gu^* \leq h \quad \text{(原始/对偶可行)}$$

**隐式微分**: 在 KKT 点, 对参数 (Q, q, G, h) 求导:

$$\begin{bmatrix} Q & G^T \\ D(\lambda)G & D(Gu^* - h) \end{bmatrix} \begin{bmatrix} du^* \\ d\lambda \end{bmatrix} = -\begin{bmatrix} dQ \cdot u^* + dq + dG^T \cdot \lambda \\ D(\lambda) \cdot (dG \cdot u^* - dh) \end{bmatrix}$$

解这个线性系统 → $\frac{\partial u^*}{\partial q}$, $\frac{\partial u^*}{\partial h}$, ... → 梯度可以通过 QP 反向传播!

### 5.3 qpth 库

```python
from qpth.qp import QPFunction

# 可微分 QP (训练用)
u_safe = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
    Q,      # (B, n, n) 二次项矩阵
    q,      # (B, n) 线性项 (参考控制)
    G,      # (B, m, n) 不等式约束矩阵
    h,      # (B, m) 不等式约束上界
    e,      # (B, p, n) 等式约束矩阵 (通常为空)
    e       # (B, p) 等式约束上界
)

# QPFunction 内部用 Primal-Dual Interior Point Method
# 通过 KKT 隐式微分实现自动求导
# loss.backward() 时梯度自动通过 QP 层传播!
```

### 5.4 推理时

```python
# 推理时不需要可微分, 用 cvxopt 更稳定
from cvxopt import solvers
sol = solvers.qp(matrix(Q), matrix(q), matrix(G), matrix(h))
u_safe = sol['x']
```

---

## 6. 2D 机器人示例 (最简实现)

### 6.1 场景

```
机器人状态: [px, py, θ, v]   (位置x, 位置y, 朝向角, 速度)
控制: [ω, a]                (角速度, 加速度)
障碍物: 位于 (40, 15), 半径 R=6

目标: 从起点移动到终点, 同时避开障碍物
```

### 6.2 网络实现 (`models.py`)

```python
class BarrierNet(nn.Module):
    def __init__(self):
        # 共享主干: [5→128]
        self.fc1 = nn.Linear(5, 128)
        
        # q_head: 参考控制 [128→32→2]
        self.fc21 = nn.Linear(128, 32)
        self.fc31 = nn.Linear(32, 2)  # q = [ω_ref, a_ref]
        
        # p_head: CBF参数 [128→32→2]
        self.fc22 = nn.Linear(128, 32)
        self.fc32 = nn.Linear(32, 2)
        # p = 4*sigmoid(raw) → (0, 4) 范围, 保证正

    def forward(self, x):
        # 共享特征
        feat = F.relu(self.fc1(x))
        
        # 双分支
        q = self.fc31(F.relu(self.fc21(feat)))      # 参考控制
        p = 4 * sigmoid(self.fc32(F.relu(self.fc22(feat))))  # CBF参数
        
        # HOCBF 约束构建 (相对度2)
        barrier = (px-40)² + (py-15)² - 6²
        barrier_dot = 2*(px-40)*v*cos(θ) + 2*(py-15)*v*sin(θ)
        Lf2b = 2*v²
        LgLfbu1 = -2*(px-40)*v*sin(θ) + 2*(py-15)*v*cos(θ)
        LgLfbu2 = 2*(px-40)*cos(θ) + 2*(py-15)*sin(θ)
        
        G = [[-LgLfbu1, -LgLfbu2]]  # (B, 1, 2)
        h = Lf2b + (p1+p2)*barrier_dot + p1*p2*barrier  # (B, 1)
        
        # QP求解
        Q = eye(2)  # min ½u² + q·u
        u_safe = QPFunction()(Q, q, G, h, e, e)
        
        return u_safe  # 保证安全的 [ω, a]
```

### 6.3 对比: FCNet vs BarrierNet

```python
class FCNet(nn.Module):
    """普通全连接网络, 无 QP 层"""
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc21(x))
        return self.fc31(x)  # 直接输出 [ω, a], 无安全保证!
```

**区别**: FCNet 直接输出控制 → 可能撞障碍物。BarrierNet 通过 QP 修正 → 数学上保证避开。

---

## 7. 自动驾驶场景 (完整实现)

### 7.1 场景

```
CARLA 仿真器中的车辆跟随 + 避障:
  状态: [s, d, μ, v, δ, κ]
    s: 纵向位置, d: 横向偏移
    μ: 航向角误差, v: 速度
    δ: 方向盘转角, κ: 道路曲率

  控制: [a, ω]  (加速度, 方向盘转角速率)

  输入: 前视相机图像 (3通道, 时序)
```

### 7.2 三种 CBF 约束

```python
# 1. 避障 CBF (HOCBF, 相对度2)
barrier_obs = (s - obs_s)² + (d - obs_d)² - R²  # 到障碍物距离 > R

# 2. 车道保持左边界 CBF
barrier_left = threshold - d       # 不能太靠左

# 3. 车道保持右边界 CBF  
barrier_right = d + threshold      # 不能太靠右

# 三个约束堆叠:
G = [G_obs, G_left, G_right]  # (B, 3, 2)  ← 3个约束, 2个控制
h = [h_obs, h_left, h_right]  # (B, 3)
```

### 7.3 三种模型模式

```python
# 'deri' 模式: 输出速度+方向盘 + 安全加速度+方向盘速率
# NN 输出 [v, δ] 作为运动学参考, QP 输出 [a, ω] 满足动力学约束

# 'inte' 模式: 增量控制
# NN 输出 [a_ref, ω_ref], QP 从当前控制出发微调

# 'direct' 模式: 直接控制
# NN 输出 q, QP 直接输出 u = q + Δu
```

### 7.4 state_net (可选)

```python
# BarrierNet 可选择从图像中估计状态
if use_state_net:
    ds  = ds_mlp(features)   # 预测纵向距离差
    dd  = dd_mlp(features)   # 预测横向距离差
    mu  = mu_mlp(features)   # 预测航向角误差
    d   = d_mlp(features)    # 预测横向位置
    kappa = kappa_mlp(features)  # 预测曲率
    
    # 用 ground truth 约束预测 (tol=5%)
    ds = clamp(ds.detach(), gt_ds*0.95, gt_ds*1.05)
```

---

## 8. 完整数据流

### 8.1 训练数据流

```
┌─ 感知 ─────────────────────────────────────────────┐
│                                                     │
│  图像序列: (B, T, 3, H, W)                          │
│    ↓                                                │
│  CNN (5层): 每帧独立提取特征                         │
│    → (B×T, 64)                                      │
│    ↓                                                │
│  LSTM: 时序融合                                      │
│    → (B×T, 64)                                      │
│                                                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─ 双分支 ───────────────────────────────────────────┐
│                                                     │
│  q_mlp: [64→32→32→2]                               │
│    → q = [a_ref, ω_ref]  (NN 想要的加速度+角速度)     │
│                                                     │
│  p_mlp: [64→32→32→2]                               │
│    → p = [p1, p2]  (CBF参数, 控制安全约束紧度)       │
│                                                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─ CBF 约束构建 ──────────────────────────────────────┐
│                                                     │
│  避障:                                               │
│    barrier = ds² + dd² - R²                         │
│    barrier_dot = 2·ds·v·cos(μ+β)/(1-dκ) + 2·dd·v·sin(μ+β) │
│    Lf2b = ... (漂移项的二阶李导数)                    │
│    LgLfbu = [∂b̈/∂a, ∂b̈/∂ω]  (控制对二阶导的影响)     │
│    → G_obs, h_obs                                    │
│                                                     │
│  车道保持左/右:                                      │
│    barrier = ±d ± threshold                        │
│    → 相对度2 → 类似构造                               │
│    → G_left, h_left, G_right, h_right               │
│                                                     │
│  堆叠: G = [G_obs; G_left; G_right]                 │
│        h = [h_obs; h_left; h_right]                 │
│                                                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─ QP 求解 ──────────────────────────────────────────┐
│                                                     │
│  训练: qpth.QPFunction (可微分, KKT隐式微分)          │
│    u* = QP(Q=I, q, G, h)                            │
│                                                     │
│  推理: cvxopt.solvers.qp (数值稳定)                   │
│                                                     │
│  若 q 本身满足约束 → u* ≈ q (QP 不干预)               │
│  若 q 违反约束 → u* 是最接近 q 的安全控制              │
│                                                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─ 损失 ──────────────────────────────────────────────┐
│                                                     │
│  loss = MSE(u*, u_expert)                           │
│  模仿学习: 让 QP 输出尽量接近专家轨迹                  │
│                                                     │
│  梯度流: loss → u* → QPFunction → q → nn → ...     │
│         loss → u* → QPFunction → p → nn → ...     │
│                                                     │
│  NN 学到: 输出一个 q, 使得 QP 修正后 u* 接近专家       │
│  NN 也学到: 输出合适的 p, 让约束不过于激进/保守        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 8.2 推理数据流

```
摄像头图像 (实时)
  → CNN → 特征
  → LSTM → 时序特征
  → q_mlp → q (参考控制)
  → p_mlp → p (CBF 参数)
  → 构建 CBF 约束 (基于当前车辆状态 + 障碍物)
  → cvxopt QP 求解 → u_safe
  → 发送到车辆执行器
```

---

## 9. 与 SafePVC/SBC 的关系

### 9.1 对比

| | BarrierNet (CBF-QP) | SafePVC (SBC) |
|---|---|---|
| **安全机制** | 确定性约束 | 概率验证 |
| **时机** | 在线推理每步 | 离线训练时 |
| **保证类型** | "这个动作安全" | "系统 96.58% 概率永远安全" |
| **输出** | u_safe (安全控制) | B(s) (风险评分) + 概率界 |
| **核心数学** | CBF + QP 凸优化 | 上鞅 + IBP 形式化验证 |
| **梯度** | KKT 隐式微分 | 标准反向传播 |
| **可组合性** | ✅ 作为独立层插入 | ✅ 作为独立验证器 |

### 9.2 互补关系 (v3 的核心贡献)

```
训练阶段:
  SafePVC 训练 SBC → 验证控制器概率安全 → 96.58%

推理阶段:
  控制器输出 u_ref
    ↓
  BarrierNet QP 层 → u_safe (确定性地 CBF 安全)
    ↓
  执行
```

**为什么两者互补**:
- SBC 不能阻止单次坏动作 → QP 可以
- QP 不能提供全局概率保证 → SBC 可以
- SBC 在 OOD 时失效 → QP 独立于 NN 工作
- QP 需要 CBF 参数 → SBC 的 B(s) 可用来指导 (V3B/V3D)

---

## 10. 关键超参数

| 超参数 | 值 (自动驾驶) | 说明 |
|--------|-------------|------|
| CNN 层 | [3→24→36→48→64→64] | 5层卷积, kernel=5/3 |
| LSTM 隐藏 | 64 | 时序融合 |
| q_mlp | [64→32→32→2] | 参考控制头 |
| p_mlp | [64→32→32→2] | CBF 参数头 |
| p 范围 | 4×sigmoid (0~4) | 正且有界 |
| QP 求解器 | qpth (训练) / cvxopt (推理) | KKT微分 / 数值稳定 |
| 障碍物半径 R | 7.9m | 膨胀安全半径 |
| 车道CBF阈值 | 2.0m | 横向偏移限制 |
| model_type | deri | 输出 v,δ + QP→a,ω |
| 学习率 | 1e-3 | Adam |
| batch_size | 64 | |
| 图像尺寸 | 3×H×W | 前视相机 |

---

## 11. 总结

```
BarrierNet 的核心创新:
  
  1. 将 CBF 安全约束嵌入 NN 的 QP 层
     → NN 输出参考控制 + CBF参数
     → QP 修正为确定安全的控制
  
  2. 通过 KKT 隐式微分使 QP 可微分
     → 梯度可以通过 QP 层反向传播
     → NN 可以端到端训练
  
  3. HOCBF 处理相对度 > 1 的约束
     → 扩展到更复杂的动力学系统
  
  4. p 参数由 NN 学习 (不是手工调)
     → 自适应不同场景的保守程度

与 v3 实验的关系:
  - v2: 尝试将 BarrierNet 双分支嵌入 SafePVC → SBC 受损
  - v3: 将 QP 从训练中移除, 仅推理时使用 → 两全其美
  - V3A/V3D: 简化版 BarrierNet (只用 QP, 不用双分支训练)
```
