# 双重安全架构: SafePVC + BarrierNet 集成系统 — 完整技术文档

> 论文组合项目 — 从原理到实验的完整记录
> 2026-07-20

---

## 目录

- [第一部分: 背景与原理](#第一部分背景与原理)
  - [1. 项目目标](#1-项目目标)
  - [2. SafePVC 框架](#2-safepvc-框架)
  - [3. BarrierNet 框架](#3-barriernet-框架)
  - [4. 两个框架的互补性](#4-两个框架的互补性)
- [第二部分: 系统架构](#第二部分系统架构)
  - [5. 完整系统架构](#5-完整系统架构)
  - [6. 每个时间步详解](#6-每个时间步详解)
  - [7. 训练管线](#7-训练管线)
- [第三部分: 实验](#第三部分实验)
  - [8. v2 vs v3 设计演进](#8-v2-vs-v3-设计演进)
  - [9. 实验配置](#9-实验配置)
  - [10. 实验结果](#10-实验结果)
  - [11. 结果解读与结论](#11-结果解读与结论)
- [附录](#附录)

---

# 第一部分: 背景与原理

---

## 1. 项目目标

### 1.1 核心问题

**基于视觉的神经网络控制器无法提供安全保证。**

一个典型的端到端视觉控制系统:
```
相机图像 → NN感知 → NN控制 → 动作
```
这个 NN 是一个黑箱——你不知道它什么时候会做出危险决策。

### 1.2 解决方案: 双重安全

本项目将两种互补的安全机制结合起来:

```
┌────────────────────────────────────────────────────────┐
│                  双重安全架构                            │
│                                                        │
│  安全层 1: SBC (随机屏障证书)                            │
│    来源: SafePVC 论文                                    │
│    功能: 离线验证, 提供概率安全保证                       │
│    输出: "控制器以 96.58% 概率永远安全"                   │
│                                                        │
│  安全层 2: CBF-QP (控制屏障函数 + 二次规划)               │
│    来源: BarrierNet 论文                                 │
│    功能: 在线防护, 每步确定性地修正不安全动作              │
│    输出: "这个具体动作经过数学验证, 一定安全"              │
│                                                        │
│  组合优势: 全局概率信心 + 局部确定性保障                  │
└────────────────────────────────────────────────────────┘
```

### 1.3 AEBS 应用场景

实验基于 **自动紧急制动系统 (AEBS)**:

```
场景: 自车跟随前车

状态: s = [d, v]
  d: 到前车距离 (5-16m)
  v: 自车速度 (0-3 m/s)

控制: u ∈ [-3, 3] m/s² (加速度)

安全目标: 保持安全时距 t_gap = 1.5s
  即: d ≥ v × 1.5  (不能太近太快)

动力学 (离散, dt=0.05s):
  d_{t+1} = d_t - v_t · dt
  v_{t+1} = clip(v_t - u_t · dt, 0, 3)
```

---

## 2. SafePVC 框架

### 2.1 概述

SafePVC = **S**tochastic **B**arrier **C**ertificate (随机屏障证书)

核心思想: 训练一个神经网络函数 B(s), 作为系统的"安全评分", 然后用 IBP 形式化验证它满足数学条件, 从而导出概率安全下界。

### 2.2 四个组件

```
SafePVC 由四个神经网络组成:

┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐
│ gen_net  │ → │state_net │ → │controller_net│    │ l_model  │
│ 图像生成 │    │ 状态估计 │    │  PPO控制器   │    │ SBC 函数 │
│ (frozen) │    │ (frozen) │    │  (可微调)    │    │ (★训练)  │
└──────────┘    └──────────┘    └─────────────┘    └──────────┘
    ↓               ↓                ↓                  ↓
  z+d → img     img → d̂         [d̂,v] → u         [d,v] → B(s)
```

| 组件 | 输入 | 输出 | 架构 | 参数状态 |
|------|------|------|------|:------:|
| gen_net | z(4维) + d(1维) | img(32×32=1024) | [5→256→512→1024]+tanh | ❄ frozen |
| state_net | img(1024) | d̂(1维) | [1024→256→64→1]+LayerNorm+ReLU | ❄ frozen |
| controller_net | [d̂, v](2维) | u(1维) | SB3 PPO: [2→256→256]→[256→1] | 🔧 微调 |
| l_model (SBC) | [d, v](2维) | B(s)(1维,≥0) | [2→32→16→8→1]+tanh+square | ★ 训练 |

### 2.3 SBC 的四个条件

SBC 函数 B(s) 必须满足:

| # | 条件 | 代码实现 |
|---|------|---------|
| 1 | $B(s) \geq 0$ | `square_output=True` 自动保证 |
| 2 | $B(s_0) \leq 1$ (初始区域) | `relu(max(B_init) - 1)` |
| 3 | $B(s_u) \geq 1/(1-p)$ (不安全区域) | `relu(20 - min(B_unsafe))` for p=0.95 |
| 4 | $\mathbb{E}[B(s_{t+1})] \leq B(s_t)$ (期望递减) | `max(E[B_next] - B + 0.1, 0) × 1000` |

### 2.4 IBP 形式化验证

IBP = Interval Bound Propagation (区间边界传播)

```
输入: 网格单元的区间 [d_low, d_high] × [v_low, v_high]
  ↓
通过网络逐层传播区间 (不是单点!)
  ↓
输出: B(s) 在该单元内的严格上下界 [B_low, B_high]

在 100×100 网格 (10000个单元) 上验证:
  ∀单元: E[B(s_next)] ≤ B(s) - lipschitz·K ?
  
全部通过 → 通过率 > 99.9% → 验证成功
→ 概率下界 = 1 - B_init_ub / B_unsafe_lb
```

### 2.5 训练流程

```
Loop.run() 主循环 (最多100轮):

  每轮迭代:
    ┌─ Step A: train('l', 10 epochs) ───────────────────┐
    │  训练 SBC 网络 B(s)                                  │
    │  损失 = martingale × 1000 + region + lip             │
    │  16条三角噪声轨迹估计 E[B(s_next)]                    │
    └────────────────────────────────────────────────────┘
              ↓
    ┌─ Step B: verifier.check_dec_cond(K=1.2) ──────────┐
    │  IBP 在 100×100 网格上严格验证期望递减条件            │
    │  统计违反单元数                                       │
    └────────────────────────────────────────────────────┘
              ↓
    ┌─ Step C: 如果通过 → 计算概率界 ────────────────────┐
    │  prob = 1 - (ub_init-domain_min)/(lb_unsafe-domain_min) │
    │  96.58% → 保存模型                                   │
    └────────────────────────────────────────────────────┘
              ↓
    ┌─ Step D: train('p', 1 epoch) ─────────────────────┐
    │  用验证反例训练控制器                                  │
    │  损失 = martingale × 10 + MSE(teacher) × 10 + lip   │
    │  128条噪声轨迹 (更多样本)                              │
    └────────────────────────────────────────────────────┘
              ↓
         下一轮迭代...
```

---

## 3. BarrierNet 框架

### 3.1 概述

BarrierNet = **CBF-QP 安全层** (Control Barrier Function + Quadratic Program)

核心思想: 在 NN 控制器输出后面接一个 QP 凸优化层。QP 在满足 CBF 安全约束的前提下, 寻找最接近原始输出的安全控制。

### 3.2 CBF 数学

**屏障函数**: $b(s)$ — $b(s) \geq 0$ 表示状态 s 是安全的

**CBF 条件**: 存在 p > 0, 使得:
$$\dot{b}(s) + p \cdot b(s) \geq 0$$

在 AEBS 中: $b(s) = d - v \cdot t_{gap}$

$$\dot{b} = -v + t_{gap} \cdot u$$

离散 QP 约束: $-t_{gap} \cdot u \leq -v + p \cdot (d - v \cdot t_{gap})$

### 3.3 QP 安全层

```
QP 求解:
  min  ½(u - u_ref)²          ← 尽量接近原始控制
  s.t. -t_gap·u ≤ -v + p·b    ← CBF 安全约束
       -3 ≤ u ≤ 3              ← 控制限幅

输出: u_safe — 在满足所有约束下最接近 u_ref 的控制
```

### 3.4 BarrierNet 双分支架构

```
输入: 图像序列 + 车辆状态

  ┌─ CNN ────┐
  │ 5层卷积   │ → 图像特征 (64维)
  └──────────┘
       ↓
  ┌─ LSTM ───┐
  │ 时序融合  │ → 感知特征 (64维)
  └──────────┘
       ↓
  ┌────┴────┐
  ↓         ↓
q_mlp     p_mlp
[64→32    [64→32
 →32→2]    →32→2]
  ↓         ↓
  q         p
(参考控制) (CBF参数)
  └────┬────┘
       ↓
  ┌─ QP 层 ───────────────────────┐
  │ min ½u² + q·u                 │
  │ s.t. G(s,p)·u ≤ h(s,p)        │
  │      u ∈ [u_min, u_max]       │
  └───────────────────────────────┘
       ↓
     u_safe (保证安全的控制)
```

### 3.5 可微分 QP (qpth)

BarrierNet 的关键创新: QP 层可通过 KKT 隐式微分反向传播梯度。

```
训练时:
  loss = MSE(u_safe, u_expert)
  loss.backward()  →  梯度自动通过 QP 层
  → 更新 q_mlp, p_mlp, CNN, LSTM

这使 NN 学会:
  - 输出一个 q, 使得 QP 修正后仍接近专家
  - 输出合适的 p, 让约束不过于激进或保守
```

---

## 4. 两个框架的互补性

### 4.1 核心洞察

SBC 和 CBF-QP 解决的是**不同层次**的安全问题:

| 维度 | SBC (SafePVC) | CBF-QP (BarrierNet) |
|------|:---:|:---:|
| **问题** | "系统整体有多安全?" | "这个具体动作安全吗?" |
| **保证** | 概率性 (96.58%) | 确定性 (100%) |
| **时域** | 无限时域 | 当前单步 |
| **时机** | 离线训练时 | 在线推理每步 |
| **输出** | 概率界 | u_safe |
| **盲区** | 不能阻止单次坏动作 | 不能提供全局概率保证 |

### 4.2 互补矩阵

```
失效模式                        SBC 能防?    QP 能防?    组合能防?
────────────────────────────────────────────────────────────
NN 在训练分布内偶然失误           ✅          ✅           ✅✅
NN 遭遇 OOD 输入                 ❌          ✅           ✅
攻击者修改 NN 权重               ❌          ✅           ✅
长期缓慢漂移到危险               ✅          ❌           ✅
QP 参数调得太松                  N/A         ❌           ✅ (SBC监测)
QP 不可行 (无安全控制)            N/A         ⚠️           ✅ (SBC预期内)
传感器噪声                      ❌ (界松动)   ✅           ✅
```

---

# 第二部分: 系统架构

---

## 5. 完整系统架构

### 5.1 训练阶段 (v3 设计: QP 不在训练中)

```
┌──────────────────────────────────────────────────────────────┐
│                  V3 训练阶段 (无 QP 干扰!)                     │
│                                                              │
│  状态 s_t 从 100×100 均匀网格采样                              │
│      ↓                                                       │
│  扰动: s_pert = s_t + δ × U([-0.5, 0.5])                     │
│      ↓                                                       │
│  ┌─ VCLS 前向 (★ 直接模式, 无 QP!) ──────────────────┐       │
│  │ gen_net(z, d_norm) → img (32×32)         ❄        │       │
│  │ state_net(img) → d̂                       ❄        │       │
│  │ controller_net([d̂, v]) → u               🔧       │       │
│  └───────────────────────────────────────────────────┘       │
│      ↓                                                       │
│  动力学: s_{t+1} = f(s_pert, u) + 16条三角噪声               │
│      ↓                                                       │
│  ┌─ SBC 训练 ───────────────────────────────────────┐       │
│  │ l_model(s_pert) → B(s_t)                        │       │
│  │ l_model(s_{t+1}[16]) → B(s_{t+1})[16]           │       │
│  │ E[B] = mean(B(s_{t+1}), dim=1)                  │       │
│  │ loss = martingale + region + lip                 │       │
│  └──────────────────────────────────────────────────┘       │
│      ↓                                                       │
│  IBP 验证 → 概率界 → 控制器微调 (反例) → 下一轮               │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 推理阶段 (v3 设计: QP 安全盾激活)

```
┌──────────────────────────────────────────────────────────────┐
│                V3 推理阶段 (QP 安全盾激活)                      │
│                                                              │
│  对每个时间步:                                                 │
│                                                              │
│  ① 感知:                                                     │
│     gen_net(z, d) → img → state_net(img) → d̂                │
│                                                              │
│  ② 控制器:                                                    │
│     u_ref = controller_net([d̂, v])                           │
│                                                              │
│  ③ ★ QP 安全盾:                                              │
│     ┌───────────────────────────────────────┐                │
│     │ 计算 CBF 参数 p:                      │                │
│     │   V3A: p = 2.0 (固定)                 │                │
│     │   V3D: p = f(b(s)) (Barrier 自适应)   │                │
│     │                                       │                │
│     │ QP: min ½(u-u_ref)²                  │                │
│     │     s.t. -t_gap·u ≤ -v + p·b(s)      │                │
│     │          -3 ≤ u ≤ 3                   │                │
│     │                                       │                │
│     │ IF |u_safe - u_ref| > 0.01:           │                │
│     │   → 干预! 输出 u_safe                 │                │
│     │ ELSE:                                  │                │
│     │   → 不干预, 输出 u_ref                │                │
│     └───────────────────────────────────────┘                │
│                                                              │
│  ④ 执行: s_{t+1} = f(s_t, u_safe)                            │
│                                                              │
│  ⑤ 记录: u_safe, u_ref, p, B(s), intervened, margin         │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 每个时间步详解

### 6.1 训练时 (一个 batch 的 SBC 训练步)

```
输入:
  y_batch: [B, 2]  — 从 100×100 网格随机采样的归一化状态 [d_norm, v]
  z_batch: [B, 4]  — 随机隐向量, U(-1, 1)

━━━ 第1步: 状态扰动 ━━━
  s_rand = rand(B, 2) - 0.5         # U([-0.5, 0.5])
  y_pert = y_batch + grid_delta × s_rand
  → 输出: [B, 2] 扰动后的归一化状态

━━━ 第2步: VCLS 前向 ━━━
  # gen_net: 隐向量 + 距离 → 图像
  img = gen_net(z_batch, y_pert[:,0:1])     # [B, 1024]
  
  # state_net: 图像 → 估计距离
  d_est = state_net(img)                     # [B, 1]
  
  # controller_net: [估计距离, 真实速度] → 控制
  ctrl_in = cat([d_est, y_pert[:,1:2]], dim=1)  # [B, 2]
  u = controller_net(ctrl_in)                # [B, 1]

  → 输出: u [B, 1]

━━━ 第3步: 动力学 + 噪声 ━━━
  # 确定性下一状态 (归一化坐标)
  d_next = y_pert[:,0] - y_pert[:,1] × 0.05
  v_next = clip(y_pert[:,1] - u × 0.05, 0, 3)
  s_next_det = stack([d_next, v_next], dim=1)  # [B, 2]
  
  # 加噪声 (16条独立轨迹)
  noise = triangular(B, 16, 2) × noise_scale  # [B, 16, 2]
  s_next_random = s_next_det.unsqueeze(1) + noise  # [B, 16, 2]

  → 输出: s_next_random [B, 16, 2]

━━━ 第4步: SBC 计算 ━━━
  # 当前状态 B 值
  B_now = l_model(y_pert)                  # [B, 1]
  
  # 下一状态 B 值 (16条轨迹)
  B_next = l_model(s_next_random.reshape(-1, 2))  # [B×16, 1]
  B_next = B_next.reshape(B, 16)                   # [B, 16]
  
  # 期望 (16条轨迹平均)
  E_B_next = B_next.mean(dim=1)            # [B]

  → 输出: B_now [B], E_B_next [B]

━━━ 第5步: 损失计算 ━━━
  # Martingale: E[B_next] 不能大于 B_now
  mart = mean(relu(E_B_next - B_now + 0.1)) × 1000
  
  # Region: 初始区 ≤ 1, 不安全区 ≥ 20
  region_init = relu(max(B(init_samples)) - 1.0)
  region_unsafe = relu(20.0 - min(B(unsafe_samples)))
  region = region_init + region_unsafe
  
  # Lipschitz: ||∇B||₂ ≤ 4.0
  lip = relu(||grad(B, s)||₂ - 4.0).mean()
  
  total_loss = mart + 0.001 × lip + region

━━━ 第6步: 反向传播 ━━━
  total_loss.backward()
  clip_grad(l_model.params, 5.0)
  l_optimizer.step()
  → 仅更新 l_model (SBC网络) 参数
  → gen_net, state_net 保持 frozen
```

### 6.2 推理时 (单个时间步, V3A 配置)

```
输入:
  当前状态: d (真实距离, 米), v (真实速度, m/s)
  隐向量: z = zeros(4)  (确定性推理)

━━━ 第1步: 归一化 ━━━
  d_norm = d / std1

━━━ 第2步: 控制器前向 ━━━
  img = gen_net(z, d_norm)           # [1, 1024]
  d_est = state_net(img)             # [1, 1]
  u_ref = controller_net([d_est, v]) # [1, 1] → 标量

━━━ 第3步: QP 安全盾 ━━━
  # 计算 barrier
  b = d - v × 1.5
  
  # V3A: 固定 p
  p = 2.0
  
  # 构建 CBF 约束
  G_cbf = -1.5      # -t_gap
  h_cbf = -v + 2.0 × b
  
  # QP: min ½(u-u_ref)² s.t. G·u ≤ h, -3≤u≤3
  # 解析解: u_safe = clip(u_ref, lower_bound, upper_bound)
  
  # 约束给出的 bounds:
  # G_cbf × u ≤ h_cbf → -1.5×u ≤ h_cbf → u ≥ h_cbf/(-1.5)
  lower = max(-3.0, h_cbf / G_cbf)   # 因为 G_cbf < 0
  upper = 3.0
  u_safe = clip(u_ref, lower, upper)

━━━ 第4步: 判断是否干预 ━━━
  intervened = |u_safe - u_ref| > 0.01

━━━ 第5步: 执行动力学 ━━━
  d_next = d - v × 0.05
  v_next = clip(v - u_safe × 0.05, 0, 3)

  → 输出: d_next, v_next (下一状态)
  → 记录: u_ref, u_safe, p, intervened, margin
```

---

## 7. 训练管线

### 7.1 训练顺序

```
阶段 1: 预备训练 (独立, SBC之前完成)

  ┌─ cGAN 训练 gen_net ────────────────────┐
  │  数据集: Downsampled.h5                  │
  │  任务: 条件图像生成                      │
  │  输出: mlp_supervised.pth               │
  └─────────────────────────────────────────┘
              ↓
  ┌─ 监督学习训练 state_net ───────────────┐
  │  输入: gen_net 生成的图像               │
  │  标签: 真实距离                          │
  │  损失: MSE + Lipschitz 正则化           │
  │  输出: state_net_trained.pth            │
  └─────────────────────────────────────────┘
              ↓
  ┌─ PPO 强化学习训练 controller ──────────┐
  │  环境: AebsEnv (Gymnasium)              │
  │  算法: PPO (SB3)                        │
  │  步数: 200,000                          │
  │  输出: best_model.zip                   │
  └─────────────────────────────────────────┘

阶段 2: SBC 训练 + 验证 (核心)

  ┌─ VTLearner 初始化 ─────────────────────┐
  │  加载: gen_net, state_net, PPO policy   │
  │  初始化: l_model (随机权重)              │
  │  优化器: Adam(lr=3e-3) for l_model      │
  │         Adam(lr=5e-2) for controller     │
  └─────────────────────────────────────────┘
              ↓
  ┌─ Loop.run() ───────────────────────────┐
  │  for iter in range(100):                 │
  │    train('l', 10)  # 训练SBC            │
  │    verify()         # IBP验证            │
  │    train('p', 1)   # 微调控制器         │
  │  → 最佳概率界: 96.58%                   │
  │  → 保存: l_model (SBC), p_net (修改后)   │
  └─────────────────────────────────────────┘

阶段 3: QP 安全盾部署 (v3 新增, 无需额外训练)

  ┌─ 构建 QP 盾 ───────────────────────────┐
  │  V3A: FixedQPShield(p=2.0, t_gap=1.5)   │
  │  V3D: BarrierModulatedQPShield(         │
  │         p_min=0.5, margin_scale=2.0)     │
  │  → 直接插入推理管线                      │
  │  → 不需要任何训练!                       │
  └─────────────────────────────────────────┘
```

### 7.2 关键训练技巧

| 技巧 | 说明 |
|------|------|
| **三角噪声** | 用 triangular() 代替高斯 — 有界 → IBP 可严格包围 |
| **16/128 条轨迹** | SBC 训练用 16 条, 控制器训练用 128 条 (更准确) |
| **s_next 梯度阻断** | controller → s_next → SBC 的梯度被 detach() 阻断 |
| **Lipschitz 正则化** | 限制 B(s) 和 controller 的梯度范数 → IBP 包围更紧 |
| **CEGIS** | 验证失败的反例 → 控制器训练数据 → 下一轮验证 |

---

# 第三部分: 实验

---

## 8. v2 vs v3 设计演进

### 8.1 v2: QP 嵌入训练 (失败)

```
v2 设计:
  训练循环包含 QP 层:
    controller → q_head (u_ref) + p_head (CBF参数)
    → QP → u_safe → 动力学 → SBC 验证

问题:
  QP 层增加了闭环系统的非线性
  → Lipschitz 常数增大
  → IBP 包围变松 (更保守)
  → SBC 概率界从 95.8% 降至 87.9% (-8pp!)
```

### 8.2 v3: QP 仅推理 (当前方案)

```
v3 设计:
  训练循环: controller → u (直接输出) → SBC 验证
  推理阶段: controller → u_ref → QP → u_safe

优势:
  ✓ SBC 训练不受 QP 干扰 → 概率界 96.58%
  ✓ 推理时仍有 QP 保护
  ✓ 模块化设计, 可以独立替换
```

---

## 9. 实验配置

### 9.1 五种对比配置

| 代号 | QP训练 | QP推理 | p参数来源 | 代码 | 状态 |
|------|:---:|:---:|------|------|:---:|
| **B** | ✗ | ✗ | N/A | 原始 PPO controller | 基准 |
| **V2** | ✓ | ✓ | NN 学习 | `qp_controller.py` | ❌ |
| **V3A** | ✗ | ✓ | **固定 p=2.0** | `FixedQPShield` | ✅ **最优** |
| **V3B** | ✗ | ✓ | SBC 调制 p=f(B(s)) | `SBCModulatedQPShield` | ❌ |
| **V3C** | ✗ | ✓ | 训练 p-network | `TrainedQPShield` | △ |
| **V3D** | ✗ | ✓ | **Barrier调制 p=f(b(s))** | `BarrierModulatedQPShield` | ✅ **自适应** |

### 9.2 关键参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| t_gap | 1.5s | 安全时距 |
| dt | 0.05s | 离散时间步 |
| CBF p (V3A最优) | 2.0 | 固定值, 平衡安全与干预 |
| p_min (V3D最优) | 0.5 | Barrier调制最小p |
| p_max | 4.0 | CBF参数上限 |
| margin_scale (V3D) | 2.0 | 调制平滑度 |
| u 范围 | [-3, 3] m/s² | 控制限幅 |
| SBC 架构 | [2,32,16,8,1] | tanh + square |
| SBC 训练 | 120 轮, Adam(lr=3e-3) | 概率界 96.58% |
| 控制器微调 | Adam(lr=1e-2) | 仅 controller_net |

### 9.3 测试场景

| 类型 | 数量 | 距离范围 | 速度范围 | 噪声 |
|------|:---:|---------|---------|------|
| safe | 15 | 10-16m | 0.5-2.0m/s | 0 |
| following | 15 | 6-10m | 0.3-1.5m/s | 0 |
| moderate | 15 | 7-12m | 1.0-2.5m/s | 0 |
| approaching | 15 | 8-14m | 0.5-1.5m/s | 0 |
| dangerous | 15 | 5.0-6.5m | 2.0-3.0m/s | 0-2.0 |
| dangerous_noisy | 15 | 5.5-7.5m | 2.0-3.0m/s | 1.0-3.0 |
| high_noise | 15 | 7-12m | 1.5-2.5m/s | 2.0-5.0 |
| extreme | 15 | 5.0-6.0m | 2.5-3.0m/s | 0-3.0 |

---

## 10. 实验结果

### 10.1 SBC 训练结果

| 指标 | 值 |
|------|-----|
| 训练轮数 | 120 |
| 训练时间 | 413s (~7分钟) |
| **概率下界** | **96.58%** |
| B(s) 范围 | [0.022, 35.569] |
| B(s) 均值 | 26.656 |

### 10.2 主要对比结果 (120场景 × 3次运行)

| 配置 | 干预率 | CBF安全率 | CBF违反率 | 平均p | 平均裕度 |
|------|:-----:|:--------:|:--------:|:----:|:------:|
| Baseline | N/A | N/A | N/A | N/A | 5.89 |
| **V3A p=2.0** ✨ | **46.4%** | **100.0%** | **0.0%** | 2.00 | 10.68 |
| V3A p=4.0 | 46.7% | 100.0% | 0.0% | 4.00 | 22.49 |
| **V3D p_min=0.5** ✨ | **49.7%** | **100.0%** | **0.0%** | 0.93 | 2.93 |
| V3A p=0.5 | 53.9% | 99.7% | 0.3% | 0.50 | 1.94 |
| V3C Trained-p | 54.7% | 100.0% | 0.0% | 0.64 | 2.16 |
| V3B p_min=0.1 | 86.1% | 98.3% | **1.7%** | 0.24 | 1.80 |
| V3D p_min=0.1 | 83.3% | 99.4% | 0.6% | 0.58 | 1.93 |

### 10.3 按场景类型 (V3A p=2.0)

| 场景类型 | V3A 干预率 | 安全率 | 解读 |
|---------|:--------:|:----:|------|
| extreme | **100%** | 100% | 最危险场景, 每步都介入 ✅ |
| dangerous | **86.7%** | 100% | 高风险, 高频介入 |
| dangerous_noisy | **75.6%** | 100% | 噪声叠加, 适度介入 |
| high_noise | **42.2%** | 100% | 噪声主导, 选择性介入 |
| approaching | **33.3%** | 100% | 正常行驶, 低频介入 |
| safe | **28.9%** | 100% | 安全区域, 尽量不干预 |
| moderate | **2.2%** | 100% | 中距低速, 几乎不介入 |
| following | **2.2%** | 100% | 跟车低速, 几乎不介入 |

### 10.4 受控实验: 故障控制器

| 初始状态 | 坏控制器 | Baseline | V3A QP |
|---------|:------:|----------|:------:|
| d=10.0, v=2.0 | 始终 u=-2.0 | ❌ 39步违例 | ✅ 安全 |
| d=8.0, v=2.0 | 始终 u=-2.0 | ❌ 26步违例 | ✅ 安全 |
| d=12.0, v=2.5 | 始终 u=-2.5 | ❌ 51步违例 | ✅ 安全 |

### 10.5 v2 vs v3 对比

| 指标 | v2 (QP训练) | v3 (V3A p=2.0) |
|------|:----------:|:-------------:|
| SBC 概率界 | 87.9% | **96.6% (+8.7pp)** |
| 运行时安全 | ✅ | ✅ |
| CBF 安全率 | ~99.7% | **100%** |
| 训练时间 | +52% | 同 baseline |
| 模块化 | 耦合 | **解耦** |
| 需要调参 | λ_cbf, lr | **仅 p 值** |

---

## 11. 结果解读与结论

### 11.1 核心发现

**1. V3A (固定 p=2.0) 是最优配置**
- 100% CBF 安全率, 最低干预率 (46.4%)
- 简单、可靠、无需训练
- 危险场景 87-100% 干预, 安全场景仅 2-29% 干预
- 理想的安全机制: 该出手时才出手

**2. V3D (Barrier 调制 p_min=0.5) 是最优自适应方案**
- 同样 100% CBF 安全率
- 根据 barrier b(s) 自动调节 p: 危险时紧, 安全时松
- 不需要手工选 p 值, 物理含义清晰

**3. V3B (SBC 调制) 效果不佳**
- SBC B(s) 值在所有测试状态都 ≈ 25-35
- sigmoid((25-5)/0.5) ≈ 1 → p ≈ p_max → 无法区分
- 根因: SBC 只在初始区和不安全区有训练约束, 中间区域 B(s) 自由浮动

**4. v3 成功实现了解耦**
- SBC 训练不受 QP 干扰 → 概率界保持 96.58% (vs v2 的 87.9%)
- QP 在推理时独立工作 → 提供确定性 CBF 安全保证

### 11.2 为什么 PPO 控制器已经很安全?

在这个简单的 1D AEBS 任务中, PPO 学到了"完美"的安全策略。即使对抗噪声也很难让它输出危险动作。QP 在这个场景中更像是"安全带"——平时系着不影响驾驶, 但万一出事就是最后的保护。

受控实验证明: **当控制器真的"发疯"时, QP 100% 阻止违例**。

### 11.3 两个安全指标的区分

| 指标 | 含义 | 本次实验中 |
|------|------|:--------:|
| **CBF 违反** | QP 约束在单步是否满足 | 1-2% (某些配置) |
| **Barrier 变负** | 下一步是否物理进入不安全 | 总是 0% (dt=0.05 太短) |

CBF 违反是可观测且 QP 可直接消除的安全风险。Barrier 变负在单步测试中物理不可能, 需要多步累积。

### 11.4 实验结论

1. **SBC + CBF-QP 双重安全架构可行且互补**: SBC 提供 96.58% 全局概率保证, QP 提供每步确定性的 CBF 安全

2. **QP 必须与 SBC 训练解耦**: v2 的教训 — QP 嵌入训练导致 SBC 界降 8pp。v3 的解耦设计两全其美

3. **V3A p=2.0 是生产部署推荐配置**: 简单、100%安全、最小干预

4. **V3D Barrier 调制是自适应场景的最优选择**: 自动根据状态危险程度调节保护力度

5. **受控实验证明 QP 机制有效**: 故障控制器 100% 被 QP 阻止

---

## 附录

### A. 文档索引

| 文档 | 内容 |
|------|------|
| `sbc_deep_dive.md` | SafePVC/SBC 原理、架构、数据流深度解析 |
| `barriernet_deep_dive.md` | BarrierNet/CBF-QP 原理、架构、数据流深度解析 |
| `project_structure.md` | 完整项目结构、代码文件说明、依赖关系 |
| `v3_improved_results.md` | 改进版完整实验结果 |
| `qp_shield_benefit_report.md` | QP 价值验证详细报告 |
| `sweep_results.md` | 参数扫描 26 配置 × 2 运行结果 |
| ★ **本文档** | 从原理到实验的完整技术文档 |

### B. 关键代码文件

| 文件 | 功能 |
|------|------|
| `artical-F122/Aebs/VT/train.py` | VTLearner — SBC + Controller 训练 |
| `artical-F122/Aebs/VT/verify.py` | VTVerifier — IBP 形式化验证 |
| `artical-F122/Aebs/VT/loop.py` | Loop — 主训练循环 |
| `src/models/cbf_constraints.py` | CBF 约束数学实现 |
| `src/models/sbc_modulated_qp.py` | V3A/V3B QP 盾 |
| `src/models/qp_p_network.py` | V3C/V3D QP 盾 + p-network 训练 |
| `src/eval/train_improved_sbc.py` | 改进 SBC 训练 (120轮, 96.58%) |
| `src/eval/v3_full_evaluation.py` | 完整评估脚本 |
| `src/eval/qp_benefit_experiments.py` | QP 价值证明实验 |
| `BarrierNet/Driving/models/barrier_net.py` | BarrierNet 完整实现 (CNN+LSTM+QP) |
| `BarrierNet/2D_Robot/models.py` | BarrierNet 最简实现 |

### C. 运行命令

```bash
cd /root/paper-combination/artical-F122

# 训练 SBC
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/train_improved_sbc.py

# 完整评估
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/v3_full_evaluation.py

# QP 价值实验
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/qp_benefit_experiments.py
```
