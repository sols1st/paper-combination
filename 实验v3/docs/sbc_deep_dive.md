# SafePVC 论文深度解析: SBC原理、架构、数据流

> 论文: *Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems*
> 基于 artical-F122 代码库的完整分析

---

## 目录

1. [核心问题](#1-核心问题)
2. [SBC 数学定义与四个条件](#2-sbc-数学定义)
3. [完整系统架构](#3-完整系统架构)
4. [gen_net: cGAN 图像生成器](#4-gen_net)
5. [state_net: 状态估计网络](#5-state_net)
6. [PPO 控制器](#6-ppo-控制器)
7. [完整项目结构](#7-完整项目结构)
8. [SBC 网络: B(s) 的设计](#9-sbc-网络)
9. [训练流程: VTLearner](#10-训练流程)
10. [形式化验证: VTVerifier + IBP](#11-形式化验证)
11. [主训练循环: Loop](#12-主训练循环)
12. [完整数据流 (端到端)](#13-完整数据流)
13. [环境设定: AEBS 场景](#14-环境设定)
14. [关键超参数](#15-关键超参数)
15. [总结: SBC 做了什么](#16-总结)

---

## 1. 核心问题: 视觉控制器的安全保证

### 1.1 要解决的问题

SafePVC 针对的是**基于视觉的神经网络控制系统** (Vision-based NNCS):

```
相机 → gen_net → 图像 → state_net → 状态估计 → controller_net → 控制动作 u
```

这种端到端 NN 系统的问题: **无法提供形式化的安全保证**——你不知道它什么时候会做出危险决策。

### 1.2 SafePVC 的解决方案

引入 **SBC (随机屏障证书, Stochastic Barrier Certificate)**:

- SBC 是一个函数 $B(s): \mathbb{R}^n \rightarrow \mathbb{R}_{\geq 0}$
- 如果存在这样的 $B(s)$ 满足四个条件, 则系统**以概率保证**不进入不安全区域
- $B(s)$ 是一个**神经网络** (MLP), 通过训练+形式化验证获得

**核心思想**: 用神经网络学习一个 "安全评分函数" B(s), 然后用 IBP 严格验证它确实满足 SBC 的数学条件。训练和验证交替进行, 直到 IBP 验证通过。

### 1.3 安全保证的层级

```
SBC 提供的保证:
  "控制器从初始区域出发, 以至少 96.58% 的概率, 
   在无限时域内, 永远不进入不安全区域"

这个保证的层次:
  概率性 ← 不是 100%, 但可量化 (96.58%)
  无限时域 ← 不是有限步, 而是永远
  形式化 ← IBP 严格证明, 不是采样估计
```

---

## 2. SBC 数学定义与四个条件

### 2.1 SBC 的定义

SBC 是一个函数 $B: \mathcal{S} \rightarrow \mathbb{R}_{\geq 0}$。如果它满足以下四个条件, 则系统从初始区域 $\mathcal{S}_0$ 出发, 进入不安全区域 $\mathcal{S}_u$ 的概率 $\leq p$。

### 2.2 四个条件

#### 条件 1: 非负性

$$B(s) \geq 0 \quad \forall s \in \mathcal{S}$$

**实现**: MLP 使用 `square_output=True` + `tanh`, 最终输出 $x^2 \geq 0$, 自动满足。

#### 条件 2: 初始区域上界

$$B(s) \leq 1 \quad \forall s \in \mathcal{S}_0$$

**物理含义**: 在"正常起点", 风险评分不超过 1。

**实现**: `region_loss_init = relu(max(B_init) - 1.0)`

#### 条件 3: 不安全区域下界

$$B(s) \geq \frac{1}{1-p_{target}} \quad \forall s \in \mathcal{S}_u$$

例如 $p_{target} = 0.95 \rightarrow B(s) \geq 20$。

**物理含义**: 在"绝对危险"的区域, 风险评分必须足够大。

**实现**: `region_loss_unsafe = relu(1/(1-p) - min(B_unsafe))`

#### 条件 4: 期望递减 (上鞅性质)

$$\mathbb{E}_{\omega}[B(f(s, \pi(s), \omega))] \leq B(s) \quad \forall s \in \mathcal{S}$$

其中 $\omega$ 是动力学噪声, $\mathbb{E}_{\omega}$ 是对噪声的期望。

**物理含义**: 平均来说, 系统的"风险评分"不会随时间增加——系统不会漂移到更危险的区域。

**实现**: `martingale_loss = mean(max(E[B(s_next)] - B(s) + ε, 0))`

### 2.3 概率界的推导

如果四个条件成立, IBP 可以在网格上计算精确的界限:

$$\text{概率下界} = 1 - \frac{\max_{s \in \mathcal{S}_0} B(s) - \min_{s \in \mathcal{S}} B(s)}{\min_{s \in \mathcal{S}_u} B(s) - \min_{s \in \mathcal{S}} B(s)}$$

代码中:
```python
ub_n = ub_init - domain_min     # B(init)上界 - 全局最小
lb_n = lb_unsafe - domain_min   # B(unsafe)下界 - 全局最小
ratio = lb_n / ub_n
prob = 1 - 1/ratio              # 概率下界
```

---

## 3. 完整系统架构

### 3.1 三个神经网络

```
┌─────────────────────────────────────────────────────────────────┐
│                    SafePVC 系统架构                               │
│                                                                 │
│  ① gen_net (cGAN 生成器, frozen)                                │
│     输入: z (隐向量 4维) + d (距离)                               │
│     输出: 32×32 灰度图像 (模拟相机)                               │
│     来源: cGAN 预训练                                            │
│                                                                 │
│  ② state_net (状态估计, frozen)                                 │
│     输入: 32×32 图像                                             │
│     输出: [d̂, v̂] (估计距离和速度)                                │
│     来源: 监督学习预训练                                          │
│                                                                 │
│  ③ controller_net (PPO控制器, 可微调)                            │
│     输入: [d̂, v̂, v] (状态估计 + 真实速度)                        │
│     输出: u (加速度, 标量)                                        │
│     来源: PPO 预训练, SBC 训练中微调                              │
│                                                                 │
│  ④ l_model (SBC 网络, ★ 训练目标)                                │
│     输入: [d, v] (真实状态, 2维)                                  │
│     输出: B(s) ≥ 0 (安全评分)                                     │
│     架构: MLP[2→32→16→8→1], tanh + square                       │
│                                                                 │
│  VCLS = ① ∘ ② ∘ ③  (端到端闭环系统)                              │
│  SBC 验证的是 VCLS 的安全性                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 VCLS (Verifiable Closed-Loop System)

```python
# 完整前向传播
z = randn(4)                     # 随机隐向量
s = [d, v]                       # 真实状态 (归一化)
img = gen_net(z, d)              # ① 生成图像
state_est = state_net(img)       # ② 估计状态
u = controller_net([state_est, v])  # ③ 控制输出
s_next = dynamics(s, u) + noise  # 动力学前向
B_now = l_model(s)              # ④ SBC 当前值
B_next = l_model(s_next)        # SBC 下一状态值
```

---

## 4. gen_net: cGAN 图像生成器

### 4.1 结构

```python
# cGAN/taxi_models_and_data.py
class AebsMLPGenerator(nn.Module):
    # 输入: z (latent vector, 4维) + d (距离, 1维标量)
    # 输出: 32×32 = 1024 维灰度图像
    # 架构: [5→256] → [256→512] → [512→1024] → tanh → [-1, 1]范围
```

### 4.2 训练方式

- **条件 GAN (cGAN)**: 生成器接受隐向量 z + 条件 d (距离), 生成对应距离下的"路面图像"
- **判别器**: 判断图像是否真实 (来自真实数据集)
- **预训练**: 在 `Downsampled.h5` 数据集上训练
- **frozen**: SBC 训练时 gen_net 参数冻结, 不参与梯度更新

### 4.3 输入输出

```
z = torch.randn(4)          # 随机噪声, ~N(0,1)
d = 10.0 / std1             # 归一化距离
→ gen_net(z, d)             # 32×32 灰度图像
```

### 4.4 作用

模拟真实相机: 给定距离, 生成对应的"前车视图"。不同 z 生成同一距离下的不同视角/光照变化。这是 SafePVC "视觉" 部分的核心, 因为真实控制器需要通过相机感知环境。

---

## 5. state_net: 状态估计网络

### 5.1 结构

```python
# Combined_network/model.py
class SubNet(nn.Module):
    # 输入: 32×32=1024 维图像 (展平)
    # 架构: [1024] → Linear+LayerNorm+ReLU → [256] → Linear+LayerNorm+ReLU → [64] → Linear → [1]
    # 输出: d̂ (估计距离, 标量)
```

**注意**: state_net 只估计距离 d, 不估计速度 v。速度 v 直接从系统状态获取 (不需要从图像估计)。

### 5.2 训练方式

```python
# StateEstimate_train.py
# 监督学习: 输入图像 → 预测距离 → MSE loss vs 真实距离

for y_batch in data_loader:           # y_batch = [d_true, v_true]
    img = gen_net(z, y_batch[:,0])    # 根据真实距离生成图像
    d_pred = state_net(img)           # 从图像预测距离
    loss = MSE(d_pred, y_batch[:,0])  # 与真实距离对比
    loss += lip_weight × lip_penalty   # Lipschitz 正则化

# Lipschitz 正则化: 限制 state_net 的谱范数 ≤ 2.0
# 目的: 让 state_net 的梯度有界 → IBP 可以紧密包围
```

### 5.3 为什么需要 Lipschitz 约束?

```
如果没有 Lipschitz 约束:
  图像微小变化 → 状态估计剧烈变化
  → 控制器输出剧烈变化
  → 闭环系统不可预测
  → IBP 包围非常松 (conservative)

有 Lipschitz 约束:
  |state_net(img1) - state_net(img2)| ≤ 2.0 × |img1 - img2|
  → 输入扰动被控制
  → IBP 可以给出紧密的包围
  → SBC 验证可以成功
```

---

## 6. PPO 控制器

### 6.1 什么是 PPO

PPO (Proximal Policy Optimization) 是强化学习算法, 训练一个策略网络 $\pi(u|s)$:

```
目标: 最大化期望累计奖励
方法: Actor-Critic
  - Actor (策略网络): 根据状态输出动作
  - Critic (价值网络): 估计状态价值
  - PPO 核心: 限制策略更新幅度 (clip), 防止训练崩溃
```

### 6.2 PPO 训练

```python
# Controller_train.py

env = AebsEnv(std1)       # Gymnasium 环境
model = PPO(
    "MlpPolicy",           # 策略网络: MLP [2→64→64→action]
    env,
    learning_rate=3e-4,
    n_steps=2048,          # 每次收集2048步经验
    batch_size=64,         # 小批量更新
    n_epochs=10,           # 每次更新10轮
    gamma=0.99,            # 折扣因子
    gae_lambda=0.95,       # GAE 平滑
    ent_coef=0.01,         # 熵正则化 (鼓励探索)
)

model.learn(total_timesteps=200000)  # 训练20万步
```

### 6.3 AEBS 的奖励函数

```python
# env.py - step()
SAFETY_DIST = 6.0     # 安全距离 6m
SAFETY_SPEED = 0.5    # 安全速度 0.5m/s

reward = 0.0

# A. 接近奖励: 每步接近前车 → 正奖励
progress = d - d_next                 # 正值 = 在接近
reward += 2.0 * progress              # 鼓励接近 (但不能撞)

# B. 时间惩罚: 每步小惩罚, 鼓励快速完成
reward -= 0.001

# C. 终止:
#    成功: d ≤ 6.0 且 v ≤ 0.5 → 大正奖励 (安全刹停!)
#    碰撞: d ≤ 5.0 → 大负奖励
#    超速: v > 3.0 → 负奖励
```

**PPO 学到了什么策略?**
- 远处 (d>10m, v高): 可以选择加速/匀速
- 中距离 (6m<d<10m): 开始减速
- 近处 (d<6m): 必须减速到 v≤0.5 安全刹停

### 6.4 PPO 网络在 SBC 中的角色

```python
# VTLearner 中加载 PPO
model = PPO.load('./Aebs/controller/best_model/best_model.zip')

# 提取 Actor 网络的两个部分:
mlp_extractor = model.policy.mlp_extractor.policy_net   # 特征提取: [2→256→256]
action_net = model.policy.action_net                     # 动作输出: [256→1]

# 组装成端到端网络:
p_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)

# 前向传播:
# s = [d, v] → gen_net → img → state_net → d̂
#             → [d̂, v] → mlp_extractor → features
#                       → action_net → u (加速度)
```

### 6.5 PPO 网络架构详解

```
mlp_extractor (policy_net):     action_net:
  [d̂, v] (2维)                    features (256维)
    ↓                                ↓
  Linear(2, 256)                  Linear(256, 1)
    ↓                                ↓
  ReLU                            u = mean (加速度均值)
    ↓                             log_std (对数标准差, 探索用)
  Linear(256, 256)
    ↓
  ReLU → features (256维)
```

**SB3 PPO 的完整策略**:
```
obs = [d, v]
  → mlp_extractor(obs) → features
  → action_net(features) → [mean, log_std]
  → Normal(mean, exp(log_std)) → sample → action
```

SBC 训练时只用了确定性的 mean 输出 (忽略随机采样部分), 所以 controller_net 输出是确定性的。

---

## 7. 完整项目结构

### 7.1 项目总览

```
SafePVC 项目由四个独立训练的组件构成, 最终在 SBC 训练循环中集成:

┌───────────────────────────────────────────────────────────────┐
│                     SafePVC 完整流水线                          │
│                                                               │
│  阶段 1: 感知模块训练 (独立, SBC 前完成)                        │
│  ┌──────────┐    ┌──────────┐                                 │
│  │ gen_net  │    │state_net │                                 │
│  │ cGAN训练 │    │ 监督学习  │                                 │
│  │ 图像生成 │    │ 状态估计  │                                 │
│  └──────────┘    └──────────┘                                 │
│       ↓               ↓                                       │
│  阶段 2: 控制器训练 (独立, SBC 前完成)                          │
│  ┌──────────────────────────┐                                 │
│  │     PPO 强化学习          │                                 │
│  │  200k步 × AEBS 环境       │                                 │
│  │  学习安全的刹车策略        │                                 │
│  └──────────────────────────┘                                 │
│       ↓                                                       │
│  阶段 3: SBC 训练+验证 (集成所有组件)  ★ 核心                   │
│  ┌──────────────────────────────────────────────────┐        │
│  │  VTLearner + VTVerifier + Loop                    │        │
│  │  gen_net(frozen) → state_net(frozen)              │        │
│  │  → controller_net(可微调) → u                     │        │
│  │  → dynamics → SBC B(s) 训练 → IBP 验证            │        │
│  └──────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────┘
```

### 7.2 各组件关系

| 组件 | 代码位置 | 输入 | 输出 | 训练状态 | 训练方式 |
|------|---------|------|------|---------|---------|
| **gen_net** | `cGAN/taxi_models_and_data.py` | z(4) + d(1) | img(1024) | ❄ frozen | cGAN 预训练 |
| **state_net** | `Combined_network/model.py:SubNet` | img(1024) | d̂(1) | ❄ frozen | 监督学习 |
| **mlp_extractor** | SB3 PPO 内部 | [d̂, v](2) | features(256) | 🔧 微调 | PPO 预训练 |
| **action_net** | SB3 PPO 内部 | features(256) | u(1) | 🔧 微调 | PPO 预训练 |
| **l_model** | `Aebs/VT/utils.py:MLP` | [d, v](2) | B(s)(1) | ★ 训练 | SBC 训练 |
| **VCLS** | 上述组合 | z, [d,v] | u | 混合 | 端到端 |

### 7.3 文件结构

```
artical-F122/
├── Aebs/
│   ├── system/
│   │   ├── env.py              # Aebs 环境 + AebsEnv (PPO gym环境)
│   │   ├── combined.py         # 加载完整模型流水线的脚本
│   │   └── estimate.py         # StateEstimate_train 的引用
│   ├── VT/                     # ★ SBC 核心
│   │   ├── utils.py            # MLP, martingale_loss, triangular
│   │   ├── train.py            # VTLearner (SBC+Controller训练)
│   │   ├── verify.py           # VTVerifier (IBP验证)
│   │   └── loop.py             # 主训练循环
│   └── controller/
│       ├── Controller_train.py # PPO 训练
│       ├── StateEstimate_train.py # state_net 训练
│       └── best_model/         # 训练好的模型
├── cGAN/
│   └── taxi_models_and_data.py # AebsMLPGenerator (gen_net)
├── Combined_network/
│   └── model.py                # AebsEnd2EndNet, SubNet
└── auto_LiRPA/                 # IBP 库 (第三方)
```

### 7.4 数据流完整版

```
┌─ 训练 SBC (train_step_l) ──────────────────────────────────────┐
│                                                                 │
│  网格采样: s = [d_norm, v] from 100×100 grid                     │
│     ↓                                                           │
│  扰动: s_pert = s + grid_delta × U([-0.5, 0.5])                  │
│     ↓                                                           │
│  ┌─ VCLS 前向 ──────────────────────────────────────────┐       │
│  │ gen_net(z, d_norm)     → img (32×32)   ❄ frozen     │       │
│  │ state_net(img)          → d̂             ❄ frozen     │       │
│  │ mlp_extractor([d̂, v])  → features(256)  🔧 微调     │       │
│  │ action_net(features)    → u             🔧 微调     │       │
│  └────────────────────────────────────────────────────┘       │
│     ↓                                                           │
│  动力学: s_next_det = f(s_pert, u)                               │
│     d_next = d - v·dt                                           │
│     v_next = clip(v - u·dt, 0, 3)                               │
│     ↓                                                           │
│  噪声: noise = triangular(B, 16, 2) × noise_scale               │
│     s_next = s_next_det + noise                                 │
│     ↓                                                           │
│  ┌─ SBC 计算 ──────────────────────────────────────────┐       │
│  │ l_model(s_pert)        → B(s)         ★ 训练        │       │
│  │ l_model(s_next[16])    → B(s_next)    ★ 训练        │       │
│  │ E[B(s_next)] = mean(B(s_next), dim=1)                │       │
│  └────────────────────────────────────────────────────┘       │
│     ↓                                                           │
│  损失:                                                           │
│    martingale_loss = max(E[B(s_next)] - B(s) + 0.1, 0) × 1000  │
│    region_loss = relu(max(B_init)-1) + relu(20-min(B_unsafe))  │
│    lip_loss = relu(||∇B||₂ - 4.0)                               │
│     ↓                                                           │
│  反向传播 → 只更新 l_model 和 controller_net                      │
│            gen_net, state_net 保持 frozen                       │
└─────────────────────────────────────────────────────────────────┘

┌─ IBP 验证 (check_dec_cond) ─────────────────────────────────────┐
│                                                                 │
│  对 100×100 网格每个单元 [s_low, s_high]:                         │
│     ↓                                                           │
│  IBP 传播 B(s):  输入区间 → l_model → [B_low, B_high]            │
│  IBP 传播 VCLS:  输入区间 → gen→state→controller → u            │
│  IBP 传播 E[B]:  s + noise_grid → l_model → 上界 ub              │
│     ↓                                                           │
│  检查: ub ≤ B_low - lipschitz·K ?                                │
│    全部通过 → 验证成功 → 计算概率界                                │
│    有违反 → 记录反例 → train('p') 微调控制器                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. SBC 网络: B(s) 的设计

### 9.1 网络架构

```python
class MLP(nn.Module):
    def __init__(self, features=[2, 32, 16, 8, 1], activation="tanh", square_output=True):
        # features[i] → features[i+1] 的 Linear 层
        # 前 n-1 层: Linear + tanh
        # 最后层: Linear → (如果 square_output) x²

# 实例化
l_model = MLP([2, 32, 16, 8, 1], activation="tanh", square_output=True)

# 数据流:
# [d, v] → Linear(2,32) → tanh → Linear(32,16) → tanh
#        → Linear(16,8) → tanh → Linear(8,1) → x² → B(s) ≥ 0
```

### 9.2 设计要点

| 设计选择 | 理由 |
|---------|------|
| `tanh` 激活 | 有界输出 (-1,1), 利于 IBP 紧密包围 |
| `square_output=True` | 自动保证 B(s) ≥ 0 |
| 架构 [2,32,16,8,1] | 足够表达力 + 可被 IBP 高效验证 |
| 输入归一化到 [~0.5, ~2.4] | 和 IBP 的区间传播配合 |

---

## 10. 训练流程: VTLearner

### 10.1 训练概览

```
VTLearner 负责:
  1. 加载预训练模型 (gen_net, state_net, PPO)
  2. 管理两个优化器:
     - l_optimizer: Adam(lr=3e-3), 优化 SBC 网络
     - p_optimizer: Adam(lr=5e-2), 优化 controller_net
  3. 提供三种训练模式: 'l', 'p', 'both'
```

### 10.2 训练 SBC (`train_step_l`)

```
输入: z (随机隐向量), y (状态样本), current_delta (网格步长)

步骤:
  1. 扰动状态: y_pert = y + current_delta × random([-0.5, 0.5])
     ↑ 在网格单元内随机采样, 数据增强

  2. 前向传播:
     a = p_net(z, y_pert)                    # 控制器输出
     s_next = env.v_next(y_pert, a)          # 确定性下一状态

  3. 噪声采样 (16条轨迹估计期望):
     noise = triangular(B, 16, 2)            # 16组三角噪声
     s_next_random = s_next + noise × noise_scale

  4. 计算 B(s) 和 B(s_next):
     l = l_model(y_pert)                     # B(s)
     l_next = l_model(s_next_random)         # B(s_next), [B, 16]
     exp_l = l_next.mean(dim=1)              # E[B(s_next)]

  5. 损失:
     martingale_loss = max(exp_l - l + 0.1, 0) × 1000
     region_loss_init = relu(max(B_init) - 1.0)       # ≤1
     region_loss_unsafe = relu(20 - min(B_unsafe))     # ≥20
     lip_loss = relu(||∇B||₂ - l_lip)                  # 限制梯度

  total_loss = martingale × 1000 + lip_coeff × lip_loss + region_loss
```

**关键实现细节**:

1. **三角噪声代替高斯**: `triangular()` 生成有界噪声 → IBP 可严格包围
2. **16条轨迹近似期望**: 在 16 个噪声样本上取平均来估计 $\mathbb{E}[B(s_{t+1})]$
3. **区域损失分开采样**: `sample_init(13, 256)` 和 `sample_unsafe(17, 256)` 分别采样

### 10.3 训练控制器 (`train_step_p`)

```
输入: 同上

步骤:
  1. 扰动状态: y_pert = y + delta × random([-0.5, 0.5])

  2. 前向传播 (128条噪声, 更多样本):
     a_p = p_net(z, y_pert)
     s_next = env.v_next(y_pert, a_p)
     noise = triangular(B, 128, 2)           # 128条!
     s_next_random = s_next + noise × noise_scale

  3. 计算 SBC 值的期望递减:
     l_p = l_model(y_pert).detach()
     l_next = l_model(s_next_random)
     exp_l = l_next.mean(dim=1)
     martingale_loss = max(exp_l - l_p + 0.1, 0) × 10

  4. MSE 与教师网络对齐:
     teacher_u = old_p_net(z, y_pert)         # 原始 PPO 输出
     mse_loss = MSE(a_p, teacher_u) × 10

  5. Lipschitz 约束:
     计算 controller_net 对输入的梯度范数
     lip_loss = relu(||∇u controller||₂ - p_lip)

  total_loss = martingale × 10 + mse × 10 + lip_coeff × lip_loss
```

**为什么控制器训练用 128 条噪声?** 控制器训练需要更准确的期望估计 (128 > 16), 因为控制器的输出直接影响 SBC 的期望递减条件。

### 10.4 联合训练 (`train_step_joint`)

```
一次前向 + 一次反向传播, 同时更新 l_model 和 p_net。

关键: s_next_for_l = s_next_random.detach()  # 阻断控制器→SBC的梯度
      ↑ 防止 SBC 的梯度通过动力学反向传到控制器
      ↑ 两个网络的训练信号分离
```

---

## 11. 形式化验证: VTVerifier + IBP

### 11.1 IBP 原理 (Interval Bound Propagation)

IBP 不传播单个值, 而是传播**区间**:

```
输入: [d_low, d_high] × [v_low, v_high]  (一个网格单元)
  ↓
每层网络: 对区间的上下界做运算
  Linear:   [W⁺·low + W⁻·high, W⁺·high + W⁻·low]
  tanh:     利用单调性, 对上下界分别算
  square:   [min(low²,high²,0), max(low²,high²)]
  ↓
输出: [B_low, B_high]  (B(s) 在这个网格单元内的上下界)
```

**这是 sound (可靠) 的**: 真实的 B(s) 值一定落在 [B_low, B_high] 内。但可能**不紧** (conservative)。

auto_LiRPA 库提供了完整的 IBP 实现。

### 11.2 验证流程 (`check_dec_cond`)

```
输入: k_except_l = 1.2 (Lipschitz 放大因子)

步骤:
  1. 计算 IBP 界限:
     ub_init = max_{s∈S₀} B(s)       ← compute_bound_init
     lb_unsafe = min_{s∈Sᵤ} B(s)     ← compute_bound_unsafe
     domain_min = min_{s∈S} B(s)     ← compute_bound_domain

  2. 对 100×100 网格的每个单元:
     a) 取网格中心点 s
     b) 计算控制器输出: a = p_net(z, s)
     c) 计算 SBC 值: l = l_model(s)
     d) 计算局部 Lipschitz: lip = ||∇B(s)||₂
     e) 过滤: 只检查 B(s) 接近不安全阈值的单元 (优化性能)
     f) 检查期望递减: E[B(s_next)] ≤ B(s) - lip·K ?
        ↑ compute_expected_l: IBP 计算 E[B(s_next)] 的严格上界
        ↑ K = k·δ = 1.2 × 网格半径

  3. 统计: 有多少个单元违反期望递减?
     violations < 0.1% → 验证通过 ✅
```

### 11.3 期望上界的 IBP 计算 (`compute_expected_l`)

```
对单个状态 s:
  确定性子状态: s_det = f(s, u)
  
  噪声网格: 将噪声空间离散化为 10×10 网格
  每个噪声网格单元有:
    - 概率质量 pmass (均匀噪声的体积占比)
    - 下界 lb, 上界 ub
  
  对每个噪声单元:
    noisy_s 的区间 = s_det + [lb, ub]
    IBP 传播 → B(noisy_s) 的上界 ub_B
  
  期望上界 = Σ pmass × ub_B  (加权平均)
```

### 11.4 概率界计算

```
验证通过后:
  ub_init_norm = ub_init - domain_min
  lb_unsafe_norm = lb_unsafe - domain_min
  ratio = lb_unsafe_norm / ub_init_norm
  prob = 1 - 1/ratio

例如: ub_init=0.5, lb_unsafe=20, domain_min=0.02
  ub_init_norm = 0.48
  lb_unsafe_norm = 19.98
  ratio = 41.625
  prob = 1 - 1/41.625 = 0.976 = 97.6%
```

---

## 12. 主训练循环: Loop

### 12.1 完整算法

```python
Loop.run(timeout=3600):  # 1小时超时

  1. 预填充: 生成 100×100 状态网格 → train_buffer
     
  2. 迭代循环 (最多100轮):
     for iter in range(100):
     
       # Step A: 训练 SBC
       train('l', num_epochs=10)
       # 10个epoch, 每个epoch遍历整个网格
       
       # Step B: IBP 验证
       sat, violations, info, buffer = verifier.check_dec_cond(K=1.2)
       
       # Step C: 如果通过, 计算概率界
       if sat:
         ub_init = verifier.compute_bound_init(grid)
         lb_unsafe = verifier.compute_bound_unsafe(grid)
         prob = 1 - 1/(lb_unsafe/ub_init)
         if prob > max_prob: 保存模型
       
       # Step D: 训练控制器 (用反例)
       train('p', num_epochs=1, violation_buffer)
       # 只训练1个epoch, 在验证失败的反例上
       
       iter += 1
```

### 12.2 训练-验证交替

```
每次迭代的节奏:

  train('l', 10 epochs)  →  训练 SBC (让它更好地满足四个条件)
       ↓
  verify                  →  严格检查 SBC 是否真满足条件
       ↓ (如果失败, 收集反例)
  train('p', 1 epoch)    →  用反例训练控制器 (修正导致失败的状态)
       ↓
  下一轮 train('l', 10)  →  SBC 适应新的控制器
```

**这是 CEGIS (反例引导归纳合成) 的核心**: 
- SBC 试图找到满足条件的函数
- 验证器找到 SBC 不满足条件的反例
- 控制器被修正以避免这些反例
- 循环直到验证通过

---

## 13. 完整数据流 (端到端)

### 13.1 训练数据流

```
┌─ 数据准备 ────────────────────────────────────────────────────┐
│                                                               │
│  状态空间: [d_min, d_max] × [v_min, v_max]                    │
│          = [5.0/σ, 16.0/σ] × [0.0, 3.0]                       │
│                                                               │
│  离散化: 100×100 均匀网格                                      │
│  → 10000 个网格单元, 每个单元有中心点 + 上下界                  │
│  → 所有单元的中心点组成 train_ds                               │
│                                                               │
│  输入: 网格中心点 s = [d_norm, v]                              │
│  扰动: s_pert = s + Δ × U([-0.5, 0.5])                        │
│        Δ = (状态空间上界 - 下界) / 100                          │
└───────────────────────────────────────────────────────────────┘

┌─ SBC 训练 (train_step_l) ───────────────────────────────────┐
│                                                               │
│  y_pert = [d_norm, v] + δ  (扰动后的状态)                      │
│    │                                                          │
│    ├→ p_net(z, y_pert) → a (控制动作)                         │
│    │   ┌─ gen_net(z, d) → img (32×32)                        │
│    │   ├─ state_net(img) → [d̂, v̂]                            │
│    │   └─ controller_net([d̂, v̂, v]) → u                      │
│    │                                                          │
│    ├→ env.v_next(y_pert, a) → s_next_det                      │
│    │   d_next = d - v·dt                                      │
│    │   v_next = clip(v - u·dt, 0, 3)                          │
│    │                                                          │
│    ├→ s_next_det + noise(16条) → s_next_random                │
│    │                                                          │
│    ├→ l_model(y_pert) → B(s)                                  │
│    ├→ l_model(s_next_random) → B(s_next) [B×16]              │
│    │                                                          │
│    └→ 损失:                                                    │
│        martingale: max(E[B(s_next)] - B(s) + 0.1, 0) × 1000  │
│        init:       relu(max(B(init)) - 1.0)                   │
│        unsafe:     relu(1/(1-p) - min(B(unsafe)))             │
│        lip:        relu(||∇B||₂ - 4.0)                        │
└───────────────────────────────────────────────────────────────┘

┌─ IBP 验证 (check_dec_cond) ─────────────────────────────────┐
│                                                               │
│  对 100×100 网格的每个单元:                                    │
│                                                               │
│  网格单元 [s_low, s_high] 的中心 s                             │
│    │                                                          │
│    ├→ B(s) 的区间: IBP(l_model, [s_low, s_high])             │
│    │   → [B_low, B_high]                                      │
│    │                                                          │
│    ├→ 控制器: u = p_net(z, s)                                 │
│    │                                                          │
│    ├→ s_next 的区间:                                           │
│    │   确定部分 s_det = dynamics(s, u)                         │
│    │   + 噪声网格 (10×10) × 概率质量                            │
│    │   → IBP(l_model, s_det + noise_interval)                │
│    │   → E[B(s_next)] 的严格上界                               │
│    │                                                          │
│    └→ 检查: E[B(s_next)] ≤ B(s) - lip·K ?                    │
│        违反 → 记录反例 → 用于控制器训练                         │
│        通过 → 满足期望递减                                      │
│                                                               │
│  全网格通过 → 计算概率下界                                      │
│    prob = 1 - (ub_init - domain_min)/(lb_unsafe - domain_min) │
└───────────────────────────────────────────────────────────────┘
```

### 13.2 关键数据维度

| 数据 | 维度 | 说明 |
|------|------|------|
| 状态 s | [B, 2] | [d_norm, v] |
| 隐向量 z | [B, 4] | ~U(-1,1) |
| 图像 img | [B, 1024] | 32×32 展平 |
| 控制 u | [B, 1] | 标量加速度 |
| SBC 输出 B(s) | [B, 1] | ≥0 的安全评分 |
| 噪声轨迹 | [B, 16, 2] 或 [B, 128, 2] | 多条噪声样本 |
| 训练网格 | [10000, 2] | 100×100 离散化 |
| 验证网格 | [10000, 2] | 同训练网格 (同分辨率) |

### 13.3 噪声建模

```
AEBS 中使用三角分布噪声 (不是高斯):

triangular(shape):
  U ~ Uniform(0, 1)
  if U ≤ 0.5: return -1 + √(2U)    # 偏向 -1
  else:        return 1 - √(2(1-U))  # 偏向 +1
  → 输出范围: [-1, 1], 在 -1 和 +1 处概率密度最高

噪声尺度: noise_scale = (state_high - state_low) × 0.01
  → 对归一化状态, noise_scale ≈ 0.023

为什么用三角分布?
  ✓ 有界 → IBP 可严格包围
  ✓ 近似正态但尾部有界 → 比均匀分布更真实
```

---

## 14. 环境设定: AEBS 场景

### 14.1 AEBS (自动紧急制动系统)

```
场景: 自车跟随前车, 需要保持安全距离

状态: s = [d, v]
  d: 距离 (归一化: ÷σ, 原始范围 5-16m)
  v: 速度 (0-3 m/s)

控制: u ∈ [-3, 3] m/s² (加速度)

动力学 (离散, dt=0.05s):
  d_{t+1} = d_t - v_t · dt
  v_{t+1} = clip(v_t - u_t · dt, 0, 3)

注意约定: v_next = v - u·dt
  → u > 0 时 v 减小 ("加速减速")
  → u < 0 时 v 增大
```

### 14.2 安全区域定义

```python
# 初始区域 (系统从这出发)
init_spaces = [
    Box([15.0/σ, 2.5], [16.0/σ, 3.0])  # 远处, 高速
]

# 不安全区域 (绝对不想进入)
unsafe_spaces = [
    Box([5.0/σ, 0.5], [6.0/σ, 3.0])    # 很近, 任意速度
]

# 目标: 从初始区域出发, 永远不进入不安全区域
# SBC 条件: B(init) ≤ 1, B(unsafe) ≥ 1/(1-p) = 20 (p=0.95)
```

---

## 15. 关键超参数

| 超参数 | 值 | 位置 | 说明 |
|--------|-----|------|------|
| **SBC 架构** | [2,32,16,8,1] | `MLP(features)` | tanh + square |
| **SBC 学习率** | 3e-3 | `l_optimizer` | Adam |
| **控制器学习率** | 5e-2 | `p_optimizer` | Adam (较高, 因为只微调) |
| **martingale ε** | 0.1 | `martingale_loss(l, exp_l, eps=0.1)` | 期望递减的容忍度 |
| **martingale 权重** | 1000 (SBC) / 10 (controller) | `loss_l`, `loss_p` | SBC 训练中 martingale 是主导损失 |
| **MSE 权重** | 10 | `loss_p` | 控制器与 teacher 的对齐强度 |
| **Lipschitz 上限 (SBC)** | 4.0 | `l_lip` | B(s) 梯度范数上限 |
| **Lipschitz 上限 (controller)** | 2.0 | `p_lip` | 控制器梯度范数上限 |
| **Lipschitz 正则化系数** | 0.001 | `lip_coeff` | 小权重, 辅助作用 |
| **噪声轨迹数 (SBC训练)** | 16 | `triangular(B, 16, 2)` | 更多→更准, 但更慢 |
| **噪声轨迹数 (控制器训练)** | 128 | `triangular(B, 128, 2)` | 控制器需要更准的期望 |
| **网格分辨率** | 100×100 | `train_space_split` | 10000个单元 |
| **验证 K 值** | 1.2 | `k_except_l` | Lipschitz 放大因子 |
| **噪声因子** | 0.01 | `Aebs(factor=0.01)` | 噪声尺度 = 状态范围 × 0.01 |
| **dt** | 0.05s | `dt` | 离散时间步长 |
| **目标概率** | 0.95 | `reach_prob` | p_target → B_thresh = 20 |
| **IBP batch size** | 2048 | `VTVerifier(batch_size=2048)` | 验证时每批处理数 |
| **训练 batch size** | 256 or 2048 | `train_epoch(batch_size)` | 训练时每批处理数 |
| **每轮训练 epoch** | 10 (SBC) / 1 (controller) | `train('l', 10)` | SBC 多训练, controller 微调 |

---

## 16. 总结: SBC 做了什么

```
SBC 不是一个约束, 而是一个 "评分函数 + 验证框架":

  1. 定义 B(s): 神经网络, 状态 → 风险评分

  2. 训练 B(s): 
     - 在初始区域 B(s) ≤ 1
     - 在不安全区域 B(s) ≥ 20  
     - 在所有区域 E[B(s_next)] ≤ B(s) (期望不增)

  3. 验证 B(s):
     - IBP 在 100×100 网格上严格证明上述条件成立
     - 发现反例 → 修正控制器 → 重新训练 SBC → 重新验证

  4. 输出结果:
     - B(s) 函数本身 (可用于指导 QP, 如 V3B)
     - 概率下界: P(安全) ≥ 96.58%
     - 这个界是形式化保证, 不是采样估计
```

**SBC 和 CBF 的关系**:

| | SBC | CBF |
|---|---|---|
| 性质 | 概率验证函数 | 确定性安全约束 |
| 输出 | B(s) ∈ ℝ⁺ | 对 u 的约束 |
| 保证类型 | "96.58% 概率永远安全" | "这一步的 u 保证安全" |
| 时机 | 离线训练+验证 | 在线推理每步 |
| 互补 | 给全局信心 | 给局部保障 |
