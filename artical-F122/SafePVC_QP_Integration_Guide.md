# 在 SafePVC 论文框架中加入 QP 安全滤波层

> 参考：BarrierNet — Differentiable Control Barrier Functions via QP Layers
> 目标：在 SafePVC（Provably Probabilistic Safe Vision-Based Controller）的控制流水线中，加入一个**可微分 QP（二次规划）安全滤波层**，使神经网络输出的控制动作在运行时即被 CBF（Control Barrier Function）约束过滤，实现"在线安全保障 + 离线形式化验证"的双重安全机制。

---

## 目录

1. [原始 SafePVC 框架回顾](#1-原始-safePVC-框架回顾)
2. [集成 QP 安全滤波的动机与改进分析](#2-集成-qp-安全滤波的动机与改进分析)
3. [BarrierNet 的 QP 核心思想](#3-barriernet-的-qp-核心思想)
4. [加入 QP 后的整体架构变化](#4-加入-qp-后的整体架构变化)
5. [修改后的网络结构详解](#5-修改后的网络结构详解)
6. [QP 层的数学公式](#6-qp-层的数学公式)
7. [详细数据流](#7-详细数据流)
8. [训练流程变化](#8-训练流程变化)
9. [损失函数变化](#9-损失函数变化)
10. [对 SBC 验证的影响](#10-对-sbc-验证的影响)
11. [具体实施步骤](#11-具体实施步骤)
12. [以 X-Plane11 滑行为例的完整推导](#12-以-x-plane11-滑行为例的完整推导)
13. [以 CARLA 紧急制动为例的完整推导](#13-以-carla-紧急制动为例的完整推导)
14. [代码结构建议](#14-代码结构建议)
15. [对比总结表](#15-对比总结表)

---

## 1. 原始 SafePVC 框架回顾

### 1.1 原始数据流（无 QP）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    原始 SafePVC 运行时数据流                             │
│                                                                         │
│  状态 s_t ──► 感知 MLP g(s_t, z_t) ──► 图像 o_t ──► 控制器 π(o_t) ──► u_t │
│                                                                         │
│  u_t ──► 动力学 f(s_t, u_t) ──► s_{t+1}                                │
│                                                                         │
│  ★ 安全性完全依赖网络权重，无运行时安全约束                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 原始控制律

控制动作由神经网络直接输出，无任何在线修正：

$$u_t = \pi_\theta(o_t) = \pi_\theta(g(s_t, z_t))$$

### 1.3 原始安全保障方式

- **离线**：通过随机障碍证书（SBC）$B(s)$ 的形式化验证来证明概率安全性
- **运行时**：无安全保障——安全性完全依赖训练过程中"烘焙"进权重的隐式安全行为
- **问题**：若控制器遇到训练分布外的状态，无法保证输出的动作满足安全约束

---

## 2. 集成 QP 安全滤波的动机与改进分析

本节系统性地分析原始 SafePVC 框架存在的核心问题，阐述引入 CBF-QP 安全滤波层的动机，并详细说明这一集成所带来的多维度改进。

### 2.1 原始 SafePVC 存在的核心问题

#### 2.1.1 问题一：运行时无显式安全保障——"隐式安全"的根本局限

原始 SafePVC 的安全性完全依赖 SBC（Stochastic Barrier Certificate）的**离线形式化验证**。在运行时，控制器 $\pi_\theta(o_t)$ 直接输出控制动作 $u_t$，没有任何在线安全约束机制。这意味着：

- **安全性是"烘焙"在权重中的隐式属性**：网络在训练过程中通过损失函数和 SBC 反例的引导，逐渐"学会"输出安全动作。但这种安全行为是隐式的——无法在运行时验证当前动作是否安全。
- **面对 OOD（分布外）输入时脆弱**：当感知系统产生的观测 $o_t$ 偏离训练分布时（例如光照突变、遮挡、传感器噪声异常），控制器可能输出在训练中从未见过的不安全动作，而此时没有任何机制阻止该动作被执行。
- **无实时纠错能力**：即使网络输出了明显违反物理安全约束的动作（如转向角度导致立即碰撞），系统也只能"盲目执行"——安全性完全取决于网络权重是否恰好编码了正确的安全行为。

用一个比喻来说：原始 SafePVC 就像一位**只在考试中验证过能力的司机**——虽然通过了理论测试（SBC 验证），但实际驾驶时没有安全带、没有刹车辅助、没有任何主动安全系统。

#### 2.1.2 问题二：缺乏前向不变性保证——安全集不被显式维护

SafePVC 使用 SBC 来验证概率安全性，其核心是证明 SBC 函数 $B(s)$ 的期望递减条件：

$$\mathbb{E}[B(s_{t+1}) | s_t] \leq \gamma \cdot B(s_t) - \epsilon$$

然而，这种概率层面的保证存在两个根本性局限：

1. **不保证单次执行的安全性**：SBC 提供的是统计意义上的安全下界 $\mathbb{P}[\text{Safe}] \geq p$，而非每次执行的确定性保证。在单次运行中，系统仍有可能进入不安全状态。
2. **缺乏安全集的前向不变性**：控制理论中的经典安全定义是**前向不变性**（forward invariance）——如果当前状态在安全集内（$b(s_t) \geq 0$），则下一时刻状态也必须在安全集内（$b(s_{t+1}) \geq 0$）。原始 SafePVC 没有这种逐步保证，其安全性是"全局统计"而非"逐步约束"的。

#### 2.1.3 问题三：SBC 验证的保守性——验证通过难度大

SBC 的形式化验证依赖 $\alpha$-$\beta$-CROWN 等区间传播方法来计算 Lipschitz 常数并验证期望递减条件。这种验证方法本质上是**保守的**（conservative），因为它需要用 Lipschitz 上界来覆盖所有可能的状态转移：

$$K = \tau \cdot L_B \cdot (1 + L_f \sqrt{1 + (L_\pi L_g)^2})$$

由于控制器 $\pi_\theta$ 没有任何运行时约束，验证器必须假设它可能输出**任意值**（在 Lipschitz 锥的范围内），这导致：

- Lipschitz 常数 $L_\pi$ 被高估
- 安全界 $\beta$（SBC 在安全集上的上界）难以压缩
- 验证条件 (iv)（期望递减条件）在复杂场景中难以满足
- 随着任务复杂度增加，验证通过的难度急剧上升

#### 2.1.4 问题四：安全行为的泛化性不足

由于安全性完全由训练数据驱动，当部署环境与训练环境存在差异时（domain gap），控制器的安全行为可能无法泛化：

- **训练覆盖有限**：RL 预训练和 SBC 反例只能覆盖有限的状态空间区域
- **长尾场景缺失**：罕见的危险场景（如极端天气、突发障碍物）在训练中采样不足
- **安全边界模糊**：网络学到的"安全行为"没有明确的数学边界，在安全临界区域的决策不可预测

### 2.2 CBF-QP 如何解决这些问题

引入 CBF-QP 安全滤波层后，从以下三个维度系统性地解决了上述问题：

#### 2.2.1 维度一：运行时确定性安全保障——从"隐式"到"显式"

**核心机制**：在每个时间步 $t$，QP 安全滤波层将控制器的输出约束在 CBF 定义的安全集内：

$$u_t^* = \arg\min_{u} \frac{1}{2}\|u\|^2 + q_t^T u \quad \text{s.t.} \quad G(s_t, p_t) u \leq h(s_t, p_t)$$

**关键改进**：

- **逐步安全保证**：无论控制器输出什么，QP 求解器都会确保最终执行的动作 $u_t^*$ 满足 CBF 约束，从而保证 $b(s_t) \geq 0 \Rightarrow b(s_{t+1}) \geq 0$（安全集前向不变性）
- **运行时纠错**：如果控制器因 OOD 输入输出了危险动作，QP 层会自动将其修正为最近的安全动作——相当于一个"安全网"
- **硬约束保证**：安全约束是数学上的硬约束（$Gu \leq h$），而非统计意义上的软约束

```
┌───────────────────────────────────────────────────────────────────┐
│              运行时安全保障机制对比                                  │
│                                                                   │
│  原始 SafePVC:                                                    │
│  NN 输出 u_t ──────────────────────────────► 执行 u_t            │
│  （无任何检查，直接执行）                                          │
│                                                                   │
│  SafePVC + QP:                                                    │
│  NN 输出 q_t ──► CBF 约束检查 ──► QP 修正 ──► 执行 u_t*         │
│                   │                   │                           │
│                   │  if 不安全:        │  投影到最近的安全动作     │
│                   │  if 安全:          │  保持不变（最小修改）     │
│                   └───────────────────┘                           │
└───────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 维度二：双重安全机制——确定性 + 概率性的互补

引入 QP 后，系统同时拥有两层安全保障，二者在性质上互补：

| 维度 | CBF-QP（在线，确定性） | SBC（离线，概率性） |
|------|----------------------|-------------------|
| **保障类型** | 前向不变性：$b(s_t) \geq 0 \Rightarrow b(s_{t+1}) \geq 0$ | 概率安全界：$\mathbb{P}[\text{Safe}_{0:T}] \geq p$ |
| **作用时间** | 运行时每一步 | 离线一次性验证 |
| **时间范围** | 单步保证（瞬时） | 无限时域保证（$T \to \infty$） |
| **覆盖范围** | 当前状态附近的安全集 | 整个状态空间的安全概率 |
| **对扰动的处理** | 确定性约束不考虑随机扰动 | 显式建模扰动分布 $\Delta s \sim \mu$ |
| **局限性** | 不提供长期概率界 | 保守的 Lipschitz 上界可能过于严格 |

**互补效应**：
- CBF-QP 弥补了 SBC 缺乏运行时保障的缺陷——即使 SBC 验证的概率界不够紧，运行时的 CBF 约束也能防止系统进入不安全状态
- SBC 弥补了 CBF-QP 只能保证单步安全的缺陷——SBC 的无限时域概率界提供了 CBF 无法给出的长期安全保证
- 二者的结合使得系统同时具有**瞬时确定性安全**和**长期概率安全**

#### 2.2.3 维度三：缓解 SBC 验证的保守性——使验证更容易通过

这是集成 QP 的一个常被忽视但极为关键的收益。QP 层通过约束控制器的输出空间，直接降低了 SBC 验证的难度：

**1. 减小控制器有效 Lipschitz 常数**

原始控制器的 Lipschitz 常数 $L_\pi$ 覆盖整个输出空间。加入 QP 后，QP 层的投影操作将输出限制在安全约束定义的可行域内，有效减小了闭环映射的 Lipschitz 常数：

$$L_{\text{closed-loop}}^{\text{original}} = L_f \cdot L_\pi \cdot L_g \quad \longrightarrow \quad L_{\text{closed-loop}}^{\text{QP}} = L_f \cdot L_{\text{QP}} \cdot L_\pi \cdot L_g$$

其中 QP 层的 Lipschitz 常数 $L_{\text{QP}}$ 在某些条件下可以小于 1（当 QP 解在约束边界上"压缩"了输入变化的传播时）。

**2. 缩小不安全区域的覆盖需求**

SBC 验证需要覆盖所有可能的状态转移。原始 SafePVC 中，控制器可能将系统带入任何状态（包括不安全区域），验证器需要用足够大的 $\beta$ 值来覆盖这些转移。QP 层通过 CBF 约束阻止系统进入不安全区域，显著缩小了需要覆盖的状态空间范围。

**3. 更紧的安全界**

由于 QP 层将系统轨迹约束在安全集内，SBC 函数 $B(s)$ 在实际运行中只需要在安全集内有效，无需对不安全区域的状态给出有意义的界。这使得 SBC 网络可以专注于安全集内的精确建模，而非浪费容量去覆盖不可达的不安全区域。

```
┌──────────────────────────────────────────────────────────────────┐
│               SBC 验证难度的直观对比                               │
│                                                                  │
│  原始 SafePVC:                                                   │
│  ┌───────────────────────────────────────┐                      │
│  │  状态空间 S                            │                      │
│  │  ┌───────────┐  ┌──────────────────┐  │                      │
│  │  │ 安全集 C  │  │   不安全集        │  │                      │
│  │  │ (需验证)  │  │ (也需验证！       │  │                      │
│  │  │           │  │  因为控制器可能    │  │                      │
│  │  │           │  │  把系统带到这里)   │  │                      │
│  │  └───────────┘  └──────────────────┘  │                      │
│  │  验证范围: 整个 S                       │                      │
│  └───────────────────────────────────────┘                      │
│                                                                  │
│  SafePVC + QP:                                                   │
│  ┌───────────────────────────────────────┐                      │
│  │  状态空间 S                            │                      │
│  │  ┌───────────┐  ┌──────────────────┐  │                      │
│  │  │ 安全集 C  │  │   不安全集        │  │                      │
│  │  │ (需验证)  │  │ (CBF-QP 阻止进入  │  │                      │
│  │  │           │  │  无需覆盖)        │  │                      │
│  │  │           │  │                   │  │                      │
│  │  └───────────┘  └──────────────────┘  │                      │
│  │  验证范围: 主要限于 C                   │                      │
│  └───────────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 与单独使用 BarrierNet 相比的改进

虽然本方案的 QP 层设计借鉴了 BarrierNet，但单独使用 BarrierNet 存在以下局限，本方案通过集成 SafePVC 框架进行了系统性弥补：

#### 2.3.1 BarrierNet 缺乏离线形式化验证

BarrierNet 的安全性完全依赖 CBF-QP 的运行时约束，没有任何离线验证机制。这意味着：

- **无法给出概率安全界**：BarrierNet 只能保证"如果 CBF 约束被满足，则系统安全"，但无法回答"系统有多大概率保持安全"这一问题
- **CBF 构造的正确性无法验证**：如果障碍函数 $b(s)$ 本身定义有误（如安全集建模不准确），运行时 CBF 约束可能保护的是一个错误的安全集
- **对感知误差无鲁棒性保证**：BarrierNet 假设状态 $s_t$ 是准确已知的，当使用视觉感知时，感知误差可能导致 CBF 约束基于错误的状态信息

**本方案的改进**：
- SBC 提供了 $\mathbb{P}[\text{Safe}] \geq p$ 的概率安全界，即使 CBF 约束偶尔失效（如 QP 无可行解），SBC 的概率界仍然有效
- SBC 的验证过程通过反例驱动（CEGIS）可以暴露 $b(s)$ 定义中的问题
- SafePVC 的感知模型（cGAN + 蒸馏 MLP）带有 Lipschitz 约束，显式处理了感知误差

#### 2.3.2 BarrierNet 的 CBF 参数是启发式固定的

BarrierNet 虽然使用 NN 学习 CBF 的 class-K 函数参数，但其学习过程缺乏与形式化验证的交互。参数的"好坏"只能通过实验效果来判断，没有理论保证。

**本方案的改进**：
- CBF 参数 $p_t$ 由 NN 分支学习，同时受到 SBC 验证的间接约束——如果参数导致闭环系统不满足 SBC 条件，训练过程会通过反例驱动机制修正参数
- 形成了"**NN 学习 CBF 参数 → QP 在线约束 → SBC 离线验证 → 反例反馈修正 NN**"的完整闭环

#### 2.3.3 BarrierNet 缺乏无限时域安全保证

CBF 的前向不变性本质上是**单步保证**：$b(s_t) \geq 0 \Rightarrow b(s_{t+1}) \geq 0$。在离散时间系统中，这种保证可能因离散化误差而累积，长期安全性无法从单步保证直接推出。

**本方案的改进**：
- SBC 提供的无限时域概率界 $\mathbb{P}[\forall t \geq 0: s_t \in \mathcal{C}] \geq p$ 恰好弥补了这一缺陷
- 即使离散化导致单步 CBF 保证存在小误差，SBC 的无限时域界仍然提供了全局安全保证

### 2.4 集成改进的详细总结

#### 2.4.1 安全性改进

| 安全属性 | 原始 SafePVC | SafePVC + QP | 改进幅度 |
|---------|-------------|--------------|---------|
| 运行时安全约束 | ❌ 无 | ✅ CBF-QP 逐步约束 | **从无到有** |
| 前向不变性 | ❌ 无显式保证 | ✅ $b(s) \geq 0$ 维持 | **从无到有** |
| OOD 鲁棒性 | ❌ 脆弱 | ✅ QP 自动修正 | **显著提升** |
| 概率安全下界 | $\mathbb{P} \geq p$ | $\mathbb{P} \geq p'$（$p' > p$） | **预期提升 3~8%** |
| 安全约束可解释性 | ❌ 隐式（权重） | ✅ 显式（数学约束） | **质的飞跃** |
| 长期安全保证 | ✅ SBC 无限时域 | ✅ SBC + CBF 双重 | **增强** |

#### 2.4.2 可验证性改进

| 验证属性 | 原始 SafePVC | SafePVC + QP | 改进说明 |
|---------|-------------|--------------|---------|
| SBC 验证通过率 | 低（保守） | 更高（QP 缩小搜索空间） | QP 约束使不安全状态不可达，降低验证保守性 |
| Lipschitz 常数 | 大 | 更小 | QP 层限制输出空间，减小闭环 Lipschitz 常数 |
| 反例数量 | 多 | 更少 | QP 阻止不安全行为，减少了 SBC 反例的产生 |
| 训练收敛速度 | 慢 | 更快 | CBF 约束提供了额外的安全先验，加速学习 |

#### 2.4.3 控制性能改进

| 性能指标 | 原始 SafePVC | SafePVC + QP | 改进说明 |
|---------|-------------|--------------|---------|
| 安全临界区域行为 | 不可预测 | CBF 约束保证安全 | 在安全边界附近，QP 层提供确定性保护 |
| 控制平滑性 | 可能抖动 | QP 最小化控制偏差 | $\min \frac{1}{2}\|u\|^2$ 项天然促进平滑控制 |
| 与 RL 策略的一致性 | 隐式 | 显式（MSE + QP 联合优化） | MSE 损失确保 QP 输出接近 RL 策略 |
| 多约束处理能力 | 隐式（全靠 NN） | 显式（CBF 约束堆叠） | 多个安全约束可通过矩阵堆叠同时处理 |

#### 2.4.4 工程实用性改进

| 工程属性 | 原始 SafePVC | SafePVC + QP | 改进说明 |
|---------|-------------|--------------|---------|
| 安全规则可编码 | ❌ 无法显式编码 | ✅ 物理约束直接写入 CBF | 如速度限制、距离限制可直接建模 |
| 安全调试 | ❌ 难以定位不安全行为来源 | ✅ 可分析 CBF 约束激活情况 | 可以知道哪些约束在何时被激活 |
| 安全认证 | ❌ 仅靠统计指标 | ✅ 确定性 + 概率性双重指标 | 更易通过安全审查 |
| 系统模块化 | 安全与耦合 | 安全层可独立替换 | QP 层可作为独立模块添加到不同控制器上 |

### 2.5 改进的理论基础

本节从理论角度解释为什么集成 QP 能够带来上述改进。

#### 2.5.1 安全性的分解与组合

**定理（安全性分解）**：设系统在安全集 $\mathcal{C} = \{s: b(s) \geq 0\}$ 中运行。如果：
1. CBF-QP 保证前向不变性：$s_t \in \mathcal{C} \Rightarrow s_{t+1} \in \mathcal{C}$（在连续时间意义下）
2. SBC 保证概率收敛：$\mathbb{P}[\lim_{t\to\infty} B(s_t) = 0] \geq p$

则系统在双重机制下的安全性不低于单独使用 SBC 的安全性，且在有限时间内的安全性严格更强。

**直观理解**：CBF-QP 像一个"安全围栏"将系统约束在安全集内，SBC 像一个"导航仪"引导系统在安全集内向目标收敛。二者分工明确，各司其职。

#### 2.5.2 QP 最优性与最小干预原则

QP 的目标函数 $\min \frac{1}{2}\|u\|^2 + q_t^T u$ 体现了**最小干预原则**：

- 当控制器的参考输出 $q_t$ 本身是安全的（满足 CBF 约束），QP 的解 $u_t^* \approx q_t$，即不修改控制器输出
- 当 $q_t$ 不安全时，QP 找到与 $q_t$ 最接近的安全动作 $u_t^*$，实现对控制器意图的最小修改
- 这保证了安全性与任务性能之间的最优平衡——在安全的前提下最大化任务完成质量

#### 2.5.3 CBF 参数自适应的优势

传统 CBF 方法使用固定的 class-K 函数参数（如 $\alpha(x) = \gamma x$ 中的 $\gamma$ 由人工设定）。本方案通过 NN 分支学习自适应参数 $p_t = 4 \cdot \sigma(\text{NN}(\cdot))$，带来以下优势：

- **状态依赖性**：不同状态下使用不同的 CBF 增益——远离安全边界时使用较小增益（更宽松，允许更大控制自由度），接近安全边界时使用较大增益（更保守，更强制地推回安全集）
- **任务-安全平衡**：NN 通过端到端训练自动学习任务性能与安全性之间的最优权衡
- **避免过度保守**：固定参数 CBF 在所有状态下使用相同的保守度，可能在安全区域过度限制控制器；自适应参数避免了这种不必要的限制

---

## 3. BarrierNet 的 QP 核心思想

### 3.1 核心理念

BarrierNet 将一个**可微分的 QP 求解器**作为神经网络的最后一层，在运行时对 NN 输出的控制动作进行 CBF 约束过滤：

$$u^* = \arg\min_{u} \frac{1}{2}\|u\|^2 + q^T u \quad \text{s.t.} \quad Gu \leq h$$

其中：
- $q$ = NN 分支 1 的输出（参考控制）
- $G, h$ = 由 CBF 约束构造，其中 CBF 的 class-K 函数参数由 NN 分支 2 学习

### 3.2 BarrierNet 的双分支结构

```
┌──────────────────────────────────────────────────────┐
│              BarrierNet 双分支架构                     │
│                                                      │
│  输入 x ──► 共享隐藏层 ──┬──► 分支1 ──► q (参考控制)   │
│                         │                            │
│                         └──► 分支2 ──► 4·σ(p) (CBF参数)│
│                                                      │
│  ┌─────────────────────────────────────────┐         │
│  │  QP 层：min ½‖u‖² + qᵀu  s.t. Gu ≤ h   │         │
│  │  其中 G, h 由 CBF + p 构造              │         │
│  │  → 输出安全控制 u*                       │         │
│  └─────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────┘
```

### 3.3 关键实现细节

- **训练时**：使用 `qpth.QPFunction`（可微分，梯度可通过 KKT 条件的隐函数定理反传）
- **推理时**：使用 `cvxopt` 或 `cvxpy`（不可微分，但更快更稳定）
- **CBF 参数正性保证**：`p = 4 · sigmoid(raw_output)`，确保 class-K 函数参数严格为正

---

## 4. 加入 QP 后的整体架构变化

### 4.1 修改后的运行时数据流

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    修改后 SafePVC + QP 运行时数据流                               │
│                                                                                  │
│  状态 s_t ──► 感知 MLP g(s_t, z_t) ──► 图像 o_t                                 │
│                                                                                  │
│  o_t ──► 控制器 π_θ(o_t) ──┬──► 分支1 ──► q_t (参考控制, ℝⁿ)                     │
│                            │                                                     │
│                            └──► 分支2 ──► p_t = 4·σ(·) (CBF 参数, ℝᵏ)           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────┐                        │
│  │  s_t, q_t, p_t ──► QP 安全滤波层                      │                        │
│  │  min ½‖u‖² + q_tᵀ u                                  │                        │
│  │  s.t.  CBF 约束: G(s_t, p_t) u ≤ h(s_t, p_t)        │                        │
│  │  → 输出安全控制 u_t*                                   │                        │
│  └──────────────────────────────────────────────────────┘                        │
│                                                                                  │
│  u_t* ──► 动力学 f(s_t, u_t*) ──► s_{t+1}                                       │
│                                                                                  │
│  ★ 运行时安全约束（CBF-QP）+ 离线形式化验证（SBC）= 双重保障                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 核心变化总结

| 维度 | 原始 SafePVC | 修改后 SafePVC + QP |
|------|-------------|---------------------|
| 控制器输出 | 直接输出 $u_t$ | 输出参考控制 $q_t$ + CBF 参数 $p_t$ |
| 运行时安全 | 无（纯隐式） | QP-CBF 在线安全滤波 |
| 控制律 | $u_t = \pi(o_t)$ | $u_t^* = \text{QP}(q_t, p_t, s_t)$ |
| 安全保障 | 仅 SBC 离线验证 | SBC 验证 + CBF 在线保障（双重） |
| 网络结构 | 单输出头 | 双分支输出头 |
| 训练损失 | $\mathcal{L}_P$ (SBC+MSE) | $\mathcal{L}_P$ + CBF 约束损失 |

---

## 5. 修改后的网络结构详解

### 5.1 控制器网络 $\pi_\theta$ 的结构修改

**原始结构**（单输出头）：

```
o_t (图像/状态估计)
  │
  ▼
┌──────────────┐
│  FC + ReLU   │  隐藏层 1
├──────────────┤
│  FC + ReLU   │  隐藏层 2
├──────────────┤
│  FC + ReLU   │  隐藏层 3
├──────────────┤
│  FC (线性)   │  输出层 → u_t ∈ ℝⁿ
└──────────────┘
```

**修改后结构**（双分支输出头，参考 BarrierNet）：

```
o_t (图像/状态估计)
  │
  ▼
┌──────────────────────────────────────────┐
│          共享主干（Shared Backbone）       │
│                                          │
│  FC + ReLU   隐藏层 1  (dim_hidden)      │
│  FC + ReLU   隐藏层 2  (dim_hidden)      │
│  FC + ReLU   隐藏层 3  (dim_hidden)      │
│                                          │
│  ┌────────────────┬──────────────────┐   │
│  │  分支 1 (q)    │  分支 2 (p)      │   │
│  │                │                  │   │
│  │  FC (线性)     │  FC (线性)       │   │
│  │  → q_t ∈ ℝⁿ   │  → raw ∈ ℝᵏ     │   │
│  │  (参考控制)    │  → p_t = 4·σ(raw)│   │
│  │                │  (CBF 参数)      │   │
│  └────────────────┴──────────────────┘   │
└──────────────────────────────────────────┘
         │                      │
         ▼                      ▼
    q_t (参考控制)         p_t (CBF 增益)
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌────────────────────┐
         │   QP 安全滤波层     │
         │   + CBF 约束构造   │
         │   + 状态 s_t       │
         │                    │
         │  → u_t* ∈ ℝⁿ      │
         └────────────────────┘
```

### 5.2 各分支的输出维度

以论文中的两个场景为例：

| 场景 | 控制维度 $n$ | CBF 参数维度 $k$ | 说明 |
|------|-------------|-----------------|------|
| **X-Plane11 滑行** | 2 ($\phi_k$, $a_k$) | 2 ($p_1$, $p_2$) | HOCBF 需要 2 个 class-K 参数 |
| **CARLA 紧急制动** | 1 ($a_k$) | 1 ($p$) | OCBF 仅需 1 个 class-K 参数 |

### 5.3 激活函数选择

- **共享主干**：ReLU 或 Tanh（与原始 SafePVC 保持一致）
- **分支 1（参考控制）**：**线性输出**（不加激活），因为参考控制可正可负
- **分支 2（CBF 参数）**：$4 \cdot \text{Sigmoid}(\cdot)$，确保参数在 $(0, 4)$ 范围内，满足 class-K 函数的正性要求

### 5.4 为什么用 $4 \cdot \text{Sigmoid}$？

参考 BarrierNet 的实现：

```python
p = 4 * torch.sigmoid(raw_output)  # 输出范围 (0, 4)
```

- `sigmoid` 输出 $(0, 1)$，乘以 4 得到 $(0, 4)$
- 保证 CBF 的 class-K 函数参数严格为正（数学要求）
- 4 是一个经验上界，可根据具体问题调整（如改为 $10 \cdot \text{sigmoid}$ 以允许更大增益）

---

## 6. QP 层的数学公式

### 6.1 通用 QP 形式

在每个时间步 $t$，求解以下二次规划：

$$\boxed{u_t^* = \arg\min_{u \in \mathbb{R}^n} \quad \frac{1}{2} u^T Q u + q_t^T u}$$

$$\text{s.t.} \quad G(s_t, p_t) \, u \leq h(s_t, p_t)$$

其中：
- $Q = I_{n \times n}$（单位矩阵），表示最小化控制偏差
- $q_t \in \mathbb{R}^n$：NN 分支 1 输出的参考控制（线性代价项）
- $G(s_t, p_t) \in \mathbb{R}^{m \times n}$：CBF 约束矩阵（$m$ 为约束个数）
- $h(s_t, p_t) \in \mathbb{R}^m$：CBF 约束上界

### 6.2 CBF 约束的构造（HOCBF 情形）

对于相对阶为 2 的系统（如 X-Plane11 滑行），需要使用**高阶 CBF (HOCBF)**。

**定义障碍函数**：$b(s)$ 使得安全集 $\mathcal{C} = \{s : b(s) \geq 0\}$

**HOCBF 约束**：

$$L_g L_f b(s) \cdot u + L_f^2 b(s) + (p_1 + p_2) \cdot L_f b(s) + p_1 \cdot p_2 \cdot b(s) \geq 0$$

转换为 QP 标准不等式形式 $Gu \leq h$：

$$\underbrace{-L_g L_f b(s)}_{G(s,p)} \cdot u \leq \underbrace{L_f^2 b(s) + (p_1 + p_2) \cdot L_f b(s) + p_1 \cdot p_2 \cdot b(s)}_{h(s,p)}$$

其中：
- $L_f b(s)$：$b$ 沿 $f$ 的李导数
- $L_g L_f b(s)$：$b$ 沿 $f$ 再沿 $g$ 的二阶李导数
- $L_f^2 b(s)$：$b$ 沿 $f$ 的二阶李导数
- $p_1, p_2 > 0$：NN 学习的 class-K 函数参数

### 6.3 CBF 约束的构造（OCBF 情形）

对于相对阶为 1 的系统（如 CARLA 紧急制动），使用**标准 CBF (OCBF)**：

$$L_g b(s) \cdot u + L_f b(s) + p \cdot b(s) \geq 0$$

转换为 $Gu \leq h$：

$$\underbrace{-L_g b(s)}_{G(s,p)} \cdot u \leq \underbrace{L_f b(s) + p \cdot b(s)}_{h(s,p)}$$

### 6.4 多约束场景

当存在多个安全约束时（如避障 + 保持车道），将所有约束堆叠：

$$G = \begin{bmatrix} G_1 \\ G_2 \\ \vdots \\ G_m \end{bmatrix}, \quad h = \begin{bmatrix} h_1 \\ h_2 \\ \vdots \\ h_m \end{bmatrix}$$

其中每个 $(G_i, h_i)$ 对应一个独立的 CBF 约束。

### 6.5 控制输入边界约束

除了 CBF 约束，还应加入控制输入的物理边界：

$$u_{\min} \leq u \leq u_{\max}$$

转换为 QP 不等式：

$$\begin{bmatrix} I \\ -I \end{bmatrix} u \leq \begin{bmatrix} u_{\max} \\ -u_{\min} \end{bmatrix}$$

与 CBF 约束堆叠后，完整的约束矩阵为：

$$G_{\text{full}} = \begin{bmatrix} G_{\text{CBF}} \\ I \\ -I \end{bmatrix}, \quad h_{\text{full}} = \begin{bmatrix} h_{\text{CBF}} \\ u_{\max} \\ -u_{\min} \end{bmatrix}$$

---

## 7. 详细数据流

### 7.1 训练阶段数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     训练阶段完整数据流                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  阶段 1：感知模型训练（不变）                                        │    │
│  │  s, z ──► cGAN G(s,z) ──► 图像数据                                  │    │
│  │  s, z ──► MLP g(s,z) ──► 蒸馏逼近 G                                │    │
│  │  损失: L_distill = ‖G(s,z) - g(s,z)‖² + λ_lip · L_lip             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  阶段 2：RL 预训练（不变）                                           │    │
│  │  o_t ──► π₀(o_t) ──► u_t ──► 环境 ──► 奖励                        │    │
│  │  损失: PPO clipped surrogate                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  阶段 3：控制器 + QP 联合训练（★ 新增/修改）                         │    │
│  │                                                                     │    │
│  │  s_t ──► g(s_t, z₀) ──► o_t ──► π_θ(o_t) ──┬──► q_t (参考控制)    │    │
│  │                                              │                       │    │
│  │                                              └──► p_t (CBF 参数)    │    │
│  │                                                     │               │    │
│  │  s_t ─────────────────────────────┐                 │               │    │
│  │                                   ▼                 ▼               │    │
│  │  ┌────────────────────────────────────────────────────────┐         │    │
│  │  │  CBF 约束构造 (利用 s_t, p_t 计算 G, h)                │         │    │
│  │  │  b(s), Lfb(s), Lf²b(s), LgLfb(s) ──► G, h            │         │    │
│  │  └────────────────────────────────────────────────────────┘         │    │
│  │                                   │                                 │    │
│  │                                   ▼                                 │    │
│  │  ┌────────────────────────────────────────────────────────┐         │    │
│  │  │  QP 求解 (qpth.QPFunction，可微分)                     │         │    │
│  │  │  min ½‖u‖² + q_tᵀ u  s.t.  G·u ≤ h                   │         │    │
│  │  │  → u_t* (安全控制)                                     │         │    │
│  │  └────────────────────────────────────────────────────────┘         │    │
│  │                                   │                                 │    │
│  │                                   ▼                                 │    │
│  │  u_t* ──► f(s_t, u_t*) ──► s_{t+1} = f(s_t, u_t*) + Δs            │    │
│  │                                   │                                 │    │
│  │                                   ▼                                 │    │
│  │  ┌────────────────────────────────────────────────────────┐         │    │
│  │  │  损失计算                                              │         │    │
│  │  │  L_P = L_dec_P + λ_P · L_lip_P + λ_M · L_mse          │         │    │
│  │  │       + λ_CBF · L_CBF    (★ 新增 CBF 损失项)           │         │    │
│  │  └────────────────────────────────────────────────────────┘         │    │
│  │                                   │                                 │    │
│  │                                   ▼                                 │    │
│  │  梯度反传（经过 QPFunction 的隐式微分）──► 更新 θ                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  阶段 4：SBC 验证（修改）                                           │    │
│  │  在闭环系统中，控制律变为 u_t* = QP(q_t, p_t, s_t)                  │    │
│  │  SBC 需验证含 QP 层的闭环系统                                       │    │
│  │  反例 → 交替更新 SBC 和 (π + QP)                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 推理/部署阶段数据流

```
┌──────────────────────────────────────────────────────────────────┐
│                  推理/部署阶段数据流                               │
│                                                                  │
│  摄像头 ──► 图像 o_t                                             │
│                │                                                 │
│                ▼                                                 │
│  ┌────────────────────────────────────┐                          │
│  │  控制器 π_θ (双分支)               │                          │
│  │  ├─ 分支1 → q_t (参考控制)        │                          │
│  │  └─ 分支2 → p_t = 4·σ(·) (CBF参数)│                          │
│  └────────────┬───────────────────────┘                          │
│               │                                                  │
│               ▼                                                  │
│  ┌────────────────────────────────────┐                          │
│  │  QP 求解器 (cvxopt/cvxpy)         │                          │
│  │  ★ 推理时用非可微分求解器，更快     │                          │
│  │  min ½‖u‖² + q_tᵀ u              │                          │
│  │  s.t. G(s_t, p_t)·u ≤ h(s_t,p_t) │                          │
│  │       u_min ≤ u ≤ u_max           │                          │
│  │  → u_t*                            │                          │
│  └────────────┬───────────────────────┘                          │
│               │                                                  │
│               ▼                                                  │
│  u_t* ──► 执行器 ──► 车辆/飞机                                   │
│                                                                  │
│  ★ 每一步都有 CBF 安全保证：即使 NN 输出异常，QP 也会修正        │
└──────────────────────────────────────────────────────────────────┘
```

### 7.3 梯度反传路径（训练时）

```
损失 L
  │
  ▼
  ∂L/∂u_t*   (QP 输出的梯度)
  │
  ▼
  ┌─────────────────────────────────────────────────┐
  │  qpth.QPFunction 的隐式微分                      │
  │                                                  │
  │  利用 KKT 条件的隐函数定理：                      │
  │  ∂u*/∂q = -M⁻¹ · ∂KKT/∂q                       │
  │  ∂u*/∂G = ...                                    │
  │  ∂u*/∂h = ...                                    │
  │                                                  │
  │  其中 M 是 KKT 系统的 Jacobian                   │
  └────┬──────────────┬──────────────┬───────────────┘
       │              │              │
       ▼              ▼              ▼
  ∂L/∂q_t        ∂L/∂G          ∂L/∂h
  (分支1 梯度)    (通过 s_t)     (通过 s_t, p_t)
       │              │              │
       │              │              ▼
       │              │         ∂L/∂p_t
       │              │         (分支2 梯度)
       │              │              │
       ▼              ▼              ▼
  ┌─────────────────────────────────────────────┐
  │  标准 PyTorch 反向传播                       │
  │  更新 π_θ 的共享主干 + 两个分支的权重        │
  └─────────────────────────────────────────────┘
```

---

## 8. 训练流程变化

### 8.1 原始训练流程

```
1. 训练 cGAN G(s,z)
2. 蒸馏 G → MLP g(s,z)
3. PPO 预训练控制器 π₀
4. 构建 VCLS
5. 估计扰动分布 Δs ~ μ
6. 循环:
   a. 训练 SBC B(s)  (损失 L_L)
   b. 每 10 轮，训练控制器 π  (损失 L_P)
   c. 重新估计扰动分布
   d. 验证 SBC 条件 (α-β-CROWN + Theorem 3.1)
   e. 若验证通过 → 终止
```

### 8.2 修改后的训练流程

```
1. 训练 cGAN G(s,z)                          ← 不变
2. 蒸馏 G → MLP g(s,z)                       ← 不变
3. PPO 预训练控制器 π₀                        ← 不变
4. ★ 修改控制器为双分支结构 π_θ（参考 5.1 节）
5. ★ 定义 CBF 障碍函数 b(s) 和约束构造
6. 构建 VCLS（★ 含 QP 层）
7. 估计扰动分布 Δs ~ μ
8. 循环:
   a. 训练 SBC B(s)  (损失 L_L)              ← 不变
   b. 每 10 轮，训练控制器 π_θ + QP:         ← 修改
      i.   前向传播：o → π_θ → (q, p) → CBF构造 → QP → u*
      ii.  计算损失 L_P + λ_CBF · L_CBF     ← ★ 新增 CBF 损失
      iii. 反向传播（梯度穿过 QPFunction）
      iv.  更新 θ
   c. 重新估计扰动分布
   d. 验证 SBC 条件（★ 闭环系统包含 QP 层）
   e. 若验证通过 → 终止
```

### 8.3 关键修改点

1. **步骤 4**：控制器网络从单输出改为双分支
2. **步骤 5（新增）**：定义 CBF 障碍函数 $b(s)$
3. **步骤 6**：VCLS 的闭环映射变为 $s_{t+1} = f(s_t, \text{QP}(\pi_\theta(g(s_t, z_0)), s_t)) + \Delta s$
4. **步骤 8.b**：训练损失增加 CBF 项
5. **步骤 8.d**：SBC 验证的 Lipschitz 常数需考虑 QP 层的 Lipschitz 性质

---

## 9. 损失函数变化

### 9.1 原始控制器损失

$$\mathcal{L}_P = \mathcal{L}_{\text{dec\_P}} + \lambda_P \mathcal{L}_{\text{lip\_P}} + \lambda_M \mathcal{L}_{\text{mse}}$$

### 9.2 修改后的控制器损失

$$\boxed{\mathcal{L}_P^{\text{new}} = \mathcal{L}_{\text{dec\_P}} + \lambda_P \mathcal{L}_{\text{lip\_P}} + \lambda_M \mathcal{L}_{\text{mse}} + \lambda_{\text{CBF}} \mathcal{L}_{\text{CBF}}}$$

### 9.3 新增的 CBF 损失 $\mathcal{L}_{\text{CBF}}$

此损失用于**辅助训练**，鼓励 NN 学习输出使得 CBF 约束更容易被满足的参考控制和参数：

$$\mathcal{L}_{\text{CBF}} = \mathbb{E}_{s_t} \left[ \max\left(0, \; -\left(L_g L_f b(s_t) \cdot u_t^* + L_f^2 b(s_t) + (p_1 + p_2) L_f b(s_t) + p_1 p_2 \cdot b(s_t)\right)\right) \right]$$

直观理解：
- 如果 QP 求解成功且 CBF 约束满足 → 括号内 $\geq 0$ → $\mathcal{L}_{\text{CBF}} = 0$
- 如果约束被违反（QP 无可行解时） → $\mathcal{L}_{\text{CBF}} > 0$ → 惩罚网络

### 9.4 $\mathcal{L}_{\text{mse}}$ 的变化

原始的 MSE 损失是控制器输出与 RL 预训练策略之间的距离。加入 QP 后，MSE 应改为 QP 输出与 RL 策略之间的距离：

$$\mathcal{L}_{\text{mse}}^{\text{new}} = \mathbb{E}_{o_t} \left[ \| u_t^* - \pi_0(o_t) \|_2^2 \right]$$

其中 $u_t^* = \text{QP}(q_t, p_t, s_t)$ 是 QP 过滤后的安全控制。

这样网络会学习输出使得 QP 过滤后的控制尽量接近 RL 预训练策略。

### 9.5 超参数建议

| 超参数 | 建议值 | 说明 |
|--------|--------|------|
| $\lambda_{\text{CBF}}$ | 1.0 ~ 10.0 | CBF 损失权重，初始可设为 1.0，若约束违反严重可增大 |
| $\lambda_M$ | 0.1 ~ 1.0 | MSE 权重，保持与原始论文一致 |
| $\lambda_P$ | 0.01 ~ 0.1 | Lipschitz 正则化权重 |

---

## 10. 对 SBC 验证的影响

### 10.1 闭环系统的变化

**原始闭环**：
$$s_{t+1} = \tilde{F}(s_t, z_0, \Delta s) = f(s_t, \pi(g(s_t, z_0))) + \Delta s$$

**修改后闭环**：
$$s_{t+1} = \tilde{F}^{\text{QP}}(s_t, z_0, \Delta s) = f\left(s_t, \; \text{QP}\left(\pi_\theta(g(s_t, z_0)), \; s_t\right)\right) + \Delta s$$

### 10.2 Lipschitz 常数的变化

原始 Theorem 3.1 中的 Lipschitz 常数：

$$K = \tau \cdot L_B \cdot (1 + L_f \sqrt{1 + (L_\pi L_g)^2})$$

修改后需加入 QP 层的 Lipschitz 常数 $L_{\text{QP}}$：

$$K^{\text{new}} = \tau \cdot L_B \cdot \left(1 + L_f \sqrt{1 + (L_{\text{QP}} \cdot L_\pi \cdot L_g)^2}\right)$$

其中 $L_{\text{QP}}$ 是 QP 解关于其输入的 Lipschitz 常数。

**关键优势**：由于 QP 层在运行时提供了 CBF 安全保障，SBC 验证的条件 (iv)（期望递减条件）更容易满足——QP 已经将系统状态约束在安全集内，减少了 SBC 需要"覆盖"的不安全区域。

### 10.3 SBC 与 CBF 的关系

| | SBC（离线验证） | CBF-QP（在线保障） |
|---|---|---|
| **作用时间** | 离线 | 运行时每一步 |
| **保障类型** | 概率安全性 $\mathbb{P}[\text{Safe}] \geq p$ | 确定性前向不变性 $b(s_t) \geq 0 \Rightarrow b(s_{t+1}) \geq 0$ |
| **覆盖范围** | 整个状态空间（通过网格验证） | 当前轨迹附近 |
| **局限性** | 依赖 Lipschitz 界，可能保守 | 仅保证单步安全，不提供无限时域概率界 |
| **互补性** | CBF 使 SBC 更易验证 | SBC 提供 CBF 无法给出的概率界 |

---

## 11. 具体实施步骤

### 步骤 1：定义障碍函数 $b(s)$

根据具体场景，定义安全集的障碍函数：

```python
def barrier_function(state):
    """
    state: 系统状态张量
    返回: b(s)，b(s) >= 0 表示安全
    """
    # 例：与障碍物的距离平方减去安全半径平方
    # b(s) = ||s - s_obs||^2 - R^2
    raise NotImplementedError
```

### 步骤 2：计算 CBF 约束中的各李导数

```python
def compute_cbf_components(state, dynamics_fn):
    """
    计算 HOCBF 所需的各阶李导数
    返回: b, Lfb, Lf2b, LgLfb
    """
    # b(s): 障碍函数值
    # Lf b(s): b 沿 f 的李导数
    # Lf^2 b(s): b 沿 f 的二阶李导数
    # Lg Lf b(s): b 沿 f 再沿 g 的李导数（控制影响项）
    raise NotImplementedError
```

### 步骤 3：修改控制器网络为双分支

```python
import torch
import torch.nn as nn
from qpth.qp import QPFunction
from torch.autograd import Variable

class SafePVCController(nn.Module):
    def __init__(self, input_dim, control_dim, cbf_param_dim, hidden_dim=256):
        super().__init__()
        self.control_dim = control_dim
        self.cbf_param_dim = cbf_param_dim

        # 共享主干
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 分支 1: 参考控制 (线性输出)
        self.q_head = nn.Linear(hidden_dim, control_dim)

        # 分支 2: CBF 参数 (4*sigmoid 保证正性)
        self.p_head = nn.Linear(hidden_dim, cbf_param_dim)

    def forward(self, obs, state=None, solver='qpth'):
        """
        obs: 观测输入 (图像或状态估计)
        state: 低维状态 (用于 CBF 约束构造)
        solver: 'qpth' (训练) 或 'cvxopt' (推理)
        """
        features = self.shared(obs)

        # 分支 1: 参考控制
        q = self.q_head(features)  # (batch, control_dim)

        # 分支 2: CBF 参数
        p_raw = self.p_head(features)
        p = 4.0 * torch.sigmoid(p_raw)  # (batch, cbf_param_dim), 保证 > 0

        # QP 安全滤波
        if state is not None:
            u_safe = self.solve_qp(q, p, state, solver)
            return u_safe, q, p
        else:
            return q  # 无状态时退化为直接输出

    def solve_qp(self, q, p, state, solver='qpth'):
        """
        求解 CBF-QP:
        min  0.5 * u^T Q u + q^T u
        s.t. G(s, p) u <= h(s, p)
        """
        batch_size = q.shape[0]
        n = self.control_dim

        # 代价矩阵 Q = I
        Q = Variable(torch.eye(n).unsqueeze(0).expand(batch_size, n, n))

        # 构造 CBF 约束 G, h
        G, h = self.build_cbf_constraints(state, p)

        # 无等式约束
        e = Variable(torch.Tensor())

        if solver == 'qpth':
            # 训练时：可微分 QP 求解
            u_safe = QPFunction(verbose=-1)(
                Q.double(), q.double(),
                G.double(), h.double(),
                e, e
            )
            return u_safe.float()
        else:
            # 推理时：cvxopt 求解
            return self.solve_qp_cvxopt(Q, q, G, h)

    def build_cbf_constraints(self, state, p):
        """
        ★ 需要根据具体场景实现 ★
        根据状态 s 和 CBF 参数 p 构造约束矩阵 G 和上界 h
        """
        raise NotImplementedError("需要根据具体场景实现")

    def solve_qp_cvxopt(self, Q, q, G, h):
        """推理时使用 cvxopt 求解"""
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        batch_size = q.shape[0]
        results = []
        for i in range(batch_size):
            Q_np = Q[i].detach().cpu().numpy()
            q_np = q[i].detach().cpu().numpy()
            G_np = G[i].detach().cpu().numpy()
            h_np = h[i].detach().cpu().numpy()

            sol = solvers.qp(
                matrix(Q_np), matrix(q_np),
                matrix(G_np), matrix(h_np)
            )
            results.append(torch.tensor(sol['x'], dtype=q.dtype).squeeze())

        return torch.stack(results).to(q.device)
```

### 步骤 4：实现场景特定的 CBF 约束

#### X-Plane11 滑行场景（HOCBF）

```python
def build_cbf_constraints_xplane(self, state, p):
    """
    state: (p_x, p_y, θ, v) — 位置、航向角、速度
    p: (p1, p2) — HOCBF class-K 参数
    障碍物: 圆心 (obs_x, obs_y), 半径 R
    """
    px, py, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    p1, p2 = p[:, 0], p[:, 1]

    obs_x, obs_y, R = 0.0, 0.0, 5.0  # 根据实际场景设定

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # 障碍函数
    b = (px - obs_x)**2 + (py - obs_y)**2 - R**2

    # 一阶李导数 Lf b
    Lfb = 2*(px - obs_x)*v*cos_theta + 2*(py - obs_y)*v*sin_theta

    # 二阶李导数
    Lf2b = 2 * v**2  # Lf²b

    # 控制影响项 Lg Lf b (对 u1=θ̇ 和 u2=a 的偏导)
    LgLfb_u1 = -2*(px - obs_x)*v*sin_theta + 2*(py - obs_y)*v*cos_theta
    LgLfb_u2 = 2*(px - obs_x)*cos_theta + 2*(py - obs_y)*sin_theta

    batch_size = px.shape[0]

    # G = -[LgLfb_u1, LgLfb_u2]  shape: (batch, 1, 2)
    G = torch.cat([-LgLfb_u1.unsqueeze(1), -LgLfb_u2.unsqueeze(1)], dim=1)
    G = G.reshape(batch_size, 1, 2)

    # h = Lf²b + (p1+p2)·Lf b + p1·p2·b  shape: (batch, 1)
    h = Lf2b + (p1 + p2)*Lfb + p1*p2*b
    h = h.reshape(batch_size, 1)

    return G, h
```

#### CARLA 紧急制动场景（OCBF）

```python
def build_cbf_constraints_carla(self, state, p):
    """
    state: (d, v) — 距离、速度
    p: (p1,) — OCBF class-K 参数
    安全规则: 保持安全距离 d_safe = v * t_gap
    """
    d, v = state[:, 0], state[:, 1]
    p1 = p[:, 0]

    t_gap = 1.5  # 安全时间间隔 (秒)

    # 障碍函数: b(s) = d - v * t_gap (安全裕度)
    b = d - v * t_gap

    # 李导数
    # 动力学: d_{k+1} = d_k - v_k * dt,  v_{k+1} = v_k - u * dt
    # Lf b = ∂b/∂d · f_d + ∂b/∂v · f_v
    #      = 1 · (-v) + (-t_gap) · (0)  = -v
    Lfb = -v

    # Lg b = ∂b/∂v · g_v = (-t_gap) · (-dt) = t_gap * dt
    # 注意: 控制 u 是加速度 (减速为负), v_{k+1} = v_k + u * dt
    dt = 0.1  # 时间步长
    Lgb = -t_gap * dt  # 控制 u 对 b 的影响

    batch_size = d.shape[0]

    # G = -Lgb  shape: (batch, 1, 1)
    G = (-Lgb * torch.ones(batch_size, 1, 1))

    # h = Lfb + p1 * b  shape: (batch, 1)
    h = (Lfb + p1 * b).reshape(batch_size, 1)

    return G, h
```

### 步骤 5：修改训练循环

```python
# === 控制器训练步骤 (每个 epoch) ===

for s_t, o_t, s_next_gt in dataloader:
    # 1. 前向传播: 控制器 + QP
    u_safe, q_ref, p_cbf = controller(o_t, state=s_t, solver='qpth')

    # 2. 闭环动力学
    s_next_pred = dynamics(s_t, u_safe)

    # 3. 计算损失
    # 3a. SBC 期望递减损失 (用反例)
    B_st = sbc_network(s_t)
    B_snext = sbc_network(s_next_pred)
    L_dec_P = torch.mean(torch.relu(B_snext - gamma * B_st + epsilon))

    # 3b. Lipschitz 正则化
    L_lip_P = lipschitz_regularization(controller)

    # 3c. MSE: QP 输出 vs RL 预训练策略
    with torch.no_grad():
        u_rl = pretrained_policy(o_t)
    L_mse = torch.mean((u_safe - u_rl)**2)

    # 3d. ★ 新增: CBF 约束损失
    b_val = barrier_function(s_t)
    cbf_violation = compute_cbf_violation(s_t, u_safe, p_cbf)
    L_CBF = torch.mean(torch.relu(-cbf_violation))

    # 4. 总损失
    loss = (L_dec_P
            + lambda_P * L_lip_P
            + lambda_M * L_mse
            + lambda_CBF * L_CBF)

    # 5. 反向传播 (梯度自动穿过 QPFunction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 12. 以 X-Plane11 滑行为例的完整推导

### 12.1 系统动力学

$$p_{k+1} = p_k + v \Delta t \sin\theta_k$$
$$\theta_{k+1} = \theta_k + \frac{v}{L} \Delta t \tan\phi_k$$

控制输入：$u = (\dot{\phi}_k, a_k)$（转向角速率和加速度）

### 12.2 安全场景定义

假设飞机需要在跑道上滑行，避开跑道边的障碍物：
- 障碍物位置：$(obs_x, obs_y)$
- 安全半径：$R$

### 12.3 障碍函数

$$b(s) = (p_x - obs_x)^2 + (p_y - obs_y)^2 - R^2$$

### 12.4 HOCBF 约束推导

由于控制 $u$ 不直接出现在 $\dot{b}(s)$ 中（相对阶为 2），需要二阶条件：

**一阶导数**：
$$\dot{b} = 2(p_x - obs_x)\dot{p}_x + 2(p_y - obs_y)\dot{p}_y = 2(p_x - obs_x)v\cos\theta + 2(p_y - obs_y)v\sin\theta$$

**二阶导数**（含控制项）：
$$\ddot{b} = L_f^2 b + L_g L_f b \cdot u$$

其中：
- $L_f^2 b = 2v^2$
- $L_g L_f b = \begin{bmatrix} -2(p_x - obs_x)v\sin\theta + 2(p_y - obs_y)v\cos\theta \\ 2(p_x - obs_x)\cos\theta + 2(p_y - obs_y)\sin\theta \end{bmatrix}^T$

**HOCBF 条件**：
$$L_g L_f b \cdot u + L_f^2 b + (p_1 + p_2)\dot{b} + p_1 p_2 b \geq 0$$

### 12.5 对应的 QP

$$\min_{u \in \mathbb{R}^2} \quad \frac{1}{2}\|u\|^2 + q^T u$$

$$\text{s.t.} \quad -L_g L_f b \cdot u \leq L_f^2 b + (p_1 + p_2)\dot{b} + p_1 p_2 b$$

$$\quad u_{\min} \leq u \leq u_{\max}$$

### 12.6 网络结构参数

```
输入: o_t ∈ ℝ^{H×W} (灰度图像) 或状态估计 ŝ ∈ ℝ⁴
共享主干: FC(4→256) → ReLU → FC(256→256) → ReLU → FC(256→256) → ReLU
分支1 (q): FC(256→2)  — 参考控制 (转向角速率, 加速度)
分支2 (p): FC(256→2) → 4·sigmoid — CBF 参数 (p₁, p₂)
QP: 2 变量, 1 CBF 约束 + 4 边界约束
```

---

## 13. 以 CARLA 紧急制动为例的完整推导

### 13.1 系统动力学

$$d_{k+1} = d_k - v_k \Delta t$$
$$v_{k+1} = v_k - a_k \Delta t$$

控制输入：$u = a_k$（制动加速度，标量）

### 13.2 安全场景定义

自车需要在前车突然制动时安全停车：
- $d$：与前车的距离
- $v$：自车速度
- 安全规则：保持 $t_{\text{gap}}$ 秒的安全时距

### 13.3 障碍函数

$$b(s) = d - v \cdot t_{\text{gap}}$$

$b(s) \geq 0$ 表示距离足以在 $t_{\text{gap}}$ 秒内安全停车。

### 13.4 OCBF 约束推导

控制 $u = a$ 直接出现在 $\dot{b}$ 中（相对阶为 1），使用标准 CBF：

$$\dot{b} = \frac{\partial b}{\partial d}\dot{d} + \frac{\partial b}{\partial v}\dot{v} = 1 \cdot (-v) + (-t_{\text{gap}}) \cdot (-a) = -v + t_{\text{gap}} \cdot a$$

**OCBF 条件**：
$$\dot{b} + p \cdot b \geq 0$$
$$(-v + t_{\text{gap}} \cdot a) + p(d - v \cdot t_{\text{gap}}) \geq 0$$
$$t_{\text{gap}} \cdot a \geq v - p(d - v \cdot t_{\text{gap}})$$

### 13.5 对应的 QP

$$\min_{a \in \mathbb{R}} \quad \frac{1}{2}a^2 + q \cdot a$$

$$\text{s.t.} \quad -t_{\text{gap}} \cdot a \leq -v + p(d - v \cdot t_{\text{gap}})$$

$$\quad a_{\min} \leq a \leq a_{\max}$$

### 13.6 网络结构参数

```
输入: o_t ∈ ℝ^{H×W} (前方摄像头图像) 或状态估计 ŝ ∈ ℝ²
共享主干: FC(2→256) → ReLU → FC(256→256) → ReLU → FC(256→256) → ReLU
分支1 (q): FC(256→1)  — 参考加速度 (标量)
分支2 (p): FC(256→1) → 4·sigmoid — CBF 参数 p
QP: 1 变量, 1 CBF 约束 + 2 边界约束
```

---

## 14. 代码结构建议

### 14.1 目录结构

```
artical-F122/
├── Combined_network/
│   ├── models/
│   │   ├── cgan.py               # cGAN 生成器 (不变)
│   │   ├── distill_mlp.py        # 蒸馏 MLP (不变)
│   │   ├── controller.py         # ★ 修改: 双分支控制器
│   │   ├── qp_layer.py           # ★ 新增: QP 安全滤波层
│   │   ├── cbf_constraints.py    # ★ 新增: CBF 约束构造
│   │   └── sbc_network.py        # SBC 网络 (不变)
│   ├── training/
│   │   ├── train_sbc.py          # SBC 训练 (不变)
│   │   ├── train_controller.py   # ★ 修改: 含 QP 的控制器训练
│   │   └── verify.py             # ★ 修改: 验证含 QP 的闭环系统
│   └── utils/
│       ├── dynamics.py           # 动力学模型 (不变)
│       └── lipschitz.py          # Lipschitz 估计 (★ 修改: 含 QP 层)
```

### 14.2 核心模块划分

```
┌─────────────────────────────────────────────────┐
│  controller.py                                   │
│  ┌─────────────────────────────────────┐        │
│  │  SafePVCController(nn.Module)       │        │
│  │  - shared backbone                  │        │
│  │  - q_head (参考控制)                │        │
│  │  - p_head (CBF 参数)                │        │
│  │  - forward() → (u_safe, q, p)      │        │
│  └────────────┬────────────────────────┘        │
│               │                                  │
│               ▼                                  │
│  ┌─────────────────────────────────────┐        │
│  │  qp_layer.py                        │        │
│  │  QPSafeFilter(nn.Module)            │        │
│  │  - solve_qp(q, p, state)           │        │
│  │  - 训练: qpth.QPFunction           │        │
│  │  - 推理: cvxopt                     │        │
│  └────────────┬────────────────────────┘        │
│               │                                  │
│               ▼                                  │
│  ┌─────────────────────────────────────┐        │
│  │  cbf_constraints.py                 │        │
│  │  XPlaneCBF / CARLACBF              │        │
│  │  - barrier_function(state)          │        │
│  │  - compute_lie_derivatives(state)   │        │
│  │  - build_constraints(state, p)      │        │
│  │    → (G, h) for QP                 │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

---

## 15. 对比总结表

### 15.1 方法对比

| 方面 | SafePVC（原始） | SafePVC + QP（修改后） | BarrierNet |
|------|-----------------|----------------------|------------|
| **运行时安全** | ❌ 无 | ✅ CBF-QP | ✅ CBF-QP |
| **离线验证** | ✅ SBC | ✅ SBC | ❌ 无 |
| **概率安全界** | ✅ $\mathbb{P} \geq p$ | ✅ $\mathbb{P} \geq p$（更强） | ❌ 无 |
| **感知模型** | ✅ cGAN+MLP | ✅ cGAN+MLP | ❌ 仅 Driving 有 StateNet |
| **控制输出** | 直接输出 | QP 过滤后输出 | QP 过滤后输出 |
| **CBF 参数** | 无 | NN 学习 | NN 学习 |
| **无限时域** | ✅ | ✅ | ❌（仅瞬时） |

### 15.2 工作量评估

| 任务 | 难度 | 说明 |
|------|------|------|
| 修改控制器为双分支 | ⭐ 低 | 添加一个线性层 + sigmoid |
| 实现 CBF 约束构造 | ⭐⭐ 中 | 需要推导具体场景的李导数 |
| 集成 QP 层 | ⭐⭐ 中 | 使用 qpth 库，代码量小 |
| 修改训练损失 | ⭐ 低 | 添加 CBF 损失项 |
| 修改 SBC 验证 | ⭐⭐⭐ 高 | 需处理 QP 层的 Lipschitz 常数 |
| 调参 | ⭐⭐ 中 | 新增 $\lambda_{\text{CBF}}$ 等超参数 |

### 15.3 预期收益

1. **运行时安全保障**：即使遇到训练分布外的状态，CBF-QP 也能防止不安全动作被执行
2. **更容易通过 SBC 验证**：QP 层限制了不安全行为，SBC 的期望递减条件更容易满足
3. **更高的概率安全下界**：双重安全机制有望提升 $p$ 值（如从 92.1% 提升至 >95%）
4. **更强的鲁棒性**：CBF 约束可以显式编码物理安全规则（距离、速度限制等）

---

## 附录 A：依赖库安装

```bash
# QP 求解器
pip install qpth        # 可微分 QP（训练用）
pip install cvxopt      # 非可微分 QP（推理用）
pip install cvxpy       # 备选 QP 求解器

# 验证工具（原有依赖）
pip install auto-LiRPA  # α-β-CROWN 区间传播
```

## 附录 B：qpFunction 调用参数说明

```python
from qpth.qp import QPFunction

# QPFunction 的标准形式:
# min  0.5 * x^T Q x + q^T x
# s.t. Gx <= h
#      Ax = b

x = QPFunction(verbose=-1)(Q, q, G, h, A, b)
# Q: (batch, n, n) — 二次代价矩阵
# q: (batch, n)    — 线性代价向量
# G: (batch, m, n) — 不等式约束矩阵
# h: (batch, m)    — 不等式约束上界
# A: (batch, p, n) — 等式约束矩阵 (本方案中为空)
# b: (batch, p)    — 等式约束右端 (本方案中为空)
# x: (batch, n)    — 最优解

# 无等式约束时:
e = torch.Tensor()  # 空张量
x = QPFunction(verbose=-1)(Q, q, G, h, e, e)
```

## 附录 C：常见问题与解决方案

### C.1 QP 无可行解

**现象**：QP 求解器返回 None 或异常值

**原因**：CBF 约束与输入边界约束冲突

**解决方案**：
1. 引入松弛变量 $\delta \geq 0$，将硬约束转为软约束：
   $$G u \leq h + \delta, \quad \text{代价增加} \quad \lambda_\delta \delta^2$$
2. 减小 CBF 参数 $p$ 的上界
3. 增大控制输入边界

### C.2 梯度爆炸（QP 层）

**现象**：训练不稳定，loss 突然变为 NaN

**解决方案**：
1. 对 QP 输入做梯度裁剪：`torch.nn.utils.clip_grad_norm_`
2. 使用 `torch.double()` 进行 QP 计算（提高数值精度）
3. 减小学习率

### C.3 SBC 验证不通过

**现象**：$\alpha$-$\beta$-CROWN 无法验证 SBC 条件

**原因**：QP 层引入了额外的非线性，增大了 Lipschitz 常数

**解决方案**：
1. 增加 SBC 网络容量
2. 增加反例数量
3. 在 QP 层之前对控制器输出做 Lipschitz 正则化
4. 考虑将 QP 层在验证时近似为分段线性函数
