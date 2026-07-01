# BarrierNet + SafePVC 结合方案：基于可微控制障碍函数的可验证概率安全视觉控制器

> **文档版本**: v1.0 | **日期**: 2026/06/26  
> **项目基础**: SafePVC (artical-F122) — Provably Probabilistic Safe Controller Synthesis for Vision-Based NNCS  
> **结合方法**: BarrierNet — Differentiable Control Barrier Functions for Learning of Safe Robot Control (IEEE TRO 2023)

---

## 目录

- [一、研究背景与动机](#一研究背景与动机)
- [二、三篇论文原理详解](#二三篇论文原理详解)
  - [2.1 BarrierNet 论文原理](#21-barriernet-论文原理)
  - [2.2 DiffCBF for Vision-based Driving 论文原理](#22-diffcbf-for-vision-based-driving-论文原理)
  - [2.3 SafePVC (720_file_Paper) 论文原理](#23-safePVC-720_file_paper-论文原理)
- [三、近年相关研究调查](#三近年相关研究调查)
- [四、可行性分析与创新贡献](#四可行性分析与创新贡献)
- [五、结合方案设计](#五结合方案设计)
  - [5.1 总体思路](#51-总体思路)
  - [5.2 新系统架构](#52-新系统架构)
  - [5.3 数学原理](#53-数学原理)
- [六、完整项目框架结构](#六完整项目框架结构)
- [七、数据流详解](#七数据流详解)
- [八、网络结构详解](#八网络结构详解)
- [九、变量定义与符号表](#九变量定义与符号表)
- [十、代码修改方案](#十代码修改方案)
  - [10.1 新增文件清单](#101-新增文件清单)
  - [10.2 修改现有文件](#102-修改现有文件)
  - [10.3 新增模块实现](#103-新增模块实现)
- [十一、训练流程](#十一训练流程)
- [十二、验证流程](#十二验证流程)
- [十三、潜在挑战与解决方案](#十三潜在挑战与解决方案)
- [十四、实验设计](#十四实验设计)
- [十五、总结](#十五总结)

---

## 一、研究背景与动机

### 1.1 问题陈述

视觉神经网络控制系统 (Vision-based Neural Network Control Systems, NNCS) 的安全保障面临双重挑战：

1. **高维感知空间**：视觉输入维度极高（如 32×32=1024 维），传统形式化验证方法难以直接处理
2. **随机扰动**：环境不确定性（光照、天气等）通过感知系统传播到控制决策，导致闭环系统的随机性

### 1.2 两条技术路线

| 特性 | SafePVC (当前项目) | BarrierNet (结合目标) |
|------|-------------------|---------------------|
| **安全保证类型** | 概率安全（随机障碍证书 SBC） | 确定性安全（可微控制障碍函数 dCBF） |
| **理论基础** | Martingale 理论 + 鞅超水平 | 高阶控制障碍函数 (HOCBF) + 可微 QP |
| **验证方法** | IBP 区间传播 + CEGIS 反例引导 | 端到端训练 + Lyapunov 型保证 |
| **保守性** | 概率性保证（非绝对） | 可学习降低保守性 |
| **计算开销** | 网格离散化 + IBP 验证 | 可微 QP 前向/后向传播 |
| **环境适应性** | 固定扰动分布 | 环境自适应（penalty functions） |
| **控制平滑性** | 无特殊处理 | 输出高相对度控制量（加速度/转向率） |

### 1.3 结合动机

**核心想法**：将 BarrierNet 的**可微 QP 安全层**嵌入 SafePVC 的**概率安全验证框架**中，同时获得：
- BarrierNet 的**确定性安全保证** + **低保守性** + **环境自适应性**
- SafePVC 的**形式化概率验证** + **CEGIS 反例引导训练**

这产生了一个**双层安全保障系统**：
- **内层** (BarrierNet)：实时运行，通过 dCBF 约束确保每一步控制都满足安全约束
- **外层** (SBC 验证)：离线验证，通过鞅理论证明闭环系统在无限时间域上的概率安全

---

## 二、三篇论文原理详解

### 2.1 BarrierNet 论文原理

> **论文**: *BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control*  
> **作者**: Wei Xiao, Tsun-Hsuan Wang, Ramin Hasani 等 (MIT CSAIL)  
> **发表**: IEEE Transactions on Robotics, Vol. 39, No. 3, June 2023

#### 2.1.1 控制障碍函数 (CBF) 基础

**仿射控制系统**：

$$\dot{x} = f(x) + g(x)u \quad \text{...(1)}$$

其中 $x \in \mathbb{R}^n$ 为系统状态，$u \in U \subset \mathbb{R}^q$ 为控制输入。

**Class K 函数**：连续函数 $\alpha: [0,a) \to [0,\infty)$，严格递增且 $\alpha(0)=0$。扩展 Class K 函数 $\beta: \mathbb{R} \to \mathbb{R}$ 定义在全域上。

**相对度 (Relative Degree)**：函数 $b(x)$ 对系统 (1) 的相对度 $m$ 是使得控制 $u$ 首次显式出现在 $b$ 的第 $m$ 阶导数中所需的微分次数。

#### 2.1.2 高阶控制障碍函数 (HOCBF)

对于安全约束 $b(x) \geq 0$（相对度 $m$），定义函数序列：

$$\psi_0(x) := b(x)$$
$$\psi_i(x) := \dot{\psi}_{i-1}(x) + \alpha_i(\psi_{i-1}(x)), \quad i \in \{1, \ldots, m\} \quad \text{...(2)}$$

其中 $\alpha_i(\cdot)$ 是 $(m-i)$ 阶可微的 Class K 函数。

**HOCBF 约束**（保证安全的核心条件）：

$$\sup_{u \in U} \left[ L_f^m b(x) + L_g L_f^{m-1} b(x) u + O(b(x)) + \alpha_m(\psi_{m-1}(x)) \right] \geq 0 \quad \text{...(4)}$$

其中 $L_f^m$ 和 $L_g$ 分别是沿 $f$ 和 $g$ 的 Lie 导数，$O(b(x)) = \sum_{i=1}^{m-1} L_f^i (\alpha_{m-i} \circ \psi_{m-i-1})(x)$。

**定理 1**：如果存在控制器 $u(t)$ 满足上述 HOCBF 约束，则集合 $C_1 \cap \cdots \cap C_m$ 是前向不变的（即系统始终安全）。

##### 2.1.2a Lie 导数的直觉与逐步计算

**Lie 导数**是理解 CBF 的核心数学工具。它的物理意义是：**一个标量函数 $b(x)$ 沿着向量场 $f(x)$ 的方向导数**。

**定义**：函数 $b: \mathbb{R}^n \to \mathbb{R}$ 沿向量场 $f: \mathbb{R}^n \to \mathbb{R}^n$ 的 Lie 导数定义为：

$$L_f b(x) = \nabla b(x) \cdot f(x) = \frac{\partial b}{\partial x} f(x) = \sum_{i=1}^n \frac{\partial b}{\partial x_i} f_i(x)$$

**物理含义**：如果系统沿着 $\dot{x} = f(x)$ 运动（无控制），那么 $b(x)$ 随时间的变化率就是 $L_f b(x)$。

**高阶 Lie 导数**：

$$L_f^2 b(x) = L_f(L_f b)(x) = \nabla (L_f b(x)) \cdot f(x)$$

即对 $L_f b(x)$ 再沿 $f(x)$ 求一次方向导数。

**沿 $g(x)$ 的 Lie 导数**：

$$L_g b(x) = \nabla b(x) \cdot g(x)$$

**混合 Lie 导数** $L_g L_f b(x)$：

$$L_g L_f b(x) = \nabla (L_f b(x)) \cdot g(x)$$

这表示**先沿 $f$ 求导，再沿 $g$ 求导**，它刻画了控制 $u$ 对 $b$ 的变化的影响。

**具体例子 — AEBS 系统**：

系统动力学（连续时间版本）：
$$\dot{d} = -v, \quad \dot{v} = -a$$

写成仿射形式 $\dot{x} = f(x) + g(x)u$，其中 $x = [d, v]^T$，$u = a$：

$$f(x) = \begin{bmatrix} -v \\ 0 \end{bmatrix}, \quad g(x) = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

> **注意**：这里 $f(x)$ 是**漂移项**（不含控制的部分），$g(x)u$ 是**控制项**。$f_d = -v$ 是因为距离随速度减小，$f_v = 0$ 是因为无控制时速度不变，$g_d = 0$ 是因为加速度不直接影响距离，$g_v = -1$ 是因为加速度直接改变速度。

安全约束函数：

$$b(x) = d - d_{\text{safe}} - \phi v$$

**第一步：梯度**

$$\nabla b = \left[\frac{\partial b}{\partial d}, \frac{\partial b}{\partial v}\right] = [1, -\phi]$$

**第二步：$L_f b(x)$（漂移对 $b$ 的影响）**

$$L_f b(x) = [1, -\phi] \begin{bmatrix} -v \\ 0 \end{bmatrix} = -v$$

**解读**：无控制时，$b$ 的变化率为 $-v$，即以速度 $v$ 在减小——这是因为距离在缩短。

**第三步：$L_g b(x)$（控制对 $b$ 的直接影响）**

$$L_g b(x) = [1, -\phi] \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \phi$$

**解读**：单位加速度使 $b$ 以速率 $\phi$ 增加——因为刹车使速度减小，而 $b$ 中 $v$ 的系数为 $-\phi$。

**第四步：判断相对度**

$L_g b(x) = \phi \neq 0$，所以控制 $u$ 在 $b$ 的**一阶导数**中显式出现，**相对度 $m = 1$**。

**第五步：如果需要相对度 2（如控制是 jerk）**

$$L_f^2 b(x) = \nabla(-v) \cdot f(x) = [0, -1] \begin{bmatrix} -v \\ 0 \end{bmatrix} = 0$$

$$L_g L_f b(x) = \nabla(-v) \cdot g(x) = [0, -1] \begin{bmatrix} 0 \\ -1 \end{bmatrix} = 1$$

此时 $L_g b = \phi$（控制 $a$ 已在一阶出现），但如果控制是 jerk（$a$ 的导数），则需要到二阶导数才出现 jerk，相对度变为 2。

##### 2.1.2b 前向不变性的直觉解释

**定义**：集合 $C$ 对系统 $\dot{x} = f(x) + g(x)u$ 是**前向不变的 (forward invariant)**，如果从 $C$ 中任何初始状态出发，系统在适当的控制 $u$ 下永远留在 $C$ 中：

$$x(0) \in C \implies x(t) \in C, \quad \forall t \geq 0$$

**直觉**：想象一个池塘（安全集 $C$）和一条鱼（系统状态）。前向不变性意味着鱼永远不会跳出池塘。

**与 CBF 的关系**：安全集 $C = \{x: b(x) \geq 0\}$。CBF 的核心思想是：**如果在边界 $\partial C = \{x: b(x) = 0\}$ 上，系统的"速度方向"指向 $C$ 内部，那么系统就不会离开 $C$**。

数学上，在边界 $b(x) = 0$ 处，需要：

$$\dot{b}(x) = L_f b(x) + L_g b(x) u + \alpha(b(x)) \geq 0$$

当 $b(x) = 0$ 时 $\alpha(b(x)) = \alpha(0) = 0$，所以简化为：

$$L_f b(x) + L_g b(x) u \geq 0$$

这就是"在边界上速度指向内部"的条件。

**为什么加上 $\alpha(b(x))$？**

$\alpha(b(x))$ 提供了一个"缓冲区"：即使 $b(x)$ 已经很小但还没有到 0，$\alpha(b(x)) > 0$ 就会要求控制更早地开始修正。这使得 $b(x)$ 被"推回"安全集内部，而不是只在边界上才被拦住。

##### 2.1.2c HOCBF 序列函数的逐步构造（以 $m=2$ 为例）

当安全约束 $b(x) \geq 0$ 的相对度为 2 时（如障碍物避让），需要构造两层的 CBF 序列：

**第 0 层**：$\psi_0(x) = b(x)$

**第 1 层**：$\psi_1(x) = \dot{\psi}_0(x) + \alpha_1(\psi_0(x)) = L_f b(x) + \alpha_1(b(x))$

**第 2 层**：

$$\psi_2(x) = \dot{\psi}_1(x) + \alpha_2(\psi_1(x))$$
$$= \frac{d}{dt}[L_f b(x) + \alpha_1(b(x))] + \alpha_2(\psi_1(x))$$
$$= L_f^2 b(x) + L_g L_f b(x) u + L_f \alpha_1(b(x)) + \alpha_2(\psi_1(x))$$

HOCBF 约束要求 $\psi_2(x) \geq 0$，即：

$$L_f^2 b(x) + L_g L_f b(x) u + O(b(x)) + \alpha_2(\psi_1(x)) \geq 0$$

其中 $O(b(x)) = L_f(\alpha_1 \circ \psi_0)(x) = \alpha_1'(b(x)) L_f b(x)$。

**定理 1 的证明思路**（归纳法）：

1. 假设 $\psi_2(x) \geq 0$ 成立
2. 由 CBF 的性质，$\psi_1(x) \geq 0$ 被保证（因为 $\psi_2 \geq 0$ 意味着 $\psi_1$ 的"速度方向"指向 $\geq 0$ 的方向）
3. 同理，$\psi_1(x) \geq 0$ 保证 $\psi_0(x) = b(x) \geq 0$
4. 因此安全约束 $b(x) \geq 0$ 被保证

这就是 HOCBF 的**递归保证机制**：最高层的约束成立 → 倒数第二层成立 → ... → 原始安全约束成立。

#### 2.1.3 HOCBF 的保守性问题

传统 HOCBF 的保守性来自 **Class K 函数 $\alpha_i$ 的固定选择**：

- **$\alpha_i$ 太陡峭** → 约束仅在接近不安全集边界时才激活 → 需要极大控制输入 → 可能与控制约束冲突
- **$\alpha_i$ 太平缓** → 约束在远离不安全集时就激活 → 系统过度保守（远离障碍物）

#### 2.1.4 可微控制障碍函数 (dCBF)

**核心创新**：将 Class K 函数乘以**环境依赖的正值惩罚函数** $p_i(z) > 0$：

$$\psi_i(x, z, z_d) := \dot{\psi}_{i-1}(x, z, z_d) + p_i(z) \alpha_i(\psi_{i-1}(x, z, z_d)), \quad i \in \{1, \ldots, m\} \quad \text{...(5)}$$

其中：
- $z \in \mathbb{R}^d$ 是神经网络的输入特征（如图像经过 CNN 后的特征）
- $z_d = (z^{(1)}, \ldots, z^{(m-1)})$ 是 $z$ 的导数
- $p_i: \mathbb{R}^d \to \mathbb{R}_{>0}$ 是上游神经网络的输出或可训练参数

**dCBF 约束**（替代硬约束）：

$$L_f^m b(x) + L_g L_f^{m-1} b(x) u + O'(b(x), z, z_d) + p_m(z) \alpha_m(\psi_{m-1}(x, z, z_d)) \geq 0 \quad \text{...(6)}$$

**关键定理**：如果 $p_i(z)$ 可微且正值，或 $p_i$ 为可训练参数，则 dCBF 仍然保证安全性。

**推论 1**：当 $z_d = 0$（即 $p_i(z)$ 在每个离散时间步内为常数）且 $\alpha_i$ 为线性函数时，安全性仍然保证。

##### 2.1.4a dCBF 安全性定理的完整证明

**定理 2 (dCBF 安全性)**：考虑系统 $\dot{x} = f(x) + g(x)u$ 和 dCBF 序列函数 $\psi_i$ 定义如 (5)。如果存在控制器 $u(t)$ 使得 dCBF 约束 (6) 对所有 $t \geq 0$ 成立，且 $p_i(z(t)) > 0$ 对所有 $i, t$ 成立，则集合 $C_1 \cap \cdots \cap C_m$ 是前向不变的。

**证明**（对 $m$ 进行归纳）：

**基础情况 $m = 1$**（相对度 1）：

dCBF 约束简化为：

$$L_f b(x) + L_g b(x) u + p_1(z) \alpha_1(b(x)) \geq 0$$

即 $\dot{b}(x) + p_1(z) \alpha_1(b(x)) \geq 0$。

定义 $C_1 = \{x : \psi_0(x) = b(x) \geq 0\}$。

**关键步骤**：我们需要证明从 $b(x(0)) \geq 0$ 出发，$b(x(t)) \geq 0$ 对所有 $t \geq 0$ 成立。

**反证法**：假设存在时间 $t^*$ 使得 $b(x(t^*)) < 0$。由连续性，存在最早时间 $t_0$ 使得 $b(x(t_0)) = 0$，且在 $t_0$ 之后的某个小区间内 $b(x(t)) < 0$。

在 $t = t_0$ 处：$b(x(t_0)) = 0$，因此 $\alpha_1(b(x(t_0))) = \alpha_1(0) = 0$。

由 dCBF 约束：$\dot{b}(x(t_0)) \geq -p_1(z(t_0)) \alpha_1(0) = 0$

这意味着在 $b = 0$ 处，$b$ 的变化率非负，即 $b$ 不会继续减小。这与假设矛盾。

但更精确的论证需要使用**比较引理 (Comparison Lemma)**：

由 $\dot{b} \geq -p_1(z) \alpha_1(b)$，考虑比较方程 $\dot{y} = -p_1(z(t)) \alpha_1(y)$，$y(0) = b(x(0)) \geq 0$。

由于 $p_1(z(t)) > 0$ 且 $\alpha_1$ 是 Class K 函数（严格递增，$\alpha_1(0) = 0$），$y = 0$ 是该比较方程的平衡点。由比较引理，$b(x(t)) \geq y(t) \geq 0$ 对所有 $t \geq 0$ 成立。$\blacksquare$

**归纳步骤**（从 $m-1$ 到 $m$）：

假设 dCBF 约束 (6) 成立，即 $\psi_m(x, z, z_d) \geq 0$，这等价于：

$$\dot{\psi}_{m-1} + p_m(z) \alpha_m(\psi_{m-1}) \geq 0$$

由与基础情况相同的论证，$\psi_{m-1}(x, z, z_d) \geq 0$ 对所有 $t \geq 0$ 成立。

但 $\psi_{m-1} \geq 0$ 本身具有 dCBF 的形式：

$$\dot{\psi}_{m-2} + p_{m-1}(z) \alpha_{m-1}(\psi_{m-2}) \geq 0$$

由归纳假设，$\psi_{m-2} \geq 0$ 成立。依次递推，最终得到 $\psi_0(x) = b(x) \geq 0$。$\blacksquare$

##### 2.1.4b 推论 1 的证明（离散时间常数 $p_i$ 情况）

**推论 1**：当 $z_d = 0$（$p_i$ 在每个时间步内为常数）且 $\alpha_i$ 为线性函数 $\alpha_i(r) = c_i r$（$c_i > 0$）时，安全性仍然保证。

**证明**：

当 $\alpha_i(r) = c_i r$ 时，序列函数变为：

$$\psi_1 = \dot{b} + p_1 c_1 b$$

这是一个线性微分算子。$\psi_1 \geq 0$ 意味着 $\dot{b} \geq -p_1 c_1 b$，解为 $b(t) \geq b(0) e^{-p_1 c_1 t}$。

由于 $p_1 > 0, c_1 > 0$，指数衰减项 $e^{-p_1 c_1 t} > 0$，所以 $b(0) \geq 0 \implies b(t) \geq 0$。

当 $p_i$ 在离散时间步 $[t_k, t_{k+1})$ 内为常数时，上述论证在每个时间步内成立。由于连续性，$b(t_k)$ 的值作为下一步的初始条件，归纳可得全局安全性。$\blacksquare$

##### 2.1.4c dCBF 与传统 CBF / AdaCBF 的对比

| 特性 | 传统 HOCBF | AdaCBF (自适应 CBF) | dCBF (BarrierNet) |
|------|-----------|-------------------|-------------------|
| **Class K 函数** | 固定 $\alpha_i(\cdot)$ | 状态依赖 $\alpha_i(x)$ | 环境依赖 $p_i(z) \alpha_i(\cdot)$ |
| **保守性** | 高（固定参数） | 中（状态自适应） | 低（环境自适应 + 可学习） |
| **可训练性** | ❌ 不可训练 | ❌ 手工设计适应律 | ✅ 端到端可训练 |
| **安全性证明** | 标准 CBF 理论 | Lyapunov-like 论证 | 比较引理 + 归纳法 |
| **梯度传播** | N/A | N/A | ✅ 通过 KKT 条件 |
| **感知融合** | 需要精确状态 | 需要精确状态 | 直接从感知特征 $z$ 提取 |

**$p_i(z)$ 如何降低保守性 — 直觉解释**：

考虑一个自动驾驶场景：
- **传统 HOCBF**：无论前方是空旷道路还是拥堵路段，都使用相同的 $\alpha$ 值。在空旷道路上过于保守（不必要地减速），在拥堵时可能过于激进。
- **dCBF**：当 CNN 识别到前方空旷时，$p_i(z)$ 输出较小值，放松约束，允许更自由的驾驶；当识别到拥堵或行人时，$p_i(z)$ 输出较大值，收紧约束，要求更早刹车。

**数值例子**：设 $b(x) = d - 6$（安全距离 6m），当前 $d = 10$，$v = 2$。

传统 HOCBF（$\alpha(r) = r$）：$\dot{b} + b \geq 0 \implies -v + (d-6) \geq 0 \implies -2 + 4 = 2 \geq 0$ ✓

dCBF（$p(z) = 0.5$，空旷场景）：$\dot{b} + 0.5 b \geq 0 \implies -2 + 0.5 \times 4 = 0 \geq 0$ ✓（约束更紧但可满足）

dCBF（$p(z) = 2.0$，拥堵场景）：$\dot{b} + 2.0 b \geq 0 \implies -2 + 2.0 \times 4 = 6 \geq 0$ ✓（约束宽松，允许更多自由）

> **注意**：$p_i$ 越大，$\alpha$ 的"等效斜率"越大，要求在 $b$ 还很大时就开始修正——更保守。$p_i$ 越小，允许 $b$ 更接近 0 才修正——更激进但更高效。这就是可学习的 $p_i(z)$ 能在安全与性能之间找到最优平衡的原因。

#### 2.1.5 BarrierNet 定义

BarrierNet 将 dCBF 嵌入**可微二次规划 (differentiable QP)**：

$$u^*(t) = \arg\min_{u(t)} \frac{1}{2} u(t)^T H(z|\theta_h) u(t) + F^T(z|\theta_f) u(t) \quad \text{...(7)}$$

**约束条件**（$j \in S$，所有安全约束）：

$$L_f^m b_j(x) + L_g L_f^{m-1} b_j(x) u + O'(b_j, z, z_d) + p_m(z|\theta_{pm}) \alpha_m(\psi_{m-1}) \geq 0$$
$$u_{\min} \leq u \leq u_{\max} \quad \text{...(8)}$$

**参数说明**：
| 参数 | 含义 | 来源 |
|------|------|------|
| $H(z\|\theta_h) \succ 0$ | 正定代价矩阵 | 上游网络输出或可训练参数 |
| $F(z\|\theta_f)$ | 参考控制（标称控制器输出） | 上游网络输出 |
| $p_i(z\|\theta_{pi})$ | 惩罚函数（调节保守性） | 上游网络输出 |
| $u^*$ | 安全控制输出 | QP 的解 |

**解读**：$H^{-1}F$ 可理解为参考控制（无约束时的最优控制），QP 找到在满足所有 dCBF 安全约束下最接近参考控制的控制量。

#### 2.1.6 前向传播与反向传播

**前向传播**：求解上述 QP（使用 interior point method，复杂度 $O(d^3)$，$d$ 为控制维度）。

**反向传播**：利用 KKT 条件，通过对 Lagrangian 求导得到所有参数的梯度：

$$\begin{bmatrix} du \\ d\lambda \end{bmatrix} = \begin{bmatrix} H & G^T D(\lambda^*) \\ G & D(Gu^* - h) \end{bmatrix}^{-1} \begin{bmatrix} \frac{\partial \ell}{\partial u^*} \\ 0 \end{bmatrix} \quad \text{...(9)}$$

其中 $G, h$ 由 dCBF 约束构成，$D(\cdot)$ 创建对角矩阵。

参数梯度：

$$\nabla_H = \frac{1}{2}(du \cdot u^{*T} + u^* \cdot du^T)$$
$$\nabla_F = du$$
$$\nabla_G = D(\lambda^*)(d\lambda \cdot u^{*T} + \lambda^* \cdot du^{*T})$$
$$\nabla_h = -D(\lambda^*) d\lambda \quad \text{...(11)}$$

这些梯度再通过链式法则传播到 $\theta_h, \theta_f, \theta_p$。

##### 2.1.6a 二次规划 (QP) 的标准形式与 KKT 条件完整推导

**BarrierNet QP 的标准形式**：

将 dCBF 约束和控制界限写成统一的线性不等式约束 $Gu \leq h$：

$$\min_u \quad \frac{1}{2} u^T H u + F^T u$$
$$\text{s.t.} \quad Gu \leq h$$

其中 $G$ 和 $h$ 由以下约束组合而成：

1. **dCBF 约束**（每个安全约束 $j$）：

$$-L_g L_f^{m-1} b_j(x) \cdot u \leq L_f^m b_j(x) + O'(b_j, z, z_d) + p_m(z) \alpha_m(\psi_{m-1})$$

2. **控制上界**：$u \leq u_{\max} \implies I \cdot u \leq u_{\max}$

3. **控制下界**：$-u \leq -u_{\min} \implies -I \cdot u \leq -u_{\min}$

将所有约束堆叠：

$$G = \begin{bmatrix} -L_g L_f^{m-1} b_1(x) \\ \vdots \\ -L_g L_f^{m-1} b_{|S|}(x) \\ I \\ -I \end{bmatrix}, \quad h = \begin{bmatrix} L_f^m b_1 + O'_1 + p_m \alpha_m(\psi_{m-1}^{(1)}) \\ \vdots \\ L_f^m b_{|S|} + O'_{|S|} + p_m \alpha_m(\psi_{m-1}^{(|S|)}) \\ u_{\max} \\ -u_{\min} \end{bmatrix}$$

**Lagrangian 构造**：

$$\mathcal{L}(u, \lambda) = \frac{1}{2} u^T H u + F^T u + \lambda^T (Gu - h)$$

其中 $\lambda \geq 0$ 是 Lagrange 乘子向量。

**KKT 条件**（必要条件，对凸 QP 也是充分条件）：

**(1) 驻点条件 (Stationarity)**：

$$\nabla_u \mathcal{L} = Hu + F + G^T \lambda = 0 \quad \text{...(KKT-1)}$$

**(2) 原始可行性 (Primal Feasibility)**：

$$Gu - h \leq 0 \quad \text{...(KKT-2)}$$

**(3) 对偶可行性 (Dual Feasibility)**：

$$\lambda \geq 0 \quad \text{...(KKT-3)}$$

**(4) 互补松弛 (Complementary Slackness)**：

$$\lambda_i (G_i u - h_i) = 0, \quad \forall i \quad \text{...(KKT-4)}$$

互补松弛的含义：如果约束 $i$ 不活跃（$G_i u < h_i$），则对应的乘子 $\lambda_i = 0$；如果 $\lambda_i > 0$，则约束 $i$ 必须活跃（$G_i u = h_i$，即约束恰好在边界上）。

##### 2.1.6b 反向传播梯度的完整推导

**目标**：给定损失函数 $\ell(u^*)$（上游损失对 QP 最优解的依赖），计算 $\frac{\partial \ell}{\partial \theta}$，其中 $\theta$ 包括 $\theta_h, \theta_f, \theta_p$。

**核心思路**：利用**隐函数定理 (Implicit Function Theorem)** 对 KKT 条件求微分。

**步骤 1**：将 KKT 条件写为方程组 $R(u^*, \lambda^*, \theta) = 0$

从 (KKT-1) 和 (KKT-4) 出发：

$$R_1 = Hu^* + F + G^T \lambda^* = 0$$
$$R_2 = D(\lambda^*)(Gu^* - h) = 0$$

其中 $D(\lambda^*)$ 是以 $\lambda^*$ 为对角元素的对角矩阵。

**步骤 2**：对 $R$ 进行全微分

对 $R_1$ 微分：

$$dR_1 = H \, du^* + dH \, u^* + dF + dG^T \lambda^* + G^T d\lambda^* = 0$$

整理（只保留 $H, F, G, h$ 的变化，因为它们依赖于 $\theta$）：

$$H \, du^* + G^T d\lambda^* = -dH \, u^* - dF - dG^T \lambda^*$$

对 $R_2$ 微分：

$$dR_2 = D(d\lambda^*)(Gu^* - h) + D(\lambda^*)(G \, du^* + dG \, u^* - dh) = 0$$

利用互补松弛 $D(\lambda^*)(Gu^* - h) = 0$ 以及对角矩阵的性质：

$$D(Gu^* - h) d\lambda^* + D(\lambda^*) G \, du^* = -D(\lambda^*)(dG \, u^* - dh)$$

**步骤 3**：写成矩阵形式

$$\begin{bmatrix} H & G^T \\ D(\lambda^*) G & D(Gu^* - h) \end{bmatrix} \begin{bmatrix} du^* \\ d\lambda^* \end{bmatrix} = -\begin{bmatrix} dH \, u^* + dF + dG^T \lambda^* \\ D(\lambda^*)(dG \, u^* - dh) \end{bmatrix}$$

**注意**：原文中使用了不同的等价形式——将右端项简化为仅包含上游梯度 $\frac{\partial \ell}{\partial u^*}$：

$$\begin{bmatrix} du^* \\ d\lambda^* \end{bmatrix} = \begin{bmatrix} H & G^T D(\lambda^*) \\ G & D(Gu^* - h) \end{bmatrix}^{-1} \begin{bmatrix} \frac{\partial \ell}{\partial u^*} \\ 0 \end{bmatrix}$$

这里 $\frac{\partial \ell}{\partial u^*}$ 是损失对 $u^*$ 的梯度（由上游网络提供），矩阵求逆的复杂度为 $O((q + |S|)^3)$，其中 $q$ 是控制维度，$|S|$ 是约束数量。

**步骤 4**：参数梯度

得到 $du^*$ 和 $d\lambda^*$ 后，各参数的梯度为：

$$\nabla_H \ell = \frac{1}{2}(du^* \cdot u^{*T} + u^* \cdot du^{*T})$$

$$\nabla_F \ell = du^*$$

$$\nabla_G \ell = D(\lambda^*)(d\lambda^* \cdot u^{*T} + \lambda^* \cdot du^{*T})$$

$$\nabla_h \ell = -D(\lambda^*) d\lambda^*$$

**步骤 5**：链式法则传播到神经网络参数

$$\frac{\partial \ell}{\partial \theta_h} = \frac{\partial \ell}{\partial H} \cdot \frac{\partial H}{\partial \theta_h} \quad \text{（标准自动微分）}$$

$$\frac{\partial \ell}{\partial \theta_f} = \frac{\partial \ell}{\partial F} \cdot \frac{\partial F}{\partial \theta_f}$$

$$\frac{\partial \ell}{\partial \theta_p} = \frac{\partial \ell}{\partial G} \cdot \frac{\partial G}{\partial p} \cdot \frac{\partial p}{\partial \theta_p} + \frac{\partial \ell}{\partial h} \cdot \frac{\partial h}{\partial p} \cdot \frac{\partial p}{\partial \theta_p}$$

> **计算复杂度**：对于 AEBS 场景，$q = 1$（加速度），$|S| = 2$（一个 dCBF + 控制上下界），矩阵维度为 $3 \times 3$，求逆几乎是零开销的。这使得 BarrierNet 在低维控制场景中极为高效。

##### 2.1.6c 数值例子 — BarrierNet QP 前向与反向传播

**场景**：AEBS，当前状态 $d = 10$, $v = 2$, 安全距离 $d_{\text{safe}} = 6$。

设 $b(x) = d - d_{\text{safe}} = 4$，相对度 $m = 1$（控制为加速度 $a$）。

$f(x) = [-v, 0]^T$, $g(x) = [0, -1]^T$

**Step 1 — 计算约束参数**：

$L_f b = [1, 0] \cdot [-v, 0]^T = -v = -2$

$L_g b = [1, 0] \cdot [0, -1]^T = 0$

> 等等，这里 $b(x) = d - d_{\text{safe}}$，$\nabla b = [1, 0]$。$L_g b = 0$ 意味着控制不直接影响 $b$——这是**相对度 2** 的情况。

改用 $b(x) = d - d_{\text{safe}} - \phi v$（带速度项），$\nabla b = [1, -\phi]$。设 $\phi = 1$。

$L_f b = [1, -1] \cdot [-2, 0]^T = -2$

$L_g b = [1, -1] \cdot [0, -1]^T = 1$

**Step 2 — 构造 QP**：

设 $H = 2$（标量），$F = -u_{\text{ref}} \cdot H = -2 \times 2 = -4$（参考控制 $u_{\text{ref}} = 2$，希望加速）

dCBF 约束：$L_f b + L_g b \cdot u + p \cdot \alpha(b) \geq 0$

设 $p = 1.0$，$\alpha(r) = r$，则 $b(x) = 10 - 6 - 1 \times 2 = 2$

$-2 + 1 \cdot u + 1.0 \times 2 \geq 0 \implies u \geq 0$

控制界限：$-3 \leq u \leq 3$

QP 变为：$\min_u \frac{1}{2} \times 2 \times u^2 - 4u = u^2 - 4u$

s.t. $u \geq 0$，$u \leq 3$，$u \geq -3$

**Step 3 — 求解**：

无约束最优：$2u - 4 = 0 \implies u = 2$

检查约束：$u = 2 \geq 0$ ✓，$u = 2 \leq 3$ ✓

因此 $u^* = 2$，$\lambda^* = [0, 0, 0]^T$（所有约束不活跃）

**Step 4 — 反向传播**：

设上游损失 $\ell = (u^* - u_{\text{target}})^2$，$u_{\text{target}} = 1$

$\frac{\partial \ell}{\partial u^*} = 2(u^* - 1) = 2$

由于所有 $\lambda^* = 0$（无活跃约束），KKT 矩阵简化为 $H = 2$：

$du^* = H^{-1} \frac{\partial \ell}{\partial u^*} = 2 / 2 = 1$

$\nabla_H = du^* \cdot u^* = 1 \times 2 = 2$

$\nabla_F = du^* = 1$

$\nabla_G = 0$（因为 $\lambda^* = 0$）

$\nabla_h = 0$

**解读**：当安全约束不活跃时，BarrierNet 等价于 $u^* = H^{-1}(-F) = u_{\text{ref}}$，梯度直接传播到 $H$ 和 $F$，与标准神经网络无异。BarrierNet 的"安全干预"仅在约束活跃时（$\lambda^* > 0$）才体现。

#### 2.1.7 视觉自动驾驶应用

在端到端自动驾驶中，BarrierNet 的架构为：

```
前视图 → CNN → LSTM → MLP分支 → {H(z), F(z), p_i(z)} → BarrierNet (dQP) → 安全控制
                    ↓
              状态估计分支 → {d, μ, Δs, d_obs} → BarrierNet 状态输入
```

**多层级可解释设计**：
- CNN 层输出：监督学习位置信息 (loss 1)
- MLP 分支输出：监督学习速度和转向角 (loss 2)
- 求导后得到：加速度和转向率作为 BarrierNet 参考控制
- BarrierNet 输出：最终安全控制 (loss 3)

**处理可变数量障碍物**：使用**大圆盘覆盖法**，用 N 个固定数量的圆盘覆盖最多 N 个障碍物，无实际障碍物时将圆盘移到路边。

---

### 2.2 DiffCBF for Vision-based Driving 论文原理

> **论文**: *Differentiable Control Barrier Functions for Vision-based End-to-End Autonomous Driving*  
> **作者**: Wei Xiao, Tsun-Hsuan Wang 等 (MIT CSAIL)  
> **发表**: arXiv:2203.02401, March 2022

#### 2.2.1 核心贡献

这是 BarrierNet 在视觉端到端自动驾驶中的具体应用论文。主要贡献：

1. **多层级可解释端到端架构**：不同深度的神经元学习不同层次的信息
2. **视觉状态估计**：在缺少 ground truth 的情况下从图像推断状态
3. **Sim-to-Real 部署**：在 VISTA 仿真器中训练，部署到真实自动驾驶车辆

#### 2.2.2 车辆动力学模型

采用自行车模型（相对于参考轨迹）：

$$\dot{s} = \frac{v \cos(\mu + \beta)}{1 - d\kappa}$$
$$\dot{d} = v \sin(\mu + \beta)$$
$$\dot{\mu} = \frac{v \sin \beta}{l_r} - \frac{\kappa v \cos(\mu + \beta)}{1 - d\kappa}$$
$$\dot{v} = a$$
$$\dot{\delta} = \omega \quad \text{...(29)}$$

其中：
- $s$：沿轨迹进度距离
- $d$：到车道中心的横向偏移
- $\mu$：航向角误差（$\mu = \theta - \varphi$）
- $v$：线速度
- $\delta$：转向角
- $\kappa$：参考轨迹曲率
- $l_r$：后轴到质心的距离
- $\beta = \arctan(\frac{l_r}{l_r + l_f} \tan \delta)$：侧滑角

**控制输入**：$u_{jerk}$（加速度变化率）和 $u_{steer}$（转向加速度），使得输出是**高相对度**的，保证控制平滑性。

#### 2.2.3 安全约束设计

**障碍物避让**：使用圆盘覆盖法，每个障碍物 $j$ 的安全约束为：

$$b_j(x) = (x - x_{o,j})^2 + (y - y_{o,j})^2 - R_j^2 \geq 0$$

**车道保持**：

$$b_{\text{lane}}(x) = d_{\text{lane\_width}} - |d| \geq 0$$

#### 2.2.4 NMPC 训练数据生成

使用非线性模型预测控制 (NMPC) 作为特权控制器生成训练标签：
- 预见域 (receding horizon) = 20 步
- 在 MATLAB 虚拟仿真环境中实现
- 使用 VISTA 数据驱动仿真器增强真实数据
- 总计约 40 万张训练图像

---

### 2.3 SafePVC (720_file_Paper) 论文原理

> **论文**: *Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems*  
> **发表**: DAC '26, July 26-29, 2026, Long Beach, CA

#### 2.3.1 系统模型

**离散时间动力系统**：

$$s_{t+1} = f(s_t, u_t), \quad o_t = g(s_t, z_t), \quad u_t = \pi(o_t)$$

其中：
- $s_t \in S \subseteq \mathbb{R}^m$：系统状态
- $u_t \in U \subseteq \mathbb{R}^n$：控制输入
- $o_t \in O \subseteq \mathbb{R}^{H \times W}$：视觉观测
- $z_t \in Z \subseteq \mathbb{R}^p$：未观测的环境扰动

**随机性来源**：$z_t$ 影响视觉观测 $o_t$，通过策略 $\pi(o_t)$ 传播到控制，最终影响状态演化。

**假设 1 (状态扰动独立性)**：

$$\Delta s = F(s, z) - F(s, z_0) \sim \mu$$

其中 $\mu$ 是与当前状态 $s$ 独立的 Borel 概率测度。闭环动力学可写为：

$$s' = \tilde{F}(s, z_0, \Delta s) = F(s, z_0) + \Delta s$$

#### 2.3.2 概率安全定义

**定义 2.1 (概率安全)**：给定不安全集 $X_u \subseteq S$ 和概率阈值 $p \in [0,1)$，找到控制策略 $\pi$ 使得：

$$\mathbb{P}_{s_0}(\text{Safe}(X_u)) \geq p$$

其中 $\text{Safe}(X_u) = \{(s_t, \Delta s)_{t \in \mathbb{N}_0} \mid \forall t, s_t \notin X_u\}$。

#### 2.3.3 随机障碍证书 (SBC)

**定理 2.2**：连续函数 $B: S \to \mathbb{R}$ 满足 SBC 条件，当且仅当：

| 条件 | 数学表达 | 含义 |
|------|---------|------|
| (i) 非负性 | $B(s) \geq 0, \forall s \in S$ | 障碍函数非负 |
| (ii) 初始集 | $B(s) \leq 1, \forall s \in S_0$ | 初始状态处值小 |
| (iii) 不安全集 | $B(s) \geq \frac{1}{1-p}, \forall s \in X_u$ | 不安全状态处值大 |
| (iv) 期望递减 | $B(s) \geq \mathbb{E}_{\Delta s \sim \mu}[B(\tilde{F}(s, z_0, \Delta s))] + \epsilon$ | 鞅超水平条件 |

**安全保证**：如果找到 SBC，则 $\mathbb{P}(\text{进入不安全区域}) \leq 1 - p$。

**证明思路**（基于鞅理论）：$B(s_t)$ 是一个**超鞅 (supermartingale)**，由 Doob 不等式推导出到达不安全集的概率上界。

##### 2.3.3a 鞅理论基础

**定义（滤子与鞅）**：

设 $(\Omega, \mathcal{F}, \mathbb{P})$ 为概率空间，$\{\mathcal{F}_t\}_{t \geq 0}$ 是一个**滤子 (filtration)**，即一列递增的 $\sigma$-代数：$\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}$。

随机过程 $\{X_t\}_{t \geq 0}$ 关于 $\{\mathcal{F}_t\}$ 是：
- **鞅 (martingale)**：$\mathbb{E}[X_{t+1} | \mathcal{F}_t] = X_t$
- **超鞅 (supermartingale)**：$\mathbb{E}[X_{t+1} | \mathcal{F}_t] \leq X_t$
- **亚鞅 (submartingale)**：$\mathbb{E}[X_{t+1} | \mathcal{F}_t] \geq X_t$

**直觉**：鞅是"公平的赌博"——平均来说不赚不赔。超鞅是"对你不利的赌博"——平均来说在赔钱。

##### 2.3.3b $B(s_t)$ 构成超鞅的完整证明

**设定**：在我们的系统中，闭环动力学为 $s_{t+1} = \tilde{F}(s_t, z_0, \Delta s_t)$，其中 $\Delta s_t$ 是 i.i.d. 扰动。定义自然滤子 $\mathcal{F}_t = \sigma(s_0, s_1, \ldots, s_t)$。

**证明 $B(s_t)$ 是超鞅**：

由 SBC 条件 (iv)：

$$B(s_t) \geq \mathbb{E}_{\Delta s_t \sim \mu}[B(\tilde{F}(s_t, z_0, \Delta s_t))] + \epsilon$$

注意 $\mathbb{E}_{\Delta s_t}[B(\tilde{F}(s_t, z_0, \Delta s_t))]$ 恰好等于 $\mathbb{E}[B(s_{t+1}) | \mathcal{F}_t]$，因为：
1. $s_t$ 在给定 $\mathcal{F}_t$ 时是已知的（确定的）
2. $\Delta s_t$ 独立于 $\mathcal{F}_t$（由假设 1：扰动与状态独立）

因此：

$$\mathbb{E}[B(s_{t+1}) | \mathcal{F}_t] = \mathbb{E}_{\Delta s_t}[B(\tilde{F}(s_t, z_0, \Delta s_t))] \leq B(s_t) - \epsilon < B(s_t)$$

由于 $\epsilon > 0$，$B(s_t)$ 甚至是一个**严格超鞅**。$\blacksquare$

##### 2.3.3c Doob 停时不等式与概率安全界的完整推导

**Doob 停时定理 (Optional Stopping Theorem)**：

如果 $\{X_t\}$ 是非负超鞅，$\tau$ 是停时（即 $\{\tau = t\} \in \mathcal{F}_t$），则 $\mathbb{E}[X_\tau] \leq \mathbb{E}[X_0]$（在适当条件下）。

**定义到达不安全集的停时**：

$$\tau_u = \inf\{t \geq 0 : s_t \in X_u\}$$

即第一次进入不安全集的时间。如果永远不进入，$\tau_u = \infty$。

**定义截断停时**：$\tau_T = \min(\tau_u, T)$，这是有界停时。

**证明步骤**：

**Step 1**：由于 $B(s_t)$ 是非负超鞅，由 Doob 停时定理：

$$\mathbb{E}[B(s_{\tau_T})] \leq \mathbb{E}[B(s_0)]$$

**Step 2**：分解期望

$$\mathbb{E}[B(s_{\tau_T})] = \mathbb{E}[B(s_{\tau_T}) | \tau_u \leq T] \cdot \mathbb{P}(\tau_u \leq T) + \mathbb{E}[B(s_{\tau_T}) | \tau_u > T] \cdot \mathbb{P}(\tau_u > T)$$

**Step 3**：下界估计

当 $\tau_u \leq T$ 时，$s_{\tau_T} = s_{\tau_u} \in X_u$，由 SBC 条件 (iii)：$B(s_{\tau_u}) \geq \frac{1}{1-p}$

当 $\tau_u > T$ 时，由 SBC 条件 (i)：$B(s_T) \geq 0$

因此：

$$\mathbb{E}[B(s_{\tau_T})] \geq \frac{1}{1-p} \cdot \mathbb{P}(\tau_u \leq T) + 0 \cdot \mathbb{P}(\tau_u > T) = \frac{\mathbb{P}(\tau_u \leq T)}{1-p}$$

**Step 4**：结合 Step 1 和 Step 3

$$\frac{\mathbb{P}(\tau_u \leq T)}{1-p} \leq \mathbb{E}[B(s_{\tau_T})] \leq \mathbb{E}[B(s_0)]$$

**Step 5**：利用初始条件

由 SBC 条件 (ii)：$B(s_0) \leq 1$（因为 $s_0 \in S_0$）

$$\frac{\mathbb{P}(\tau_u \leq T)}{1-p} \leq 1$$

$$\mathbb{P}(\tau_u \leq T) \leq 1 - p$$

**Step 6**：取 $T \to \infty$

由单调收敛定理，$\mathbb{P}(\tau_u < \infty) = \lim_{T \to \infty} \mathbb{P}(\tau_u \leq T) \leq 1 - p$

因此：

$$\mathbb{P}(\text{永远安全}) = 1 - \mathbb{P}(\tau_u < \infty) \geq 1 - (1-p) = p$$

$\blacksquare$

##### 2.3.3d 数值例子 — AEBS 场景的 SBC 概率界

**场景设定**：
- 初始集 $S_0$: $d \in [15, 16], v \in [2.5, 3.0]$
- 不安全集 $X_u$: $d \in [5, 6], v \in [0.5, 3.0]$（距离太近且速度太快）
- 目标安全概率：$p = 0.90$

**SBC 要求**：
- $B(s) \leq 1$ 在 $S_0$ 上
- $B(s) \geq \frac{1}{1-0.9} = 10$ 在 $X_u$ 上
- $B(s) \geq \mathbb{E}[B(s')] + \epsilon$ 全局成立

**直觉**：$B(s)$ 就像一个"危险度计"——在初始状态时危险度低（$\leq 1$），在不安全状态时危险度高（$\geq 10$），而且危险度平均来说在下降（超鞅性质）。从低危险度出发，到达高危险度的概率被鞅理论限制在 $1/10 = 10\%$ 以内。

**概率界推导**：

$$\mathbb{P}(\text{到达} X_u) \leq \frac{\max_{s \in S_0} B(s)}{\min_{s \in X_u} B(s)} \leq \frac{1}{10} = 10\%$$

$$\mathbb{P}(\text{安全}) \geq 1 - 10\% = 90\%$$

**$\epsilon$ 的作用**：$\epsilon > 0$ 确保 $B(s_t)$ 是**严格递减**的超鞅，而不仅仅是非递增。这使得 $B(s_t)$ 以速率 $\epsilon$ 趋向于 0，从而提供了额外的安全裕度。在训练中，$\epsilon$ 通常取 $0.1$。

#### 2.3.4 框架组件

**1. cGAN 感知模型**

$$\min_G \max_D \mathbb{E}_{(o,s) \sim p_{data}}[\log D(o,s)] + \mathbb{E}_{s, z}[\log(1 - D(G(s,z), s))]$$

用生成器 $G(s, z)$ 逼近真实观测模型 $g(s, z)$。

**cGAN 训练原理详解**：

cGAN 的训练是一个 minimax 博弈：

- **判别器 $D$**：输入 $(o, s)$，输出标量概率 $D(o,s) \in [0,1]$。目标：区分真实图像 $o$ 和生成图像 $G(s,z)$。
- **生成器 $G$**：输入 $(s, z)$，输出合成图像。目标：生成判别器无法区分的"假"图像。

**为什么用 cGAN 而非简单的回归网络？**

1. **多模态性**：同一状态 $s$ 在不同扰动 $z$ 下对应不同的图像 $o$。回归网络只能输出条件均值 $\mathbb{E}[o|s]$，丢失了多样性。cGAN 通过随机种子 $z$ 生成不同的图像。
2. **真实感**：GAN 的对抗训练使生成图像更逼真，这对后续的视觉控制器至关重要。
3. **扰动建模**：$z$ 显式建模了环境不确定性（如光照、天气），使得验证时可以系统地遍历扰动空间。

**AEBS 场景的具体实现**：

AebsMLPGenerator 的结构：
```
输入: [z(4维), d_norm(1维)] = 5维
Linear(5→256) → ReLU
Linear(256→256) → ReLU  ×3
Linear(256→1024) → Tanh
输出: reshape 为 (1, 32, 32) 的灰度图像
```

生成器的输入 $z \in \mathbb{R}^4$ 是随机采样的扰动向量，$d_{\text{norm}}$ 是当前归一化距离。输出是一张 32×32 的合成图像，模拟在该距离和扰动条件下的摄像头观测。

**2. MLP 蒸馏（带 Lipschitz 正则化）**

$$\mathcal{L}_{\text{distill}} = \|G(s,z) - g(s,z)\|_2^2 + \lambda_{\text{lip}} \mathcal{L}_{\text{lip}}$$

将 cGAN 蒸馏为紧凑型 MLP 模型 $g(s, z)$，使其可被形式化验证工具处理。

**蒸馏的必要性与方法**：

cGAN 的生成器 $G$ 通常是一个大型网络（数十万参数），直接使用 IBP 验证会产生极宽的区间界（conservativity explosion）。蒸馏的目标是将 $G$ 的知识转移到一个更小、更可验证的网络 $g$ 中。

**蒸馏损失函数的分解**：

$$\mathcal{L}_{\text{distill}} = \underbrace{\|G(s,z) - g(s,z)\|_2^2}_{\text{行为模仿}} + \underbrace{\lambda_{\text{lip}} \sum_{i=1}^L \|W_i\|_2}_{\text{Lipschitz 正则化}}$$

- **行为模仿**：确保蒸馏后的网络输出与 cGAN 一致
- **Lipschitz 正则化**：约束每层权重的谱范数，使 $g$ 的 Lipschitz 常数可控。这对后续的 Lipschitz-based 验证（定理 3.1）至关重要。

**蒸馏网络的架构选择**：

SubNet: Linear(1024→256) → LayerNorm → ReLU → Linear(256→64) → LayerNorm → ReLU → Linear(64→1)

参数量：$1024 \times 256 + 256 \times 64 + 64 \times 1 \approx 279K$（远小于 cGAN 生成器）

选择 LayerNorm（而非 BatchNorm）的原因：LayerNorm 不依赖 batch 统计量，推理时行为确定，更适合 IBP 验证。

**3. PPO 控制器预训练**

$$L(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

**PPO 原理详解**：

PPO 是一种**策略梯度**方法，通过限制策略更新步长来保证训练稳定性。

**比率函数**：$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$

衡量新策略 $\pi_\theta$ 和旧策略 $\pi_{\theta_{\text{old}}}$ 在同一状态-动作对上的概率比。

**优势函数**：$\hat{A}_t = R_t - V(s_t)$

$R_t$ 是实际回报，$V(s_t)$ 是价值网络估计的基线。$\hat{A}_t > 0$ 表示动作 $a_t$ 比平均好。

**裁剪机制**：$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$

将比率限制在 $[1-\epsilon, 1+\epsilon]$ 范围内（通常 $\epsilon = 0.2$），防止策略更新过大。

**为什么用 $\min$？**：当 $\hat{A}_t > 0$ 时，$\min$ 防止 $r_t$ 增长过大（过度增加好动作的概率）；当 $\hat{A}_t < 0$ 时，$\min$ 防止 $r_t$ 减小过多（过度减少坏动作的概率）。这创建了一个"保守"的策略更新。

**AEBS 场景的奖励设计**：

$$r = \underbrace{2.0 \times (d - d_{\text{next}})}_{\text{接近目标}} - \underbrace{0.001}_{\text{时间惩罚}} + \underbrace{\text{safety\_bonus}}_{\text{安全奖励/惩罚}}$$

- 安全奖励：当 $d_{\text{next}} \leq 6$ 且 $v_{\text{next}} \leq 0.5$ 时 $+2$（成功停车）
- 安全惩罚：当 $d_{\text{next}} \leq 6$ 且 $v_{\text{next}} > 0.5$ 时 $-(v_{\text{next}} - 0.5) \times 3$（太快的惩罚）

**4. SBC 神经网络训练损失**

$$\mathcal{L}_L = \mathcal{L}_{\text{dec\_L}} + \lambda_R \mathcal{L}_{\text{region}} + \lambda_L \mathcal{L}_{\text{lip\_L}}$$

- **鞅期望递减损失**：$\mathcal{L}_{\text{dec\_L}} = \mathbb{E}_{s_t}[\max(0, \mathbb{E}[B(s_{t+1})|s_t] - \gamma B(s_t) + \epsilon)]$
- **区域约束损失**：$\mathcal{L}_{\text{region}}$ 惩罚 $B(s)$ 在 $S_0$ 和 $X_u$ 上的违反

**区域约束损失的详细定义**：

$$\mathcal{L}_{\text{region}} = \underbrace{\mathbb{E}_{s \in S_0}[\max(0, B(s) - 1)]}_{\text{初始集: } B(s) \leq 1} + \underbrace{\mathbb{E}_{s \in X_u}[\max(0, \frac{1}{1-p} - B(s))]}_{\text{不安全集: } B(s) \geq \frac{1}{1-p}}$$

对于 AEBS 场景（$p = 0.90$，$\frac{1}{1-p} = 10$）：

- **初始集项**：从 $S_0 = \{d \in [15,16], v \in [2.5,3.0]\}$ 中采样，惩罚 $B(s) > 1$
- **不安全集项**：从 $X_u = \{d \in [5,6], v \in [0.5,3.0]\}$ 中采样，惩罚 $B(s) < 10$

区域约束确保 $B(s)$ 的"形状"正确：在安全区域值小，在危险区域值大。没有这个约束，网络可能输出一个常数函数（鞅条件满足但概率界无意义）。
- **Lipschitz 正则化**：$\mathcal{L}_{\text{lip\_L}}$ 约束 $B(s)$ 的 Lipschitz 常数

**5. 策略网络训练损失**

$$\mathcal{L}_P = \mathcal{L}_{\text{dec\_P}} + \lambda_P \mathcal{L}_{\text{lip\_P}} + \lambda_M \mathcal{L}_{\text{mse}}$$

- $\mathcal{L}_{\text{dec\_P}}$：鞅损失（梯度流过控制器）
- $\mathcal{L}_{\text{lip\_P}}$：控制器 Lipschitz 正则化
- $\mathcal{L}_{\text{mse}} = \mathbb{E}_o[\|\pi(o) - \pi_0(o)\|_2^2]$：与预训练控制器的行为蒸馏

#### 2.3.5 验证定理

**定理 3.1**（基于 Lipschitz 的验证）：设 $B$ 的 Lipschitz 常数为 $L_B$，$f, \pi, g$ 的 Lipschitz 常数分别为 $L_f, L_\pi, L_g$，$\|s - \tilde{s}\|_2 \leq \tau$。定义：

$$K = \tau \cdot L_B \cdot \left(1 + L_f \sqrt{1 + (L_\pi L_g)^2}\right)$$

如果 $B(\tilde{s}) - \mathbb{E}[B(\tilde{F}(\tilde{s}, z_0, \Delta s))] - K \geq \epsilon$，则对任意 $\|s - \tilde{s}\|_2 \leq \tau$ 也有 $B(s) - \mathbb{E}[B(\tilde{F}(s, z_0, \Delta s))] \geq \epsilon$。

**意义**：只需在离散化网格中心验证，Lipschitz 连续性保证邻域内也满足条件。

##### 2.3.5a 定理 3.1 的完整证明

**目标**：证明如果 SBC 递减条件在网格中心 $\tilde{s}$ 处满足（考虑 Lipschitz 修正 $K$），则在整个网格单元 $\{s : \|s - \tilde{s}\|_2 \leq \tau\}$ 内也满足。

**定义**：设 $\Phi(s) = B(s) - \mathbb{E}_{\Delta s \sim \mu}[B(\tilde{F}(s, z_0, \Delta s))]$

需证明：$\Phi(\tilde{s}) - K \geq \epsilon \implies \Phi(s) \geq \epsilon, \forall \|s - \tilde{s}\| \leq \tau$

**证明**：

**Step 1 — $B(s)$ 项的 Lipschitz 界**：

$$|B(s) - B(\tilde{s})| \leq L_B \|s - \tilde{s}\|_2 \leq L_B \tau$$

因此 $B(s) \geq B(\tilde{s}) - L_B \tau$。

**Step 2 — 期望项的 Lipschitz 界**：

$$|\mathbb{E}[B(\tilde{F}(s, z_0, \Delta s))] - \mathbb{E}[B(\tilde{F}(\tilde{s}, z_0, \Delta s))]| \leq \mathbb{E}[|B(\tilde{F}(s, \cdot)) - B(\tilde{F}(\tilde{s}, \cdot))|]$$

由 $B$ 的 Lipschitz 连续性：

$$\leq L_B \cdot \mathbb{E}[\|\tilde{F}(s, z_0, \Delta s) - \tilde{F}(\tilde{s}, z_0, \Delta s)\|_2]$$

**Step 3 — 闭环动力学的 Lipschitz 界**：

$$\tilde{F}(s, z_0, \Delta s) = F(s, z_0) + \Delta s$$

所以：

$$\|\tilde{F}(s, z_0, \Delta s) - \tilde{F}(\tilde{s}, z_0, \Delta s)\|_2 = \|F(s, z_0) - F(\tilde{s}, z_0)\|_2$$

其中 $F(s, z_0) = f(s, \pi(g(s, z_0)))$ 是标称闭环动力学。

**Step 4 — 复合函数的 Lipschitz 常数**：

$F(s, z_0)$ 是三个函数的复合：$s \mapsto (s, g(s, z_0)) \mapsto \pi(\cdot) \mapsto f(s, \pi(\cdot))$

$$\|F(s, z_0) - F(\tilde{s}, z_0)\|_2 = \|f(s, \pi(g(s, z_0))) - f(\tilde{s}, \pi(g(\tilde{s}, z_0)))\|_2$$

$f$ 关于其两个参数 $(s, u)$ 的 Lipschitz 常数为 $L_f$：

$$\leq L_f \sqrt{\|s - \tilde{s}\|_2^2 + \|\pi(g(s, z_0)) - \pi(g(\tilde{s}, z_0))\|_2^2}$$

$\pi \circ g$ 的 Lipschitz 常数为 $L_\pi \cdot L_g$：

$$\leq L_f \sqrt{\|s - \tilde{s}\|_2^2 + (L_\pi L_g)^2 \|s - \tilde{s}\|_2^2}$$

$$= L_f \|s - \tilde{s}\|_2 \sqrt{1 + (L_\pi L_g)^2}$$

**Step 5 — 合并所有界**：

$$\Phi(\tilde{s}) - \Phi(s) = [B(\tilde{s}) - B(s)] + [\mathbb{E} B(F(s,\cdot)) - \mathbb{E} B(F(\tilde{s},\cdot))]$$

$$\leq L_B \tau + L_B \cdot L_f \tau \sqrt{1 + (L_\pi L_g)^2}$$

$$= \tau L_B (1 + L_f \sqrt{1 + (L_\pi L_g)^2}) = K$$

因此：

$$\Phi(s) \geq \Phi(\tilde{s}) - K \geq \epsilon$$

$\blacksquare$

##### 2.3.5b Lipschitz 常数的实际计算方法

**神经网络 $B(s)$ 的 Lipschitz 常数**：

对于 MLP 网络 $B(s) = W_L \sigma(W_{L-1} \cdots \sigma(W_1 s))$，Lipschitz 常数的上界为：

$$L_B \leq \prod_{i=1}^L \|W_i\|_2$$

其中 $\|W_i\|_2$ 是权重矩阵的谱范数（最大奇异值）。

**实际计算**：
- **精确谱范数**：对每个权重矩阵做 SVD 分解，$O(n^3)$
- **幂迭代法 (Power Iteration)**：近似最大奇异值，$O(n^2)$ 每次迭代
- **训练时约束**：使用谱归一化 (Spectral Normalization) 在训练过程中限制 $\|W_i\|_2 \leq c$

**闭环动力学的 Lipschitz 常数**：

$$L_F = L_f \sqrt{1 + (L_\pi L_g)^2}$$

各组件的 Lipschitz 常数：
- $L_f$：动力学模型 $f(s, u)$ 已知，可以直接计算
- $L_\pi$：策略网络的谱范数乘积
- $L_g$：感知模型（MLP 蒸馏后）的谱范数乘积

##### 2.3.5c 网格离散化参数的选择

**定理 3.1 的核心推论**：网格步长 $\tau$ 需要足够小使得 $K < \Phi(\tilde{s}) - \epsilon$。

对于 AEBS 场景：
- 状态空间 $S$: $d \in [5, 16], v \in [0, 3]$
- 网格划分 $100 \times 100$
- 网格步长：$\Delta d = 0.11$, $\Delta v = 0.03$
- 网格中心到边缘的最大距离：$\tau = \frac{1}{2}\sqrt{0.11^2 + 0.03^2} \approx 0.057$

若 $L_B = 4.0$（由 Lipschitz 正则化控制），$L_F = 2.5$（闭环系统 Lipschitz）：

$$K = 0.057 \times 4.0 \times (1 + 2.5) = 0.057 \times 14.0 = 0.798$$

若 $\epsilon = 0.1$，则需要 $\Phi(\tilde{s}) \geq K + \epsilon = 0.898$ 才能在该网格单元内保证条件成立。

**更密的网格** → $\tau$ 更小 → $K$ 更小 → 更容易验证通过 → 但计算成本更高（$O(n^d)$，$n$ 为每维网格数，$d$ 为状态维度）。

#### 2.3.6 CEGIS 循环

```
Algorithm SafePVC:
1. 训练 cGAN → 蒸馏为 MLP g(s,z)
2. PPO 预训练 π₀
3. 构建 VCLS = Concat(g, π₀)
4. 数据驱动估计扰动分布 Δs

**扰动分布 $\Delta s$ 的估计方法**：

SafePVC 的一个关键假设是：状态转移的随机性可以用与当前状态独立的扰动 $\Delta s$ 来建模：

$$s_{t+1} = F(s_t, z_0) + \Delta s_t, \quad \Delta s_t \sim \mu$$

**估计步骤**：

1. **收集数据**：在真实环境（或高保真仿真器）中运行标称控制器，记录 $(s_t, u_t, s_{t+1})$ 轨迹
2. **计算标称转移**：使用已知的标称动力学 $F(s_t, z_0)$（无扰动时的状态转移）
3. **计算残差**：$\Delta s_t = s_{t+1} - F(s_t, z_0)$
4. **拟合分布**：对残差进行统计拟合

**AEBS 场景的具体实现**：

标称动力学（确定性部分）：

$$d_{\text{next}}^{\text{nom}} = d - v \cdot \Delta t, \quad v_{\text{next}}^{\text{nom}} = v - a \cdot \Delta t$$

扰动：$\Delta d = d_{\text{next}}^{\text{actual}} - d_{\text{next}}^{\text{nom}}$，$\Delta v = v_{\text{next}}^{\text{actual}} - v_{\text{next}}^{\text{nom}}$

在 AEBS 代码中，扰动分布被建模为均匀分布：

$$\Delta s \sim \text{Uniform}(\text{noise\_bounds}[0], \text{noise\_bounds}[1])$$

其中 $\text{noise\_bounds} = \pm 5\% \times (\text{observation\_space\_high} - \text{observation\_space\_low})$

**验证中的噪声离散化**：

连续均匀分布被离散化为 $10 \times 10$ 的网格（每个状态维度 10 个区间），每个网格单元的概率质量为：

$$\text{pmass}_i = \frac{\text{volume}(\text{cell}_i)}{\text{volume}(\text{noise\_space})}$$

对于均匀分布，这等价于 $\text{pmass}_i = \frac{1}{10 \times 10} = 0.01$（每个单元等概率）。

期望近似为：

$$\mathbb{E}[B(s')] \approx \sum_{i=1}^{100} \text{pmass}_i \cdot B^U(s_{\text{next}} + \text{cell}_i)$$

其中 $B^U$ 是通过 IBP 计算的上界。
5. while SBC 未验证:
   a. 训练 SBC 网络 B(s)
   b. IBP 验证 B(s) 的条件
   c. 如果验证通过 → 计算概率安全下界
   d. 提取反例 (counterexamples)
   e. 交替更新 π 和 B
6. 返回 (B, π)
```

##### 2.3.6a CEGIS 循环的详细工作机制

**CEGIS (Counter-Example Guided Inductive Synthesis)** 是一种迭代的合成方法，核心思想是：**验证失败的状态（反例）被用来指导下一轮训练**。

**每次迭代的详细步骤**：

**Step 1 — 训练 L-Net (SBC 网络 $B(s)$)**：

- 训练 10 个 epoch
- 训练数据：状态空间网格点 + 上轮反例
- 损失函数：
  $$\mathcal{L}_L = 1000 \times \text{martingale\_loss}(B, s, \pi) + \text{lip\_loss}(B) + \text{region\_loss}(B)$$
- 使用 16 个三角分布噪声样本近似 $\mathbb{E}[B(s')]$

**鞅损失的实现细节**：

```python
def martingale_loss(l, l_next, eps=0.0):
    # l = B(s_t): 当前状态的 SBC 值
    # l_next = B(s_{t+1}): 下一步状态的 SBC 值（含噪声）
    # eps: 最小递减量 ε
    diff = l_next - l  # 期望递增
    return torch.mean(torch.clamp(diff + eps, min=0.0))
    # clamp: 只在 diff + eps > 0 时产生损失（违反递减条件）
```

**三角分布噪声的优势**：

$$f(x) = \begin{cases} 1 + x & x \in [-1, 0] \\ 1 - x & x \in [0, 1] \end{cases}$$

三角分布相比均匀分布，更集中于中心（均值 0），更符合实际扰动的统计特性。实现：

```python
def triangular(shape):
    U = torch.rand(shape)
    p1 = -1 + torch.sqrt(2 * U)
    p2 = 1 - torch.sqrt(2 * (1 - U))
    return torch.where(U <= 0.5, p1, p2)
```

**Step 2 — IBP 验证**：

- 在 $100 \times 100$ 网格上验证鞅递减条件
- 使用 IBP 计算 $B(s)$ 和 $B(s')$ 的区间界
- 考虑噪声网格 $10 \times 10$（每个状态维度 10 个噪声区间）
- 加入 Lipschitz 修正 $K = \text{lip\_const} \times \delta$
- 如果违反率 $\leq 0.1\%$ → 验证通过

**Step 3 — 反例提取**：

验证失败的状态被收集到 `violation_buffer`：

```python
if v > 0:  # v = 违反数量
    temp = s_batch[violating_indices].cpu().numpy()
    self.violation_buffer.append(temp)
```

这些反例在下一轮训练中被加入训练集，迫使网络在"困难的"状态上也满足条件。

**Step 4 — 训练 P-Net (策略网络 $\pi$)**：

- 训练 1 个 epoch
- 冻结 L-Net，梯度流过策略网络
- 损失函数：
  $$\mathcal{L}_P = 10 \times \text{martingale\_loss}(B, s, \pi) + \text{lip\_loss}(\pi) + 10 \times \text{MSE}(\pi, \pi_0)$$
- 使用 128 个噪声样本（比 L-Net 训练更精确）

**Step 5 — 联合训练（可选）**：

同时更新 L-Net 和 P-Net，在两者之间取得平衡。

##### 2.3.6b 收敛性与终止条件

**终止条件**（满足任一）：

1. **验证通过**：IBP 验证的违反率 $\leq 0.1\%$ → 成功
2. **最大迭代次数**：100 次迭代
3. **时间限制**：1 小时
4. **概率界满足**：$1 - 1/B_{\text{ratio}} \geq p$

**收敛性分析**：

CEGIS 不保证收敛（可能陷入无限循环），但在实践中，随着反例的积累，训练数据覆盖了越来越多的"困难"区域，SBC 网络和策略网络逐渐协同进化到满足验证条件的状态。

**关键超参数**：

| 参数 | 值 | 作用 |
|------|------|------|
| L-Net epochs | 10 | 每轮 SBC 训练次数 |
| P-Net epochs | 1 | 每轮策略训练次数 |
| L-Net lip | 4.0 | SBC 网络 Lipschitz 约束 |
| P-Net lip | 2.0 | 策略网络 Lipschitz 约束 |
| $\epsilon$ | 0.1 | 鞅最小递减量 |
| $\gamma$ | 1.0 | 折扣因子 |
| reach_prob | 0.95 | 目标安全概率 |
| factor | 0.05 | 噪声幅度（状态范围的 5%） |

#### 2.3.7 AEBS 基准测试

**系统动力学**：

$$d_{k+1} = d_k - v_k \Delta t, \quad v_{k+1} = v_k - a_k \Delta t$$

- 状态：$s = (d, v)$，$d \in [5, 16]$ m（到目标的距离），$v \in [0, 3]$ m/s（速度）
- 控制：$a \in [-3, 3]$ m/s²（加速度）
- $\Delta t = 0.05$ s

**安全属性**：初始状态 $d \in [15, 16], v \in [2.5, 3.0]$（远且快），要求在到达 $d \in [5, 6]$ 前速度降到 $v < 0.5$ m/s，概率至少 90%。

---

## 三、近年相关研究调查

本节对与本研究方向相同或高度相似的近年论文进行全面梳理，按**方法相似度**从高到低分类，并简述每篇论文的核心方法、贡献及与本工作的异同。

### 3.1 高度相似工作（直接竞争/重叠）

#### 3.1.1 ABNet: Adaptive explicit-Barrier Net for Safe and Scalable Robot Learning

| 项目 | 内容 |
|------|------|
| **作者** | Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Daniela Rus (MIT CSAIL) |
| **发表** | [ICML 2025](https://icml.cc/virtual/2025/poster/43514) / [arXiv 2024](https://arxiv.org/abs/2406.13025) |
| **代码** | [GitHub: Weixy21/ABNet](https://github.com/Weixy21/ABNet) |

**核心方法**：ABNet 是 BarrierNet 的直接扩展，在原始 BarrierNet 框架上引入**注意力机制 (attention mechanism)** 来动态加权多个安全约束。每个"BarrierNet head"负责一个安全子任务，注意力层将多个 head 的输出融合，同时保持数学上的安全保证。

**与本工作的异同**：
- **相同点**：都基于 BarrierNet 的 dCBF + 可微 QP 框架；都来自 MIT CSAIL 同一研究组
- **不同点**：ABNet 关注**可扩展性**（多约束、多任务场景），本工作关注**概率安全验证**（SBC 鞅理论）；ABNet 不提供概率安全保证，本工作的双层架构是互补的
- **潜在结合**：ABNet 的多头架构可以与本工作的 SBC 验证结合，形成"可扩展 + 概率安全"的完整方案

#### 3.1.2 Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees

| 项目 | 内容 |
|------|------|
| **作者** | Xinhang Ma, Junlin Wu, Hussein Sibai, Yiannis Kantaros 等 |
| **发表** | [AAAI 2026](https://arxiv.org/abs/2503.00191) / [arXiv 2025](https://arxiv.org/abs/2503.00191) |

**核心方法**：针对**视觉控制器**合成**半概率安全保证**的控制器。核心思路是将视觉感知模块和控制器模块分别处理，通过组合分析（compositional analysis）在感知误差有界的条件下提供概率安全保证。

**与本工作的异同**：
- **极为相似**：同样处理视觉控制器 + 概率安全保证的组合问题
- **不同点**：本文使用半概率（semi-probabilistic）保证（在感知误差有界时保证安全），本工作使用 SBC 鞅理论提供全局概率保证；本文不使用 CBF/dCBF，而是依赖组合验证
- **竞争关系**：这是最直接的竞争工作，需要在实验中对比安全概率下界和控制器性能

#### 3.1.3 Stochastic Neural Control Barrier Functions (SNCBF)

| 项目 | 内容 |
|------|------|
| **作者** | Zhang, Tayal 等 |
| **发表** | [arXiv 2025](https://arxiv.org/abs/2506.21697) |

**核心方法**：提出**随机神经控制障碍函数 (SNCBF)**，将经典 CBF 扩展到随机系统。提出两种框架：(1) **无验证合成** (verification-free synthesis) 用于光滑 SNCBF；(2) **验证在环** (verification-in-the-loop) 用于光滑和 ReLU SNCBF。通过超鞅条件将 CBF 约束与概率安全界联系起来。

**与本工作的异同**：
- **高度相似**：同样结合了 CBF + 随机系统 + 概率安全保证
- **不同点**：SNCBF 直接在 CBF 定义中引入随机性（随机 CBF），本工作使用确定性 CBF (BarrierNet) + 独立的概率验证 (SBC)；SNCBF 不涉及视觉感知
- **互补性**：SNCBF 的随机 CBF 理论可以为 BarrierNet 的 dCBF 提供概率化的理论扩展

#### 3.1.4 Controlled Supermartingale Functions for Stochastic Differential Equations

| 项目 | 内容 |
|------|------|
| **作者** | Sriram Sankaranarayanan 等 |
| **发表** | [CDC 2025](https://home.cs.colorado.edu/~srirams/papers/supermartingales-sde-cdc-2025.pdf) |

**核心方法**：研究**受控超鞅函数**的构造方法，用于合成保证随机微分方程安全性的反馈控制律。核心是将超鞅条件与 CBF 约束统一，通过 Lyapunov-like 分析得到概率安全保证。

**与本工作的异同**：
- **理论高度相似**：本工作的 SBC 验证同样基于超鞅理论（Doob 不等式），本文提供了更一般化的受控超鞅函数框架
- **不同点**：本文针对连续时间 SDE，本工作针对离散时间系统；本文不涉及视觉感知或 BarrierNet
- **理论借鉴**：本文的受控超鞅函数理论可以直接用于本工作双层架构的概率安全分析

### 3.2 方法相近工作（共享核心技术组件）

#### 3.2.1 CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **作者** | Yang, Werner 等 |
| **发表** | [arXiv 2025](https://arxiv.org/abs/2510.14959) |

**核心方法**：提出在 RL 训练过程中同时使用 CBF 进行**安全过滤** (safety filtering) 和**奖励塑形** (reward shaping) 的双重策略。安全过滤确保训练时的即时安全，奖励塑形引导策略学习内在安全的行为。两者互补，使训练出的策略内化安全约束。

**与本工作的异同**：
- **相似点**：都在训练过程中使用 CBF 保证安全
- **不同点**：CBF-RL 使用传统 CBF（非可微），不提供形式化验证保证；本工作使用可微 CBF (dCBF) + SBC 形式化验证
- **可借鉴**：CBF-RL 的奖励塑形思路可以引入本工作的 PPO 预训练阶段

#### 3.2.2 Verification of Neural CBFs with Symbolic Derivative Bound Propagation

| 项目 | 内容 |
|------|------|
| **作者** | Hu 等 |
| **发表** | [ICML 2025](https://proceedings.mlr.press/v270/hu25a.html) / [PDF](https://raw.githubusercontent.com/mlresearch/v270/main/assets/hu25a/hu25a.pdf) |

**核心方法**：提出**符号化导数界传播** (Symbolic Derivative Bound Propagation) 来验证神经 CBF。核心创新是用符号表达式（而非数值区间）来传播导数界，获得比 IBP 更紧的验证界，从而减少验证的保守性。

**与本工作的异同**：
- **相似点**：都涉及 Neural CBF 的形式化验证
- **不同点**：本文使用符号化方法替代 IBP，获得更紧的界；本工作使用 IBP + Lipschitz 修正
- **可替代方案**：本文的符号化方法可以替代本工作中的 IBP 验证，提高验证精度

#### 3.2.3 Scalable Verification of Neural Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **作者** | 多作者 |
| **发表** | [arXiv 2026](https://arxiv.org/html/2511.06341v2) |

**核心方法**：提出可扩展的 Neural CBF 验证框架，克服 SMT-based 验证器的可扩展性限制。通过分层验证和并行化策略，将验证时间从指数级降低到多项式级。

**与本工作的异同**：
- **相似点**：都面临 Neural CBF/CBF 验证的可扩展性问题
- **不同点**：本文关注验证算法本身的可扩展性，本工作关注将验证嵌入训练循环 (CEGIS)
- **可结合**：本文的可扩展验证方法可以加速本工作的 IBP 验证步骤

#### 3.2.4 Learning a Formally Verified Control Barrier Function in Stochastic Environment

| 项目 | 内容 |
|------|------|
| **作者** | StochLab 团队 |
| **发表** | [arXiv 2024](https://arxiv.org/abs/2403.19332) / [StochLab](https://www.stochlab.com/projects/NCBF.html) |

**核心方法**：提出在**随机环境**中合成**形式化验证的连续时间神经 CBF** 的算法。在单一优化循环中同时合成和验证 CBF，使用 SOS (Sum-of-Squares) 或 SMT 验证器提供形式化保证。

**与本工作的异同**：
- **高度相似**：同样在随机环境中合成经过验证的 Neural CBF
- **不同点**：本文使用连续时间框架和 SOS/SMT 验证，本工作使用离散时间框架和 IBP 验证；本文不涉及视觉感知
- **方法借鉴**：本文的"合成+验证一体化"思路与本工作的 CEGIS 循环类似

#### 3.2.5 SafeMind: A Risk-Aware Differentiable Control Framework

| 项目 | 内容 |
|------|------|
| **发表** | [arXiv 2026](https://arxiv.org/abs/2604.09474) |

**核心方法**：提出**风险感知可微控制框架**，使用随机 CBF 结合**方差感知**的端到端训练。在 CBF 约束中引入方差项，使得安全保证不仅考虑均值还考虑不确定性的大小。

**与本工作的异同**：
- **相似点**：同样结合可微 CBF 与端到端训练
- **不同点**：SafeMind 在 CBF 中直接引入方差（风险度量），本工作使用 SBC 鞅理论提供概率保证
- **可借鉴**：方差感知的 CBF 约束可以作为本工作 dCBF 的增强

#### 3.2.6 Probabilistic Safety Guarantees for Learned Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **发表** | [MDPI Mathematics 2025](https://www.mdpi.com/2227-7390/14/3/516) |

**核心方法**：为**学习得到的 CBF** 建立严格的理论基础，通过**显式概率界**将神经网络的逼近误差与安全保证联系起来。证明了当逼近误差在某个范围内时，学习得到的 CBF 仍然以一定概率保证安全。

**与本工作的异同**：
- **理论高度相关**：本工作的 BarrierNet 也是"学习得到的 CBF"，本文的概率界理论可以直接应用于分析本工作 dCBF 的概率安全性
- **不同点**：本文提供理论分析框架，不提供具体的算法；本工作是完整的算法+验证+实验方案

### 3.3 CEGIS 框架相关工作

#### 3.3.1 Formal Synthesis of Neural Barrier Certificates via Counterexample Guided Learning

| 项目 | 内容 |
|------|------|
| **发表** | [ACM TOSEM 2023](https://dl.acm.org/doi/10.1145/3609125) |

**核心方法**：提出基于 CEGIS 的神经障碍证书自动合成方法。训练器（learner）生成候选障碍证书，验证器（verifier）使用 SMT 求解器检查，反例被反馈到训练集中迭代改进。

**与本工作的异同**：
- **框架相似**：都使用 CEGIS 循环（训练→验证→反例→再训练）
- **不同点**：本文使用 SMT 验证器，本工作使用 IBP 验证器（更快但更保守）；本文不涉及 CBF 或概率安全
- **框架参考**：本文的 CEGIS 框架设计是本工作 CEGIS 循环的重要参考

#### 3.3.2 Formal Synthesis of Safe KAN Controllers

| 项目 | 内容 |
|------|------|
| **发表** | [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0035.pdf) |

**核心方法**：使用 **Kolmogorov-Arnold Network (KAN)** 替代传统 MLP 作为安全控制器和障碍证书的网络架构。KAN 在每个边上使用可学习的单变量函数替代传统 MLP 的固定激活函数，具有更好的可解释性和更紧凑的网络结构。通过 CEGIS 循环进行合成和验证。

**与本工作的异同**：
- **相似点**：都使用 CEGIS 合成安全控制器
- **不同点**：本文使用 KAN 架构，本工作使用 MLP；KAN 可能更适合 IBP 验证（网络更紧凑，传播更精确）
- **可替代方案**：本工作的 SBC 网络和策略网络都可以尝试用 KAN 替代 MLP

#### 3.3.3 k-Inductive Neural Barrier Certificates for Unknown Nonlinear Systems

| 项目 | 内容 |
|------|------|
| **发表** | [arXiv 2026](https://arxiv.org/pdf/2605.20108) |

**核心方法**：提出 **$k$-归纳神经障碍证书**，将传统的一步归纳（1-inductive）推广到 $k$ 步归纳。即如果 $B(s_t), B(s_{t+1}), \ldots, B(s_{t+k-1})$ 都满足条件，则 $B(s_{t+k})$ 也满足。这放松了对 $B$ 函数的要求，使更容易找到合格的障碍证书。

**与本工作的异同**：
- **方法创新**：$k$-归纳可以显著降低 SBC 条件的严格性
- **可结合**：本工作的 SBC 鞅递减条件可以推广为 $k$-步鞅条件，提高验证通过率

#### 3.3.4 Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **发表** | [TU Eindhoven 2025](https://research.tue.nl/files/362365933/Shakhesi_RDTCBFs_pure_tue.pdf) |

**核心方法**：针对**离散时间**系统，使用 CEGIS 合成**鲁棒 CBF**。在 CBF 条件中考虑模型不确定性和外部扰动，确保 CBF 在最坏情况下仍然满足安全约束。

**与本工作的异同**：
- **高度相似**：同样处理离散时间系统 + CEGIS + CBF 合成
- **不同点**：本文使用鲁棒 CBF（最坏情况分析），本工作使用概率安全（平均情况分析）
- **互补性**：鲁棒性分析与概率分析的结合可以提供更全面的安全保证

#### 3.3.5 Simultaneous Synthesis and Verification of Neural Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **发表** | [TU Delft 2024](https://autonomousrobots.nl/msc_projects_finished/24-sunnywang-neuralcontrolbarrierfunctions) |
| **代码** | [GitHub: tud-amr/ncbf-simultaneous-synthesis-and-verification](https://github.com/tud-amr/ncbf-simultaneous-synthesis-and-verification) |

**核心方法**：将验证方案**嵌入训练循环**中，同时合成和验证神经 CBF。每次训练步骤都进行部分验证检查，而非等到训练完成后再验证。这减少了 CEGIS 循环的次数，加速收敛。

**与本工作的异同**：
- **框架相似**：都在训练循环中嵌入验证
- **不同点**：本文使用持续验证（每个 step 都检查），本工作使用间歇验证（每 10 epoch 检查一次）
- **改进方向**：本工作可以借鉴本文的持续验证策略，减少训练-验证的迭代次数

### 3.4 视觉安全控制相关工作

#### 3.4.1 Enforcing Safety for Vision-Based Controllers via CBFs and Neural Radiance Fields

| 项目 | 内容 |
|------|------|
| **发表** | [ResearchGate 2023](https://www.researchgate.net/publication/372131019_Enforcing_safety_for_vision-based_controllers_via_Control_Barrier_Functions_and_Neural_Radiance_Fields) |

**核心方法**：使用**神经辐射场 (NeRF)** 从 RGB-D 图像中重建 3D 场景，然后在 3D 空间中构造 CBF 来保证视觉控制器的安全。NeRF 提供了高维图像到低维安全相关状态（如障碍物距离）的桥梁。

**与本工作的异同**：
- **相似点**：都处理视觉控制器的安全保障问题，都使用 CBF
- **不同点**：本文使用 NeRF 进行 3D 重建，本工作使用 cGAN + MLP 蒸馏进行感知建模；本文不涉及概率安全验证
- **替代方案**：NeRF 可以替代本工作的 cGAN 感知模型，提供更精确的 3D 感知

#### 3.4.2 Point Cloud-Based CBF Regression for Safe Vision-Based Control

| 项目 | 内容 |
|------|------|
| **发表** | [ICRA 2024 (Berkeley Hybrid Robotics)](https://hybrid-robotics.berkeley.edu/publications/ICRA2024_Point_Cloud_CBF.pdf) |

**核心方法**：在**点云数据**上合成 CBF，用于安全的视觉控制。使用低计算开销的 CBF 回归方法从点云中提取安全相关信息，避免了完整的 3D 重建。

**与本工作的异同**：
- **相似点**：都处理视觉/感知输入下的安全控制
- **不同点**：本文使用点云（3D 感知），本工作使用 2D 图像 + cGAN 建模
- **扩展方向**：如果将本工作扩展到 3D 场景，点云 CBF 是重要的参考

#### 3.4.3 Towards Verified Vision-Based Neural Network Controllers

| 项目 | 内容 |
|------|------|
| **发表** | [Washington University Thesis 2024](https://openscholarship.wustl.edu/cgi/viewcontent.cgi?article=2185&context=eng_etds) |

**核心方法**：设计一个**端到端的可验证视觉控制器框架**，通过开发感知模块的抽象（abstraction）来使形式化验证工具能够处理视觉输入。核心是将高维图像映射到低维抽象空间，然后在抽象空间中进行验证。

**与本工作的异同**：
- **极为相似**：同样追求视觉控制器的形式化验证
- **不同点**：本文使用抽象化方法，本工作使用 cGAN + MLP 蒸馏；本文不使用 CBF
- **竞争关系**：这是另一个视觉控制器验证的完整方案

#### 3.4.4 Towards Safety Assured End-to-End Vision-Based Control

| 项目 | 内容 |
|------|------|
| **发表** | [ScienceDirect (IFAC) 2023](https://www.sciencedirect.com/science/article/abs/pii/S2405896323017810) |

**核心方法**：采用**解耦方法**——同时学习一个最优端到端控制器和一个状态预测端到端模型。状态预测模型提供安全相关信息，用于验证控制器的安全性。

**与本工作的异同**：
- **相似点**：都处理端到端视觉控制的安全保证
- **不同点**：本文使用解耦架构（感知+控制分离），本工作使用 cGAN 建模感知不确定性

### 3.5 随机 CBF 与概率安全相关工作

#### 3.5.1 Control Barrier Functions for Stochastic Systems and Safety-critical Control

| 项目 | 内容 |
|------|------|
| **发表** | [arXiv 2024 更新](https://arxiv.org/html/2209.08728v5)（综述） |

**核心方法**：全面综述了随机系统 CBF 的理论发展，包括：(1) 超鞅 CBF 的定义和性质；(2) 随机系统中 CBF 的各种变体；(3) 安全概率界的计算方法。

**与本工作的关系**：本工作的理论基础之一，双层架构的概率安全分析可以参考本文的理论框架。

#### 3.5.2 Robust Safety under Stochastic Uncertainty with Discrete-Time CBFs

| 项目 | 内容 |
|------|------|
| **发表** | [RSS 2023](https://www.roboticsproceedings.org/rss19/p084.pdf) |

**核心方法**：为离散时间随机系统开发有限时间安全界。使用离散时间 CBF 约束，在最坏情况扰动下保证系统在有限时间内不进入不安全集。

**与本工作的异同**：
- **相似点**：都处理离散时间随机系统的安全保证
- **不同点**：本文使用最坏情况分析（鲁棒 CBF），本工作使用概率分析（SBC 鞅理论）

#### 3.5.3 Safety Guarantees for Neural Network Dynamic Systems via Stochastic Barrier Functions

| 项目 | 内容 |
|------|------|
| **发表** | [NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/3f1f3e38d1ce5653afb81505d3e26618-Paper-Conference.pdf) |

**核心方法**：为**神经网络动力系统**提供安全认证和控制。使用**随机障碍函数 (SBF)** 类似 Lyapunov 函数的方法，通过 SOS 优化合成 SBF，提供概率安全保证。

**与本工作的异同**：
- **方法相似**：同样使用随机障碍函数 + 概率安全保证
- **不同点**：本文使用 SOS 优化（适用于多项式系统），本工作使用神经网络 + IBP 验证（适用于一般非线性系统）
- **基准比较**：本文是重要的基准工作，本工作的实验应与本文的结果进行对比

#### 3.5.4 Ensuring Safety Through Stochastic Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **发表** | [Benelux Meeting 2025](https://www.beneluxmeeting.nl/2025/uploads/papers/bmsc2025_370.pdf) |

**核心方法**：使用**神经网络识别未知系统动力学**，然后使用随机 CBF (SCBF) 约束安全概率。提出了 NN 逼近与真实动力学之间的非渐近误差界的未来研究方向。

**与本工作的异同**：
- **相似点**：都使用神经网络 + 随机 CBF 处理不确定系统
- **不同点**：本文用 NN 学习动力学，本工作用 NN 学习 CBF 和策略

#### 3.5.5 Probabilistic Shielding for Safe Reinforcement Learning

| 项目 | 内容 |
|------|------|
| **发表** | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33767) |

**核心方法**：提出**概率盾牌 (probabilistic shielding)** 方法，为随机动力学中的安全 RL 提供可扩展的、具有严格形式化保证的方法。盾牌在策略输出之后进行安全过滤，保证安全概率不低于阈值。

**与本工作的异同**：
- **相似点**：都提供概率安全保证 + 与 RL 的结合
- **不同点**：概率盾牌是后处理方法（不改策略），本工作的 BarrierNet 是嵌入式安全层（通过训练改进策略）
- **可对比**：概率盾牌可以作为本工作的 baseline 对比方法

#### 3.5.6 Scenario Generation for Risk-Aware RL with Probabilistic Barrier Certificates

| 项目 | 内容 |
|------|------|
| **发表** | [arXiv 2026](https://arxiv.org/html/2606.04812v1) |

**核心方法**：通过**轨迹采样**构造概率障碍证书来验证策略。使用场景生成方法从经验数据中估计安全概率，而非依赖精确的系统模型。

**与本工作的异同**：
- **相似点**：都提供概率安全保证
- **不同点**：本文使用数据驱动的场景方法（无模型），本工作使用 IBP 验证（基于模型）

### 3.6 可微 CBF / 端到端安全学习相关工作

#### 3.6.1 End-to-End Safe RL through Barrier Functions (RL-CBF)

| 项目 | 内容 |
|------|------|
| **发表** | [AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4213/4091) |

**核心方法**：开创性地将无模型 RL 与基于模型的 CBF 控制器结合。RL 策略的输出经过 CBF-based 安全过滤器修正后再执行，不安全动作被投影到安全集上。

**与本工作的关系**：这是 CBF + RL 结合的先驱工作，本工作的 PPO 预训练 + BarrierNet 安全过滤继承了这一思路。

#### 3.6.2 Learning Differentiable Safety-Critical Control using CBFs

| 项目 | 内容 |
|------|------|
| **发表** | [ECC 2022 (Berkeley)](https://hybrid-robotics.berkeley.edu/publications/ECC2022_Differentiable_CBF.pdf) |

**核心方法**：将可微凸优化应用于 CBF-based QP，实现端到端训练。使用可微优化层的隐函数定理进行反向传播。处理了递推可行性问题。

**与本工作的关系**：BarrierNet 的同期工作，共享可微 QP 的技术路线。本文的可微优化技术是本工作 BarrierNet 层实现的重要参考。

#### 3.6.3 Safe RL Using Robust Control Barrier Functions

| 项目 | 内容 |
|------|------|
| **作者** | Emam, Notomista 等 |
| **代码** | [GitHub: yemam3/SAC-RCBF](https://github.com/yemam3/SAC-RCBF) / [Mod-RL-RCBF](https://github.com/yemam3/Mod-RL-RCBF) |

**核心方法**：将鲁棒 CBF 作为可微层嵌入基于模型的 RL 中。策略通过 CBF-QP 安全层传播，梯度可以回传到策略网络。

**与本工作的关系**：提供了可微 CBF-QP 层的开源实现，是本工作 BarrierNet 层代码实现的重要参考。

#### 3.6.4 Model-Free Safe RL through Neural Barrier Certificate

| 项目 | 内容 |
|------|------|
| **发表** | [IEEE Robotics 2023](https://www.researchgate.net/publication/367302208) |
| **代码** | [GitHub: jjyyxx/srlnbc](https://github.com/jjyyxx/srlnbc) |

**核心方法**：在**无模型**条件下使用神经障碍证书求解安全约束下的最优控制。通过数据驱动方法学习障碍证书，在 RL 训练中作为安全约束。

**与本工作的异同**：
- **相似点**：都使用神经网络 + 障碍证书保证安全
- **不同点**：本文无模型（纯数据驱动），本工作有模型（已知动力学 + IBP 验证）

#### 3.6.5 Joint Differentiable Optimization and Verification for Certified RL

| 项目 | 内容 |
|------|------|
| **发表** | [ACM 2023](https://dl.acm.org/doi/10.1145/3576841.3585919) |

**核心方法**：通过双层优化问题联合进行 RL 和形式化验证。上层优化 RL 策略性能，下层验证安全性，两者通过可微优化联合训练。

**与本工作的异同**：
- **极为相似**：同样联合训练和验证
- **不同点**：本文使用双层优化框架，本工作使用 CEGIS 交替训练框架

#### 3.6.6 Differentiable High Order CBF-Based Safe RL

| 项目 | 内容 |
|------|------|
| **发表** | [北京理工大学](https://pure.bit.edu.cn/zh/publications/differential-high-order-control-barrier-function-based-safe-reinf/) |

**核心方法**：设计基于微分高阶 CBF 的安全约束，用于安全 RL。将 HOCBF 的递推序列函数微分化，使得安全约束可以端到端训练。

**与本工作的关系**：如果本工作扩展到相对度 >1 的系统（如 jerk 控制），本文的微分 HOCBF 是重要参考。

### 3.7 其他相关工作

| 论文 | 年份/会议 | 核心方法 | 与本工作的关系 |
|------|-----------|---------|---------------|
| **Exact Verification of ReLU Neural CBFs** ([NeurIPS 2023](https://openreview.net/forum?id=1h2TAUEfc4)) | 2023 | ReLU Neural CBF 精确验证（混合整数规划） | 验证方法的基准 |
| **SMT-Based CEGIS of Neural BC** ([ScienceDirect 2023](https://epub.ub.uni-muenchen.de/122149/1/1-s2.0-S2405896323016233-main__1_.pdf)) | 2023 | SMT + CEGIS 合成 Neural BC | CEGIS 框架参考 |
| **Data-Driven Barrier Certificate Generation** ([ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S1383762125000918)) | 2025 | 深度学习 + 符号回归生成 BC | 数据驱动方法参考 |
| **Incremental Synthesis of Safe Controller** ([Springer 2026](https://link.springer.com/chapter/10.1007/978-3-032-26204-2_22)) | 2026 | 增量式安全控制器合成 | BC 引导增量合成 |
| **Safe Reach Set Computation via Neural BC** ([arXiv 2024](https://arxiv.org/html/2404.18813v1)) | 2024 | Neural BC 计算可达集 | 可达性分析参考 |
| **Collision Avoidance Using CBF** ([MDPI 2024](https://www.mdpi.com/2079-9292/14/3/557)) | 2024 | CBF + QP 避障 | CBF 在自动驾驶中的应用 |
| **Safe RL via Probabilistic Logic Shields** ([IJCAI 2023](https://www.ijcai.org/proceedings/2023/637)) | 2023 | 概率逻辑编程建模安全约束 | 概率安全的方法参考 |
| **Neural CBFs for Safe Navigation** ([arXiv 2024](https://arxiv.org/html/2407.19907v1)) | 2024 | 数据驱动 Neural CBF 合成 | 无模型 CBF 学习参考 |
| **Verified Safe RL for Neural Network Dynamic Systems** ([NeurIPS 2024](https://arxiv.org/html/2405.15994v1)) | NeurIPS 2024 | 有限域验证 + 经验安全约束 | 有限时间安全保证参考 |
| **Formally Verifying DRL with Neural Lyapunov BC** ([arXiv 2024](https://arxiv.org/pdf/2405.14058)) | 2024 | Neural Lyapunov-Barrier 验证 DRL | Lyapunov + Barrier 联合验证 |
| **Formal Methods in Robot Policy Learning and Verification** ([arXiv 2026](https://arxiv.org/pdf/2602.06971)) | 2026 | 机器人策略的形式化方法综述 | 全面的领域综述 |

### 3.8 领域空白与机会

经过对以上 30+ 篇相关论文的全面调查，以下方向存在**明确的研究空白**：

| 研究空白 | 说明 | 本工作的填补方式 |
|----------|------|-----------------|
| **BarrierNet + 形式化概率验证** | ABNet (ICML 2025) 扩展了 BarrierNet 的可扩展性，SNCBF (2025) 扩展了随机 CBF，但**无人将 BarrierNet 的 dCBF + dQP 与 SBC 鞅验证框架结合** | 双层安全架构 |
| **dCBF 的概率安全保证** | BarrierNet/ABNet 仅提供确定性安全保证，缺少概率化分析；SNCBF 提供概率保证但不使用 dCBF | dCBF 确定性保证 + SBC 概率验证 |
| **视觉控制器的双层安全** | 视觉控制器验证工作（Ma et al. AAAI 2026）使用组合验证但不使用 CBF；CBF+视觉工作（NeRF-CBF, Point Cloud CBF）不进行形式化验证 | cGAN 感知 + BarrierNet 控制 + SBC 验证 |
| **CEGIS 中的 BarrierNet** | 现有 CEGIS 框架（ACM 2023, IJCAI 2025, TU Delft 2024）未利用 BarrierNet 的环境自适应 penalty 函数 | 反例引导的 penalty 网络训练 |
| **内层确定性 + 外层概率** | 现有工作要么使用确定性保证（CBF），要么使用概率保证（SBC），**无人在同一架构中同时提供两种保证** | 内层 BarrierNet + 外层 SBC |

---

## 四、可行性分析与创新贡献

### 4.1 可行性论证

| 方面 | 可行性 | 理由 |
|------|--------|------|
| **理论可行性** | ✅ 高 | BarrierNet 的安全保证是确定性的，SBC 的概率保证是互补的，两者数学基础兼容 |
| **技术可行性** | ✅ 高 | 两个系统共享 PyTorch 框架，auto_LiRPA 支持 QP 层的绑定传播 |
| **计算可行性** | ⚠️ 中 | BarrierNet dQP 增加前向传播开销，但 AEBS 场景控制维度低 (1D)，QP 求解快 |
| **验证可行性** | ⚠️ 中 | dQP 层需要处理 IBP 传播，可能需要近似或线性化 |
| **发表可行性** | ✅ 高 | 明确的创新点 + 填补空白 + 实验可行 |

### 4.2 创新贡献

**贡献 1：双层安全保证架构 (Dual-Layer Safety Architecture)**
- 首次提出结合**运行时安全保证** (BarrierNet/dCBF) 和**离线形式化验证** (SBC/鞅理论) 的双层架构
- 内层 BarrierNet 确保每步控制满足安全约束（硬保证）
- 外层 SBC 证明闭环系统的长期概率安全（软保证）

**贡献 2：概率安全可微控制障碍函数 (Probabilistic dCBF)**
- 在 dCBF 中引入随机扰动建模，使 BarrierNet 的安全保证从确定性扩展到概率性
- 推导出 dCBF 约束在扰动下的概率满足条件

**贡献 3：CEGIS 引导的 BarrierNet 训练**
- 利用 SBC 验证产生的反例 (counterexamples) 指导 BarrierNet 的 penalty 函数学习
- 相比 BarrierNet 原始的监督学习，CEGIS 方法可以探索更困难的状态

**贡献 4：增强的验证可处理性**
- BarrierNet 的确定性安全约束缩小了需要验证的状态空间
- SBC 只需验证 BarrierNet 约束之外的"残余风险"

### 4.3 与现有工作的差异化

| 方法 | 安全保障 | 端到端 | 概率性 | 形式化验证 | 环境自适应 |
|------|---------|--------|--------|-----------|-----------|
| 原始 BarrierNet | ✅ 确定性 | ✅ | ❌ | ❌ | ✅ |
| 原始 SafePVC | ✅ 概率性 | ✅ | ✅ | ✅ IBP | ❌ |
| Verified Safe RL (NeurIPS'24) | ✅ 有限域 | ✅ | ❌ | ✅ | ❌ |
| Diff. Verification (2025) | ✅ | ✅ | ❌ | ✅ | ❌ |
| **本研究 (BarrierNet+SafePVC)** | **✅ 双层** | **✅** | **✅** | **✅ IBP** | **✅** |

---

## 五、结合方案设计

### 5.1 总体思路

```
原始 SafePVC:
  图像生成 → 状态估计 → 控制器(MLP/PPO) → 加速度
                                            ↓
                                    SBC 验证 (鞅理论)

结合 BarrierNet 后:
  图像生成 → 状态估计 → 控制器(MLP/PPO) → 参考加速度 → BarrierNet(dQP) → 安全加速度
                                                          ↑
                                                    dCBF + penalty 网络
                                                          ↓
                                                    SBC 验证 (鞅理论 + dCBF 约束增强)
```

### 5.2 新系统架构

#### 5.2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         BarrierNet-SafePVC 系统                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    可验证闭环系统 (VCLS)                         │        │
│  │                                                                  │        │
│  │  s,d → [cGAN/MLP Gen] → img → [StateNet] → ŝ                   │        │
│  │                    z(潜在扰动)              ↓                    │        │
│  │                                    [PolicyNet] → û(参考控制)     │        │
│  │                                          ↓                       │        │
│  │                                  [PenaltyNet] → p₁,p₂,...       │        │
│  │                                          ↓                       │        │
│  │                                  [H-Net] → H, [F-Net] → F      │        │
│  │                                          ↓                       │        │
│  │                           ┌──── [BarrierNet Layer] ────┐        │        │
│  │                           │  dQP: min ½uᵀHu + Fᵀu     │        │        │
│  │                           │  s.t. dCBF constraints     │        │        │
│  │                           │       u_min ≤ u ≤ u_max    │        │        │
│  │                           └────────────────────────────┘        │        │
│  │                                          ↓                       │        │
│  │                                    u* (安全控制)                  │        │
│  │                                          ↓                       │        │
│  │                               [Dynamics] → s' = f(s, u*)        │        │
│  │                                          ↓                       │        │
│  │                              [扰动模型] → s'' = s' + Δs          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                          ↓                                   │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    SBC 验证器 (CEGIS 循环)                       │        │
│  │                                                                  │        │
│  │  状态空间网格化 → IBP 验证 {                                      │        │
│  │    1. SBC 条件验证: B(s) ≤ E[B(s')] + ε (鞅递减)               │        │
│  │    2. dCBF 约束验证: dCBF condition 在网格上成立                  │        │
│  │    3. 区域约束: B(s_init) ≤ 1, B(s_unsafe) ≥ 1/(1-p)           │        │
│  │  }                                                              │        │
│  │  → 如果违反 → 提取反例 → 交替更新 Policy/Penalty + SBC           │        │
│  │  → 如果满足 → 计算 P(safe) ≥ 1 - B(s_init)/B(s_unsafe)         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2.2 关键设计理念

**1. BarrierNet 作为安全投影层**

在原始 SafePVC 中，策略网络直接输出加速度。在新系统中，策略网络输出**参考加速度** $\hat{u}$，BarrierNet 将其投影到安全集内：

$$u^* = \text{BarrierNet}(\hat{u}, s, p_i(z))$$

这确保了即使策略网络输出不安全控制，最终控制 $u^*$ 仍然满足 dCBF 约束。

**2. dCBF 约束增强 SBC 验证**

在 SBC 验证中，原本需要验证整个状态空间上的鞅递减条件。有了 BarrierNet 的 dCBF 约束，我们知道：
- 在安全集内部（远离边界），dCBF 约束不活跃，SBC 正常验证
- 在安全集边界附近，dCBF 约束主动修正控制，**缩小了需要验证的区域**

**3. CEGIS 反馈优化 BarrierNet 参数**

当 SBC 验证发现反例时，不仅更新策略网络和 SBC 网络，还更新 BarrierNet 的 penalty 函数 $p_i(z)$，使其更适应困难状态。

### 5.3 数学原理

#### 5.3.1 AEBS 系统的 BarrierNet 设计

**安全约束**：车辆必须在到达 $d = 6$ m 前减速到 $v < 0.5$ m/s。

定义安全约束函数：

$$b(d, v) = d - d_{\text{safe}} - \phi \cdot v$$

其中 $d_{\text{safe}} = 5.0$ m 是最小安全距离，$\phi$ 是反应时间参数。

**相对度分析**：

$$\dot{b} = \dot{d} - \phi \dot{v} = -v - \phi(-a) = -v + \phi a$$

控制 $u = a$ 在一阶导数中出现，所以相对度 $m = 1$。

**注意**：对于 AEBS 这个 2D 系统，安全约束的相对度为 1，所以我们使用标准 CBF 而非 HOCBF。但如果要约束 jerk（加速度变化率），则相对度为 2，需要 HOCBF。

**方案选择**：我们提供两种方案：

**方案 A（相对度 1，简单）**：安全约束直接对加速度

$$b(s) = d - d_{\text{safe}} - \phi v \geq 0$$

$$\dot{b}(s) = -v + \phi a$$

CBF 约束：

$$-v + \phi a + p_1(z) \alpha_1(b(s)) \geq 0$$

即：

$$a \geq \frac{v - p_1(z) \alpha_1(b(s))}{\phi}$$

**方案 B（相对度 2，更平滑）**：安全约束对 jerk

如果控制输入改为 jerk $u_j$（加速度变化率），$\dot{a} = u_j$：

$$b(s) = d - d_{\text{safe}} - \phi v$$
$$\dot{b} = -v + \phi a \quad \text{(不含 } u_j \text{)}$$
$$\ddot{b} = -\dot{v} + \phi \dot{a} = a + \phi u_j \quad \text{(含 } u_j \text{)}$$

HOCBF 约束：

$$L_f^2 b + L_g L_f b \cdot u_j + (p_1 + p_2) L_f b + (\dot{p}_1 + p_1 p_2) b \geq 0$$

$$a + \phi u_j + (p_1 + p_2)(-v + \phi a) + (\dot{p}_1 + p_1 p_2)(d - d_{\text{safe}} - \phi v) \geq 0$$

**推荐使用方案 A**，因为：
1. AEBS 系统控制维度为 1，方案 A 的 QP 只有 1 个决策变量，求解极快
2. 当前 SafePVC 已经使用加速度作为控制，改动最小
3. 方案 B 可以留作未来工作

#### 5.3.2 BarrierNet QP 公式化 (方案 A)

对于 AEBS 系统：

**决策变量**：$u = a$ (加速度，标量)

**代价函数**：

$$\min_a \frac{1}{2} h (a - f)^2$$

其中 $h > 0$ 是正定标量（方案 A 中 $H$ 退化为标量），$f$ 是参考加速度（策略网络输出）。

展开：$\frac{1}{2} h a^2 - h f a + \frac{1}{2} h f^2$，所以 $H = h, F = -hf$。

**约束条件**：

1. **dCBF 安全约束**：

$$\phi a \geq v - p_1(z) \cdot \alpha_1(b(s))$$

即 $G \cdot a \leq h_{\text{cbf}}$，其中 $G = -\phi, h_{\text{cbf}} = -v + p_1(z) \alpha_1(b(s))$。

2. **控制约束**：

$$a_{\min} \leq a \leq a_{\max}$$

即 $-3 \leq a \leq 3$。

**解析解**（对于 1D QP）：

$$a^* = \text{clip}\left(\max\left(f, \frac{v - p_1(z) \alpha_1(b(s))}{\phi}\right), a_{\min}, a_{\max}\right)$$

这个解析解使得前向和反向传播都非常高效。

#### 5.3.3 概率安全与 dCBF 的结合

**定理（双层安全保证）**：

设 $b(s) \geq 0$ 是由 BarrierNet 的 dCBF 强制的安全约束，$B(s)$ 是 SBC。如果：

1. BarrierNet 保证 $b(s_t) \geq 0$ 对所有 $t$（确定性保证）
2. SBC 条件 $B(s) \geq \mathbb{E}[B(s')|s] + \epsilon$ 在 $\{s: b(s) \geq 0\}$ 上成立

则系统在概率意义上安全，且 $\mathbb{P}(\text{safe}) \geq 1 - 1/B_{\text{ratio}}$。

**证明思路**：BarrierNet 的 dCBF 约束保证了状态始终在安全集 $\{s: b(s) \geq 0\}$ 中。SBC 只需要在这个受限的状态空间上验证，这大幅降低了验证难度并提高了安全概率下界。

##### 5.3.3a 双层安全保证定理的完整证明

**定理（双层安全保证，完整陈述）**：

考虑闭环系统 $s_{t+1} = \tilde{F}(s_t, z_0, \Delta s_t)$，其中控制 $u_t$ 由 BarrierNet 生成。设：

1. **内层 (dCBF)**：存在 dCBF $b(s)$ 使得 BarrierNet 的 QP 保证 $b(s_t) \geq 0$ 对所有 $t \geq 0$ 成立
2. **外层 (SBC)**：存在连续函数 $B: S \to \mathbb{R}_{\geq 0}$ 满足：
   - (a) $B(s) \leq 1, \forall s \in S_0 \cap \{s: b(s) \geq 0\}$
   - (b) $B(s) \geq \frac{1}{1-p}, \forall s \in X_u$
   - (c) $B(s) \geq \mathbb{E}[B(\tilde{F}(s, z_0, \Delta s))] + \epsilon, \forall s \in \{s: b(s) \geq 0\}$

则 $\mathbb{P}_{s_0}(\text{Safe}(X_u)) \geq p$ 对所有 $s_0 \in S_0 \cap \{s: b(s) \geq 0\}$ 成立。

**证明**：

**Step 1 — dCBF 约束下的状态空间限制**：

由 dCBF 的确定性保证，从任何满足 $b(s_0) \geq 0$ 的初始状态出发，闭环系统轨迹始终在 $S_{\text{safe}} = \{s: b(s) \geq 0\}$ 中。

这意味着 $S_{\text{safe}}$ 是闭环系统的**不变集**。

**Step 2 — 限制域上的 SBC 验证**：

由于系统永远在 $S_{\text{safe}}$ 中，SBC 条件只需要在 $S_{\text{safe}}$ 上验证。条件 (c) 保证了这一点。

**Step 3 — 受限鞅论证**：

定义 $B_t = B(s_t)$。由条件 (c) 和 Step 1：

$$\mathbb{E}[B_{t+1} | \mathcal{F}_t] = \mathbb{E}[B(s_{t+1}) | s_t] \leq B(s_t) - \epsilon = B_t - \epsilon$$

这证明 $B_t$ 是超鞅（在 $S_{\text{safe}}$ 上），与定理 2.2 的证明相同。

**Step 4 — 概率界**：

由 Doob 停时不等式（与 2.3.3c 完全相同）：

$$\mathbb{P}(\tau_u < \infty) \leq \frac{\max_{s \in S_0 \cap S_{\text{safe}}} B(s)}{\min_{s \in X_u} B(s)} \leq \frac{1}{1/(1-p)} = 1-p$$

因此 $\mathbb{P}(\text{Safe}) \geq p$。$\blacksquare$

**关键优势 — 为什么双层比单层更好**：

| 方面 | 仅 SBC (SafePVC) | 双层 (BarrierNet+SBC) |
|------|-----------------|----------------------|
| **SBC 验证域** | 整个状态空间 $S$ | 受限域 $S \cap \{b(s) \geq 0\}$ |
| **$B(s)$ 需要覆盖的范围** | 从 $S_0$ 到 $X_u$ 的所有路径 | 仅 dCBF 安全集内的路径 |
| **$B(s)$ 的 Lipschitz 要求** | 需要在整个 $S$ 上满足条件 | 仅在 $S_{\text{safe}}$ 上满足 |
| **概率下界** | $1 - 1/B_{\text{ratio}}$ | 更高（因为验证域更小，$B_{\text{ratio}}$ 更容易做大） |
| **反例数量** | 更多（全空间搜索） | 更少（受限空间搜索） |

**直觉**：dCBF 就像一个"围栏"，把系统限制在安全区域内。SBC 只需要在围栏内验证，不需要担心围栏外的情况。这使得 SBC 的学习和验证都更容易。

##### 5.3.3b 双层安全的数值例子

**场景**：AEBS，$d \in [5, 16], v \in [0, 3]$

**仅 SBC**：$B(s)$ 需要在整个 $[5,16] \times [0,3]$ 上满足鞅递减条件。

设 $S_0 = \{d \in [15,16], v \in [2.5,3.0]\}$，$X_u = \{d \in [5,6], v \in [0.5,3.0]\}$

$B_{\text{ratio}} = \frac{\min_{X_u} B}{\max_{S_0} B}$。如果 $\max_{S_0} B = 1$，需要 $\min_{X_u} B \geq 10$。

在 $S$ 的某些区域（如 $d = 8, v = 1.5$），动力学可能导致 $B$ 的非单调行为，增加验证难度。

**双层方案**：dCBF 保证 $d \geq d_{\text{safe}} = 6$（即状态永远在 $d \geq 6$ 的区域）。

SBC 只需在 $d \in [6, 16], v \in [0, 3]$ 上验证，排除了 $d \in [5, 6)$ 的危险区域。

这使得：
1. $B(s)$ 不需要在 $d < 6$ 区域有意义
2. 验证网格更小 → 反例更少 → 收敛更快
3. 安全概率下界更高

#### 5.3.4 Penalty 函数设计

对于 AEBS 系统，penalty 函数 $p_1(z)$ 的设计：

**输入 $z$ 的构成**：
- $z$ 来自上游网络的输出（策略网络的中间层特征）
- 或者直接使用状态 $s = (d, v)$ 作为 $z$（简化方案）

**网络结构**：

$$p_1(z) = \text{Softplus}(\text{MLP}_p(z)) + p_{\min}$$

其中 $p_{\min} > 0$ 确保 penalty 始终为正，Softplus 保证连续可微。

**学习目标**：
- 远离不安全集时，$p_1$ 较大 → CBF 约束更松 → 性能更好
- 接近不安全集时，$p_1$ 较小 → CBF 约束更紧 → 更安全

#### 5.3.5 SBC 损失函数修改

在原始 SafePVC 的 SBC 损失中，加入 BarrierNet 约束的信息：

$$\mathcal{L}_L = \underbrace{\mathcal{L}_{\text{dec\_L}}}_{\text{鞅递减}} + \underbrace{\lambda_R \mathcal{L}_{\text{region}}}_{\text{区域约束}} + \underbrace{\lambda_L \mathcal{L}_{\text{lip\_L}}}_{\text{Lipschitz}} + \underbrace{\lambda_{\text{cbf}} \mathcal{L}_{\text{cbf\_guide}}}_{\text{新增: CBF 引导}}$$

**CBF 引导损失**：

$$\mathcal{L}_{\text{cbf\_guide}} = \mathbb{E}_s \left[ \max(0, B(s) - B_{\text{target}}(s)) \right]$$

其中 $B_{\text{target}}(s)$ 是基于 dCBF 值构造的目标函数：

$$B_{\text{target}}(s) = \frac{1}{1 + \exp(\kappa \cdot b(s))}$$

当 $b(s)$ 大（远离不安全集）时 $B_{\text{target}}$ 小，当 $b(s)$ 接近 0 时 $B_{\text{target}}$ 接近 1。这引导 SBC 网络学习一个与 dCBF 一致的值函数形状。

---

## 六、完整项目框架结构

```
artical-F122/                                    # 项目根目录
├── environment.yml                              # Conda 环境
├── environment_fixed.yml                        # 环境备份
├── README.md                                    # 项目说明
├── .gitignore                                   # Git 忽略规则
│
├── Combined_network/
│   ├── model.py                                 # [修改] 端到端网络 + BarrierNet 层
│   └── barrier_qp.py                            # [新增] 可微 QP 层实现
│
├── cGAN/
│   ├── cGAN_common.py                           # cGAN 通用训练
│   ├── spectral_norm.py                         # 谱归一化
│   └── taxi_models_and_data.py                  # 生成器/判别器 + 数据加载
│
├── Aebs/
│   ├── cGAN/
│   │   ├── train_gans.py                        # cGAN 训练
│   │   ├── train_mlp.py                         # MLP 观测模型训练
│   │   ├── view.py                              # 可视化
│   │   └── mlp_supervised_ld4/
│   │       └── mlp_supervised.pth               # 预训练观测模型权重
│   │
│   ├── connect/
│   │   ├── genGANData.py                        # CARLA 数据收集
│   │   ├── downSample.py                        # 图像下采样
│   │   └── view.py                              # 数据集查看
│   │
│   ├── controller/
│   │   ├── StateEstimate_train.py               # 状态估计网络训练
│   │   ├── Controller_train.py                  # PPO 控制器训练
│   │   ├── state_net_trained.pth                # 预训练状态估计权重
│   │   └── best_model/
│   │       └── best_model.zip                   # 预训练 PPO 策略
│   │
│   ├── data/
│   │   └── Downsampled.h5                       # HDF5 数据集
│   │
│   ├── system/
│   │   ├── env.py                               # [修改] 环境定义 + dCBF 参数
│   │   ├── estimate.py                          # 扰动估计
│   │   └── combined.py                          # [修改] 含 BarrierNet 的闭环模拟
│   │
│   ├── barrier/                                 # [新增] BarrierNet 模块目录
│   │   ├── __init__.py                          # 模块导出
│   │   ├── dcbf.py                              # dCBF 定义与计算
│   │   ├── penalty_net.py                       # Penalty 函数网络
│   │   ├── barrier_net_layer.py                 # BarrierNet 层 (dQP 前向/反向)
│   │   └── safety_filter.py                     # 安全过滤器封装
│   │
│   └── VT/
│       ├── loop.py                              # [修改] CEGIS 主循环 + BarrierNet 集成
│       ├── train.py                             # [修改] VTLearner + BarrierNet 参数训练
│       ├── verify.py                            # [修改] VTVerifier + dCBF 验证
│       └── utils.py                             # [修改] MLP + 新增工具函数
│
└── auto_LiRPA/                                  # 神经网络验证库
    ├── auto_LiRPA/
    │   ├── __init__.py
    │   ├── bound_general.py
    │   ├── bound_ops.py
    │   ├── forward_bound.py
    │   ├── backward_bound.py
    │   ├── optimized_bounds.py
    │   ├── interval_bound.py
    │   └── operators/
    ├── examples/
    ├── tests/
    └── doc/
```

---

## 七、数据流详解

### 7.1 前向数据流 (训练/推理)

```
阶段 1: 观测生成 (冻结)
  输入: z ∈ [-1,1]^4 (潜在扰动), d_norm (归一化距离)
  → AebsMLPGenerator(z, d_norm) → img ∈ [-1,1]^(1×32×32)
  → img_flat = img.view(-1, 1024)

阶段 2: 状态估计 (冻结)
  输入: img_flat (1024维)
  → SubNet(img_flat) → state_hat ∈ R (估计距离)
  → controller_input = cat([state_hat, v], dim=1) ∈ R^2

阶段 3: 参考控制生成 (可训练)
  输入: controller_input (2维)
  → CombinedPolicyNetwork(controller_input) → u_ref ∈ R (参考加速度)

阶段 4: Penalty 网络 (可训练) [新增]
  输入: s = [d_norm, v] (2维) 或 controller_input (2维)
  → PenaltyNet(s) → p_1 ∈ R_{>0} (penalty 值)

阶段 5: BarrierNet dQP 层 (可训练) [新增]
  输入: u_ref, s=[d,v], p_1, H (或 h), 安全约束参数
  → dCBF 约束: G·u ≤ h_cbf
     G = -φ
     h_cbf = -v + p_1 · α_1(b(s))
     b(s) = d·std1 - d_safe - φ·v
  → 求解 1D QP:
     u* = clip(max(u_ref, (v - p_1·α_1(b(s)))/φ), u_min, u_max)
  → 输出: u* ∈ R (安全加速度)

阶段 6: 动力学 (确定性 + 随机扰动)
  → d_next = d - v·Δt
  → v_next = clip(v - u*·Δt, 0, 3)
  → s_next_det = [d_next_norm, v_next]
  → s_next = s_next_det + Δs  (Δs ~ 均匀分布扰动)
```

### 7.2 验证数据流 (IBP)

```
阶段 A: 网格生成
  状态空间 [d_min, d_max] × [v_min, v_max]
  → 100×100 网格 → 10000 个中心点

阶段 B: BarrierNet 绑定传播
  对每个网格 cell [s_lb, s_ub]:
    1. 计算 b(s) 的区间 [b_lb, b_ub]
       b(s) = d·std1 - d_safe - φ·v
       b_lb = d_lb·std1 - d_safe - φ·v_ub
       b_ub = d_ub·std1 - d_safe - φ·v_lb
    
    2. 计算 p_1(s) 的区间 [p1_lb, p1_ub] (通过 IBP)
       PenaltyNet 使用 IBP 传播
    
    3. 计算 dCBF 约束的区间
       h_cbf = -v + p_1·α_1(b(s))
       (区间运算)
    
    4. 计算 u* 的区间 [u*_lb, u*_ub]
       基于解析解的区间传播
    
    5. 计算 s_next 的区间
       s_next_lb, s_next_ub = dynamics(s_lb, s_ub, u*_lb, u*_ub)

阶段 C: SBC IBP 验证
  对扰动的 s_next 区间:
    B_ub = IBP(B, s_next_lb + noise_lb, s_next_ub + noise_ub)
    E[B(s')] ≈ Σ pmass_i · B_ub_i
    
    检查: E[B(s')] + K + ε ≤ B(s)?
    如果违反 → 记录反例
```

### 7.2a IBP (Interval Bound Propagation) 原理详解

**IBP** 是神经网络形式化验证的核心技术，用于计算：**给定输入区间 $[x^L, x^U]$，神经网络输出的区间 $[y^L, y^U]$ 是什么？**

#### 逐层传播规则

**1. 线性层 $y = Wx + b$**：

$$y_i^L = \sum_j \min(W_{ij} x_j^L, W_{ij} x_j^U) + b_i$$

$$y_i^U = \sum_j \max(W_{ij} x_j^L, W_{ij} x_j^U) + b_i$$

直觉：对于每个权重 $W_{ij}$，如果 $W_{ij} > 0$，则 $x_j$ 越大输出越大；如果 $W_{ij} < 0$，则 $x_j$ 越小输出越大。

**2. ReLU 激活 $y = \max(0, x)$**：

$$y^L = \max(0, x^L), \quad y^U = \max(0, x^U)$$

**3. Tanh 激活 $y = \tanh(x)$**：

$$y^L = \tanh(x^L), \quad y^U = \tanh(x^U)$$

（因为 $\tanh$ 是单调递增函数）

**4. Softplus 激活 $y = \ln(1 + e^x)$**：

$$y^L = \ln(1 + e^{x^L}), \quad y^U = \ln(1 + e^{x^U})$$

（Softplus 也是单调递增函数）

#### IBP 数值例子

**网络**：$y = \text{ReLU}(Wx + b)$，其中 $W = \begin{bmatrix} 1 & -2 \\ 3 & 1 \end{bmatrix}$，$b = \begin{bmatrix} 0.5 \\ -1 \end{bmatrix}$

**输入区间**：$x \in [x^L, x^U] = \begin{bmatrix} [0, 1] \\ [0.5, 1.5] \end{bmatrix}$

**线性层传播**：

$y_1^L = \min(1 \times 0, 1 \times 1) + \min(-2 \times 0.5, -2 \times 1.5) + 0.5 = 0 + (-3) + 0.5 = -2.5$

$y_1^U = \max(1 \times 0, 1 \times 1) + \max(-2 \times 0.5, -2 \times 1.5) + 0.5 = 1 + (-1) + 0.5 = 0.5$

$y_2^L = \min(3 \times 0, 3 \times 1) + \min(1 \times 0.5, 1 \times 1.5) - 1 = 0 + 0.5 - 1 = -0.5$

$y_2^U = \max(3 \times 0, 3 \times 1) + \max(1 \times 0.5, 1 \times 1.5) - 1 = 3 + 1.5 - 1 = 3.5$

线性层输出区间：$y \in \begin{bmatrix} [-2.5, 0.5] \\ [-0.5, 3.5] \end{bmatrix}$

**ReLU 传播**：

$\text{ReLU}(y) \in \begin{bmatrix} [\max(0, -2.5), \max(0, 0.5)] \\ [\max(0, -0.5), \max(0, 3.5)] \end{bmatrix} = \begin{bmatrix} [0, 0.5] \\ [0, 3.5] \end{bmatrix}$

#### IBP 的保守性与 CROWN 的改进

**IBP 的问题**：IBP 是**逐层独立**的区间估计，每层都假设输入区间内的值可以任意组合，导致输出区间过宽（保守）。

**例子**：$y = x_1 - x_2$，$x_1, x_2 \in [0, 1]$。真实范围 $y \in [-1, 1]$。但如果 $x_1 = x_2$（同一个变量），真实范围是 $y = 0$。IBP 无法捕捉这种相关性。

**CROWN (Complete Robustness Verification)**：使用**线性松弛**替代 IBP 的非线性激活近似。对于 ReLU，CROWN 用上下界的线性函数近似：

$$\alpha^L x + \beta^L \leq \text{ReLU}(x) \leq \alpha^U x + \beta^U$$

其中 $\alpha, \beta$ 根据 $x$ 的区间 $[x^L, x^U]$ 优化选择。这比 IBP 的区间传播更紧。

**在本项目中的选择**：使用 IBP（而非 CROWN），因为：
1. IBP 计算更快（$O(n)$ vs CROWN 的 $O(n^2)$）
2. AEBS 状态空间维度低（2D），IBP 的保守性在低维场景中可接受
3. 代码库 `auto_LiRPA` 同时支持 IBP 和 CROWN，可以切换

#### 在本项目中的 IBP 应用

IBP 在三个关键环节被使用：

**1. PenaltyNet 的区间传播**：

$p_1(s)$ 是 MLP 网络，输入区间 $[s^L, s^U]$ → IBP → 输出区间 $[p_1^L, p_1^U]$

**2. SBC 网络 $B(s)$ 的区间传播**：

$B(s)$ 是 MLP 网络，输入区间 $[s^L, s^U]$ → IBP → 输出区间 $[B^L, B^U]$

**3. 期望上界的计算**：

$$\mathbb{E}[B(s')] \leq \sum_i \text{pmass}_i \cdot B^U_i$$

其中 $B^U_i$ 是对第 $i$ 个噪声网格单元的 IBP 上界，$\text{pmass}_i$ 是该单元的概率质量。

### 7.3 训练数据流 (CEGIS)

```
迭代 k:
  1. VTLearner.train_step_l():
     → 冻结 P-Net, 训练 L-Net (SBC 网络)
     → 损失: martingale_loss + lip_loss + region_loss + cbf_guide_loss [新增]
  
  2. VTVerifier.check_dec_cond():
     → IBP 验证鞅递减条件 + dCBF 约束条件 [新增]
     → 提取反例 violation_buffer
  
  3. VTLearner.train_step_p():
     → 冻结 L-Net, 训练 P-Net (策略网络) + PenaltyNet [新增]
     → 损失: martingale_loss + lip_loss + mse_loss + penalty_reg_loss [新增]
  
  4. VTLearner.train_step_barrier(): [新增]
     → 训练 PenaltyNet 和 H-Net
     → 损失: dCBF 满足度 + 控制性能
```

---

## 八、网络结构详解

### 8.1 完整网络层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│  AebsEnd2EndNet (组合端到端网络)                                 │
│  总参数: ~175K (大部分冻结)                                     │
│                                                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │ 1. gen_net: AebsMLPGenerator [冻结]          │               │
│  │    输入: z(4) + d(1) → 5 维                  │               │
│  │    Linear(5, 256) → ReLU                     │               │
│  │    Linear(256, 256) → ReLU                   │               │
│  │    Linear(256, 256) → ReLU                   │               │
│  │    Linear(256, 256) → ReLU                   │               │
│  │    Linear(256, 1024) → Tanh                  │               │
│  │    Reshape → (1, 32, 32)                     │               │
│  │    输出: img ∈ [-1,1]^(1×32×32)              │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓ flatten                               │
│  ┌──────────────────────────────────────────────┐               │
│  │ 2. state_net: SubNet [冻结]                  │               │
│  │    输入: img_flat (1024 维)                   │               │
│  │    Linear(1024, 256) → LayerNorm → ReLU      │               │
│  │    Linear(256, 64) → LayerNorm → ReLU        │               │
│  │    Linear(64, 1)                             │               │
│  │    输出: state_hat ∈ R (估计距离)             │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓ cat with v                           │
│  ┌──────────────────────────────────────────────┐               │
│  │ 3. controller_net: CombinedPolicyNet [可训练] │               │
│  │    输入: [state_hat, v] (2 维)                │               │
│  │    mlp_extractor (from PPO):                  │               │
│  │      Linear(2, 64) → Tanh                    │               │
│  │      Linear(64, 64) → Tanh                   │               │
│  │    action_net:                                │               │
│  │      Linear(64, 1)                           │               │
│  │    输出: u_ref ∈ R (参考加速度)               │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │ 4. penalty_net: PenaltyNet [可训练] [新增]    │               │
│  │    输入: s = [d_norm, v] (2 维)               │               │
│  │    Linear(2, 32) → Softplus                  │               │
│  │    Linear(32, 16) → Softplus                 │               │
│  │    Linear(16, 1) → Softplus + p_min          │               │
│  │    输出: p_1 ∈ R_{>0} (penalty 值)           │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │ 5. h_net: HNet [可训练] [新增/可选]           │               │
│  │    输入: s = [d_norm, v] (2 维)               │               │
│  │    Linear(2, 16) → Softplus                  │               │
│  │    Linear(16, 1) → Softplus + h_min          │               │
│  │    输出: h ∈ R_{>0} (QP 代价系数)            │               │
│  │    (简化方案: h 为固定常数，省略此网络)        │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │ 6. barrier_net: BarrierNetLayer [新增]        │               │
│  │    输入: u_ref, s, p_1, h                     │               │
│  │    计算: dCBF 约束 → 解析 1D QP 求解          │               │
│  │    输出: u* ∈ R (安全加速度)                   │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │ 7. l_model: SBC MLP [可训练]                  │               │
│  │    输入: s = [d_norm, v] (2 维)               │               │
│  │    Linear(2, 16) → Tanh                      │               │
│  │    Linear(16, 8) → Tanh                      │               │
│  │    Linear(8, 1) → Softplus                   │               │
│  │    输出: B(s) ∈ R_{≥0} (障碍证书值)           │               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 各网络参数统计

| 网络 | 参数数量 | 是否可训练 | 激活函数 | 输出约束 |
|------|---------|-----------|---------|---------|
| gen_net | ~540K | ❌ 冻结 | ReLU + Tanh | [-1, 1] |
| state_net | ~285K | ❌ 冻结 | LayerNorm + ReLU | R |
| controller_net | ~4.5K | ✅ 训练 | Tanh | R (参考加速度) |
| penalty_net | ~1.1K | ✅ 训练 | Softplus | R_{>0} |
| h_net | ~350 | ✅ 训练(可选) | Softplus | R_{>0} |
| barrier_net | 0 (无参数) | N/A | N/A | [u_min, u_max] |
| l_model (SBC) | ~450 | ✅ 训练 | Tanh + Softplus | R_{≥0} |

---

## 九、变量定义与符号表

### 9.1 系统变量

| 符号 | 类型 | 维度 | 含义 | 代码变量 | 取值范围 |
|------|------|------|------|---------|---------|
| $s$ | 状态 | $\mathbb{R}^2$ | 系统状态 $(d_{norm}, v)$ | `y`, `s_batch` | $[d_{min}/\sigma, d_{max}/\sigma] \times [0, 3]$ |
| $d$ | 距离 | $\mathbb{R}$ | 到目标的实际距离 | `d = s[:,0] * std1` | $[5, 16]$ m |
| $d_{norm}$ | 距离 | $\mathbb{R}$ | 归一化距离 | `s[:,0]` | $[d_{min}/\sigma, d_{max}/\sigma]$ |
| $v$ | 速度 | $\mathbb{R}$ | 车辆速度 | `s[:,1]` | $[0, 3]$ m/s |
| $a$ | 加速度 | $\mathbb{R}$ | 控制输出 | `acc`, `u*` | $[-3, 3]$ m/s² |
| $u_{ref}$ | 参考控制 | $\mathbb{R}$ | 策略网络输出 | `u_ref`, `f` | R |
| $z$ | 潜在变量 | $\mathbb{R}^4$ | 环境扰动 | `z_batch` | $[-1, 1]^4$ |
| $\Delta s$ | 扰动 | $\mathbb{R}^2$ | 状态扰动 | `noise` | $\pm 5\%$ 状态范围 |
| $\Delta t$ | 时间步 | $\mathbb{R}$ | 离散时间步长 | `dt = 0.05` | 0.05 s |
| $\sigma$ | 标准差 | $\mathbb{R}$ | 距离数据标准化系数 | `std1` | 数据集计算 |

### 9.2 BarrierNet 变量

| 符号 | 类型 | 维度 | 含义 | 代码变量 | 取值范围 |
|------|------|------|------|---------|---------|
| $b(s)$ | CBF 值 | $\mathbb{R}$ | 安全约束函数值 | `b_val` | R |
| $d_{safe}$ | 安全距离 | $\mathbb{R}$ | 最小安全距离 | `d_safe = 5.0` | 5.0 m |
| $\phi$ | 反应时间 | $\mathbb{R}$ | CBF 设计参数 | `phi` | 1.0~2.0 s |
| $p_1(z)$ | penalty | $\mathbb{R}$ | 环境自适应惩罚 | `p1` | $\mathbb{R}_{>0}$ |
| $\alpha_1(\cdot)$ | Class K | $\mathbb{R} \to \mathbb{R}$ | 线性函数 $\alpha_1(x) = k_1 x$ | 内联计算 | $\mathbb{R}_{\geq 0}$ |
| $h$ | QP 代价 | $\mathbb{R}$ | 正定代价系数 | `h` | $\mathbb{R}_{>0}$ |
| $G$ | 约束矩阵 | $\mathbb{R}$ | dCBF 约束中 $u$ 的系数 | `G = -phi` | R |
| $h_{cbf}$ | 约束偏置 | $\mathbb{R}$ | dCBF 约束右端 | `h_cbf` | R |
| $u^*$ | 安全控制 | $\mathbb{R}$ | BarrierNet 输出 | `u_star` | $[-3, 3]$ |
| $\lambda^*$ | 对偶变量 | $\mathbb{R}$ | KKT 乘子 | `lam_star` | $\mathbb{R}_{\geq 0}$ |

### 9.3 SBC 变量

| 符号 | 类型 | 维度 | 含义 | 代码变量 | 取值范围 |
|------|------|------|------|---------|---------|
| $B(s)$ | SBC 值 | $\mathbb{R}$ | 随机障碍证书值 | `l`, `B_val` | $\mathbb{R}_{\geq 0}$ |
| $\epsilon$ | 递减余量 | $\mathbb{R}$ | 鞅严格递减参数 | `eps = 0.1` | > 0 |
| $\gamma$ | 递减率 | $\mathbb{R}$ | 超鞅参数 | `gamma_decrease = 1.0` | $(0, 1]$ |
| $L_B$ | Lipschitz | $\mathbb{R}$ | $B$ 的 Lipschitz 常数 | `lips_l_batch` | $\mathbb{R}_{\geq 0}$ |
| $K$ | 离散化误差 | $\mathbb{R}$ | Lipschitz 校正项 | `K = k * delta` | $\mathbb{R}_{\geq 0}$ |
| $\delta$ | 网格步长 | $\mathbb{R}$ | 离散化步长 | `delta` | R² |
| $p$ | 安全概率 | $\mathbb{R}$ | 目标安全概率 | `reach_prob = 0.95` | $[0, 1)$ |

### 9.4 训练超参数

| 参数 | 值 | 含义 |
|------|---|------|
| `l_lr` | 3e-3 | SBC 网络学习率 |
| `p_lr` | 5e-2 | 策略网络学习率 |
| `barrier_lr` | 1e-3 | PenaltyNet 学习率 |
| `l_lip` | 4.0 | SBC Lipschitz 目标 |
| `p_lip` | 2.0 | 策略 Lipschitz 目标 |
| `eps` | 0.1 | 鞅递减余量 |
| `gamma_decrease` | 1.0 | 超鞅递减率 |
| `reach_prob` | 0.95 | 目标安全概率 |
| `grid_size` | [100, 100] | 验证网格 |
| `pmass_n` | [10, 10] | 噪声离散化 |
| `batch_size` | 2048 | 训练/验证批大小 |
| `noise_factor` | 0.05 | 扰动幅度 (5%状态范围) |
| `phi` (CBF) | 1.5 | 反应时间参数 |
| `d_safe` | 5.0 | 安全距离 |
| `p_min` | 0.1 | penalty 最小值 |
| `alpha_k1` | 1.0 | Class K 线性系数 |
| `lambda_cbf_guide` | 0.1 | CBF 引导损失权重 |
| `max_iterations` | 100 | CEGIS 最大迭代数 |
| `timeout` | 3600 s | CEGIS 超时 |

---

## 十、代码修改方案

### 10.1 新增文件清单

| 文件路径 | 用途 | 优先级 |
|---------|------|--------|
| `Aebs/barrier/__init__.py` | 模块导出 | P0 |
| `Aebs/barrier/dcbf.py` | dCBF 定义 | P0 |
| `Aebs/barrier/penalty_net.py` | Penalty 网络 | P0 |
| `Aebs/barrier/barrier_net_layer.py` | BarrierNet 层 | P0 |
| `Aebs/barrier/safety_filter.py` | 安全过滤器 | P1 |
| `Combined_network/barrier_qp.py` | 可微 QP 实现 | P0 |

### 10.2 修改现有文件

| 文件路径 | 修改内容 | 影响范围 |
|---------|---------|---------|
| `Combined_network/model.py` | AebsEnd2EndNet 增加 BarrierNet 层 | 前向传播 |
| `Aebs/system/env.py` | Aebs 类增加 dCBF 参数定义 | 环境定义 |
| `Aebs/system/combined.py` | 模拟轨迹加入 BarrierNet 过滤 | 轨迹模拟 |
| `Aebs/VT/train.py` | VTLearner 增加 BarrierNet 参数训练 | 训练逻辑 |
| `Aebs/VT/verify.py` | VTVerifier 增加 dCBF 条件验证 | 验证逻辑 |
| `Aebs/VT/loop.py` | CEGIS 循环集成 BarrierNet | 主循环 |
| `Aebs/VT/utils.py` | 新增工具函数 | 辅助函数 |

### 10.3 新增模块实现

#### 10.3.1 `Aebs/barrier/__init__.py`

```python
from .dcbf import DCBF, compute_cbf_value, compute_cbf_constraint
from .penalty_net import PenaltyNet
from .barrier_net_layer import BarrierNetLayer
from .safety_filter import SafetyFilter
```

#### 10.3.2 `Aebs/barrier/dcbf.py` — dCBF 核心计算

```python
import torch
import torch.nn as nn

class DCBF:
    """
    Differentiable Control Barrier Function for AEBS system.
    
    Safety constraint: b(s) = d - d_safe - phi * v >= 0
    CBF constraint: Lf_b + Lg_b * u + p1 * alpha1(b) >= 0
    
    For AEBS:
        d_next = d - v * dt
        v_next = v - a * dt
        So: Lf_b = -v (drift), Lg_b = phi (control effect)
    """
    
    def __init__(self, d_safe=5.0, phi=1.5, alpha_k1=1.0, std1=1.0):
        """
        Args:
            d_safe: 最小安全距离 (m)
            phi: CBF 反应时间参数 (s)
            alpha_k1: Class K 线性函数系数
            std1: 距离标准化系数
        """
        self.d_safe = d_safe
        self.phi = phi
        self.alpha_k1 = alpha_k1
        self.std1 = std1
    
    def compute_b(self, s):
        """
        计算安全约束函数值 b(s).
        
        Args:
            s: 状态张量 [B, 2], s[:,0]=d_norm, s[:,1]=v
        
        Returns:
            b_val: CBF 值 [B, 1]
        """
        d = s[:, 0] * self.std1  # 转换为实际距离
        v = s[:, 1]
        b_val = d - self.d_safe - self.phi * v
        return b_val.unsqueeze(1)
    
    def compute_lie_derivatives(self, s):
        """
        计算 b(s) 的 Lie 导数.
        
        对于 AEBS 系统:
            f(s) = [d - v*dt, v]  (漂移项)
            g(s) = [0, -dt]       (控制项)
        
        b(s) = d - d_safe - phi*v
        
        Lf_b = ∂b/∂s · f(s)
             = [1, -phi] · [-v*dt, 0] (注意这里是连续时间 Lie 导数)
        
        在离散时间下，我们用差分近似:
            b(s_next) ≈ b(s) + ∂b/∂s · (s_next - s)
        
        连续时间 Lie 导数:
            Lf_b = ∂b/∂d · f_d + ∂b/∂v · f_v
                 = 1 · 0 + (-phi) · 0  (漂移 f 中不含控制)
        
        但在 AEBS 中，f 和 g 的定义是:
            d_dot = -v     → f_d = -v, g_d = 0
            v_dot = -a     → f_v = 0,  g_v = -1 (因为 a 是控制)
        
        所以:
            Lf_b = ∂b/∂d · (-v) + ∂b/∂v · 0 = -v
            Lg_b = ∂b/∂d · 0 + ∂b/∂v · (-1) = phi
        
        Args:
            s: 状态 [B, 2]
        
        Returns:
            Lf_b: 漂移 Lie 导数 [B, 1]
            Lg_b: 控制 Lie 导数 [B, 1]
        """
        v = s[:, 1]
        Lf_b = (-v).unsqueeze(1)      # ∂b/∂d · f_d = 1·(-v) = -v
        Lg_b = torch.full_like(Lf_b, self.phi)  # ∂b/∂v · g_v = (-phi)·(-1) = phi
        return Lf_b, Lg_b
    
    def compute_cbf_constraint(self, s, u, p1):
        """
        计算 dCBF 约束的残差 (应 >= 0).
        
        dCBF constraint: Lf_b + Lg_b * u + p1 * alpha1(b) >= 0
        
        Args:
            s: 状态 [B, 2]
            u: 控制 [B, 1]
            p1: penalty 值 [B, 1]
        
        Returns:
            constraint_val: 约束残差 [B, 1] (>=0 表示安全)
        """
        b_val = self.compute_b(s)
        Lf_b, Lg_b = self.compute_lie_derivatives(s)
        
        # Class K 线性函数: alpha1(x) = k1 * x
        alpha1_b = self.alpha_k1 * b_val
        
        constraint_val = Lf_b + Lg_b * u + p1 * alpha1_b
        return constraint_val
    
    def compute_min_safe_control(self, s, u_ref, p1, u_min=-3.0, u_max=3.0):
        """
        计算满足 dCBF 约束的最小修正控制 (解析解).
        
        dCBF 约束: Lf_b + Lg_b * u + p1 * alpha1(b) >= 0
        → -v + phi * u + p1 * k1 * (d - d_safe - phi*v) >= 0
        → phi * u >= v - p1 * k1 * (d - d_safe - phi*v)
        → u >= (v - p1 * k1 * b(s)) / phi
        
        最小修正: u_safe = max(u_ref, u_min_safe)
        再 clip 到控制约束: u* = clip(u_safe, u_min, u_max)
        
        Args:
            s: 状态 [B, 2]
            u_ref: 参考控制 [B, 1]
            p1: penalty [B, 1]
            u_min, u_max: 控制约束
        
        Returns:
            u_star: 安全控制 [B, 1]
            u_min_safe: CBF 要求的最小控制 [B, 1]
        """
        b_val = self.compute_b(s)
        
        # CBF 要求的最小加速度
        u_min_safe = (s[:, 1].unsqueeze(1) - p1 * self.alpha_k1 * b_val) / self.phi
        
        # 最小修正: 取参考控制和最小安全控制的较大值
        u_safe = torch.max(u_ref, u_min_safe)
        
        # 截断到控制约束
        u_star = torch.clamp(u_safe, u_min, u_max)
        
        return u_star, u_min_safe


def compute_cbf_value(dcbf, s):
    """便捷函数"""
    return dcbf.compute_b(s)

def compute_cbf_constraint(dcbf, s, u, p1):
    """便捷函数"""
    return dcbf.compute_cbf_constraint(s, u, p1)
```

#### 10.3.3 `Aebs/barrier/penalty_net.py` — Penalty 网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PenaltyNet(nn.Module):
    """
    Penalty 函数网络: 输出环境自适应的 dCBF 惩罚值.
    
    输出必须为正值且连续可微 (dCBF 的理论要求).
    使用 Softplus 激活函数保证平滑和正值.
    """
    
    def __init__(self, input_dim=2, hidden_dims=[32, 16], 
                 output_dim=1, p_min=0.1):
        """
        Args:
            input_dim: 输入维度 (状态维度, AEBS 中为 2)
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度 (penalty 个数, 通常等于 CBF 约束数)
            p_min: penalty 最小值, 保证 p(z) > 0
        """
        super().__init__()
        self.p_min = p_min
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Softplus(beta=2.0))  # 平滑正值激活
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus(beta=2.0))
        
        self.network = nn.Sequential(*layers)
        
        # 正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, s):
        """
        Args:
            s: 状态张量 [B, input_dim]
        
        Returns:
            p: penalty 值 [B, output_dim], p > p_min > 0
        """
        p = self.network(s) + self.p_min
        return p
```

#### 10.3.4 `Aebs/barrier/barrier_net_layer.py` — BarrierNet 层

```python
import torch
import torch.nn as nn

class BarrierNetLayer(nn.Module):
    """
    BarrierNet 层: 将参考控制投影到 dCBF 安全集内.
    
    这是一个无参数的确定性层, 通过解析解求解 1D QP.
    前向传播和反向传播都通过 PyTorch autograd 自动处理.
    
    对于 AEBS (1D 控制):
        QP: min_u  0.5 * h * (u - u_ref)^2
        s.t. G*u <= h_cbf   (dCBF 约束)
             u_min <= u <= u_max   (控制约束)
    
    解析解:
        u* = clip(max(u_ref, u_min_safe), u_min, u_max)
    
    其中 u_min_safe = (v - p1 * k1 * b(s)) / phi
    """
    
    def __init__(self, dcbf, u_min=-3.0, u_max=3.0):
        """
        Args:
            dcbf: DCBF 实例
            u_min: 最小控制
            u_max: 最大控制
        """
        super().__init__()
        self.dcbf = dcbf
        self.u_min = u_min
        self.u_max = u_max
    
    def forward(self, u_ref, s, p1):
        """
        BarrierNet 前向传播.
        
        Args:
            u_ref: 参考控制 [B, 1] (策略网络输出)
            s: 状态 [B, 2]
            p1: penalty [B, 1]
        
        Returns:
            u_star: 安全控制 [B, 1]
            info: 诊断信息 dict
        """
        u_star, u_min_safe = self.dcbf.compute_min_safe_control(
            s, u_ref, p1, self.u_min, self.u_max
        )
        
        info = {
            'u_ref': u_ref.detach(),
            'u_min_safe': u_min_safe.detach(),
            'u_star': u_star.detach(),
            'cbf_active': (u_star > u_ref).float().mean().item(),  # CBF 约束活跃比例
            'cbf_value': self.dcbf.compute_b(s).detach(),
        }
        
        return u_star, info


class DifferentiableBarrierNetLayer(nn.Module):
    """
    完全可微的 BarrierNet 层 (适用于需要梯度流过多约束 QP 的场景).
    
    使用 qpth 或 cvxpylayers 求解可微 QP.
    对于 1D 情况, 使用上述解析解即可 (自动可微).
    此类提供多约束 QP 的通用接口.
    """
    
    def __init__(self, dcbf_list, u_min=-3.0, u_max=3.0, h_default=1.0):
        """
        Args:
            dcbf_list: DCBF 实例列表 (多个安全约束)
            u_min, u_max: 控制约束
            h_default: 默认 QP 代价系数
        """
        super().__init__()
        self.dcbf_list = dcbf_list
        self.u_min = u_min
        self.u_max = u_max
        self.h_default = h_default
    
    def forward(self, u_ref, s, p_list, h=None):
        """
        多约束 BarrierNet 前向传播.
        
        对于 1D 控制 + 多个 dCBF 约束:
            u_min_safe = max over all constraints of u_min_safe_j
            u_star = clip(max(u_ref, u_min_safe_overall), u_min, u_max)
        
        Args:
            u_ref: 参考控制 [B, 1]
            s: 状态 [B, obs_dim]
            p_list: penalty 列表, 每个 [B, 1]
            h: QP 代价系数 [B, 1] 或 None
        
        Returns:
            u_star: 安全控制 [B, 1]
        """
        if len(self.dcbf_list) == 0:
            return torch.clamp(u_ref, self.u_min, self.u_max), {}
        
        # 计算所有约束的 u_min_safe
        u_min_safe_list = []
        for dcbf, p1 in zip(self.dcbf_list, p_list):
            _, u_min_safe = dcbf.compute_min_safe_control(
                s, u_ref, p1, self.u_min, self.u_max
            )
            u_min_safe_list.append(u_min_safe)
        
        # 取最严格的约束
        u_min_safe_overall = torch.stack(u_min_safe_list, dim=0).max(dim=0)[0]
        
        # 最小修正 + clip
        u_safe = torch.max(u_ref, u_min_safe_overall)
        u_star = torch.clamp(u_safe, self.u_min, self.u_max)
        
        return u_star, {}
```

#### 10.3.5 `Aebs/barrier/safety_filter.py` — 安全过滤器封装

```python
import torch
import torch.nn as nn
from .dcbf import DCBF
from .penalty_net import PenaltyNet
from .barrier_net_layer import BarrierNetLayer

class SafetyFilter(nn.Module):
    """
    完整的安全过滤器: 封装 PenaltyNet + DCBF + BarrierNetLayer.
    
    将参考控制 u_ref 通过 dCBF 约束过滤为安全控制 u_star.
    """
    
    def __init__(self, d_safe=5.0, phi=1.5, alpha_k1=1.0, std1=1.0,
                 penalty_hidden=[32, 16], p_min=0.1,
                 u_min=-3.0, u_max=3.0):
        super().__init__()
        
        self.dcbf = DCBF(d_safe=d_safe, phi=phi, alpha_k1=alpha_k1, std1=std1)
        self.penalty_net = PenaltyNet(
            input_dim=2, hidden_dims=penalty_hidden, 
            output_dim=1, p_min=p_min
        )
        self.barrier_layer = BarrierNetLayer(
            dcbf=self.dcbf, u_min=u_min, u_max=u_max
        )
    
    def forward(self, u_ref, s):
        """
        Args:
            u_ref: 参考控制 [B, 1]
            s: 状态 [B, 2]
        
        Returns:
            u_star: 安全控制 [B, 1]
            p1: penalty 值 [B, 1]
            info: 诊断信息
        """
        p1 = self.penalty_net(s)
        u_star, info = self.barrier_layer(u_ref, s, p1)
        info['p1'] = p1.detach()
        return u_star, p1, info
    
    def get_cbf_value(self, s):
        """获取当前状态的 CBF 值"""
        return self.dcbf.compute_b(s)
    
    def get_cbf_constraint(self, s, u, p1):
        """获取 dCBF 约束残差"""
        return self.dcbf.compute_cbf_constraint(s, u, p1)
```

#### 10.3.6 `Combined_network/model.py` — 修改后的端到端网络

```python
# 在现有文件末尾添加/修改

class AebsEnd2EndNetWithBarrier(nn.Module):
    """
    结合 BarrierNet 的 AEBS 端到端网络.
    
    数据流:
        z, s → gen_net → img → state_net → [state_hat, v] 
            → controller_net → u_ref 
            → safety_filter(u_ref, s) → u_star
    """
    
    def __init__(self, gen_net, state_layer_sizes, mlp_extractor, action_net,
                 safety_filter):
        super().__init__()
        # 原有组件 (冻结)
        self.gen_net = gen_net
        for p in self.gen_net.parameters():
            p.requires_grad = False
        
        self.state_net = SubNet(state_layer_sizes)
        for p in self.state_net.parameters():
            p.requires_grad = False
        
        # 控制器网络 (可训练)
        self.controller_net = CombinedPolicyNetwork(mlp_extractor, action_net)
        
        # BarrierNet 安全过滤器 (可训练) [新增]
        self.safety_filter = safety_filter
    
    def forward(self, z, s):
        """
        Args:
            z: 潜在变量 [B, 4]
            s: 状态 [B, 2]
        
        Returns:
            u_star: 安全加速度 [B, 1]
        """
        with torch.no_grad():
            d = s[:, 0].unsqueeze(1)
            v = s[:, 1].unsqueeze(1)
            img = self.gen_net(z, d)
            img_flat = img.view(img.size(0), -1)
            state = self.state_net(img_flat)
        
        x = torch.cat([state, v], dim=1)
        u_ref = self.controller_net(x)  # 参考加速度
        
        # BarrierNet 安全过滤 [新增]
        u_star, p1, info = self.safety_filter(u_ref, s)
        
        return u_star
    
    def forward_with_info(self, z, s):
        """带诊断信息的前向传播"""
        with torch.no_grad():
            d = s[:, 0].unsqueeze(1)
            v = s[:, 1].unsqueeze(1)
            img = self.gen_net(z, d)
            img_flat = img.view(img.size(0), -1)
            state = self.state_net(img_flat)
        
        x = torch.cat([state, v], dim=1)
        u_ref = self.controller_net(x)
        u_star, p1, info = self.safety_filter(u_ref, s)
        
        info['u_ref'] = u_ref.detach()
        info['state_hat'] = state.detach()
        
        return u_star, info
```

---

### 10.4 修改 `Aebs/system/env.py` — 增加 dCBF 参数

```python
# 在 Aebs 类的 __init__ 中添加:

class Aebs:
    def __init__(self, factor=0.01):
        # ... 原有代码 ...
        
        # === 新增: dCBF 参数 ===
        self.d_safe = 5.0          # 安全距离 (m)
        self.phi = 1.5             # CBF 反应时间 (s)
        self.alpha_k1 = 1.0        # Class K 线性系数
        
        # dCBF 设计: b(s) = d - d_safe - phi * v >= 0
        # 当 d=5, v=0: b = 5 - 5 - 0 = 0 (边界)
        # 当 d=5, v=3: b = 5 - 5 - 4.5 = -4.5 (不安全)
        # 当 d=16, v=0: b = 16 - 5 - 0 = 11 (安全)
```

### 10.5 修改 `Aebs/VT/train.py` — VTLearner 集成 BarrierNet

```python
# 主要修改:

class VTLearner:
    def __init__(self, ..., barrier_config=None):
        # ... 原有初始化 ...
        
        # 新增: BarrierNet 安全过滤器
        from Aebs.barrier import SafetyFilter
        if barrier_config is not None:
            self.safety_filter = SafetyFilter(
                d_safe=self.env.d_safe,
                phi=self.env.phi,
                alpha_k1=self.env.alpha_k1,
                std1=self.env.std1,
                penalty_hidden=barrier_config.get('penalty_hidden', [32, 16]),
                p_min=barrier_config.get('p_min', 0.1),
                u_min=self.env.action_space.low[0],
                u_max=self.env.action_space.high[0],
            ).to(self.device)
        else:
            self.safety_filter = None
        
        # 新增: BarrierNet 参数优化器
        if self.safety_filter is not None:
            barrier_params = list(self.safety_filter.penalty_net.parameters())
            self.barrier_optimizer = torch.optim.Adam(barrier_params, lr=1e-3)
        else:
            self.barrier_optimizer = None
    
    def train_step_l(self, z, y, lip_coeff, current_delta, clip_grad=5.0):
        # ... 原有代码 ...
        
        # 修改: 在计算 s_next 时使用 BarrierNet 过滤后的控制
        if self.safety_filter is not None:
            with torch.no_grad():
                a = self.p_net(z, y_pert)
                a, p1, _ = self.safety_filter(a, y_pert)
        else:
            with torch.no_grad():
                a = self.p_net(z, y_pert)
        
        # ... 其余代码不变 ...
    
    def train_step_p(self, z, y, lip_coeff, current_delta, clip_grad=1.0):
        # ... 原有代码修改 ...
        
        # P-Net 前向传播
        a_p_raw = self.p_net(z, y_pert)
        
        # BarrierNet 安全过滤
        if self.safety_filter is not None:
            a_p, p1, _ = self.safety_filter(a_p_raw, y_pert)
        else:
            a_p = a_p_raw
        
        # ... 动力学和鞅损失计算使用 a_p ...
        
        # 新增: Penalty 正则化损失
        if self.safety_filter is not None:
            # 鼓励 penalty 不要太小 (避免过度保守)
            penalty_reg = torch.relu(1.0 - p1).mean() * 0.1
            loss_p = loss_p + penalty_reg
    
    def train_step_barrier(self, z, y, lip_coeff, current_delta):
        """
        新增: 训练 BarrierNet 的 PenaltyNet.
        
        目标: 
        1. dCBF 约束满足 (安全性)
        2. penalty 不太大 (低保守性)
        3. 与参考控制偏差小 (性能)
        """
        if self.safety_filter is None:
            return {}
        
        self.barrier_optimizer.zero_grad()
        
        # 前向传播
        u_ref = self.p_net(z, y)
        u_star, p1, info = self.safety_filter(u_ref, y)
        
        # 损失 1: CBF 约束满足
        cbf_constraint = self.safety_filter.get_cbf_constraint(y, u_star, p1)
        cbf_violation = torch.relu(-cbf_constraint).mean()
        
        # 损失 2: penalty 正则化 (不要太大也不要太小)
        penalty_reg = (torch.relu(0.5 - p1).mean() +  # 不能太小
                       torch.relu(p1 - 5.0).mean())     # 不能太大
        
        # 损失 3: 控制偏差 (希望 BarrierNet 尽量少修正)
        control_deviation = ((u_star - u_ref) ** 2).mean()
        
        loss_barrier = cbf_violation * 100 + penalty_reg * 1 + control_deviation * 0.1
        
        loss_barrier.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_filter.parameters(), 1.0)
        self.barrier_optimizer.step()
        
        return {
            "loss_barrier": loss_barrier.item(),
            "cbf_violation": cbf_violation.item(),
            "penalty_mean": p1.mean().item(),
            "control_deviation": control_deviation.item(),
        }
```

### 10.6 修改 `Aebs/VT/verify.py` — 增加 dCBF 条件验证

```python
# 在 check_dec_cond 中添加 dCBF 验证:

def check_dec_cond(self, k_except_l):
    # ... 原有验证代码 ...
    
    # 新增: dCBF 约束验证
    cbf_violations = 0
    if self.learner.safety_filter is not None:
        # 检查 dCBF 约束在所有网格点上是否满足
        # 使用 IBP 传播 penalty_net 得到 p1 的区间
        # 然后验证 dCBF 约束下界 >= 0
        
        for start in range(0, sub_grid.shape[0], self.batch_size):
            # ... 原有批次处理 ...
            
            # 新增: 计算 dCBF 约束的 IBP 区间
            # 1. 获取 penalty_net 的 IBP 输出区间
            # 2. 计算 dCBF 约束值区间
            # 3. 如果下界 < 0，记录违反
    
    # 综合判断
    total_violations = hard_violations  # 原有鞅违反 + dCBF 违反
```

### 10.7 修改 `Aebs/VT/loop.py` — CEGIS 主循环

```python
# 主要修改 __main__ 部分:

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = Aebs(0.05)
    
    # BarrierNet 配置 [新增]
    barrier_config = {
        'penalty_hidden': [32, 16],
        'p_min': 0.1,
    }
    
    vt_learner = VTLearner(
        l_model_config=[2, 16, 8, 1],
        env=env,
        p_lip=2.0,
        l_lip=4.0,
        eps=0.1,
        gamma_decrease=1.0,
        reach_prob=0.95,
        square_l_output=True,
        barrier_config=barrier_config,  # [新增]
    )
    
    # ... 其余初始化不变 ...
    
    # 在 run() 的循环中增加 barrier 训练步:
    # self.train('barrier', 5)  # 每 5 个 CEGIS 迭代训练一次 BarrierNet
```

---

## 十一、训练流程

### 11.1 完整训练流程（分阶段）

```
阶段 0: 数据准备 (不变)
  CARLA → 400 images → 下采样 32×32 → Downsampled.h5

阶段 1: 感知模型训练 (不变)
  cGAN/MLP → AebsMLPGenerator → mlp_supervised.pth

阶段 2: 状态估计训练 (不变)
  SubNet → state_net_trained.pth

阶段 3: PPO 控制器预训练 (不变)
  PPO → best_model.zip

阶段 4: SafePVC CEGIS + BarrierNet (修改后)
  迭代 1~100:
    4a. 训练 SBC 网络 l_model (10 epochs)
        损失: martingale + region + lipschitz + cbf_guide
    4b. IBP 验证:
        - 鞅递减条件 (原有)
        - dCBF 约束条件 (新增)
    4c. 如果验证通过:
        计算 P(safe) ≥ 1 - B(s_init)/B(s_unsafe)
    4d. 提取反例
    4e. 训练策略网络 p_net (1 epoch)
        损失: martingale + lipschitz + mse
    4f. 训练 BarrierNet penalty_net (新增, 每 5 迭代)
        损失: cbf_satisfaction + penalty_reg + control_deviation
```

### 11.2 各阶段训练参数

| 阶段 | 网络 | 学习率 | 批大小 | Epochs | 损失函数 |
|------|------|--------|--------|--------|---------|
| 4a | l_model (SBC) | 3e-3 | 2048 | 10/iter | $\mathcal{L}_L$ |
| 4e | p_net (策略) | 5e-2 | 2048 | 1/iter | $\mathcal{L}_P$ |
| 4f | penalty_net | 1e-3 | 2048 | 5/5iter | $\mathcal{L}_{\text{barrier}}$ |

### 11.3 训练伪代码

```python
for iteration in range(100):
    # Step 1: 训练 SBC
    for epoch in range(10):
        for batch in dataloader:
            z = random_sample([-1,1]^4)
            s = batch
            loss_l = train_step_l(z, s)  # 含 cbf_guide_loss
    
    # Step 2: IBP 验证
    sat, violations, info = verifier.check_dec_cond(K)
    if sat:
        prob = compute_safety_probability()
        print(f"P(safe) >= {prob:.1%}")
    
    # Step 3: 训练策略 (使用反例)
    for batch in violation_buffer:
        z = random_sample([-1,1]^4)
        s = batch
        loss_p = train_step_p(z, s)  # 含 safety_filter 过滤
    
    # Step 4: 训练 BarrierNet (每 5 迭代)
    if iteration % 5 == 0:
        for batch in dataloader:
            z = random_sample([-1,1]^4)
            s = batch
            loss_b = train_step_barrier(z, s)
```

---

## 十二、验证流程

### 12.1 验证条件

结合 BarrierNet 后，需要验证的条件从 3 个增加到 4 个：

| # | 条件 | 方法 | 原有/新增 |
|---|------|------|----------|
| 1 | $B(s) \geq 0$ (非负性) | 架构保证 (Softplus) | 原有 |
| 2 | $B(s) \leq 1$ on $S_0$ (初始集) | IBP 上界 | 原有 |
| 3 | $B(s) \geq 1/(1-p)$ on $X_u$ (不安全集) | IBP 下界 | 原有 |
| 4 | $\mathbb{E}[B(s')] + K + \epsilon \leq B(s)$ (鞅递减) | IBP + 噪声网格 | 原有 |
| 5 | **dCBF 约束: $G u^* \leq h_{cbf}$** | **IBP 区间传播** | **新增** |

### 12.2 dCBF 约束的 IBP 验证

对于 AEBS 的 1D dCBF 约束，验证过程：

```
对于网格 cell [s_lb, s_ub]:
  1. b(s) 的区间:
     b_lb = d_lb * std1 - d_safe - phi * v_ub
     b_ub = d_ub * std1 - d_safe - phi * v_lb
  
  2. p1(s) 的区间 (通过 IBP 传播 PenaltyNet):
     p1_lb, p1_ub = IBP(PenaltyNet, s_lb, s_ub)
  
  3. alpha1(b) 的区间 (线性):
     alpha1_lb = k1 * b_lb
     alpha1_ub = k1 * b_ub
  
  4. u_ref 的区间 (通过 IBP 传播 controller_net):
     u_ref_lb, u_ref_ub = IBP(ControllerNet, s_lb, s_ub)
  
  5. u_min_safe 的区间:
     u_min_safe = (v - p1 * k1 * b) / phi
     u_min_safe_lb = (v_lb - p1_ub * k1 * max(b_ub, 0)) / phi
     u_min_safe_ub = (v_ub - p1_lb * k1 * min(b_lb, 0)) / phi
  
  6. u* = max(u_ref, u_min_safe) 的区间:
     u_star_lb = max(u_ref_lb, u_min_safe_lb)
     u_star_ub = max(u_ref_ub, u_min_safe_ub)
  
  7. dCBF 约束验证:
     constraint = Lf_b + Lg_b * u* + p1 * alpha1(b)
     constraint_lb = -v_ub + phi * u_star_lb + p1_lb * alpha1_lb
     
     如果 constraint_lb >= 0: ✅ 通过
     如果 constraint_lb < 0: ❌ 违反 → 反例
```

### 12.3 增强的概率安全下界

有了 dCBF 约束，概率安全下界可以增强：

$$\mathbb{P}(\text{safe}) \geq 1 - \frac{B(s_0)_{\max} - B_{\min}}{B(X_u)_{\min} - B_{\min}}$$

其中 $B_{\min}$ 是 $B(s)$ 在整个状态空间上的 IBP 下界。

由于 BarrierNet 保证了 dCBF 约束 $b(s) \geq 0$，SBC 网络可以在受限状态空间 $\{s: b(s) \geq 0\}$ 上获得更紧的界。

---

## 十三、潜在挑战与解决方案

### 13.1 挑战一：IBP 传播 BarrierNet 层的精度

**问题**：BarrierNet 层包含 `max` 和 `clip` 操作，IBP 区间传播可能导致过度近似。

**解决方案**：
- 对于 1D QP 的解析解，`max` 操作的 IBP 是精确的（`max([a,b], [c,d]) = [max(a,c), max(b,d)]`）
- `clip` 操作也是精确的
- 关键不确定性在 `p1` 的 IBP 传播上，可以通过更细的网格或 alpha-CROWN 提高精度

### 13.2 挑战二：PenaltyNet 训练不稳定

**问题**：PenaltyNet 需要在安全性和低保守性之间平衡，可能出现训练震荡。

**解决方案**：
- 使用 Softplus 激活保证平滑正值
- 添加 $p_{\min}$ 下界和正则化上界
- 在 CEGIS 中交替训练而非同时训练
- 初始化 $p_1$ 为较大值（偏保守），让训练逐步放松

### 13.3 挑战三：dCBF 与控制约束冲突

**问题**：dCBF 约束要求的最小控制可能超出 $[u_{\min}, u_{\max}]$，导致 QP 不可行。

**解决方案**：
- 在设计 $\phi$ 和 $d_{safe}$ 时确保可行性：$\phi$ 足够大使得 CBF 约束在控制范围内可满足
- 在训练标签（PPO 策略）本身就满足安全约束的情况下，QP 总是可行的
- 可以添加松弛变量处理不可行情况

### 13.4 挑战四：计算开销

**问题**：增加 PenaltyNet + BarrierNet 层会增加前向传播时间，验证也需要额外的 IBP 传播。

**解决方案**：
- AEBS 的控制维度为 1，1D QP 有解析解，开销接近零
- PenaltyNet 很小（2→32→16→1，约 1100 参数），前向传播极快
- 验证中可以增加 dCBF 的过滤条件，跳过明显安全的网格点，加速验证

### 13.5 挑战五：dCBF 参数选择

**问题**：$\phi$ 和 $d_{safe}$ 的选择直接影响安全约束的形状。

**解决方案**：
- 参数化 $\phi$ 使其可学习（但需保持正值）
- 使用多组参数进行消融实验
- 参考 BarrierNet 论文中自适应选择的方法

---

## 十四、实验设计

### 14.1 消融实验

| 实验 | 方法 | 比较指标 |
|------|------|---------|
| A1 | 原始 SafePVC（无 BarrierNet） | P(safe), 迭代数 |
| A2 | SafePVC + BarrierNet (固定 penalty) | P(safe), 迭代数 |
| A3 | SafePVC + BarrierNet (可学习 penalty) | P(safe), 迭代数 |
| A4 | SafePVC + BarrierNet + CBF 引导损失 | P(safe), 迭代数 |

### 14.2 性能比较

| 实验 | 方法 | 比较指标 |
|------|------|---------|
| B1 | 纯 BarrierNet（无 SBC 验证） | 轨迹安全性, 控制平滑性 |
| B2 | 纯 SafePVC | P(safe), 控制性能 |
| B3 | 结合方法 | 两者指标 |

### 14.3 鲁棒性实验

在不同扰动强度下测试（$\pm 3\%$, $\pm 5\%$, $\pm 7\%$, $\pm 10\%$）：
- 比较 P(safe) 下界的变化
- 比较轨迹安全性的经验成功率
- 比较控制性能的退化程度

### 14.4 可扩展性实验

在未来工作中测试更复杂的场景：
- X-Plane 11 飞机滑行（2D 控制：转向角）
- 多障碍物场景（需要 HOCBF）
- 更高维度的状态空间

---

## 十四a、端到端数值计算演练

本节用一个完整的数值例子，跟踪一个时间步内系统的所有计算步骤。

### 14a.1 初始条件

**系统状态**：$d = 12.0$ m, $v = 2.0$ m/s, $\Delta t = 0.05$ s

**归一化**：设 $\text{std1} = 3.0$（训练数据的标准差）

$d_{\text{norm}} = 12.0 / 3.0 = 4.0$, $v = 2.0$

$s = [4.0, 2.0]$

**安全约束参数**：$d_{\text{safe}} = 6.0$ m, $\phi = 1.0$

### 14a.2 cGAN 感知模型 → MLP 蒸馏

**输入**：$z = [0.1, -0.2, 0.05, 0.3]$（环境扰动），$d_{\text{norm}} = 4.0$

**AebsMLPGenerator**：
- 输入拼接：$[z, d_{\text{norm}}] = [0.1, -0.2, 0.05, 0.3, 4.0]$（5维）
- Linear(5→256) → ReLU → Linear(256→256) → ReLU → ... → Linear(256→1024) → Tanh
- 输出：$\text{img} \in [-1, 1]^{1024}$（展平后的 32×32 图像）

**MLP 蒸馏后的 SubNet**：
- 输入：$\text{img}$（1024维）
- Linear(1024→256) → LayerNorm → ReLU → Linear(256→64) → LayerNorm → ReLU → Linear(64→1)
- 输出：$\hat{d}_{\text{norm}} = 3.95$（估计的归一化距离）

### 14a.3 策略网络 (PPO 预训练)

**输入**：$[\hat{d}_{\text{norm}}, v] = [3.95, 2.0]$

**PPO 网络**：
- mlp_extractor: Linear(2→64) → Tanh → Linear(64→64) → Tanh
- action_net: Linear(64→1)
- 输出：$u_{\text{ref}} = 1.5$ m/s²（参考加速度，希望加速）

### 14a.4 BarrierNet dCBF 计算

**安全函数**：$b(s) = d_{\text{norm}} \times \text{std1} - d_{\text{safe}} - \phi v = 12.0 - 6.0 - 1.0 \times 2.0 = 4.0$

**Lie 导数**：

$L_f b = \nabla b \cdot f(s) = [1, -\phi] \cdot [-v, 0]^T = -v = -2.0$

$L_g b = \nabla b \cdot g(s) = [1, -\phi] \cdot [0, -1]^T = \phi = 1.0$

**Penalty 网络**：

$p_1(s) = \text{Softplus}(\text{MLP}_p([3.95, 2.0])) + 0.1 = 0.8 + 0.1 = 0.9$

（中等值，表示当前场景不太危险）

**dCBF 约束**：

$L_f b + L_g b \cdot u + p_1 \cdot \alpha_1(b) \geq 0$

$-2.0 + 1.0 \cdot u + 0.9 \times 4.0 \geq 0$

$u \geq 2.0 - 3.6 = -1.6$

### 14a.5 BarrierNet QP 求解

**QP 参数**：$H = 2.0$, $F = -u_{\text{ref}} \cdot H = -3.0$

$$\min_u \frac{1}{2} \times 2.0 \times u^2 - 3.0u = u^2 - 3u$$

**约束**：
1. $u \geq -1.6$（dCBF）
2. $u \leq 3.0$（控制上界）
3. $u \geq -3.0$（控制下界）

**无约束最优**：$2u - 3 = 0 \implies u = 1.5$

**检查约束**：$u = 1.5 \geq -1.6$ ✓，$1.5 \leq 3.0$ ✓

**最终控制**：$u^* = 1.5$ m/s²（安全约束不活跃，参考控制直接通过）

> **对比**：如果当前状态更危险（如 $d = 8, v = 2.5$），$b = 8 - 6 - 2.5 = -0.5$，dCBF 约束会变为 $-2.5 + u + 0.9 \times (-0.5) \geq 0 \implies u \geq 2.95$，此时参考控制 $1.5$ 被安全约束覆盖，BarrierNet 输出 $u^* = 2.95$（强制刹车）。

### 14a.6 状态转移

$$d_{\text{next}} = d - v \cdot \Delta t = 12.0 - 2.0 \times 0.05 = 11.9 \text{ m}$$

$$v_{\text{next}} = v - u^* \cdot \Delta t = 2.0 - 1.5 \times 0.05 = 1.925 \text{ m/s}$$

$$s_{\text{next}} = [11.9/3.0, 1.925] = [3.967, 1.925]$$

### 14a.7 SBC 值计算与鞅验证

**SBC 网络**：$B(s) = \text{Softplus}(\text{MLP}_B(s))$

$B(s) = B([4.0, 2.0]) = 2.5$（当前状态的"危险度"）

**期望下一步 SBC 值**：

考虑 16 个三角分布噪声样本 $\Delta s_i$：

$$\mathbb{E}[B(s')] \approx \frac{1}{16} \sum_{i=1}^{16} B(s_{\text{next}} + \Delta s_i) = 2.3$$

**鞅递减条件检查**：

$$B(s) - \mathbb{E}[B(s')] - \epsilon = 2.5 - 2.3 - 0.1 = 0.1 \geq 0 \quad ✓$$

鞅条件满足：$B$ 值从 $2.5$ 平均降到 $2.3$，递减了 $0.2$，超过了 $\epsilon = 0.1$ 的要求。

### 14a.8 概率安全界计算

假设经过完整验证：

- $\max_{S_0} B = 0.98$（初始集上 B 的最大值）
- $\min_{X_u} B = 10.5$（不安全集上 B 的最小值）

$$B_{\text{ratio}} = \frac{10.5}{0.98} = 10.71$$

$$\mathbb{P}(\text{安全}) \geq 1 - \frac{1}{B_{\text{ratio}}} = 1 - \frac{1}{10.71} = 1 - 0.0934 = 90.66\%$$

满足 $p = 0.90$ 的安全要求。

---

## 十五、总结

### 15.1 核心贡献回顾

本研究提出了一种**双层安全保证架构**，将 BarrierNet 的可微控制障碍函数 (dCBF) 嵌入 SafePVC 的概率安全验证框架：

1. **内层 (BarrierNet)**：通过 dCBF 约束实时保证每步控制的安全性
2. **外层 (SBC 验证)**：通过鞅理论证明闭环系统的长期概率安全性
3. **CEGIS 反馈**：反例引导同时优化策略网络、SBC 网络和 Penalty 网络

### 15.2 实施优先级

| 优先级 | 任务 | 预计工作量 |
|--------|------|-----------|
| P0 | 实现 DCBF 类 | 1-2 小时 |
| P0 | 实现 PenaltyNet | 1 小时 |
| P0 | 实现 BarrierNetLayer | 1-2 小时 |
| P0 | 修改 AebsEnd2EndNet | 2 小时 |
| P0 | 修改 VTLearner | 3-4 小时 |
| P1 | 修改 VTVerifier | 2-3 小时 |
| P1 | 修改 CEGIS 循环 | 2 小时 |
| P1 | 调试和测试 | 4-8 小时 |
| P2 | 实验和消融 | 1-2 天 |

### 15.3 预期成果

- **安全概率下界提升**：从 SafePVC 的 ~94% 提升到 >95%
- **CEGIS 迭代数减少**：BarrierNet 的确定性安全约束缩小验证空间
- **控制性能改善**：PenaltyNet 的自适应性减少过度保守
- **论文贡献**：填补 BarrierNet 与形式化验证结合的研究空白

---

> **备注**: 本文档应配合源代码一同使用。所有数学公式中的变量与代码实现中的变量一一对应。如有任何不明确之处，请参考各节中的符号表。
