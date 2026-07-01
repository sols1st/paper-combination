# Control Barrier Function (CBF) 控制障碍函数

> **一句话**：CBF 是一种数学工具，用来保证一个控制系统**永远不会进入危险状态**。它通过在每个时刻对控制输入施加一个不等式约束来实现这一点。

---

## 1. 为什么需要 CBF？

想象你在驾驶一辆自动驾驶汽车。你需要保证：
- 车永远不会撞到前面的障碍物
- 车永远不会偏离车道
- 车永远不会超速

传统方法（如 [[Lyapunov 函数]]）可以证明系统**收敛到目标**，但不太擅长保证系统**永远不进入危险区域**。

**CBF 的核心思想**：定义一个"安全函数" $b(x)$，当系统安全时 $b(x) \geq 0$，当系统危险时 $b(x) < 0$。然后设计控制 $u$ 使得 $b(x)$ 永远不小于 0。

---

## 2. 数学定义

### 2.1 系统模型

考虑**控制仿射系统** (control-affine system)：

$$\dot{x} = f(x) + g(x)u$$

其中：
- $x \in \mathbb{R}^n$：系统状态（如位置、速度）
- $u \in U \subset \mathbb{R}^m$：控制输入（如加速度、转向角）
- $f(x)$：**漂移项** (drift) — 无控制时系统的自然演化
- $g(x)$：**控制矩阵** — 控制如何影响系统

**例子**（AEBS 自动紧急刹车）：

$$\begin{bmatrix} \dot{d} \\ \dot{v} \end{bmatrix} = \underbrace{\begin{bmatrix} -v \\ 0 \end{bmatrix}}_{f(x)} + \underbrace{\begin{bmatrix} 0 \\ -1 \end{bmatrix}}_{g(x)} \cdot a$$

$d$ 是距离，$v$ 是速度，$a$ 是加速度（控制输入）。

### 2.2 安全集

定义安全集 $C$ 为一个标量函数 $b: \mathbb{R}^n \to \mathbb{R}$ 的**超水平集** (superlevel set)：

$$C = \{x \in \mathbb{R}^n : b(x) \geq 0\}$$

边界：$\partial C = \{x : b(x) = 0\}$

不安全集：$C_{\text{unsafe}} = \{x : b(x) < 0\}$

**例子**：$b(x) = d - d_{\text{safe}}$（距离减去安全距离）
- $d > d_{\text{safe}} \implies b > 0$ → 安全
- $d = d_{\text{safe}} \implies b = 0$ → 边界
- $d < d_{\text{safe}} \implies b < 0$ → 危险

### 2.3 Class K 函数

**Class K 函数** $\alpha: \mathbb{R} \to \mathbb{R}$ 满足：
1. 连续
2. 严格递增
3. $\alpha(0) = 0$

常见选择：$\alpha(r) = kr$（$k > 0$，线性），$\alpha(r) = r^3$，$\alpha(r) = \tanh(r)$

> 直觉：$\alpha$ 是一个"弹簧"，当 $b(x)$ 偏离 0 时，$\alpha(b(x))$ 产生一个"回弹力"。

### 2.4 CBF 的正式定义

**定义**：函数 $b(x)$ 是系统 $\dot{x} = f(x) + g(x)u$ 的**控制障碍函数 (CBF)**，如果存在一个扩展 Class K 函数 $\alpha$ 使得：

$$\sup_{u \in U} \left[ L_f b(x) + L_g b(x) u + \alpha(b(x)) \right] \geq 0, \quad \forall x$$

其中 [[Lie 导数]]：
- $L_f b(x) = \nabla b(x) \cdot f(x)$：$b$ 沿 $f$ 的方向导数
- $L_g b(x) = \nabla b(x) \cdot g(x)$：$b$ 沿 $g$ 的方向导数

**等价表述**：存在控制 $u$ 使得

$$L_f b(x) + L_g b(x) u + \alpha(b(x)) \geq 0$$

即 $b$ 的变化率加上一个正项 $\alpha(b(x))$ 非负。

---

## 3. 直觉解释

### 3.1 池塘与鱼的比喻

- **池塘** = 安全集 $C$（$b(x) \geq 0$ 的区域）
- **鱼** = 系统状态 $x(t)$
- **CBF 条件** = 在池塘边界处，水的流动方向总是指向池塘内部

如果边界处的"水流"永远指向内部，鱼就不可能跳出池塘。

### 3.2 数学直觉

在边界 $\partial C$（$b(x) = 0$）处：

$$\dot{b}(x) = L_f b(x) + L_g b(x) u \geq -\alpha(b(x)) = -\alpha(0) = 0$$

即 $\dot{b}(x) \geq 0$：$b$ 不会继续减小。系统在边界处被"推回"安全区域。

在安全区域内部（$b(x) > 0$）：

$$\dot{b}(x) \geq -\alpha(b(x)) < 0$$

$b$ 允许减小，但减小速度被 $\alpha$ 限制——离边界越近，允许减小的速度越慢。

### 3.3 与 Lyapunov 函数的对比

| 特性 | [[Lyapunov 函数]] $V(x)$ | CBF $b(x)$ |
|------|--------------------------|------------|
| **目标** | 收敛到目标点 $x^*$ | 不进入不安全集 |
| **期望行为** | $V$ 递减 → 趋向目标 | $b$ 保持非负 → 保持安全 |
| **条件** | $\dot{V} \leq -\alpha(V)$ | $\dot{b} \geq -\alpha(b)$ |
| **关注区域** | 全局（所有 $x$） | 边界 $\partial C$ 最关键 |

---

## 4. CBF 的使用方式

### 4.1 安全过滤器 (Safety Filter)

最常见的使用方式：将 CBF 约束嵌入一个 [[QP (二次规划)]] 中，作为安全过滤器：

$$u^* = \arg\min_{u} \|u - u_{\text{ref}}\|^2$$
$$\text{s.t.} \quad L_f b(x) + L_g b(x) u + \alpha(b(x)) \geq 0$$
$$u_{\min} \leq u \leq u_{\max}$$

其中 $u_{\text{ref}}$ 是任何控制器（如 [[PPO]]、PID、人工操作员）的参考控制。

**解读**：QP 找到**最接近参考控制的安全控制**。如果参考控制已经安全，QP 直接输出 $u_{\text{ref}}$；如果参考控制不安全，QP 做最小修正使其安全。

### 4.2 具体计算步骤

**输入**：
- 当前状态 $x$
- 安全函数 $b(x)$ 及其梯度 $\nabla b(x)$
- 系统模型 $f(x), g(x)$
- Class K 函数 $\alpha$
- 参考控制 $u_{\text{ref}}$

**计算**：
1. 计算 $b(x)$ 的值
2. 计算 $L_f b(x) = \nabla b \cdot f(x)$
3. 计算 $L_g b(x) = \nabla b \cdot g(x)$
4. 计算 $\alpha(b(x))$
5. 构造 QP：$\min \|u - u_{\text{ref}}\|^2$ s.t. $L_g b \cdot u \geq -L_f b - \alpha(b)$
6. 求解 QP → $u^*$

**输出**：安全控制 $u^*$

### 4.3 数值例子

**AEBS 场景**：$d = 8$ m, $v = 2$ m/s, $d_{\text{safe}} = 6$ m

$b(x) = d - d_{\text{safe}} = 2$

$f(x) = [-v, 0]^T = [-2, 0]^T$, $g(x) = [0, -1]^T$

$\nabla b = [1, 0]$

$L_f b = [1, 0] \cdot [-2, 0]^T = -2$

$L_g b = [1, 0] \cdot [0, -1]^T = 0$

**问题**：$L_g b = 0$！控制 $u$（加速度）不直接影响 $b$ 的一阶导数。这是因为 $b$ 只依赖距离 $d$，而加速度直接影响的是速度 $v$，速度才影响距离。

**解决**：需要使用带速度项的安全函数或 [[HOCBF (高阶控制障碍函数)]]。

**改进**：$b(x) = d - d_{\text{safe}} - \phi v$，其中 $\phi = 1$

$\nabla b = [1, -1]$

$L_f b = [1, -1] \cdot [-2, 0]^T = -2$

$L_g b = [1, -1] \cdot [0, -1]^T = 1$ ✅

设 $\alpha(r) = r$，参考控制 $u_{\text{ref}} = 0$（不加速不减速）

QP：$\min_u u^2$ s.t. $-2 + 1 \cdot u + 1 \times 2 \geq 0 \implies u \geq 0$

解：$u^* = 0$（参考控制已满足约束）

如果 $d = 7$（更近），$b = 7 - 6 - 2 = -1$

约束：$-2 + u + (-1) \geq 0 \implies u \geq 3$

解：$u^* = 3$（强制最大刹车！）

---

## 5. 代码实现

### 5.1 PyTorch 实现

```python
import torch

class CBF:
    """
    Control Barrier Function 安全过滤器
    
    安全约束: b(s) = d - d_safe - phi * v >= 0
    CBF 约束: Lf_b + Lg_b * u + alpha * b >= 0
    """
    def __init__(self, d_safe=6.0, phi=1.0, alpha_coeff=1.0, 
                 u_min=-3.0, u_max=3.0):
        self.d_safe = d_safe
        self.phi = phi
        self.alpha_coeff = alpha_coeff
        self.u_min = u_min
        self.u_max = u_max
    
    def compute_b(self, s):
        """
        计算安全函数值 b(x)
        
        输入:
            s: tensor, shape (batch, 2), [d_norm, v]
            d_norm 是归一化距离（需要乘以 std1 还原真实距离）
        
        输出:
            b: tensor, shape (batch,), 安全函数值
        """
        d = s[:, 0]  # 假设已经还原为真实距离
        v = s[:, 1]
        b = d - self.d_safe - self.phi * v
        return b
    
    def compute_lie_derivatives(self, s):
        """
        计算 Lie 导数 Lf_b 和 Lg_b
        
        对于 b = d - d_safe - phi * v:
            ∇b = [1, -phi]
            f(x) = [-v, 0]^T  (漂移)
            g(x) = [0, -1]^T  (控制)
            
            Lf_b = ∇b · f = 1*(-v) + (-phi)*0 = -v
            Lg_b = ∇b · g = 1*0 + (-phi)*(-1) = phi
        """
        v = s[:, 1]
        Lf_b = -v                    # shape: (batch,)
        Lg_b = torch.full_like(v, self.phi)  # shape: (batch,)
        return Lf_b, Lg_b
    
    def safety_filter(self, s, u_ref):
        """
        CBF 安全过滤器：修正参考控制使其安全
        
        输入:
            s: tensor, shape (batch, 2), 当前状态 [d, v]
            u_ref: tensor, shape (batch,), 参考控制（如 PPO 输出）
        
        输出:
            u_safe: tensor, shape (batch,), 安全控制
        """
        b = self.compute_b(s)
        Lf_b, Lg_b = self.compute_lie_derivatives(s)
        
        # CBF 约束: Lf_b + Lg_b * u + alpha * b >= 0
        # 即: Lg_b * u >= -Lf_b - alpha * b
        # 即: u >= (-Lf_b - alpha * b) / Lg_b  (当 Lg_b > 0)
        
        alpha_b = self.alpha_coeff * b
        u_min_cbf = (-Lf_b - alpha_b) / Lg_b  # CBF 要求的最小控制
        
        # 取 CBF 下界和控制界限的交集
        u_lower = torch.max(u_min_cbf, torch.tensor(self.u_min))
        u_upper = torch.tensor(self.u_max)
        
        # 如果参考控制满足约束，直接使用；否则投影到安全集
        u_safe = torch.clamp(u_ref, min=u_lower, max=u_upper)
        
        return u_safe
```

### 5.2 使用示例

```python
# 创建 CBF
cbf = CBF(d_safe=6.0, phi=1.0, alpha_coeff=1.0)

# 当前状态: 距离 8m, 速度 2 m/s
s = torch.tensor([[8.0, 2.0]])

# PPO 输出的参考控制: 不刹车
u_ref = torch.tensor([0.0])

# 安全过滤
u_safe = cbf.safety_filter(s, u_ref)
print(f"参考控制: {u_ref.item():.2f}")
print(f"安全控制: {u_safe.item():.2f}")

# 当前状态: 距离 7m, 速度 2 m/s (更危险)
s2 = torch.tensor([[7.0, 2.0]])
u_safe2 = cbf.safety_filter(s2, u_ref)
print(f"危险状态下安全控制: {u_safe2.item():.2f}")  # 应输出 3.0 (最大刹车)
```

---

## 6. CBF 的局限性

| 问题 | 说明 | 解决方案 |
|------|------|---------|
| **相对度问题** | 控制 $u$ 可能不直接出现在 $b$ 的一阶导数中 | [[HOCBF (高阶控制障碍函数)]] |
| **保守性** | Class K 函数 $\alpha$ 的固定选择可能导致过度保守 | [[dCBF (可微控制障碍函数)]] |
| **可行性** | CBF 约束可能与控制界限冲突（无解） | 松弛变量、优化 $\alpha$ |
| **多约束** | 多个 CBF 可能相互冲突 | 优先级、[[QP (二次规划)]] 松弛 |
| **学习困难** | 手工设计好的 $b(x)$ 很难 | [[Neural Barrier Certificate (神经障碍证书)]] |

---

## 7. 关键定理

**定理（CBF 安全性）**：如果 $b(x)$ 是系统 $\dot{x} = f(x) + g(x)u$ 的 CBF，且存在 Lipschitz 连续控制器 $u(x)$ 满足 CBF 约束，则安全集 $C = \{x: b(x) \geq 0\}$ 是 [[前向不变性 (Forward Invariance)|前向不变的]]。

**证明思路**：
1. 在边界 $b(x) = 0$ 处，$\dot{b} \geq -\alpha(0) = 0$
2. 因此 $b$ 不会从 0 变为负数
3. 由 [[比较引理 (Comparison Lemma)]]，$b(t) \geq b(0) e^{-\alpha t} \geq 0$

---

## 8. 相关概念

- [[HOCBF (高阶控制障碍函数)]] — 处理高 [[相对度 (Relative Degree)]] 系统
- [[dCBF (可微控制障碍函数)]] — 可学习的、环境自适应的 CBF
- [[SBF (随机障碍函数)]] — CBF 在随机系统中的推广
- [[Lyapunov 函数]] — 稳定性分析的对偶工具
- [[Lie 导数]] — CBF 的核心数学工具
- [[Class K 函数]] — CBF 中用于定义衰减速率
- [[前向不变性 (Forward Invariance)]] — CBF 保证的核心性质
- [[QP (二次规划)]] — CBF 的实际使用方式
- [[Safety Filter (安全过滤器)]] — CBF 的应用模式

---

## 9. 推荐学习路径

1. 先理解 [[Lie 导数]] 和 [[Class K 函数]]
2. 理解 [[前向不变性 (Forward Invariance)]] 的概念
3. 阅读本笔记，理解 CBF 的定义和直觉
4. 运行上面的代码，动手实验
5. 进阶到 [[HOCBF (高阶控制障碍函数)]]
6. 了解 [[dCBF (可微控制障碍函数)]] 和 [[BarrierNet]]

---

> **参考论文**: 
> - Ames et al., "Control Barrier Functions: Theory and Applications," ECC 2019
> - Xu et al., "Safety-Critical Control of Nonlinear Systems," 2015
