# Class K 函数 (Class K / Class K∞ Functions)

> **一句话**：Class K 函数是一类**连续、严格递增、过原点**的函数，在 [[CBF (控制障碍函数)]] 和 [[HOCBF (高阶控制障碍函数)]] 中用来**描述安全裕度的衰减速率**。它决定了控制器在接近安全边界时"多积极地"采取行动。

---

## 1. 定义

### 1.1 Class $\mathcal{K}$ 函数

一个函数 $\alpha: [0, a) \to [0, \infty)$ 属于 **Class $\mathcal{K}$**，如果满足：

1. **连续性**：$\alpha$ 连续
2. **严格递增**：$r_1 < r_2 \Rightarrow \alpha(r_1) < \alpha(r_2)$
3. **过原点**：$\alpha(0) = 0$

### 1.2 Class $\mathcal{K}_\infty$ 函数

如果 $\alpha \in \mathcal{K}$ 且 $a = \infty$，并且：

4. **无界**：$\lim_{r \to \infty} \alpha(r) = \infty$

则 $\alpha \in \mathcal{K}_\infty$。

### 1.3 Extended Class $\mathcal{K}$ 函数

在 CBF 文献中，常见定义域为 $(-b, a)$（包含负数），$\alpha: (-b, a) \to (-\infty, \infty)$：

1. 连续
2. 严格递增
3. $\alpha(0) = 0$

这使得 $\alpha$ 可以处理 $b(x) < 0$ 的情况（已经不安全时的恢复）。

---

## 2. 常见 Class K 函数

| 函数 | 公式 | $\mathcal{K}_\infty$? | 用途 |
|------|------|---------------------|------|
| **线性** | $\alpha(r) = \gamma r$ | 是（$\gamma > 0$） | 最常用，简单 |
| **幂函数** | $\alpha(r) = r^p$ ($p > 0$) | 是 | 非线性响应 |
| **饱和** | $\alpha(r) = \tanh(r)$ | 否（有界） | 限制控制量 |
| **分段线性** | $\alpha(r) = \min(\gamma r, c)$ | 否 | 限幅 |
| **指数型** | $\alpha(r) = e^r - 1$ | 否（$r \to \infty$ 无界，但定义域受限） | 激进响应 |
| **分数幂** | $\alpha(r) = r^{1/3}$ | 是 | 原点附近更激进 |

---

## 3. 在 CBF 中的角色

### 3.1 CBF 条件回顾

在 [[CBF (控制障碍函数)]] 中，安全条件为：

$$\sup_u \left[ L_f b(x) + L_g b(x) \cdot u + \alpha(b(x)) \right] \geq 0$$

**$\alpha(b(x))$ 的含义**：

- 当 $b(x)$ 很大（远离边界）：$\alpha(b(x))$ 很大，即使 $L_f b + L_g b \cdot u$ 稍小也能满足 → 控制器**不用太积极**
- 当 $b(x) \to 0$（接近边界）：$\alpha(b(x)) \to 0$，必须靠 $L_g b \cdot u$ 补偿 → 控制器**必须积极行动**
- 当 $b(x) < 0$（已经不安全）：$\alpha(b(x)) < 0$，要求更大的补偿 → 控制器**全力恢复**

### 3.2 线性 Class K 的例子

取 $\alpha(r) = \gamma r$（最常用）：

$$L_f b + L_g b \cdot u + \gamma \cdot b(x) \geq 0$$

$\gamma$ 越大 → 要求 $b(x)$ 的衰减越快 → 控制器越保守。

---

## 4. 直觉：弹簧类比

想象一个弹簧连接着你的控制对象和安全边界：

```
     [安全区域]                    [危险区域]
     ←─────────────────────────────→
     
     ○ ←─── 控制对象
     |
     | ← 弹簧力 = α(b(x))
     |
    ║ ← 安全边界 (b=0)
```

- $\alpha(r) = \gamma r$（线性弹簧）：离边界越远，弹簧力越大
- $\alpha(r) = r^3$（硬弹簧）：远离边界时力很大，接近边界时力很小
- $\alpha(r) = r^{1/3}$（软弹簧）：接近边界时力仍然很大

---

## 5. $\alpha$ 的选择对控制行为的影响

### 5.1 数值对比

假设系统 $\dot{x} = u$，CBF $b(x) = x - 1$（要求 $x \geq 1$），当前 $x = 1.5$：

| $\alpha$ | $\alpha(b(1.5))$ | 所需最小 $u$ | 行为 |
|----------|-----------------|-------------|------|
| $0.5r$ | $0.25$ | $\geq -0.25$ | 温和，允许缓慢接近 |
| $r$ | $0.5$ | $\geq -0.5$ | 适中 |
| $2r$ | $1.0$ | $\geq -1.0$ | 激进，保持距离 |
| $r^2$ | $0.25$ | $\geq -0.25$ | 远离边界时温和 |
| $r^{0.5}$ | $0.71$ | $\geq -0.71$ | 接近边界时更积极 |

### 5.2 仿真对比图

```
b(x)
 ↑
 │    γ=0.5 ─────╲
 │                ╲── γ=1.0
 │                  ╲──── γ=2.0
 │                    ╲──────
 │                      ╲
 ────────────────────────╲───→ t
 0
```

$\gamma$ 越大，$b(x)$ 衰减越快，但安全裕度更大。

---

## 6. 在 HOCBF 中的递推使用

在 [[HOCBF (高阶控制障碍函数)]] 中，Class K 函数被**递归使用**：

$$\psi_0(x) = b(x)$$
$$\psi_1(x) = \dot{\psi}_0(x) + \alpha_1(\psi_0(x))$$
$$\psi_2(x) = \dot{\psi}_1(x) + \alpha_2(\psi_1(x))$$

每个 $\alpha_i$ 可以不同：

- $\alpha_1(r) = \gamma_1 r$：控制 $b(x)$ 接近 0 时的行为
- $\alpha_2(r) = \gamma_2 r$：控制 $\psi_1$ 接近 0 时的行为

**参数选择原则**：
- 确保每个 $\psi_i$ 的集合 $C_i = \{x: \psi_i(x) \geq 0\}$ 是非空的
- $\gamma_i$ 越大 → 响应越快，但控制量越大

---

## 7. 在 dCBF 中的变化

在 [[dCBF (可微控制障碍函数)]] 中，Class K 函数被**可学习的惩罚**替代：

$$\psi_1(x, z) = \dot{b}(x) + p_1(z) \cdot \alpha_1(b(x))$$

其中 $p_1(z) > 0$ 是神经网络输出的**自适应增益**。

- 传统 HOCBF：$\alpha_1$ 是固定的
- dCBF：$p_1(z) \cdot \alpha_1$ 随观测 $z$ 变化 → 自适应保守性

---

## 8. 代码实现

### 8.1 Class K 函数族

```python
import torch
import torch.nn as nn

class ClassKFunction(nn.Module):
    """
    Class K 函数的基类
    
    输入: r (任意形状张量)
    输出: α(r) (同形状)
    """
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class LinearClassK(ClassKFunction):
    """α(r) = γ * r"""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, r):
        return self.gamma * r

class PowerClassK(ClassKFunction):
    """α(r) = sign(r) * |r|^p (扩展到负数)"""
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p
    
    def forward(self, r):
        return r.sign() * r.abs().pow(self.p)

class TanhClassK(ClassKFunction):
    """α(r) = c * tanh(r/c) (有界)"""
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c
    
    def forward(self, r):
        return self.c * torch.tanh(r / self.c)

class SoftplusClassK(ClassKFunction):
    """α(r) = softplus(r) - softplus(0) ≈ r for large r, smooth near 0"""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.offset = torch.nn.functional.softplus(
            torch.tensor(0.0), beta=beta
        ).item()
    
    def forward(self, r):
        return torch.nn.functional.softplus(r, beta=self.beta) - self.offset
```

### 8.2 验证 Class K 性质

```python
def verify_class_k(alpha_fn, r_range=(-5, 5), n_points=1000):
    """
    验证一个函数是否满足 Class K 性质
    
    返回:
        is_class_k: 是否是 Class K
        details: 各项检查结果
    """
    r = torch.linspace(r_range[0], r_range[1], n_points)
    alpha_r = alpha_fn(r)
    
    # 1. α(0) = 0
    alpha_zero = alpha_fn(torch.tensor(0.0)).item()
    origin_ok = abs(alpha_zero) < 1e-6
    
    # 2. 严格递增 (数值检查)
    diffs = alpha_r[1:] - alpha_r[:-1]
    strictly_increasing = (diffs > -1e-6).all().item()
    
    # 3. 连续性 (数值近似)
    second_diffs = torch.abs(alpha_r[2:] - 2*alpha_r[1:-1] + alpha_r[:-2])
    continuous = second_diffs.max().item() < 0.1  # 粗糙检查
    
    is_class_k = origin_ok and strictly_increasing and continuous
    
    return {
        'is_class_k': is_class_k,
        'alpha(0)': alpha_zero,
        'strictly_increasing': strictly_increasing,
        'continuous': continuous
    }

# 测试
linear = LinearClassK(gamma=2.0)
print(verify_class_k(linear.forward))
# {'is_class_k': True, ...}

power = PowerClassK(p=0.5)
print(verify_class_k(power.forward))
# {'is_class_k': True, ...}
```

### 8.3 在 CBF-QP 中使用

```python
def cbf_qp_with_class_k(u_ref, Lf_b, Lg_b, b_val, alpha_fn, u_min=-3, u_max=3):
    """
    带自定义 Class K 函数的 CBF-QP
    
    min ||u - u_ref||^2
    s.t. Lf_b + Lg_b * u + α(b(x)) >= 0
         u_min <= u <= u_max
    
    输入:
        u_ref: 参考控制 (scalar)
        Lf_b, Lg_b: Lie 导数 (scalar)
        b_val: 当前 b(x) 值 (scalar)
        alpha_fn: Class K 函数
    """
    alpha_b = alpha_fn(torch.tensor(b_val)).item()
    
    # 约束: Lg_b * u >= -Lf_b - alpha_b
    # 即: u >= (-Lf_b - alpha_b) / Lg_b (如果 Lg_b > 0)
    # 或: u <= (-Lf_b - alpha_b) / Lg_b (如果 Lg_b < 0)
    
    cbf_bound = (-Lf_b - alpha_b) / (Lg_b + 1e-8)
    
    if Lg_b > 0:
        u_lower = max(u_min, cbf_bound)
        u_star = max(u_lower, min(u_ref, u_max))
    else:
        u_upper = min(u_max, cbf_bound)
        u_star = max(u_min, min(u_ref, u_upper))
    
    return u_star

# 对比不同 α 的效果
alpha_linear = LinearClassK(gamma=1.0)
alpha_aggressive = LinearClassK(gamma=3.0)
alpha_mild = PowerClassK(p=0.5)

u_ref = -2.0  # 驾驶员想加速（减速方向为负）
Lf_b, Lg_b, b_val = -1.0, 1.0, 0.5

print("线性 γ=1:", cbf_qp_with_class_k(u_ref, Lf_b, Lg_b, b_val, alpha_linear))
print("激进 γ=3:", cbf_qp_with_class_k(u_ref, Lf_b, Lg_b, b_val, alpha_aggressive))
print("温和 p=0.5:", cbf_qp_with_class_k(u_ref, Lf_b, Lg_b, b_val, alpha_mild))
```

---

## 9. Class KL 函数（扩展）

**Class $\mathcal{KL}$ 函数** $\beta: [0, a) \times [0, \infty) \to [0, \infty)$：

1. 对第一个参数 $s$：$\beta(\cdot, t) \in \mathcal{K}$
2. 对第二个参数 $t$：$\beta(s, \cdot)$ 递减，且 $\lim_{t \to \infty} \beta(s, t) = 0$

**用途**：描述系统的**渐近稳定性**。

在 CBF 理论中，如果 $b(x(t))$ 满足：

$$b(x(t)) \geq \beta(b(x(0)), t)$$

则 $b$ 不会衰减到 0 以下（因为 $\beta \geq 0$）。

---

## 10. 相关概念

- [[CBF (控制障碍函数)]] — $\alpha$ 的核心应用
- [[HOCBF (高阶控制障碍函数)]] — 递归使用多个 $\alpha_i$
- [[dCBF (可微控制障碍函数)]] — 可学习的自适应 $\alpha$
- [[前向不变性 (Forward Invariance)]] — $\alpha$ 确保安全集合的前向不变性
- [[比较引理 (Comparison Lemma)]] — $\alpha$ 在安全证明中的理论基础

---

> **参考**: 
> - Khalil, "Nonlinear Systems," Prentice Hall 2002, Chapter 4
> - Ames et al., "Control Barrier Functions: Theory and Applications," ECC 2019
