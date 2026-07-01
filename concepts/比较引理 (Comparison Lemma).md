# 比较引理 (Comparison Lemma)

> **一句话**：比较引理是微分方程理论中的一个基本工具——它将一个**复杂的微分不等式**与一个**简单的微分方程**进行比较，从而用简单方程的解来约束复杂系统的行为。它是 [[CBF (控制障碍函数)]] 安全性证明的核心数学工具。

---

## 1. 为什么需要比较引理？

在 [[CBF (控制障碍函数)]] 中，我们得到：

$$\dot{b}(x(t)) \geq -\alpha(b(x(t)))$$

这是一个**微分不等式**——我们不知道 $b(x(t))$ 的精确值，但知道它的变化率有一个下界。

**问题**：如何从 $\dot{b} \geq -\alpha(b)$ 推出 $b(t) \geq 0$（即安全性）？

**比较引理的回答**：将 $b(t)$ 与方程 $\dot{y} = -\alpha(y)$ 的解 $y(t)$ 比较。因为 $y(t) \geq 0$（容易证明），且 $b(t) \geq y(t)$（比较引理），所以 $b(t) \geq 0$。

---

## 2. 定理陈述

### 2.1 标量比较引理

考虑：

$$\dot{x} = f(t, x), \quad x(t_0) = x_0$$

和比较系统：

$$\dot{y} = g(t, y), \quad y(t_0) = y_0$$

**条件**：
1. $f(t, x) \leq g(t, x)$（$f$ 被 $g$ 控制）
2. $x_0 \leq y_0$（初始条件）
3. $g(t, y)$ 关于 $y$ 满足 Lipschitz 条件

**结论**：

$$x(t) \leq y(t), \quad \forall t \geq t_0$$

**直觉**：如果 $x$ 的"速度"总是小于 $y$ 的"速度"，且起点也更低，那 $x$ 永远追不上 $y$。

### 2.2 更常用的形式

如果：

$$\dot{x}(t) \leq -\alpha(x(t)), \quad x(0) = x_0$$

其中 $\alpha \in \mathcal{K}$（[[Class K 函数]]），则：

$$x(t) \leq \sigma(x_0, t)$$

其中 $\sigma(x_0, t)$ 是 $\dot{y} = -\alpha(y)$ 的解，且 $\sigma \in \mathcal{KL}$。

---

## 3. 在 CBF 安全证明中的应用

### 3.1 完整的证明链

**步骤 1**：CBF 条件给出：

$$\dot{b}(x(t)) \geq -\alpha(b(x(t)))$$

**步骤 2**：定义比较系统 $\dot{y} = -\alpha(y)$，$y(0) = b(x(0))$

**步骤 3**：由比较引理（注意不等号方向反转）：

$$b(x(t)) \geq y(t), \quad \forall t \geq 0$$

**步骤 4**：分析比较系统 $\dot{y} = -\alpha(y)$：

- 如果 $y(0) \geq 0$，则 $y(t) \geq 0$（因为 $y = 0$ 时 $\dot{y} = -\alpha(0) = 0$，不会再降）
- $y(t)$ 递减但有下界 0

**步骤 5**：因此 $b(x(t)) \geq y(t) \geq 0$，即 [[前向不变性 (Forward Invariance)]] 成立。

### 3.2 图示

```
b(t)
 ↑
 │ ● b(0) = y(0)
 │  ╲
 │   ╲  ← y(t) (比较系统的解)
 │    ╲
 │     ╲
 │      ● b(t) (实际值，总在 y(t) 上方)
 │       ╲
 │        ╲
 │─────────●──────────── y(t) → 0
 └──────────────────────→ t
 0
```

---

## 4. 详细数值例子

### 4.1 简单情形：$\alpha(r) = \gamma r$

比较系统：$\dot{y} = -\gamma y$，解为 $y(t) = y_0 e^{-\gamma t}$

CBF 不等式：$\dot{b} \geq -\gamma b$

比较引理给出：$b(t) \geq b(0) e^{-\gamma t}$

**数值**：$b(0) = 5, \gamma = 0.5$

| $t$ | $y(t) = 5 e^{-0.5t}$ | $b(t) \geq$ |
|-----|---------------------|-------------|
| 0 | 5.00 | 5.00 |
| 1 | 3.03 | 3.03 |
| 2 | 1.84 | 1.84 |
| 5 | 0.41 | 0.41 |
| 10 | 0.03 | 0.03 |
| ∞ | 0 | 0 |

$b(t)$ 指数衰减到 0，但**永远不会为负**。

### 4.2 非线性情形：$\alpha(r) = r^2$

比较系统：$\dot{y} = -y^2$

解：$y(t) = \frac{y_0}{1 + y_0 t}$

**数值**：$y_0 = 5$

| $t$ | $y(t) = \frac{5}{1 + 5t}$ |
|-----|--------------------------|
| 0 | 5.00 |
| 0.1 | 3.33 |
| 0.5 | 1.43 |
| 1 | 0.83 |
| 10 | 0.10 |

衰减速度比指数更慢（多项式衰减），但同样永远非负。

### 4.3 代码验证

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def verify_comparison_lemma(b0=5.0, gamma=0.5, T=10.0, dt=0.01):
    """
    验证比较引理:
    
    比较系统: ẏ = -γy, y(0) = b0
    实际系统: ḃ = -γb + δ(t), b(0) = b0
    
    其中 δ(t) >= 0 是 CBF 控制器提供的额外安全裕度
    """
    t = np.arange(0, T, dt)
    
    # 比较系统: ẏ = -γy
    y = odeint(lambda y, t: -gamma * y, b0, t).flatten()
    
    # 模拟实际系统 (假设额外裕度 δ = 0.5*sin(t)^2)
    b = np.zeros_like(t)
    b[0] = b0
    for i in range(len(t) - 1):
        delta = 0.5 * np.sin(t[i])**2  # 非负扰动
        b_dot = -gamma * b[i] + delta
        b[i + 1] = b[i] + b_dot * dt
    
    # 验证 b(t) >= y(t)
    assert np.all(b >= y - 1e-6), "比较引理被违反!"
    
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, 'b--', label='y(t) = b0·exp(-γt) [comparison system]')
    plt.plot(t, b, 'r-', label='b(t) [actual system with CBF]')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Comparison Lemma: b(t) ≥ y(t) ≥ 0')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_lemma_demo.png')
    
    return t, y, b

t, y, b = verify_comparison_lemma()
print(f"min(b - y) = {(b - y).min():.6f}")  # 应 >= 0
print(f"min(b) = {b.min():.6f}")  # 应 >= 0
```

---

## 5. 比较引理的不同版本

### 5.1 标准标量版本

$$\dot{x} \leq f(t, x), \quad x(t_0) = x_0$$
$$\dot{y} = f(t, y), \quad y(t_0) = x_0$$
$$\Rightarrow x(t) \leq y(t)$$

### 5.2 向量版本

$$\dot{x} \leq f(t, x) \quad (\text{分量不等式})$$

需要 $f$ 是 **quasi-monotone**（拟单调）的：

$$x_i \leq y_i, \, x_j = y_j \, (j \neq i) \implies f_i(t, x) \leq f_i(t, y)$$

### 5.3 离散时间版本

$$x_{k+1} \leq f(x_k), \quad y_{k+1} = f(y_k)$$
$$x_0 \leq y_0 \implies x_k \leq y_k$$

---

## 6. 在 dCBF 安全性证明中的应用

在 [[dCBF (可微控制障碍函数)]] 中，安全性证明更复杂：

$$\dot{\psi}_0 \geq -p_1(z) \alpha_1(\psi_0)$$

其中 $p_1(z) > 0$ 是可学习的。

**问题**：$p_1(z)$ 随时间变化，标准比较引理不能直接用。

**解决**：利用 $p_1(z) \geq p_{\min} > 0$（网络输出的正下界）：

$$\dot{\psi}_0 \geq -p_{\max} \alpha_1(\psi_0)$$

然后用比较系统 $\dot{y} = -p_{\min} \alpha_1(y)$ 来 bound $\psi_0(t)$。

---

## 7. 与 Gronwall 不等式的关系

**Gronwall 不等式**是比较引理的一个特例：

如果：

$$u(t) \leq a(t) + \int_0^t b(s) u(s) ds$$

则：

$$u(t) \leq a(t) \exp\left(\int_0^t b(s) ds\right)$$

比较引理可以看作是 Gronwall 不等式的**非线性推广**。

---

## 8. 代码：用比较引理做安全验证

```python
import torch

def safety_guarantee_via_comparison(
    b_values,        # (T,) b(x(t)) 的时间序列
    alpha_fn,        # Class K 函数
    dt=0.01,
    tol=1e-4
):
    """
    用比较引理验证安全性
    
    检查: ḃ(t) >= -α(b(t)) 是否在所有时间点成立
    
    输入:
        b_values: CBF 值的时间序列
        alpha_fn: Class K 函数 (callable)
        dt: 时间步长
    输出:
        is_safe: 是否满足 CBF 条件
        comparison_bound: 比较系统的解 y(t)
        violations: 违反的时间点
    """
    T = len(b_values)
    
    # 计算 ḃ (数值微分)
    b_dot = torch.zeros(T - 1)
    for i in range(T - 1):
        b_dot[i] = (b_values[i + 1] - b_values[i]) / dt
    
    # 计算 -α(b)
    alpha_b = torch.zeros(T - 1)
    for i in range(T - 1):
        alpha_b[i] = -alpha_fn(b_values[i])
    
    # 检查 ḃ >= -α(b)
    violations = []
    for i in range(T - 1):
        if b_dot[i] < alpha_b[i] - tol:
            violations.append({
                'time': i * dt,
                'b': b_values[i].item(),
                'b_dot': b_dot[i].item(),
                '-alpha(b)': alpha_b[i].item(),
                'gap': (b_dot[i] - alpha_b[i]).item()
            })
    
    # 计算比较系统的解
    y = torch.zeros(T)
    y[0] = b_values[0]
    for i in range(T - 1):
        y[i + 1] = y[i] - alpha_fn(y[i]) * dt
    
    is_safe = len(violations) == 0
    
    return {
        'is_safe': is_safe,
        'n_violations': len(violations),
        'comparison_bound': y,
        'min_b': b_values.min().item(),
        'min_y': y.min().item()
    }

# 测试
class LinearClassK:
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, r):
        return self.gamma * r

# 模拟一个安全的 b(t) 序列
gamma = 0.5
T_steps = 1000
dt = 0.01
b = torch.zeros(T_steps)
b[0] = 5.0
for i in range(T_steps - 1):
    b[i + 1] = b[i] * (1 - gamma * dt)  # b(t) = b0 * exp(-γt)

result = safety_guarantee_via_comparison(b, LinearClassK(gamma), dt)
print(f"安全: {result['is_safe']}")
print(f"最小 b: {result['min_b']:.4f}")
print(f"最小比较界: {result['min_y']:.4f}")
```

---

## 9. 常见误区

| 误区 | 正确理解 |
|------|---------|
| "比较引理给出精确解" | 不，只给出上/下界 |
| "$\dot{b} \geq -\alpha(b)$ 意味着 $b$ 递减" | 不一定，$\dot{b}$ 可以为正（远离边界） |
| "比较引理只用于 CBF" | 广泛用于非线性系统稳定性分析 |
| "需要全局 Lipschitz" | 只需要局部 Lipschitz 即可（在解存在的区间内） |

---

## 10. 相关概念

- [[CBF (控制障碍函数)]] — 比较引理的主要应用
- [[dCBF (可微控制障碍函数)]] — 扩展版本的安全性证明
- [[Class K 函数]] — 比较系统的核心组件
- [[前向不变性 (Forward Invariance)]] — 比较引理证明的目标性质
- [[鞅理论 (Martingale Theory)]] — 随机版本的"比较引理"（Doob 不等式）

---

> **参考**: 
> - Khalil, "Nonlinear Systems," Prentice Hall 2002, Lemma 3.4
> - Lakshmikantham & Leela, "Differential and Integral Inequalities," Academic Press 1969
