# QP (二次规划, Quadratic Programming)

> **一句话**：QP 是一类优化问题——最小化一个**二次目标函数**，满足**线性约束**。它是 [[CBF (控制障碍函数)]] 和 [[BarrierNet]] 中将安全约束转化为可求解控制问题的核心工具。

---

## 1. 标准形式

$$\min_x \quad \frac{1}{2} x^T P x + q^T x$$

$$\text{s.t.} \quad Gx \leq h$$
$$Ax = b$$

其中：
- $x \in \mathbb{R}^n$：决策变量
- $P \in \mathbb{R}^{n \times n}$：正半定矩阵（$P \succeq 0$）
- $q \in \mathbb{R}^n$：线性项
- $G, h$：不等式约束
- $A, b$：等式约束

**为什么 $P$ 要正半定？** 保证目标函数是**凸的**，存在唯一全局最优解。

---

## 2. 在 CBF 中的应用

[[CBF (控制障碍函数)]] 的安全过滤器就是一个 QP：

$$u^* = \arg\min_u \quad \frac{1}{2} \|u - u_{\text{ref}}\|^2$$

$$\text{s.t.} \quad L_f b + L_g b \cdot u + \alpha(b) \geq 0$$
$$u_{\min} \leq u \leq u_{\max}$$

**转化为标准 QP**：

$P = I$（单位矩阵），$q = -u_{\text{ref}}$

不等式约束：
- CBF：$-L_g b \cdot u \leq L_f b + \alpha(b)$
- 上界：$u \leq u_{\max}$
- 下界：$-u \leq -u_{\min}$

---

## 3. 求解方法

### 3.1 Interior Point Method（内点法）

复杂度：$O(n^3)$（$n$ 为变量维度）

适用于中等规模问题。

### 3.2 Active Set Method（活动集法）

逐步确定哪些约束是"活跃的"（在边界上），在活跃约束的交集上求解。

### 3.3 解析解（1D 情况）

当控制维度为 1（如 AEBS 的加速度）时，QP 可以解析求解：

$u^* = \text{clip}(u_{\text{ref}}, u_{\min}^{\text{cbf}}, u_{\max})$

---

## 4. 代码实现

### 4.1 使用 cvxpy

```python
import cvxpy as cp
import numpy as np

def solve_cbf_qp(u_ref, Lf_b, Lg_b, alpha_b, u_min=-3, u_max=3):
    """
    求解 CBF-QP
    
    min ||u - u_ref||^2
    s.t. Lf_b + Lg_b * u + alpha_b >= 0
         u_min <= u <= u_max
    """
    u = cp.Variable()
    
    objective = cp.Minimize(0.5 * cp.square(u - u_ref))
    
    constraints = [
        Lf_b + Lg_b * u + alpha_b >= 0,
        u >= u_min,
        u <= u_max
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return u.value
```

### 4.2 使用 PyTorch（可微）

```python
import torch
import torch.nn as nn

class DifferentiableQP(nn.Module):
    """
    可微 QP 层（用于 [[BarrierNet]]）
    
    对于 1D 问题，解析解为:
    u* = clip(-F/H, u_min, u_max)
    考虑 CBF 约束后:
    u* = clip(-F/H, max(u_min, u_min_cbf), u_max)
    """
    def forward(self, H, F, G, h, u_min=-3.0, u_max=3.0):
        """
        H: (batch, 1, 1) 正定
        F: (batch, 1) 线性项
        G: (batch, n_constraints, 1) 约束矩阵
        h: (batch, n_constraints) 约束右端
        """
        # 无约束最优
        u_unconstrained = -F / H.squeeze(-1)  # (batch, 1)
        
        # CBF 约束: G*u <= h -> u >= h/G (当 G < 0)
        # 简化: 取所有约束的最大下界
        u_min_constraints = []
        for i in range(G.shape[1]):
            g_i = G[:, i, :]  # (batch, 1)
            h_i = h[:, i:i+1]  # (batch, 1)
            # g_i * u <= h_i
            # 如果 g_i < 0: u >= h_i / g_i
            # 如果 g_i > 0: u <= h_i / g_i
            u_bound = h_i / (g_i + 1e-8)
            u_min_constraints.append(
                torch.where(g_i < 0, u_bound, 
                           torch.tensor(float('-inf')))
            )
        
        if u_min_constraints:
            u_min_cbf = torch.cat(u_min_constraints, dim=1).max(dim=1, keepdim=True)[0]
        else:
            u_min_cbf = torch.tensor(u_min)
        
        # 投影到可行集
        u_lower = torch.max(u_min_cbf, torch.tensor(u_min))
        u_star = torch.clamp(u_unconstrained, min=u_lower, max=u_max)
        
        return u_star
```

---

## 5. QP 的几何直觉

**目标函数** $\frac{1}{2} x^T P x + q^T x$ 是一个椭球（或抛物面）。

**约束** $Gx \leq h$ 定义了一个凸多面体。

**最优解** = 椭球与多面体的**最近接触点**。

- 如果无约束最优在多面体内 → 就是无约束最优
- 如果无约束最优在多面体外 → 在多面体边界上找最近点

---

## 6. 相关概念

- [[可微 QP (Differentiable QP)]] — QP 的可微版本
- [[KKT 条件]] — QP 的最优性条件
- [[CBF (控制障碍函数)]] — QP 的主要应用
- [[BarrierNet]] — 使用可微 QP 的安全框架

---

> **参考**: Boyd & Vandenberghe, "Convex Optimization," Cambridge 2004, Chapter 4
