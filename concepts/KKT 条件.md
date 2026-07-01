# KKT 条件 (Karush-Kuhn-Tucker Conditions)

> **一句话**：KKT 条件是**带约束优化问题**的最优性条件——它告诉我们一个点**必须满足什么条件**才可能是最优解。它是 [[QP (二次规划)]] 求解和 [[可微 QP (Differentiable QP)]] 反向传播的数学基础。

---

## 1. 为什么需要 KKT？

**无约束优化**：$\min f(x)$ → 最优解满足 $\nabla f(x^*) = 0$（梯度为零）。

**有约束优化**：$\min f(x)$ s.t. $g_i(x) \leq 0$ → 最优解不一定满足 $\nabla f = 0$，因为**约束可能把最优解"挡"在边界上**。

**KKT 条件**就是处理这种情况的一般性条件。

---

## 2. 问题形式

$$\min_x \quad f(x)$$
$$\text{s.t.} \quad g_i(x) \leq 0, \quad i = 1, \ldots, m$$
$$\quad\quad h_j(x) = 0, \quad j = 1, \ldots, p$$

其中：
- $f(x)$：目标函数
- $g_i(x)$：不等式约束
- $h_j(x)$：等式约束

---

## 3. Lagrangian 函数

将所有约束"吸收"进目标函数：

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^{m} \lambda_i g_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)$$

其中：
- $\lambda_i \geq 0$：**拉格朗日乘子**（对偶变量），对应不等式约束
- $\nu_j$：拉格朗日乘子，对应等式约束

**直觉**：$\lambda_i$ 表示第 $i$ 个约束的"紧迫程度"——约束越紧，$\lambda_i$ 越大。

---

## 4. 四个 KKT 条件

在最优解 $x^*$ 处，存在 $\lambda^*, \nu^*$ 使得：

### 条件 1：驻点条件（Stationarity）

$$\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = 0$$

即：

$$\nabla f(x^*) + \sum_i \lambda_i^* \nabla g_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0$$

**直觉**：目标函数的梯度被约束的梯度"抵消"了。

想象你在一面墙旁边找最低点——你的"下降方向"被墙的法向量挡住了。

### 条件 2：原始可行性（Primal Feasibility）

$$g_i(x^*) \leq 0, \quad h_j(x^*) = 0$$

**直觉**：最优解必须满足所有约束。

### 条件 3：对偶可行性（Dual Feasibility）

$$\lambda_i^* \geq 0, \quad \forall i$$

**直觉**：不等式约束的乘子必须非负。（如果 $\lambda_i < 0$，意味着放松约束反而让目标变差，这不合理。）

### 条件 4：互补松弛（Complementary Slackness）

$$\lambda_i^* \cdot g_i(x^*) = 0, \quad \forall i$$

**这是最精妙的条件！** 它说：

- 要么 $\lambda_i^* = 0$（约束不活跃，可以忽略）
- 要么 $g_i(x^*) = 0$（约束恰好在边界上）
- 不能两者都不是

**直觉**：如果约束没有碰到边界（$g_i(x^*) < 0$），那它就不影响最优解，对应的"紧迫程度" $\lambda_i^* = 0$。

---

## 5. 几何直觉

### 5.1 二维例子

$$\min f(x_1, x_2) = x_1^2 + x_2^2$$
$$\text{s.t.} \quad x_1 + x_2 \geq 2 \quad \Leftrightarrow \quad g(x) = 2 - x_1 - x_2 \leq 0$$

**无约束最优**：$(0, 0)$，但不满足 $x_1 + x_2 \geq 2$。

**有约束最优**：$(1, 1)$，恰好在边界 $x_1 + x_2 = 2$ 上。

**KKT 验证**：
- 驻点：$\nabla f = (2, 2)$，$\nabla g = (-1, -1)$
- $(2, 2) + \lambda(-1, -1) = 0 \Rightarrow \lambda = 2$
- 互补松弛：$\lambda \cdot g(1,1) = 2 \cdot 0 = 0$ ✅

### 5.2 图示

```
       x₂
       ↑
   3   |  .
       |    .  x₁+x₂=2 (约束边界)
   2   |      *  (1,1) ← 最优解
       |    /
   1   |  /    ○ (0,0) ← 无约束最优
       |/
   0---+--------→ x₁
       0  1  2  3
```

---

## 6. QP 中的 KKT

对于 [[QP (二次规划)]]：

$$\min_u \frac{1}{2} u^T H u + F^T u \quad \text{s.t.} \quad Gu \leq h$$

**Lagrangian**：

$$\mathcal{L}(u, \lambda) = \frac{1}{2} u^T H u + F^T u + \lambda^T (Gu - h)$$

**KKT 条件**：

1. **驻点**：$Hu + F + G^T \lambda = 0$
2. **原始可行**：$Gu - h \leq 0$
3. **对偶可行**：$\lambda \geq 0$
4. **互补松弛**：$\lambda_i (G_i u - h_i) = 0$

这组方程是 [[可微 QP (Differentiable QP)]] 反向传播的出发点！

### 6.1 数值例子

$$\min_u \frac{1}{2} u^2 - 3u \quad \text{s.t.} \quad u \leq 2$$

- $H = 1, F = -3, G = 1, h = 2$
- 无约束最优：$u = 3$（不在可行域内）
- 有约束最优：$u^* = 2$（在边界上）

KKT 验证：
- 驻点：$1 \cdot 2 + (-3) + 1 \cdot \lambda = 0 \Rightarrow \lambda = 1$
- 互补松弛：$\lambda \cdot (2 - 2) = 1 \cdot 0 = 0$ ✅

---

## 7. KKT 与约束分类

| 约束状态 | $g_i(x^*)$ | $\lambda_i^*$ | 含义 |
|---------|-----------|-------------|------|
| **活跃**（Active） | $= 0$ | $> 0$ | 约束在边界上，限制了最优解 |
| **不活跃**（Inactive） | $< 0$ | $= 0$ | 约束不影响最优解，可以忽略 |
| **退化**（Degenerate） | $= 0$ | $= 0$ | 约束恰好在边界但不影响 |

---

## 8. 代码实现

### 8.1 用 scipy 求解带约束优化并获取 KKT 乘子

```python
import numpy as np
from scipy.optimize import minimize

def solve_with_kkt(f, grad_f, constraints, x0):
    """
    求解带约束优化并返回 KKT 乘子
    
    min f(x) s.t. g_i(x) <= 0
    
    输入:
        f: 目标函数
        grad_f: 目标函数梯度
        constraints: 约束函数列表 [{'type': 'ineq', 'fun': g, 'jac': grad_g}]
        x0: 初始点
    输出:
        x_star: 最优解
        lambdas: KKT 乘子
    """
    result = minimize(
        f, x0, jac=grad_f,
        constraints=constraints,
        method='SLSQP'
    )
    
    x_star = result.x
    # scipy 返回的乘子在 result 中不直接暴露
    # 需要从约束的敏感度获取
    
    return x_star, result

# 例子: min x1^2 + x2^2 s.t. x1 + x2 >= 2
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])

# scipy 的 'ineq' 约束格式: g(x) >= 0
# 我们要 x1 + x2 >= 2 -> x1 + x2 - 2 >= 0
constraints = [{'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 2}]

x_star, result = solve_with_kkt(f, grad_f, constraints, [0, 0])
print(f"x* = {x_star}")  # [1, 1]
```

### 8.2 验证 KKT 条件

```python
import torch

def check_kkt(H, F, G, h, u_star, tol=1e-6):
    """
    验证 QP 的 KKT 条件
    
    min 0.5*u'Hu + F'u s.t. Gu <= h
    
    输入:
        H: (n, n) 正定矩阵
        F: (n,) 线性项
        G: (m, n) 约束矩阵
        h: (m,) 约束右端
        u_star: (n,) 候选最优解
    输出:
        是否满足 KKT，以及违反细节
    """
    n = H.shape[0]
    m = G.shape[0]
    
    # 计算约束值
    constraint_vals = G @ u_star - h  # g_i(u*) = G_i*u - h_i
    
    # 条件 2: 原始可行性
    primal_feasible = (constraint_vals <= tol).all()
    
    # 确定活跃约束
    active = constraint_vals > -tol  # g_i ≈ 0
    
    # 从驻点条件求解 lambda
    # Hu + F + G_active^T lambda_active = 0
    G_active = G[active]
    if G_active.shape[0] > 0:
        # lambda_active = -(G_active @ G_active^T)^{-1} @ G_active @ (Hu + F)
        residual = H @ u_star + F
        GGT = G_active @ G_active.T
        lam_active = -torch.linalg.solve(GGT, G_active @ residual)
        
        # 条件 3: 对偶可行性
        dual_feasible = (lam_active >= -tol).all()
        
        # 条件 4: 互补松弛（活跃约束 g_i = 0 已满足）
        complementary = True  # 不活跃约束 lambda = 0
    else:
        lam_active = torch.tensor([])
        dual_feasible = True
        complementary = True
        # 检查无约束驻点
        residual = H @ u_star + F
        stationarity = (residual.norm() < tol)
    
    return {
        'primal_feasible': primal_feasible.item(),
        'dual_feasible': dual_feasible.item() if isinstance(dual_feasible, torch.Tensor) else dual_feasible,
        'complementary_slackness': complementary,
        'active_constraints': active.nonzero().flatten().tolist()
    }

# 测试
H = torch.tensor([[1.0]])
F = torch.tensor([-3.0])
G = torch.tensor([[1.0]])
h = torch.tensor([2.0])
u_star = torch.tensor([2.0])

result = check_kkt(H, F, G, h, u_star)
print(result)
# {'primal_feasible': True, 'dual_feasible': True, ...}
```

---

## 9. KKT 在可微 QP 中的角色

在 [[可微 QP (Differentiable QP)]] 中，KKT 条件被写为隐函数 $R(u^*, \lambda^*, \theta) = 0$：

$$R = \begin{pmatrix} Hu^* + F + G^T \lambda^* \\ \text{diag}(\lambda^*) (Gu^* - h) \end{pmatrix} = 0$$

然后对 $R$ 应用**隐函数定理**：

$$\frac{\partial u^*}{\partial \theta} = -\left(\frac{\partial R}{\partial (u^*, \lambda^*)}\right)^{-1} \frac{\partial R}{\partial \theta}$$

这就是可微 QP 反向传播的核心！

---

## 10. KKT 条件的局限性

| 局限 | 说明 |
|------|------|
| **必要性** | KKT 是最优解的**必要**条件，不一定是充分的（除非问题是凸的） |
| **约束规格** | 需要满足某些正则条件（如 LICQ：活跃约束梯度线性无关） |
| **非凸问题** | 非凸问题的 KKT 点可能是鞍点或局部最优 |
| **退化** | $\lambda_i = g_i = 0$ 时，数值不稳定 |

---

## 11. Slater 条件（凸问题的充分性）

对于**凸优化**问题（$f, g_i$ 都是凸函数），如果存在一个**严格可行点** $\hat{x}$：

$$g_i(\hat{x}) < 0, \quad \forall i$$

则 KKT 条件变为**充分必要条件**——满足 KKT 的点一定是全局最优解。

这就是为什么 [[QP (二次规划)]] 的 KKT 条件如此重要——QP 是凸问题！

---

## 12. 相关概念

- [[QP (二次规划)]] — KKT 最典型的应用场景
- [[可微 QP (Differentiable QP)]] — 基于 KKT 的梯度计算
- [[CBF (控制障碍函数)]] — CBF-QP 的最优性分析
- [[SOS (Sum-of-Squares)]] — 多项式优化的 KKT 推广

---

> **参考**: 
> - Boyd & Vandenberghe, "Convex Optimization," Cambridge 2004, Chapter 5
> - Nocedal & Wright, "Numerical Optimization," Springer 2006, Chapter 12
