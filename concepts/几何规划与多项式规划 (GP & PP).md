# 几何规划与多项式规划 (Geometric & Polynomial Programming)

> **一句话**：几何规划（GP）和多项式规划（PP）是两类**特殊的优化问题**——GP 处理正变量的幂函数（单项式/多项式），PP 处理一般多项式。在安全控制中，它们用于 CBF 参数优化、控制器综合等。

---

## 1. 几何规划 (Geometric Programming)

### 1.1 单项式 (Monomial)

$$m(x) = c \cdot x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n}$$

其中 $c > 0$，$a_i \in \mathbb{R}$（指数可以是任意实数）。

**例子**：$3x^{0.5} y^{-1}$，$2x_1^2 x_2^{1.5}$

### 1.2 正项式 (Posynomial)

正项式是**单项式的和**：

$$p(x) = \sum_{k=1}^{K} c_k x_1^{a_{1k}} x_2^{a_{2k}} \cdots x_n^{a_{nk}}$$

**例子**：$2x^{0.5}y + 3x^{-1}y^2$

### 1.3 GP 标准形式

$$\min_x \quad p_0(x)$$
$$\text{s.t.} \quad p_i(x) \leq 1, \quad i = 1, \ldots, m$$
$$\quad\quad m_j(x) = 1, \quad j = 1, \ldots, p$$

其中 $p_i$ 是正项式，$m_j$ 是单项式，$x_i > 0$。

### 1.4 GP → 凸优化

**关键**：通过变量替换 $y_i = \log x_i$，GP 可以转化为**凸优化问题**！

$$\min_y \quad \log \sum_k \exp(a_k^T y + b_k)$$

这是一个 **log-sum-exp** 函数（凸函数）。

---

## 2. 多项式规划 (Polynomial Programming)

### 2.1 标准形式

$$\min_x \quad p_0(x)$$
$$\text{s.t.} \quad p_i(x) \leq 0, \quad i = 1, \ldots, m$$

其中 $p_i$ 是**多项式**。

### 2.2 与 SOS 的关系

多项式规划通常通过 [[SOS (Sum-of-Squares)]] 松弛来求解：

1. 将多项式约束转化为 SOS 约束
2. 用 SDP 求解

---

## 3. 在安全控制中的应用

### 3.1 GP 用于控制器参数优化

假设控制律为 $u = k_1 d^{a_1} v^{a_2}$（幂函数形式），优化参数使系统稳定：

$$\min_{k_1, a_1, a_2} \quad \int_0^T (d - d_{\text{ref}})^2 dt$$

这是一个 GP 问题（如果目标可以写成幂函数形式）。

### 3.2 多项式 CBF 参数优化

给定多项式 CBF $b(x) = x^T P x + q^T x + r$，优化 $P, q, r$ 使安全裕度最大：

$$\max_{P, q, r} \quad \min_{x \in S} b(x)$$

用 [[SOS (Sum-of-Squares)]] 松弛为 SDP。

---

## 4. 代码实现

### 4.1 GP 求解 (cvxpy)

```python
import cvxpy as cp
import numpy as np

def solve_gp_example():
    """
    GP 示例:
    
    min  x^0.5 * y^(-1) + 2*x^(-0.5) * y^0.5
    s.t. x * y >= 1  (即 (xy)^(-1) <= 1)
         x, y > 0
    """
    # cvxpy 的 GP 模式
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    
    # 目标函数 (正项式)
    objective = cp.Minimize(
        x**0.5 * y**(-1) + 2 * x**(-0.5) * y**0.5
    )
    
    # 约束
    constraints = [
        (x * y)**(-1) <= 1  # x*y >= 1
    ]
    
    # 求解 (指定 gp=True)
    prob = cp.Problem(objective, constraints)
    prob.solve(gp=True)
    
    print(f"最优值: {prob.value:.4f}")
    print(f"x = {x.value:.4f}, y = {y.value:.4f}")
    
    return x.value, y.value

solve_gp_example()
```

### 4.2 GP 用于 AEBS 控制器参数

```python
def gp_aebs_controller_optimization():
    """
    用 GP 优化 AEBS 控制器参数
    
    控制律: u = -k1 * (d - d_safe)^a1 * v^a2
    目标: 最小化制动能量
    约束: 安全距离
    """
    k1 = cp.Variable(pos=True)
    a1 = cp.Variable()
    a2 = cp.Variable()
    
    # 简化: 固定 a1, a2 优化 k1
    # (完整 GP 需要更复杂的建模)
    
    # 假设已知安全条件要求 k1 足够大
    # k1 * d_min^a1 * v_max^a2 >= u_max
    d_min, v_max, u_max = 5.0, 25.0, 3.0
    
    constraints = [
        k1 >= u_max / (d_min ** 1.0 * v_max ** 0.5)
    ]
    
    objective = cp.Minimize(k1)
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    print(f"最优 k1 = {k1.value:.4f}")
    return k1.value
```

### 4.3 多项式规划 (通过 SOS 松弛)

```python
import cvxpy as cp
import numpy as np

def polynomial_programming_sos():
    """
    多项式规划示例 (通过 SOS)
    
    min x^2 + y^2
    s.t. x^4 + y^4 <= 1
    
    等价于找:
    min t
    s.t. t - (x^2 + y^2) >= 0 在 x^4 + y^4 <= 1 上
    即: t - x^2 - y^2 - λ(x)(1 - x^4 - y^4) 是 SOS
    """
    # 简化: 直接求解
    x = cp.Variable()
    y = cp.Variable()
    
    objective = cp.Minimize(x**2 + y**2)
    constraints = [x**4 + y**4 <= 1]
    
    # 注意: cvxpy 不直接支持非凸多项式约束
    # 需要 SOS 松弛或其他方法
    
    # 简化: 利用对称性 x = y
    # 2x^4 <= 1 => x <= (1/2)^(1/4)
    x_opt = (0.5) ** 0.25
    print(f"最优解: x = y = {x_opt:.4f}")
    print(f"最优值: {2 * x_opt**2:.4f}")
```

---

## 5. GP vs QP vs SDP vs PP

| 类型 | 目标 | 约束 | 凸? | 工具 |
|------|------|------|-----|------|
| **QP** | 二次 | 线性 | ✅ | cvxpy |
| **SDP** | 线性 | LMI | ✅ | cvxpy, MOSEK |
| **GP** | 正项式 | 正项式 ≤ 1 | ✅ (变换后) | cvxpy(gp=True) |
| **PP** | 多项式 | 多项式 ≤ 0 | ❌ (一般) | SOS 松弛 |

---

## 6. 相关概念

- [[SOS (Sum-of-Squares)]] — 多项式规划的求解方法
- [[QP (二次规划)]] — GP/QP/SDP 的关系
- [[KKT 条件]] — 所有优化问题的最优性条件
- [[CBF (控制障碍函数)]] — GP 优化 CBF 参数

---

> **参考**: 
> - Boyd et al., "A Tutorial on Geometric Programming," Optimization and Engineering 2007
> - Lasserre, "Global Optimization with Polynomials and the Problem of Moments," SIAM 2001
