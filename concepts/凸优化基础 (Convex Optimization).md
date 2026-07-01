# 凸优化基础 (Convex Optimization Basics)

> **一句话**：凸优化是一类**目标函数和约束都是凸的**优化问题——它保证局部最优就是全局最优，并且有高效算法求解。[[QP (二次规划)]]、SDP、LP 都是凸优化的特例。在安全控制中，几乎所有优化子问题（CBF-QP、SOS、对偶问题）都是凸的。

---

## 1. 核心概念

### 1.1 凸集

集合 $C$ 是凸的，如果任意两点的连线都在 $C$ 中：

$$x, y \in C, \, \theta \in [0, 1] \implies \theta x + (1-\theta) y \in C$$

```
凸集:                非凸集:
  ○○○○                 ○  ○
 ○○○○○○               ○    ○
 ○○○○○○                ○○ ○
  ○○○○                  ○○
```

**常见凸集**：
- 仿射子空间 $\{x : Ax = b\}$
- 半空间 $\{x : a^T x \leq b\}$
- 球 $\{x : \|x\| \leq r\}$
- 椭球 $\{x : x^T P x \leq 1, P \succ 0\}$
- 多面体 $\{x : Ax \leq b\}$
- 半正定锥 $\{X : X \succeq 0\}$

### 1.2 凸函数

函数 $f$ 是凸的，如果：

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

```
凸函数:              非凸函数:
    ╲                ╲
     ╲               ╲   ╱
      ╲___            ╲_╱
         ___             ╲
     ___╱                ╱
```

**常见凸函数**：
- 线性：$f(x) = a^T x + b$
- 二次（$P \succeq 0$）：$f(x) = x^T P x$
- 范数：$\|x\|_p$
- 指数：$e^x$
- $-\log x$
- $\max(f_1, f_2)$（逐点最大）

### 1.3 凸优化问题

$$\min_x \quad f(x)$$
$$\text{s.t.} \quad g_i(x) \leq 0 \quad (g_i \text{ 凸})$$
$$\quad\quad Ax = b$$

**关键性质**：**局部最优 = 全局最优**。

---

## 2. 凸优化问题的层次

```
              凸优化
                │
    ┌───────┬──┴──┬────────┐
    │       │     │        │
   LP      QP    SDP      GP
  线性    二次   半定     几何
  规划    规划   规划     规划
```

| 类型 | 目标 | 约束 | 复杂度 | 工具 |
|------|------|------|--------|------|
| **LP** | $c^T x$ | $Ax \leq b$ | $O(n^3)$ | scipy, GLPK |
| **QP** | $\frac{1}{2}x^TPx + q^Tx$ | $Gx \leq h$ | $O(n^3)$ | cvxpy, OSQP |
| **SOCP** | $\|A_ix + b_i\| \leq c_i^T x + d_i$ | 二阶锥 | $O(n^3)$ | cvxpy, MOSEK |
| **SDP** | $\text{tr}(CX)$ | $X \succeq 0$, LMI | $O(n^6)$ | cvxpy, MOSEK |
| **GP** | 正项式 | 正项式 $\leq 1$ | $O(n^3)$ | cvxpy |

**包含关系**：LP $\subset$ QP $\subset$ SOCP $\subset$ SDP

---

## 3. 在安全控制中的应用

### 3.1 CBF-QP → QP

$$\min_u \frac{1}{2}\|u - u_{\text{ref}}\|^2 \quad \text{s.t.} \quad Au \leq b$$

这是标准 [[QP (二次规划)]]。

### 3.2 SOS → SDP

搜索 [[SOS (Sum-of-Squares)]] 多项式等价于找半正定矩阵 $Q \succeq 0$。

### 3.3 Lyapunov LMI → SDP

找 $P \succ 0$ 使得 $A^T P + PA \prec 0$ 是一个 SDP。

### 3.4 鲁棒优化 → SOCP

如果控制输入有不确定性：$\|u + \delta\| \leq u_{\max}$，$\|\delta\| \leq \epsilon$：

$$\min_u \|u - u_{\text{ref}}\|^2 \quad \text{s.t.} \quad \|Au - b\|_2 \leq c$$

这是 SOCP（二阶锥规划）。

---

## 4. 代码实现

### 4.1 cvxpy 统一接口

```python
import cvxpy as cp
import numpy as np

# LP: min c'x s.t. Ax <= b
def solve_lp():
    x = cp.Variable(2)
    c = np.array([1, 2])
    A = np.array([[1, 1], [-1, 1]])
    b = np.array([4, 2])
    
    prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= 0])
    prob.solve()
    return x.value

# QP: min 0.5 x'Px + q'x s.t. Gx <= h
def solve_qp():
    x = cp.Variable(2)
    P = np.array([[2, 0], [0, 2]])
    q = np.array([-4, -6])
    G = np.array([[1, 1], [-1, 2], [2, 1]])
    h = np.array([4, 6, 8])
    
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
        [G @ x <= h]
    )
    prob.solve()
    return x.value

# SDP: min tr(CX) s.t. tr(A_i X) = b_i, X >= 0
def solve_sdp():
    n = 3
    X = cp.Variable((n, n), symmetric=True)
    C = np.eye(n)
    
    constraints = [
        X >> 0,  # 半正定
        cp.trace(X) == 1
    ]
    
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve()
    return X.value

# SOCP
def solve_socp():
    x = cp.Variable(2)
    
    prob = cp.Problem(
        cp.Minimize(cp.norm(x - np.array([1, 2]))),
        [cp.norm(x) <= 3]
    )
    prob.solve()
    return x.value

print(f"LP: {solve_lp()}")
print(f"QP: {solve_qp()}")
print(f"SDP: {solve_sdp()}")
print(f"SOCP: {solve_socp()}")
```

### 4.2 CBF-QP 求解器

```python
def cbf_qp_solver(u_ref, Lf_b, Lg_b, b_val, gamma=1.0, 
                   u_min=-3.0, u_max=3.0):
    """
    用 cvxpy 求解 CBF-QP
    
    min ||u - u_ref||^2
    s.t. Lf_b + Lg_b * u + gamma * b >= 0
         u_min <= u <= u_max
    """
    import cvxpy as cp
    
    u = cp.Variable()
    
    objective = cp.Minimize(0.5 * cp.square(u - u_ref))
    
    constraints = [
        Lf_b + Lg_b * u + gamma * b_val >= 0,
        u >= u_min,
        u <= u_max
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    
    return u.value, prob.status

# 测试
u_ref = -2.0  # 驾驶员想加速
Lf_b = -1.0   # 自然演化（距离减小）
Lg_b = 1.0    # 控制可以改变
b_val = 0.5   # 接近安全边界

u_star, status = cbf_qp_solver(u_ref, Lf_b, Lg_b, b_val)
print(f"u* = {u_star:.4f} (status: {status})")
```

---

## 5. 求解器选择指南

| 问题类型 | 推荐求解器 | 特点 |
|---------|----------|------|
| LP | GLPK, Gurobi | 成熟 |
| QP | OSQP, Gurobi | OSQP 开源 |
| SOCP | MOSEK, ECOS | MOSEK 商用 |
| SDP | MOSEK, SCS | SCS 开源 |
| GP | SCS (cvxpy) | 自动转换 |
| 非凸 | IPOPT, SNOPT | 局部最优 |

---

## 6. 凸优化在端到端训练中的角色

在 [[BarrierNet]] 和 [[可微 QP (Differentiable QP)]] 中：

```
前向传播:  求解凸优化 → 得到最优解 u*
反向传播:  利用 [[KKT 条件]] → 计算梯度 → 更新网络参数
```

凸优化的关键作用：
1. **保证唯一解**：凸问题的最优解唯一（或凸集）
2. **可微性**：最优解关于参数可微（在约束规格下）
3. **高效求解**：多项式时间算法

---

## 7. 相关概念

- [[QP (二次规划)]] — 凸优化的核心实例
- [[KKT 条件]] — 凸优化的最优性条件
- [[对偶理论与拉格朗日松弛]] — 凸优化的对偶
- [[SOS (Sum-of-Squares)]] — SDP 在安全验证中的应用
- [[可微 QP (Differentiable QP)]] — 凸优化的可微版本
- [[CBF (控制障碍函数)]] — 凸优化在安全控制中的应用

---

> **参考**: 
> - Boyd & Vandenberghe, "Convex Optimization," Cambridge 2004
> - Bertsekas, "Convex Optimization Theory," Athena 2009
