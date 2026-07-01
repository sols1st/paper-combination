# SOS (Sum-of-Squares, 平方和规划)

> **一句话**：SOS 是一种**多项式优化技术**——它将"多项式非负"这个难以验证的条件，转化为"多项式可以写成平方和"这个可以用半定规划（SDP）求解的条件。在安全验证中，SOS 用于自动搜索多项式形式的 [[CBF (控制障碍函数)]] 和 Lyapunov 函数。

---

## 1. 直觉理解

### 1.1 核心问题

**问题**：给定多项式 $p(x)$，判断 $p(x) \geq 0$ 对所有 $x$ 是否成立？

- $p(x) = x^2 + 1$：显然 $\geq 0$（因为 $x^2 \geq 0$）
- $p(x) = x^4 - 3x^2 + 3$：不太明显
- $p(x) = x^4 + y^4 - 3x^2y^2 + 2$：更难判断

### 1.2 SOS 的思路

如果 $p(x)$ 可以写成**平方和**：

$$p(x) = \sum_i q_i(x)^2$$

则 $p(x) \geq 0$ 显然成立（每个 $q_i^2 \geq 0$）。

**关键发现**：SOS 条件可以通过**半定规划（SDP）** 来检验！

### 1.3 类比

```
非负性: p(x) ≥ 0  ← 难以直接验证

   ↓  充分条件

SOS: p(x) = Σ qᵢ(x)²  ← 可以用 SDP 验证
```

---

## 2. 数学基础

### 2.1 SOS 多项式

多项式 $p(x) \in \mathbb{R}[x]$ 是 **SOS**，如果存在多项式 $q_1, \ldots, q_m$ 使得：

$$p(x) = \sum_{i=1}^{m} q_i(x)^2$$

记为 $p \in \Sigma[x]$。

### 2.2 SOS 与 SDP 的等价性

**定理**：$p(x)$ 是 SOS 当且仅当存在半正定矩阵 $Q \succeq 0$ 使得：

$$p(x) = z(x)^T Q z(x)$$

其中 $z(x) = [1, x_1, x_2, \ldots, x_1^2, x_1 x_2, \ldots]^T$ 是**单项式基向量**。

**例子**：$p(x) = x^4 + 2x^2 + 1$

$z(x) = [1, x, x^2]^T$

$$p(x) = [1, x, x^2] \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ x \\ x^2 \end{bmatrix} = Q_{11} + 2Q_{22}x^2 + Q_{33}x^4$$

$Q = \text{diag}(1, 2, 1) \succeq 0$ → $p$ 是 SOS。

### 2.3 SOS 是凸约束

找到 $Q \succeq 0$ 使得 $z^T Q z = p(x)$ 是一个**半定规划**（SDP），是凸优化问题，可以高效求解。

---

## 3. 在安全验证中的应用

### 3.1 SOS 搜索 CBF

**目标**：找到多项式 CBF $b(x)$ 使得：

1. $b(x) \geq 0$ 在安全集 $S = \{x : g(x) \geq 0\}$
2. $L_f b + L_g b \cdot u + \alpha(b) \geq 0$

**SOS 编码**：

1. $b(x) - \lambda_1(x) g(x) \in \Sigma[x]$（$b \geq 0$ 在 $S$ 上）
2. $L_f b + L_g b \cdot u + \alpha(b) - \lambda_2(x) g(x) \in \Sigma[x]$

其中 $\lambda_1, \lambda_2$ 是 SOS 多项式（辅助乘子）。

### 3.2 SOS 搜索 Lyapunov 函数

**目标**：找到 $V(x) > 0$ 且 $\dot{V}(x) < 0$。

**SOS 编码**：

1. $V(x) - \epsilon \|x\|^2 \in \Sigma[x]$（正定）
2. $-\dot{V}(x) - \epsilon \|x\|^2 \in \Sigma[x]$（递减）

---

## 4. 代码实现

### 4.1 使用 SOSTOOLS (MATLAB)

```matlab
% SOS 搜索 CBF (MATLAB + SOSTOOLS)
% 系统: dx1 = x2, dx2 = u (二阶积分器)
% 不安全集: x1^2 + x2^2 <= 1

% 声明变量
syms x1 x2 real

% 搜索 4 次多项式 CBF
vars = [x1; x2];
[b, cb] = polyvar(vars, 4);  % b 是 4 次多项式

% SOS 约束
% 1. b >= 0 在不安全集外 (x1^2 + x2^2 >= 1)
% 即 b - lambda*(x1^2+x2^2-1) 是 SOS
[lam1, clam1] = polyvar(vars, 2);
sos = b - lam1 * (x1^2 + x2^2 - 1);

% 2. CBF 条件
% Lf_b + Lg_b * u + alpha(b) >= 0
Lf_b = diff(b, x2) * 0;  % f = [x2; 0]
Lg_b = diff(b, x2);       % g = [0; 1]
alpha_b = b;               % alpha(r) = r

sos2 = Lf_b + Lg_b * (-x2) + alpha_b;  % u = -x2 示例

% 求解 SDP
prog = sosprogram([], [b; sos; sos2]);
prog = sossolve(prog);
b_sol = sosgetpoly(prog, b);
disp('找到的 CBF:');
disp(b_sol);
```

### 4.2 使用 Python (PICOS / SumOfSquares)

```python
import numpy as np
import picos

def sos_barrier_certificate_sdp(n_states, degree, 
                                 dynamics_coeffs,
                                 unsafe_coeffs):
    """
    用 SDP 搜索 SOS 障碍证书
    
    简化版: 2D 系统, 多项式动态
    
    输入:
        n_states: 状态维度
        degree: 多项式次数
        dynamics_coeffs: 动态的多项式系数
        unsafe_coeffs: 不安全集的多项式系数
    """
    prob = picos.Problem()
    
    # 单项式基
    # z = [1, x1, x2, x1^2, x1*x2, x2^2, ...]
    n_monomials = (degree // 2 + 1) * (degree // 2 + 2) // 2
    
    # Q 矩阵 (半正定)
    Q = picos.SymmetricVariable('Q', n_monomials)
    prob.add_constraint(Q >> 0)  # Q 半正定
    
    # b(x) = z^T Q z
    # 约束: 匹配目标多项式系数
    
    # ... (需要具体实现系数匹配)
    
    prob.solve()
    
    return Q.value

# 使用 cvxpy 的简化版
import cvxpy as cp

def simple_sos_check():
    """
    简单示例: 检查 p(x) = x^4 + ax^2 + 1 是否 SOS
    
    等价于找 Q >= 0 使得 z^T Q z = p(x)
    其中 z = [1, x, x^2]
    """
    # z = [1, x, x^2]
    # z^T Q z = Q11 + 2*Q12*x + (2*Q13+Q22)*x^2 + 2*Q23*x^3 + Q33*x^4
    
    # 匹配 p(x) = x^4 + a*x^2 + 1:
    # Q11 = 1
    # Q12 = 0 (x 的系数为 0)
    # 2*Q13 + Q22 = a
    # Q23 = 0 (x^3 的系数为 0)
    # Q33 = 1
    
    Q = cp.Variable((3, 3), symmetric=True)
    
    constraints = [
        Q >> 0,              # 半正定
        Q[0, 0] == 1,       # x^0 系数
        Q[0, 1] == 0,       # x^1 系数
        2*Q[0, 2] + Q[1, 1] == 2,  # x^2 系数 (a=2)
        Q[1, 2] == 0,       # x^3 系数
        Q[2, 2] == 1        # x^4 系数
    ]
    
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()
    
    if prob.status == 'optimal':
        print("p(x) = x^4 + 2x^2 + 1 是 SOS!")
        print(f"Q = \n{Q.value}")
    else:
        print("p(x) 不是 SOS")

simple_sos_check()
```

### 4.3 手工验证 SOS

```python
import sympy as sp

def check_sos_2d(polynomial, x, y):
    """
    检查一个 2D 多项式是否 SOS
    
    方法: 尝试分解为平方和
    """
    p = sp.expand(polynomial)
    
    # 尝试完全平方分解
    sqrt_p = sp.sqrt(p)
    if sqrt_p.is_polynomial():
        return True, f"({sqrt_p})^2"
    
    # 尝试分解为两个平方和
    # p = a^2 + b^2
    # ... (需要更复杂的算法)
    
    return None, "需要 SDP 求解器"

x, y = sp.symbols('x y')

# 测试
p1 = x**4 + 2*x**2 + 1  # = (x^2 + 1)^2
result, decomp = check_sos_2d(p1, x, y)
print(f"p1 = {p1}")
print(f"SOS: {result}, 分解: {decomp}")

p2 = x**4 + y**4 + 1  # SOS
p3 = x**4 + y**4 - 3*x**2*y**2 + 2  # 不是 SOS（Motzkin 多项式变体）
```

---

## 5. SOS 的局限性

### 5.1 SOS 不等于非负

存在非负多项式**不是** SOS：

**Motzkin 多项式**：$M(x, y) = x^4 y^2 + x^2 y^4 - 3x^2 y^2 + 1 \geq 0$

但 $M$ **不是** SOS！（Hilbert 1888 年就预言了这种情况）

### 5.2 计算复杂度

| 变量数 $n$ | 次数 $d$ | SDP 大小 | 可行性 |
|-----------|---------|---------|--------|
| 2 | 4 | $6 \times 6$ | ✅ |
| 3 | 4 | $10 \times 10$ | ✅ |
| 5 | 4 | $21 \times 21$ | ✅ |
| 10 | 4 | $66 \times 66$ | ⚠️ |
| 20 | 4 | $231 \times 231$ | ❌ |

---

## 6. SOS vs Neural Barrier Certificate

| 特性 | SOS 方法 | [[Neural Barrier Certificate]] |
|------|---------|-------------------------------|
| **表达形式** | 多项式 | 神经网络 |
| **搜索方法** | SDP | 梯度下降 |
| **验证方法** | 自动（SDP 解即证明） | 需要额外验证 |
| **维度** | 低维（< 10） | 高维可处理 |
| **精度** | 精确（在 SOS 范围内） | 近似 |
| **工具** | SOSTOOLS, YALMIP | PyTorch + auto_LiRPA |

---

## 7. 相关概念

- [[CBF (控制障碍函数)]] — SOS 搜索多项式 CBF
- [[KKT 条件]] — SDP 的最优性条件
- [[Neural Barrier Certificate]] — 神经网络替代方案
- [[Reachability Analysis (可达性分析)]] — SOS 计算可达集的替代
- [[前向不变性 (Forward Invariance)]] — SOS CBF 保证的性质
- [[QP (二次规划)]] — SDP 的特殊情况

---

> **参考**: 
> - Parrilo, "Structured Semidefinite Programs and Semialgebraic Geometry Methods in Robustness and Optimization," PhD Thesis 2000
> - Papachristodoulou & Prajna, "A Tutorial on Sum of Squares Techniques," ACC 2005
> - Tan & Prajna, "SOS Methods for Barrier Certificates," 2008
