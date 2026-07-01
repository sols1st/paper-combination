# dReal

> **一句话**：dReal 是一个**非线性 SMT 求解器**，可以判定包含**实数变量和非线性函数**（如 sin、exp、神经网络）的逻辑公式是否可满足。在形式化验证中，它被用来精确验证神经网络控制器的安全属性。

---

## 1. 什么是 SMT？

**SMT (Satisfiability Modulo Theories)** = 可满足性模理论

给定一个逻辑公式（如 $\forall x, \exists y: f(x,y) > 0$），SMT 求解器判定它是否**可满足**（存在使公式为真的赋值）或**不可满足**（对所有赋值公式为假）。

常见 SMT 求解器：
- **Z3** (Microsoft)：支持线性算术、位向量等
- **CVC5**：支持多种理论
- [[SMT (可满足性模理论)]] — 详细笔记

**dReal 的特殊之处**：支持**非线性实数算术**，包括三角函数、指数函数等。

---

## 2. dReal 的核心特性

### 2.1 $\delta$-可满足性

dReal 不判定精确的可满足性，而是判定 **$\delta$-可满足性**：

- 公式 $\phi$ 是 $\delta$-可满足的，如果存在赋值使 $\phi$ 在 $\delta$ 精度内成立
- 即允许最多 $\delta$ 的数值误差

**为什么？** 非线性函数（sin, exp 等）无法精确计算，数值误差不可避免。

### 2.2 支持的理论

- 非线性实数算术 ($\mathcal{NRA}$)
- 常微分方程 ($\mathcal{ODE}$)
- 神经网络（通过分段线性近似）

---

## 3. 在安全验证中的应用

### 3.1 验证 CBF 条件

给定 [[CBF (控制障碍函数)]] $b(x)$ 和控制器 $\pi(x)$（神经网络），验证：

$$\forall x \in S: L_f b(x) + L_g b(x) \cdot \pi(x) + \alpha(b(x)) \geq 0$$

dReal 将其转化为：是否存在 $x$ 使得 $L_f b + L_g b \cdot \pi(x) + \alpha(b) < 0$？

- 如果 **unsat**（不存在）→ CBF 条件满足 ✅
- 如果 **$\delta$-sat**（存在反例）→ CBF 条件不满足 ❌，返回反例

### 3.2 验证 SBC 条件

类似地，验证 [[SBF SBC (随机障碍函数与证书)]] 的鞅递减条件：

$$\forall s: B(s) - \mathbb{E}[B(s')] \geq \epsilon$$

---

## 4. 使用方法

### 4.1 安装

```bash
# Docker 安装（推荐）
docker pull dreal/dreal4

# pip 安装
pip install dreal
```

### 4.2 Python API

```python
from dreal import *

# 声明变量
x = Variable("x")
y = Variable("y")

# 构造公式: x^2 + y^2 <= 1 AND x + y >= 2
formula = And(x**2 + y**2 <= 1, x + y >= 2)

# 设定精度
delta = 0.001

# 检查 delta-可满足性
result = CheckSat(formula, delta)

if result:
    print(f"delta-sat: x = {result[x]}, y = {result[y]}")
else:
    print("unsat (公式不可满足)")
```

### 4.3 验证神经网络

```python
from dreal import *

def verify_cbf(model, state_bounds, d_safe=6.0):
    """
    用 dReal 验证 CBF 条件
    
    model: 神经网络控制器 (PyTorch)
    state_bounds: [(d_min, d_max), (v_min, v_max)]
    """
    # 声明状态变量
    d = Variable("d")
    v = Variable("v")
    
    # 将神经网络转为 dReal 可处理的表示
    # (需要将 ReLU 转为分段线性约束)
    u = encode_nn(model, [d, v])
    
    # CBF 条件
    b = d - d_safe - v  # b(x) = d - d_safe - v
    Lf_b = -v
    Lg_b = 1.0
    alpha_b = b
    
    cbf_constraint = Lf_b + Lg_b * u + alpha_b
    
    # 验证: 是否存在违反 CBF 的状态？
    violation = And(
        d >= state_bounds[0][0],
        d <= state_bounds[0][1],
        v >= state_bounds[1][0],
        v <= state_bounds[1][1],
        cbf_constraint < 0  # 违反条件
    )
    
    result = CheckSat(violation, 0.001)
    
    if result:
        print(f"反例: d={result[d]}, v={result[v]}")
        return False, result
    else:
        print("验证通过: CBF 条件全局满足")
        return True, None
```

---

## 5. 与 IBP/CROWN 的对比

| 特性 | [[IBP (区间界传播)]] / [[CROWN (神经网络验证)]] | dReal |
|------|----------------------------------------------|-------|
| **方法** | 区间传播 | SMT 求解 |
| **精度** | 保守（可能漏报） | 精确（$\delta$ 内） |
| **速度** | 快 | 慢 |
| **可扩展性** | 高维可处理 | 高维困难 |
| **输出** | 区间界 | sat/unsat + 反例 |
| **适用** | 在线验证（CEGIS） | 离线精确验证 |

---

## 6. 相关概念

- [[SMT (可满足性模理论)]] — dReal 的理论基础
- [[IBP (区间界传播)]] — 替代验证方法
- [[CROWN (神经网络验证)]] — 替代验证方法
- [[CEGIS (反例引导合成)]] — dReal 提供反例

---

> **参考**: 
> - Gao et al., "dReal: An SMT Solver for Nonlinear Theories over the Reals," CADE 2013
> - https://github.com/dreal/dreal4
