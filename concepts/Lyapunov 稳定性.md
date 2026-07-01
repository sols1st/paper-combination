# Lyapunov 稳定性 (Lyapunov Stability)

> **一句话**：Lyapunov 稳定性是控制理论的核心概念——通过构造一个**能量函数** $V(x)$（Lyapunov 函数），证明系统在平衡点处稳定。它与 [[CBF (控制障碍函数)]] 结构高度相似：Lyapunov 证明"收敛到平衡点"，CBF 证明"不离开安全集"。

---

## 1. 直觉理解

### 1.1 碗中小球

```
         ╱         ╲
        ╱           ╲
       ╱    ○ ← 小球 ╲
      ╱       ↓       ╲
     ╱     ● 平衡点     ╲
    ╱___________________╲
    
    V(x) = 高度 = 能量
    小球滚向最低点（平衡点）
    V 在递减 → 稳定
```

- $V(x)$ = 小球的势能（高度）
- 小球总是向低处滚（$\dot{V} < 0$）
- 最终到达最低点（$V = 0$，平衡点）

### 1.2 Lyapunov 的核心思想

如果存在一个"能量函数" $V(x)$：
1. $V(x) > 0$（除了平衡点）
2. $V(x)$ 沿轨迹递减（$\dot{V} < 0$）

则系统稳定。

---

## 2. 形式定义

### 2.1 系统模型

$$\dot{x} = f(x), \quad f(0) = 0$$

平衡点 $x^* = 0$（不失一般性）。

### 2.2 稳定性类型

| 类型 | 定义 | 直觉 |
|------|------|------|
| **稳定** | $\|x(0)\| < \delta \Rightarrow \|x(t)\| < \epsilon$ | 不远离平衡点 |
| **渐近稳定** | 稳定 + $x(t) \to 0$ | 收敛到平衡点 |
| **指数稳定** | $\|x(t)\| \leq c \|x(0)\| e^{-\lambda t}$ | 指数收敛 |
| **全局渐近稳定** | 对所有 $x(0)$，$x(t) \to 0$ | 全局收敛 |

### 2.3 Lyapunov 函数

$V: \mathbb{R}^n \to \mathbb{R}$ 是 Lyapunov 函数，如果：

1. **正定**：$V(x) > 0$ 对 $x \neq 0$，$V(0) = 0$
2. **递减**：$\dot{V}(x) = \nabla V \cdot f(x) < 0$ 对 $x \neq 0$

**定理**：如果存在 Lyapunov 函数，则平衡点是**渐近稳定**的。

---

## 3. Lyapunov vs CBF

| 方面 | Lyapunov 函数 $V(x)$ | CBF $b(x)$ |
|------|---------------------|-----------|
| **目标** | 稳定性（收敛） | 安全性（不离开） |
| **集合** | $V(x) \leq c$（亚水平集） | $b(x) \geq 0$（上水平集） |
| **条件** | $\dot{V} \leq 0$ | $\dot{b} + \alpha(b) \geq 0$ |
| **方向** | $V$ 递减 | $b$ 不递减到 0 以下 |
| **前向不变** | $\{V \leq c\}$ 前向不变 | $\{b \geq 0\}$ 前向不变 |

### 3.1 对偶关系

```
Lyapunov:              CBF:
V(x) 大 → 远离平衡      b(x) 大 → 远离危险
V(x) 递减 → 趋向平衡    b(x) 不递减太快 → 保持安全
V = 0 → 平衡点          b = 0 → 安全边界

   {V ≤ c}                  {b ≥ 0}
  ┌─────────┐            ┌─────────┐
  │  ● → 0  │            │  安全区  │
  │  稳定!  │            │  不离开! │
  └─────────┘            └─────────┘
```

---

## 4. 计算方法

### 4.1 线性系统

对于 $\dot{x} = Ax$，Lyapunov 函数取二次形式 $V(x) = x^T P x$：

$$\dot{V} = x^T (A^T P + PA) x$$

稳定性条件：$A^T P + PA \prec 0$（负定）

这是一个 **LMI（线性矩阵不等式）**！

```python
import cvxpy as cp
import numpy as np

def lyapunov_lmi(A):
    """
    用 LMI 求解线性系统的 Lyapunov 函数
    
    找 P > 0 使得 A'P + PA < 0
    
    输入:
        A: (n, n) 系统矩阵
    输出:
        P: Lyapunov 矩阵
    """
    n = A.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    
    constraints = [
        P >> np.eye(n) * 0.01,  # P 正定
        A.T @ P + P @ A << -np.eye(n) * 0.01  # A'P + PA 负定
    ]
    
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve()
    
    if prob.status == 'optimal':
        return P.value
    else:
        return None  # 系统不稳定

# 测试: 稳定系统
A = np.array([[-1, 2], [-3, -4]], dtype=float)
P = lyapunov_lmi(A)
if P is not None:
    print(f"P = \n{P}")
    eigenvalues = np.linalg.eigvals(A.T @ P + P @ A)
    print(f"A'P + PA 特征值: {eigenvalues}")  # 应全为负
else:
    print("系统不稳定")
```

### 4.2 非线性系统

对于 $\dot{x} = f(x)$，没有通用方法。常用：

1. **线性化**：在平衡点附近 $\dot{x} \approx Ax$，用 LMI
2. **SOS 方法**：用 [[SOS (Sum-of-Squares)]] 搜索多项式 $V(x)$
3. **Neural Lyapunov**：用 [[Neural Barrier Certificate]] 类似方法训练 NN

### 4.3 神经网络 Lyapunov

```python
import torch
import torch.nn as nn

class NeuralLyapunov(nn.Module):
    """
    神经 Lyapunov 函数
    
    V(x) = ||NN(x)||^2 + ε||x||^2
    
    自动保证:
    1. V(0) = 0 (如果 NN(0) = 0)
    2. V(x) > 0 for x ≠ 0 (由平方保证)
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.eps = 0.01
    
    def forward(self, x):
        """V(x) = ||NN(x)||^2 + ε||x||^2"""
        nn_out = self.net(x)
        return (nn_out ** 2).sum(-1) + self.eps * (x ** 2).sum(-1)
    
    def derivative(self, x, dynamics_fn):
        """计算 V̇(x) = ∇V · f(x)"""
        x.requires_grad_(True)
        V = self.forward(x)
        
        # ∇V
        grad_V = torch.autograd.grad(
            V.sum(), x, create_graph=True
        )[0]
        
        # f(x)
        f = dynamics_fn(x)
        
        # V̇ = ∇V · f
        V_dot = (grad_V * f).sum(-1)
        
        return V_dot

def train_lyapunov(lyap_model, dynamics_fn, state_bounds, 
                    n_epochs=100, batch_size=256):
    """
    训练神经 Lyapunov 函数
    
    目标: V̇(x) < 0 for x ≠ 0
    """
    optimizer = torch.optim.Adam(lyap_model.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        # 采样状态
        x = torch.rand(batch_size, len(state_bounds))
        for i, (lo, hi) in enumerate(state_bounds):
            x[:, i] = x[:, i] * (hi - lo) + lo
        
        # V̇(x)
        V_dot = lyap_model.derivative(x, dynamics_fn)
        
        # 损失: 鼓励 V̇ < 0
        loss = torch.relu(V_dot + 0.01).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return lyap_model
```

---

## 5. Lyapunov + CBF: 安全 + 稳定

### 5.1 Control Lyapunov Function + CBF

**目标**：同时保证稳定性和安全性。

**联合 QP**：

$$\min_{u, \delta} \|u - u_{\text{ref}}\|^2 + p \delta^2$$

$$\text{s.t.} \quad L_f V + L_g V \cdot u \leq -\gamma V + \delta \quad (\text{CLF: 稳定性})$$
$$\quad\quad L_f b + L_g b \cdot u \geq -\alpha(b) \quad (\text{CBF: 安全性})$$

其中 $\delta$ 是松弛变量（允许 CLF 条件偶尔违反，但 CBF 永远满足）。

### 5.2 优先级

```
安全性 (CBF) > 稳定性 (CLF) > 性能 (u_ref)
```

- CBF 约束是**硬约束**（不能违反）
- CLF 约束有松弛变量 $\delta$（可以暂时违反）

---

## 6. 常见误区

| 误区 | 正确理解 |
|------|---------|
| "找不到 Lyapunov 函数 = 不稳定" | 不一定，可能是方法不对 |
| "Lyapunov 函数唯一" | 不唯一，有无穷多 |
| "$V > 0$ 就够了" | 还需要 $\dot{V} < 0$ |
| "Lyapunov 只能用于连续系统" | 离散版本也存在 |

---

## 7. 相关概念

- [[CBF (控制障碍函数)]] — Lyapunov 的"安全对偶"
- [[前向不变性 (Forward Invariance)]] — 两者都保证前向不变性
- [[SOS (Sum-of-Squares)]] — 搜索多项式 Lyapunov 函数
- [[Neural Barrier Certificate]] — 神经 Lyapunov 函数
- [[比较引理 (Comparison Lemma)]] — Lyapunov 稳定性证明的工具
- [[超鞅 (Supermartingale)]] — 随机版本的 Lyapunov 函数

---

> **参考**: 
> - Khalil, "Nonlinear Systems," Prentice Hall 2002, Chapter 4
> - Ames et al., "Control Barrier Function based Quadratic Programs," ACC 2014
