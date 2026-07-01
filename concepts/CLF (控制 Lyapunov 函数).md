# 控制 Lyapunov 函数 (Control Lyapunov Function, CLF)

> **一句话**：CLF 是 [[Lyapunov 稳定性]] 的控制版本——它不仅要求存在 Lyapunov 函数 $V(x)$，还要求**存在控制 $u$** 使得 $V$ 递减。在安全控制中，CLF 和 [[CBF (控制障碍函数)]] 常联合使用，同时保证**稳定性和安全性**。

---

## 1. 定义

### 1.1 系统模型

$$\dot{x} = f(x) + g(x) u$$

### 1.2 CLF 定义

$V(x)$ 是 CLF，如果 $V$ 正定（$V(x) > 0$ 对 $x \neq 0$），且：

$$\inf_u \left[ L_f V(x) + L_g V(x) \cdot u \right] < 0, \quad \forall x \neq 0$$

即：**存在某个 $u$** 使得 $\dot{V} < 0$。

### 1.3 与 Lyapunov 的区别

| | Lyapunov 函数 | CLF |
|---|---|---|
| **系统** | $\dot{x} = f(x)$（自治） | $\dot{x} = f(x) + g(x)u$（受控） |
| **条件** | $\dot{V} < 0$（已给定） | $\exists u: \dot{V} < 0$（需要设计 $u$） |
| **含义** | 系统本身稳定 | 系统可被控制到稳定 |

---

## 2. CLF 控制器设计

### 2.1 最小范数控制器

$$u^* = \arg\min_u \|u\|^2 \quad \text{s.t.} \quad L_f V + L_g V \cdot u \leq -\gamma V$$

**解析解**（Sontag 公式的推广）：

$$u^* = \begin{cases} -\frac{L_f V + \gamma V}{\|L_g V\|^2} (L_g V)^T & \text{if } L_f V + \gamma V > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 CLF-QP

$$\min_{u, \delta} \|u\|^2 + p \delta^2$$
$$\text{s.t.} \quad L_f V + L_g V \cdot u \leq -\gamma V + \delta$$

$\delta$ 是松弛变量，允许 CLF 条件偶尔不满足。

---

## 3. CLF + CBF: 安全 + 稳定

### 3.1 联合 QP

$$\min_{u, \delta} \|u - u_{\text{ref}}\|^2 + p \delta^2$$
$$\text{s.t.} \quad L_f V + L_g V \cdot u \leq -\gamma V + \delta \quad (\text{CLF: 稳定性})$$
$$\quad\quad L_f b + L_g b \cdot u \geq -\alpha(b) \quad (\text{CBF: 安全性})$$

**优先级**：安全 > 稳定 > 性能

```python
import cvxpy as cp

def clf_cbf_qp(u_ref, Lf_V, Lg_V, V_val, gamma,
                Lf_b, Lg_b, b_val, alpha,
                p_slack=100.0, u_min=-3.0, u_max=3.0):
    """
    CLF-CBF 联合 QP
    
    min ||u - u_ref||^2 + p*δ^2
    s.t. Lf_V + Lg_V*u <= -γ*V + δ  (CLF, 可松弛)
         Lf_b + Lg_b*u >= -α*b       (CBF, 硬约束)
         u_min <= u <= u_max
    """
    u = cp.Variable()
    delta = cp.Variable()
    
    objective = cp.Minimize(
        0.5 * cp.square(u - u_ref) + p_slack * cp.square(delta)
    )
    
    constraints = [
        Lf_V + Lg_V * u <= -gamma * V_val + delta,  # CLF
        Lf_b + Lg_b * u >= -alpha * b_val,           # CBF (硬约束!)
        u >= u_min,
        u <= u_max
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    
    return u.value, delta.value, prob.status
```

### 3.2 CLF-CBF 冲突

当 CLF 和 CBF 约束**冲突**时（不可能同时满足）：

- **CBF 优先**：松弛变量 $\delta$ 允许 CLF 条件违反
- 系统暂时不稳定，但保证安全
- 当远离危险后，恢复稳定性

```
时间线:
  ────[安全+稳定]──[冲突!]──[安全但不稳定]──[安全+稳定]────
                    ↑                           ↑
              CBF 优先,                 危险过去,
              CLF 松弛                  CLF 恢复
```

---

## 4. CLF 与 CBF 的兼容性

### 4.1 兼容性条件

CLF $V$ 和 CBF $b$ **兼容**，如果存在控制器同时满足两者。

**充分条件**：

$$\forall x: \{u : L_f V + L_g V \cdot u \leq -\gamma V\} \cap \{u : L_f b + L_g b \cdot u \geq -\alpha b\} \neq \emptyset$$

### 4.2 不兼容时怎么办？

1. **松弛 CLF**（推荐）：允许暂时不稳定
2. **松弛 CBF**（不推荐）：允许暂时不安全
3. **修改 $V$ 或 $b$**：重新设计
4. **修改系统**：增加控制输入

---

## 5. 代码：CLF-CBF 安全控制器

```python
import torch
import torch.nn as nn

class CLFCBFController(nn.Module):
    """
    CLF + CBF 联合控制器
    
    同时保证稳定性和安全性
    """
    def __init__(self, state_dim, V_fn, b_fn, 
                 gamma=1.0, alpha=1.0, p_slack=100.0):
        super().__init__()
        self.V_fn = V_fn  # Lyapunov 函数
        self.b_fn = b_fn  # CBF
        self.gamma = gamma
        self.alpha = alpha
        self.p_slack = p_slack
        
        # 参考控制器网络
        self.ref_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, dynamics_fn):
        """
        计算安全+稳定的控制
        
        输入:
            x: (batch, state_dim)
            dynamics_fn: callable(x) -> (f, g)
        """
        u_ref = self.ref_net(x)
        
        # 计算 CLF 相关量
        x_req = x.detach().requires_grad_(True)
        V = self.V_fn(x_req)
        grad_V = torch.autograd.grad(V.sum(), x_req)[0]
        f, g = dynamics_fn(x)
        Lf_V = (grad_V * f).sum(-1, keepdim=True)
        Lg_V = (grad_V * g).sum(-1, keepdim=True)
        
        # 计算 CBF 相关量
        b = self.b_fn(x_req)
        grad_b = torch.autograd.grad(b.sum(), x_req)[0]
        Lf_b = (grad_b * f).sum(-1, keepdim=True)
        Lg_b = (grad_b * g).sum(-1, keepdim=True)
        
        # CLF-CBF QP (解析近似)
        # CLF: Lf_V + Lg_V * u <= -gamma * V
        # CBF: Lf_b + Lg_b * u >= -alpha * b
        
        # 简化: 1D 控制
        u = u_ref.clone()
        
        # CBF 约束 (硬)
        if Lg_b.item() > 0:
            u_cbf_min = (-Lf_b - self.alpha * b) / Lg_b
            u = torch.max(u, u_cbf_min)
        else:
            u_cbf_max = (-Lf_b - self.alpha * b) / Lg_b
            u = torch.min(u, u_cbf_max)
        
        # 控制界
        u = torch.clamp(u, -3.0, 3.0)
        
        return u
```

---

## 6. CLF 与相关概念的对比

| 概念 | 目标 | 条件 | 工具 |
|------|------|------|------|
| **CLF** | 稳定性 | $\inf_u \dot{V} < 0$ | QP / Sontag |
| **CBF** | 安全性 | $\sup_u \dot{b} + \alpha(b) \geq 0$ | QP |
| **CLF+CBF** | 安全+稳定 | 联合 QP | QP + 松弛 |
| **SBC** | 概率安全 | 超鞅条件 | 神经网络 |

---

## 7. 相关概念

- [[Lyapunov 稳定性]] — CLF 的理论基础
- [[CBF (控制障碍函数)]] — 安全性对偶
- [[前向不变性 (Forward Invariance)]] — CBF 保证的性质
- [[QP (二次规划)]] — CLF-CBF 的求解工具
- [[BarrierNet]] — 端到端的安全+性能控制
- [[PPO (Proximal Policy Optimization)]] — 参考控制器训练

---

> **参考**: 
> - Ames et al., "Control Barrier Function based Quadratic Programs with Safety Critical Applications," IEEE TAC 2017
> - Sontag, "A 'Universal' Construction of Artstein's Theorem," Systems & Control Letters 1989
