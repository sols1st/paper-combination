# 可微 QP (Differentiable QP)

> **一句话**：可微 QP 是一种技术，使得 [[QP (二次规划)]] 的**解关于参数可微**。这意味着梯度可以从 QP 的最优解反向传播到 QP 的参数（如代价矩阵、约束矩阵），从而实现端到端训练。

---

## 1. 为什么需要可微 QP？

在 [[BarrierNet]] 中，QP 的参数（$H, F, G, h$）是**神经网络的输出**：

$$H = \text{Net}_H(z), \quad F = \text{Net}_F(z), \quad p = \text{Net}_p(z)$$

为了端到端训练这些网络，需要计算 $\frac{\partial \ell}{\partial \theta}$，其中 $\ell$ 是上游损失，$\theta$ 是网络参数。

**问题**：QP 的解 $u^*$ 没有显式的闭合形式（不能直接对 $u^*$ 求导）。

**解决**：利用 [[KKT 条件]] 和**隐函数定理**，得到 $u^*$ 关于参数的梯度。

---

## 2. 数学推导

### 2.1 QP 和 KKT 条件

$$\min_u \frac{1}{2} u^T H u + F^T u \quad \text{s.t.} \quad Gu \leq h$$

[[KKT 条件]]：

1. $Hu + F + G^T \lambda = 0$（驻点）
2. $Gu - h \leq 0$（可行性）
3. $\lambda \geq 0$（对偶可行性）
4. $\lambda_i (G_i u - h_i) = 0$（互补松弛）

### 2.2 隐函数定理

将 KKT 条件写为方程组 $R(u^*, \lambda^*, \theta) = 0$。

对 $R$ 全微分：

$$\begin{bmatrix} H & G^T \\ D(\lambda^*) G & D(Gu^* - h) \end{bmatrix} \begin{bmatrix} du^* \\ d\lambda^* \end{bmatrix} = -\begin{bmatrix} dF + dH \cdot u^* + dG^T \lambda^* \\ D(\lambda^*)(dG \cdot u^* - dh) \end{bmatrix}$$

其中 $D(\cdot)$ 创建对角矩阵。

### 2.3 求解梯度

通过矩阵求逆得到 $du^*$ 和 $d\lambda^*$：

$$\begin{bmatrix} du^* \\ d\lambda^* \end{bmatrix} = J^{-1} \cdot \text{rhs}$$

其中 $J$ 是 KKT 矩阵，$J^{-1}$ 的复杂度为 $O((n+m)^3)$。

### 2.4 参数梯度

$$\nabla_H \ell = \frac{1}{2}(du^* \cdot u^{*T} + u^* \cdot du^{*T})$$

$$\nabla_F \ell = du^*$$

$$\nabla_G \ell = D(\lambda^*)(d\lambda^* \cdot u^{*T} + \lambda^* \cdot du^{*T})$$

$$\nabla_h \ell = -D(\lambda^*) d\lambda^*$$

---

## 3. 直觉解释

**正常神经网络**：$y = f(x; \theta)$，直接用链式法则求 $\frac{\partial y}{\partial \theta}$。

**可微 QP**：$u^* = \text{QP}(H(\theta), F(\theta), G(\theta), h(\theta))$

- **前向传播**：求解 QP 得到 $u^*$
- **反向传播**：利用 KKT 条件的隐函数关系，得到 $\frac{\partial u^*}{\partial H}, \frac{\partial u^*}{\partial F}, \ldots$
- **链式法则**：$\frac{\partial \ell}{\partial \theta} = \frac{\partial \ell}{\partial u^*} \cdot \frac{\partial u^*}{\partial \theta}$

---

## 4. 代码实现

```python
import torch

class DifferentiableQPLayer(torch.autograd.Function):
    """
    可微 QP 层
    
    前向: 求解 QP
    反向: 通过 KKT 条件计算梯度
    """
    @staticmethod
    def forward(ctx, H, F, G, h):
        """
        求解 QP: min 0.5*u'Hu + F'u s.t. Gu <= h
        
        输入:
            H: (batch, n, n) 正定矩阵
            F: (batch, n) 线性项
            G: (batch, m, n) 约束矩阵
            h: (batch, m) 约束右端
        输出:
            u_star: (batch, n) 最优解
        """
        batch_size, n = F.shape
        m = G.shape[1]
        
        # 简化的 1D 情况 (n=1)
        if n == 1:
            u_unc = -F / H.squeeze(-1)  # 无约束最优
            
            # 检查每个约束
            u_star = u_unc.clone()
            lam = torch.zeros(batch_size, m)
            
            for i in range(m):
                g_i = G[:, i, :]  # (batch, 1)
                h_i = h[:, i:i+1]  # (batch, 1)
                
                violation = (g_i * u_star - h_i).squeeze() > 0
                if violation.any():
                    u_bound = (h_i / g_i).squeeze()
                    u_star = torch.where(violation & (g_i.squeeze() < 0),
                                        torch.max(u_star, u_bound),
                                        u_star)
                    u_star = torch.where(violation & (g_i.squeeze() > 0),
                                        torch.min(u_star, u_bound),
                                        u_star)
        
        # 保存中间变量用于反向传播
        ctx.save_for_backward(H, F, G, h, u_star)
        
        return u_star
    
    @staticmethod
    def backward(ctx, grad_u):
        """
        反向传播: 通过 KKT 条件计算梯度
        
        grad_u: 上游对 u* 的梯度
        """
        H, F, G, h, u_star = ctx.saved_tensors
        
        # 简化的 1D 情况
        # du* = H^{-1} * grad_u (当约束不活跃)
        du = grad_u / H.squeeze(-1)
        
        grad_H = -du * u_star * 0.5
        grad_F = -du
        grad_G = torch.zeros_like(G)
        grad_h = torch.zeros_like(h)
        
        return grad_H, grad_F, grad_G, grad_h
```

### 4.1 使用现成库

```python
# 使用 cvxpylayers
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

n, m = 1, 3  # 变量数, 约束数
u = cp.Variable(n)
H_param = cp.Parameter((n, n), PSD=True)
F_param = cp.Parameter(n)
G_param = cp.Parameter((m, n))
h_param = cp.Parameter(m)

objective = cp.Minimize(0.5 * cp.quad_form(u, H_param) + F_param @ u)
constraints = [G_param @ u <= h_param]
problem = cp.Problem(objective, constraints)

# 创建可微层
qp_layer = CvxpyLayer(problem, 
                       parameters=[H_param, F_param, G_param, h_param],
                       variables=[u])

# 使用
H = torch.eye(1).unsqueeze(0).requires_grad_(True)
F = torch.tensor([[-2.0]]).requires_grad_(True)
G = torch.tensor([[[-1.0], [1.0], [-1.0]]])
h = torch.tensor([[0.0, 3.0, 3.0]])

u_star, = qp_layer(H, F, G, h)
loss = (u_star - 1.0)**2
loss.backward()  # 梯度自动传播!
```

---

## 5. 在 BarrierNet 中的完整流程

```
图像 z → CNN → MLP → {H(z), F(z), p(z)} → Differentiable QP → u*
                         ↑                                          |
                         |                                          ↓
                    反向传播 ←───── 损失 ℓ(u*, u_target) ←──── 安全控制
```

---

## 6. 计算复杂度

| 操作 | 复杂度 | AEBS ($n=1, m=3$) |
|------|--------|-------------------|
| 前向求解 | $O((n+m)^3)$ | $O(64) \approx 0$ |
| 反向传播 | $O((n+m)^3)$ | $O(64) \approx 0$ |
| 参数梯度 | $O(n^2 + mn)$ | $O(4) \approx 0$ |

对于低维控制（如 AEBS），可微 QP 的开销几乎为零。

---

## 7. 相关概念

- [[QP (二次规划)]] — 基础
- [[KKT 条件]] — 反向传播的数学基础
- [[BarrierNet]] — 可微 QP 的主要应用
- [[dCBF (可微控制障碍函数)]] — 可微 QP 中的约束来源

---

> **参考**: 
> - Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in Neural Networks," ICML 2017
> - Agrawal et al., "Differentiable Convex Optimization Layers," NeurIPS 2019
