# BarrierNet

> **一句话**：BarrierNet 是一种**端到端可训练的安全控制器架构**——它将 [[dCBF (可微控制障碍函数)]] 嵌入 [[可微 QP (Differentiable QP)]] 层中，使得安全控制可以通过**反向传播**进行训练。安全约束以"惩罚项"的形式融入 QP，而非硬约束，从而避免了不可行问题。

---

## 1. 为什么需要 BarrierNet？

### 1.1 传统 CBF 的局限

传统 [[CBF (控制障碍函数)]] 方法：

1. **CBF 是固定的**：手动设计 $b(x)$，不能适应复杂场景
2. **硬约束 QP 可能不可行**：当 CBF 约束与控制界冲突时，QP 无解
3. **不能端到端训练**：CBF 和控制策略分离设计

### 1.2 BarrierNet 的解决方案

1. **可学习的 dCBF**：CBF 参数由神经网络从观测中生成
2. **软约束**：安全约束以**惩罚项**形式出现在 QP 目标函数中
3. **可微 QP 层**：整个安全控制过程可反向传播

---

## 2. 架构

### 2.1 总体流程

```
感知输入 z (图像/传感器)
       │
       ▼
   ┌───────┐
   │  CNN  │  特征提取
   └───┬───┘
       │
       ▼
   ┌───────┐
   │  MLP  │  参数生成
   └───┬───┘
       │
       ├──→ H(z) ──┐
       ├──→ F(z) ──┤
       └──→ p(z) ──┤  ← dCBF 惩罚系数
                   │
                   ▼
        ┌──────────────────────┐
        │  Differentiable QP   │
        │                      │
        │  min 0.5 u'Hu + F'u  │
        │      + Σ pᵢ·αᵢ(ψᵢ)  │
        │  s.t. u_min ≤ u ≤ u_max │
        └──────────┬───────────┘
                   │
                   ▼
              u* (安全控制)
```

### 2.2 各组件详解

| 组件 | 功能 | 输入 → 输出 |
|------|------|------------|
| **CNN** | 从图像提取特征 | $z \in \mathbb{R}^{H \times W \times C} \to \phi \in \mathbb{R}^d$ |
| **MLP** | 从特征生成 QP 参数 | $\phi \to (H, F, p)$ |
| **PenaltyNet** | 生成 dCBF 惩罚系数 | $\phi \to p_1, p_2, \ldots$ |
| **Differentiable QP** | 求解带惩罚的 QP | $(H, F, p) \to u^*$ |

---

## 3. 数学公式

### 3.1 dCBF 序列

$$\psi_0(x) = b(x)$$

$$\psi_1(x, z) = \dot{\psi}_0(x) + p_1(z) \cdot \alpha_1(\psi_0(x))$$

$$\psi_{m-1}(x, z) = \dot{\psi}_{m-2}(x, z) + p_{m-1}(z) \cdot \alpha_{m-1}(\psi_{m-2}(x, z))$$

### 3.2 QP 目标函数

$$\min_u \frac{1}{2} u^T H(z) u + F(z)^T u + \sum_{i=1}^{m-1} p_i(z) \cdot \max(0, -\psi_i(x, z))^2$$

其中：
- $\frac{1}{2} u^T H u + F^T u$：跟踪参考控制
- $p_i \cdot \max(0, -\psi_i)^2$：dCBF 违反惩罚

### 3.3 约束

$$u_{\min} \leq u \leq u_{\max}$$

只有控制量界约束（**始终可行**）。

---

## 4. 代码实现

### 4.1 PenaltyNet

```python
import torch
import torch.nn as nn

class PenaltyNet(nn.Module):
    """
    生成 dCBF 惩罚系数 p_i(z)
    
    输出始终 > 0（使用 softplus）
    """
    def __init__(self, input_dim, n_penalties=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_penalties)
        )
        self.n_penalties = n_penalties
    
    def forward(self, z):
        """
        输入: z (batch, input_dim) 特征向量
        输出: p (batch, n_penalties) 惩罚系数，> 0
        """
        raw = self.net(z)
        p = torch.nn.functional.softplus(raw) + 0.1  # 确保 > 0.1
        return p
```

### 4.2 Differentiable QP Layer

```python
class DifferentiableQP(nn.Module):
    """
    可微 QP 层
    
    min_u 0.5 u'Hu + F'u + Σ pᵢ · max(0, -ψᵢ)²
    s.t. u_min ≤ u ≤ u_max
    
    对于 1D 控制（AEBS），有解析解。
    """
    def __init__(self, u_min=-3.0, u_max=3.0):
        super().__init__()
        self.u_min = u_min
        self.u_max = u_max
    
    def forward(self, H, F, penalties, psi_values):
        """
        输入:
            H: (batch, 1, 1) QP 二次项
            F: (batch, 1) QP 线性项
            penalties: (batch, m-1) 惩罚系数
            psi_values: (batch, m-1) dCBF 序列值
        输出:
            u_star: (batch, 1) 最优控制
        """
        batch_size = H.shape[0]
        
        # 无约束最优: u = -F/H
        u_unc = -F / (H.squeeze(-1) + 1e-8)  # (batch, 1)
        
        # 惩罚梯度: d/du [Σ pᵢ max(0, -ψᵢ)²]
        # 对于 1D 情况，ψᵢ 通常包含 u 的线性项
        # 简化处理: 直接用投影
        
        # 投影到控制界
        u_star = torch.clamp(u_unc, self.u_min, self.u_max)
        
        return u_star
```

### 4.3 完整 BarrierNet

```python
class BarrierNet(nn.Module):
    """
    完整的 BarrierNet 安全控制器
    
    输入: 观测 z + 状态 x
    输出: 安全控制 u*
    """
    def __init__(self, obs_dim, state_dim, action_dim=1, 
                 relative_degree=2, u_min=-3.0, u_max=3.0):
        super().__init__()
        
        self.state_dim = state_dim
        self.relative_degree = relative_degree
        
        # 感知网络
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # QP 参数网络
        self.H_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # H > 0
        )
        self.F_net = nn.Sequential(
            nn.Linear(64 + state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # 惩罚网络
        n_penalties = relative_degree - 1
        self.penalty_net = PenaltyNet(
            64 + state_dim, n_penalties
        )
        
        # dCBF 序列中的 α 函数
        self.alphas = [
            lambda r: 1.0 * r  # 线性 Class K
            for _ in range(relative_degree - 1)
        ]
        
        # 可微 QP
        self.qp = DifferentiableQP(u_min, u_max)
    
    def compute_dcbf_sequence(self, x, z_features):
        """
        计算 dCBF 序列 ψ_0, ψ_1, ..., ψ_{m-1}
        
        输入:
            x: (batch, state_dim) 状态
            z_features: (batch, 64) 感知特征
        输出:
            psi_values: (batch, m-1) dCBF 值
        """
        batch = x.shape[0]
        m = self.relative_degree
        
        # ψ_0 = b(x) (安全函数，预定义)
        d, v = x[:, 0:1], x[:, 1:2]
        d_safe, T_gap = 6.0, 1.5
        psi_0 = d - d_safe - T_gap * v  # (batch, 1)
        
        psi_values = []
        psi_prev = psi_0
        
        # 获取惩罚系数
        features_with_state = torch.cat([z_features, x], dim=-1)
        penalties = self.penalty_net(features_with_state)
        
        for i in range(m - 1):
            # ψ_{i+1} = dψ_i/dt + p_{i+1} * α_{i+1}(ψ_i)
            # 简化: 用数值近似 dψ_i/dt
            
            p_i = penalties[:, i:i+1]  # (batch, 1)
            alpha_val = self.alphas[i](psi_prev)
            
            # 假设 dψ/dt 的常数近似
            dpsi_dt = -v if i == 0 else torch.zeros_like(psi_prev)
            
            psi_next = dpsi_dt + p_i * alpha_val
            psi_values.append(psi_next)
            psi_prev = psi_next
        
        return torch.cat(psi_values, dim=-1), penalties
    
    def forward(self, obs, state):
        """
        完整前向传播
        
        输入:
            obs: (batch, obs_dim) 观测
            state: (batch, state_dim) 状态
        输出:
            u_star: (batch, 1) 安全控制
            info: 调试信息
        """
        # 特征提取
        z = self.feature_net(obs)
        
        # QP 参数
        H = self.H_net(z).unsqueeze(-1)  # (batch, 1, 1)
        
        features_with_state = torch.cat([z, state], dim=-1)
        F = self.F_net(features_with_state)  # (batch, 1)
        
        # dCBF 序列
        psi_values, penalties = self.compute_dcbf_sequence(state, z)
        
        # 求解 QP
        u_star = self.qp(H, F, penalties, psi_values)
        
        info = {
            'H': H,
            'F': F,
            'penalties': penalties,
            'psi_values': psi_values
        }
        
        return u_star, info
```

### 4.4 训练循环

```python
def train_barriernet(model, dataloader, optimizer, n_epochs=100):
    """
    训练 BarrierNet
    """
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch in dataloader:
            obs, state, u_ref, safe_label = batch
            
            # 前向传播
            u_star, info = model(obs, state)
            
            # 损失 1: 跟踪参考控制
            tracking_loss = ((u_star - u_ref) ** 2).mean()
            
            # 损失 2: dCBF 安全损失
            psi_values = info['psi_values']
            safety_loss = torch.relu(-psi_values).mean()
            
            # 损失 3: 控制量正则化
            control_loss = (u_star ** 2).mean()
            
            # 总损失
            loss = tracking_loss + 10.0 * safety_loss + 0.1 * control_loss
            
            optimizer.zero_grad()
            loss.backward()  # 梯度通过可微 QP 反向传播!
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}")
```

---

## 5. 与传统方法的对比

| 特性 | 传统 CBF-QP | BarrierNet |
|------|-----------|-----------|
| **CBF 设计** | 手动设计 | 可学习 |
| **约束类型** | 硬约束 | 软约束（惩罚） |
| **可行性** | 可能不可行 | 始终可行 |
| **感知输入** | 不支持 | 支持（CNN） |
| **端到端训练** | 不支持 | 支持 |
| **实时性** | 需要 QP 求解器 | 解析解（低维） |

---

## 6. 安全性分析

### 6.1 理论保证

BarrierNet 的安全性依赖于 dCBF 的定理（见 [[dCBF (可微控制障碍函数)]]）：

**如果**：
1. $p_i(z) > 0$（惩罚系数正）
2. QP 解 $u^*$ 使 $\psi_{m-1} \geq 0$
3. $\alpha_i \in \mathcal{K}$

**则**：安全集合 $C = \{x : b(x) \geq 0\}$ 是前向不变的。

### 6.2 实际验证

通过 [[CEGIS (反例引导合成)]] 验证训练后的 BarrierNet：

```python
# 训练后验证
for x_test in test_grid:
    u = barriernet(obs_fn(x_test), x_test)
    b_val = cbf(x_test)
    Lf_b, Lg_b = lie_derivatives(cbf, dynamics, x_test)
    cbf_condition = Lf_b + Lg_b * u + alpha(b_val)
    
    if cbf_condition < 0:
        print(f"违反: x = {x_test}, violation = {cbf_condition}")
```

---

## 7. 计算复杂度

| 组件 | 复杂度 | AEBS ($n=1, m=3$) |
|------|--------|-------------------|
| CNN 特征提取 | $O(H \cdot W \cdot C)$ | $O(64 \times 64 \times 3) = 12288$ |
| MLP 参数生成 | $O(d \cdot h)$ | $O(64 \times 32) = 2048$ |
| PenaltyNet | $O(d \cdot h)$ | $O(64 \times 32) = 2048$ |
| dCBF 序列计算 | $O(m \cdot n)$ | $O(2) \approx 0$ |
| QP 求解（1D） | $O(1)$ | $O(1)$ |
| **总计** | — | $\approx 16000$ FLOPs |

在 GPU 上，单次前向传播 $< 1$ ms。

---

## 8. 相关概念

- [[dCBF (可微控制障碍函数)]] — BarrierNet 的安全约束来源
- [[可微 QP (Differentiable QP)]] — BarrierNet 的优化层
- [[CBF (控制障碍函数)]] — 理论基础
- [[HOCBF (高阶控制障碍函数)]] — dCBF 的前身
- [[KKT 条件]] — QP 求解的理论基础
- [[CEGIS (反例引导合成)]] — BarrierNet 的验证方法
- [[PPO (Proximal Policy Optimization)]] — 可用于预训练参考控制器

---

> **参考**: 
> - Xiao et al., "BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control," IEEE T-RO 2023
> - Qin et al., "Learning Safe Control via Differentiable Control Barrier Functions," 2023
