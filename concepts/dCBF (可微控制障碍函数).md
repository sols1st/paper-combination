# dCBF (可微控制障碍函数)

> **一句话**：dCBF 是对 [[HOCBF (高阶控制障碍函数)]] 的改进，通过引入**可学习的、环境依赖的惩罚函数 $p_i(z)$** 来动态调节 CBF 约束的松紧程度，降低保守性，同时保持安全保证。

---

## 1. 动机：传统 HOCBF 的保守性

### 1.1 问题

[[HOCBF (高阶控制障碍函数)]] 中的 [[Class K 函数]] $\alpha_i$ 是**固定**的。这意味着：

| 场景 | 固定 $\alpha$ 的表现 | 理想表现 |
|------|---------------------|---------|
| 空旷高速公路 | 过度保守，不必要地减速 | 放松约束，正常行驶 |
| 拥堵路段 | 可能不够保守 | 收紧约束，提前刹车 |
| 行人突然出现 | 可能来不及反应 | 极度收紧，紧急刹车 |

### 1.2 核心想法

**让 $\alpha_i$ 根据环境动态变化**。引入正值函数 $p_i(z) > 0$：

$$\psi_i = \dot{\psi}_{i-1} + p_i(z) \cdot \alpha_i(\psi_{i-1})$$

- $p_i(z)$ **大** → $\alpha_i$ 等效变大 → 约束更紧 → 更早干预 → 更安全但更保守
- $p_i(z)$ **小** → $\alpha_i$ 等效变小 → 约束更松 → 更晚干预 → 更高效但更冒险

---

## 2. 数学定义

### 2.1 dCBF 序列函数

对于系统 $\dot{x} = f(x) + g(x)u$，安全约束 $b(x) \geq 0$（[[相对度 (Relative Degree)]] $m$），定义：

$$\psi_0(x, z, z_d) := b(x)$$

$$\psi_i(x, z, z_d) := \dot{\psi}_{i-1}(x, z, z_d) + p_i(z) \cdot \alpha_i(\psi_{i-1}(x, z, z_d)), \quad i = 1, \ldots, m$$

其中：
- $z \in \mathbb{R}^d$：环境特征（如 CNN 从图像提取的特征）
- $z_d = (z^{(1)}, \ldots, z^{(m-1)})$：$z$ 的时间导数
- $p_i: \mathbb{R}^d \to \mathbb{R}_{>0}$：**环境依赖的惩罚函数**
- $\alpha_i$：标准 [[Class K 函数]]

### 2.2 dCBF 约束

$$L_f^m b(x) + L_g L_f^{m-1} b(x) \cdot u + O'(b, z, z_d) + p_m(z) \cdot \alpha_m(\psi_{m-1}) \geq 0$$

### 2.3 安全性定理

**定理**：如果 $p_i(z)$ 满足以下条件之一，则 dCBF 仍然保证 [[前向不变性 (Forward Invariance)]]：

1. **$p_i$ 是可训练参数**（不随时间变化）
2. **$p_i(z)$ 可微且正值**，且 $z_d$ 已知

**推论**（实用情况）：当 $\alpha_i$ 为线性函数且 $z_d = 0$（即 $p_i$ 在每个时间步内视为常数）时，安全性保证成立。

**证明核心**：利用 [[比较引理 (Comparison Lemma)]]，$\dot{\psi}_{m-1} \geq -p_m \alpha_m(\psi_{m-1})$ 的解满足 $\psi_{m-1}(t) \geq 0$，递推到 $\psi_0 = b \geq 0$。

---

## 3. 直觉解释

### 3.1 弹簧比喻

将 $\alpha_i$ 想象成一个弹簧，$p_i$ 是弹簧的刚度：

- **空旷场景**：弹簧柔软（$p_i$ 小），允许系统自由移动
- **危险场景**：弹簧刚硬（$p_i$ 大），强力推回安全区域

### 3.2 数值例子

设 $b(x) = d - 6$，当前 $d = 10, v = 2$，$\alpha(r) = r$

**传统 HOCBF**（$\alpha$ 固定为 1）：

$\dot{b} + 1 \cdot b = -v + (d-6) = -2 + 4 = 2 \geq 0$ ✅

**dCBF**（空旷场景，$p = 0.5$）：

$\dot{b} + 0.5 \cdot b = -2 + 0.5 \times 4 = 0 \geq 0$ ✅（约束更紧但仍满足）

**dCBF**（危险场景，$p = 3.0$）：

$\dot{b} + 3.0 \cdot b = -2 + 3.0 \times 4 = 10 \geq 0$ ✅（约束宽松，很容易满足）

**效果**：$p$ 大 → 约束宽松（允许更多自由），$p$ 小 → 约束严格（要求更早干预）。

---

## 4. 在 BarrierNet 中的角色

dCBF 是 [[BarrierNet]] 的核心组件。BarrierNet 将 dCBF 嵌入 [[可微 QP (Differentiable QP)]]：

$$u^* = \arg\min_u \frac{1}{2} u^T H(z) u + F(z)^T u$$

$$\text{s.t.} \quad L_f^m b + L_g L_f^{m-1} b \cdot u + O' + p_m(z) \alpha_m(\psi_{m-1}) \geq 0$$

其中 $H, F, p_m$ 都是上游神经网络的输出。

**关键创新**：整个 QP 是**可微的**，梯度可以通过 [[KKT 条件]] 反向传播到 $H, F, p_m$ 的网络参数，实现端到端训练。

---

## 5. Penalty 函数 $p_i(z)$ 的设计

### 5.1 网络结构

```python
class PenaltyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, p_min=0.1):
        super().__init__()
        self.p_min = p_min
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        """
        输入: z (环境特征)
        输出: p > 0 (惩罚值)
        """
        raw = self.net(z)
        # Softplus 保证正值且可微
        # p_min 保证下界
        return F.softplus(raw) + self.p_min
```

### 5.2 输入 $z$ 的选择

| 方案 | 输入 $z$ | 优点 | 缺点 |
|------|---------|------|------|
| 状态依赖 | $z = s$（系统状态） | 简单直接 | 不能利用感知信息 |
| 特征依赖 | $z = \text{CNN}(\text{image})$ | 利用视觉信息 | 网络更复杂 |
| 混合 | $z = [s, \text{CNN}(\text{image})]$ | 最全面 | 维度高 |

---

## 6. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCBF(nn.Module):
    """
    Differentiable Control Barrier Function
    
    将 HOCBF 的 alpha 替换为 p(z) * alpha，
    p(z) 是可学习的惩罚函数
    """
    def __init__(self, d_safe=6.0, phi=1.0, alpha_coeff=1.0,
                 p_min=0.1, state_dim=2):
        super().__init__()
        self.d_safe = d_safe
        self.phi = phi
        self.alpha_coeff = alpha_coeff
        self.p_min = p_min
        
        # 惩罚网络
        self.penalty_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def compute_b(self, s):
        """b(x) = d - d_safe - phi * v"""
        d = s[:, 0]
        v = s[:, 1]
        return d - self.d_safe - self.phi * v
    
    def compute_penalty(self, s):
        """计算 p(z) > 0"""
        raw = self.penalty_net(s)
        return F.softplus(raw) + self.p_min
    
    def compute_constraint(self, s):
        """
        计算 dCBF 约束
        
        对于 b = d - d_safe - phi*v, 相对度 m=1:
            Lf_b = -v, Lg_b = phi
            dCBF: Lf_b + Lg_b * u + p * alpha * b >= 0
            → phi * u >= v - p * alpha * b
            → u >= (v - p * alpha * b) / phi
        """
        b = self.compute_b(s)
        v = s[:, 1]
        p = self.compute_penalty(s)
        
        # dCBF 约束下界
        u_min_dcbf = (v - p.squeeze() * self.alpha_coeff * b) / self.phi
        
        return u_min_dcbf, b, p
    
    def forward(self, s, u_ref, u_min=-3.0, u_max=3.0):
        """
        dCBF 安全过滤器（可微）
        
        输入:
            s: (batch, 2) 状态
            u_ref: (batch,) 参考控制
        输出:
            u_safe: (batch,) 安全控制
            info: dict 调试信息
        """
        u_min_dcbf, b, p = self.compute_constraint(s)
        
        # 投影到安全集
        u_lower = torch.clamp(u_min_dcbf, max=u_max)
        u_safe = torch.clamp(u_ref, min=u_lower, max=u_max)
        
        info = {'b': b, 'p': p, 'u_min_dcbf': u_min_dcbf}
        return u_safe, info
```

---

## 7. 与其他方法的对比

| 方法 | $\alpha$ 类型 | 可学习 | 安全保证 | 保守性 |
|------|-------------|--------|---------|--------|
| 传统 [[CBF (控制障碍函数)]] | 固定 | ❌ | 确定性 | 高 |
| [[HOCBF (高阶控制障碍函数)]] | 固定 | ❌ | 确定性 | 高 |
| AdaCBF (自适应) | 状态依赖 | ❌ | 确定性 | 中 |
| **dCBF** (本方法) | 环境依赖 | ✅ | 确定性 | 低 |
| [[SBF (随机障碍函数)]] | 固定 | ❌ | 概率性 | 中 |

---

## 8. 相关概念

- [[CBF (控制障碍函数)]] — 基础
- [[HOCBF (高阶控制障碍函数)]] — dCBF 的前身
- [[BarrierNet]] — 使用 dCBF 的完整框架
- [[可微 QP (Differentiable QP)]] — dCBF 嵌入的可微优化层
- [[KKT 条件]] — 反向传播的数学基础
- [[Class K 函数]] — $\alpha_i$ 的定义
- [[前向不变性 (Forward Invariance)]] — 安全保证
- [[比较引理 (Comparison Lemma)]] — dCBF 安全性的证明工具

---

> **参考论文**: 
> - Wei Xiao et al., "BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control," IEEE TRO 2023
