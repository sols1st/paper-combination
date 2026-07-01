# Spectral Normalization (谱归一化)

> **一句话**：Spectral Normalization 是一种**训练时约束神经网络 Lipschitz 常数**的技术——它通过在每次前向传播时将每层权重矩阵除以最大奇异值，保证网络的 Lipschitz 常数不超过 1。由 Miyato et al. (2018) 提出，广泛用于 GAN 和安全关键网络。

---

## 1. 为什么需要 Spectral Normalization？

### 1.1 问题

在安全关键应用中，神经网络控制器 $\pi(x)$ 的输出不能随输入剧烈变化：

$$\|\pi(x_1) - \pi(x_2)\| \leq L \|x_1 - x_2\|$$

$L$ 需要小（见 [[Lipschitz 常数]]），但普通训练不控制 $L$。

### 1.2 解决方案

**Spectral Normalization**：在每个权重矩阵 $W$ 上，除以它的**谱范数**（最大奇异值 $\sigma_{\max}$）：

$$W_{\text{SN}} = \frac{W}{\sigma_{\max}(W)}$$

这保证每层的 Lipschitz 常数 $\leq 1$，整个网络的 $L \leq 1$。

---

## 2. 数学原理

### 2.1 谱范数

矩阵 $W \in \mathbb{R}^{m \times n}$ 的**谱范数**定义为：

$$\|W\|_2 = \sigma_{\max}(W) = \max_{\|v\|=1} \|Wv\|_2$$

即 $W$ 的最大奇异值。

### 2.2 为什么谱范数控制 Lipschitz？

对于线性层 $f(x) = Wx + b$：

$$\|f(x_1) - f(x_2)\| = \|W(x_1 - x_2)\| \leq \|W\|_2 \|x_1 - x_2\|$$

如果 $\|W\|_2 \leq 1$，则 $\|f(x_1) - f(x_2)\| \leq \|x_1 - x_2\|$。

### 2.3 组合效果

$$L_{\text{network}} \leq \prod_{i} \sigma_{\max}(W_i^{\text{SN}}) = \prod_i 1 = 1$$

（假设所有激活函数的 Lipschitz 常数也 $\leq 1$，如 ReLU、Tanh）

---

## 3. 算法

### 3.1 幂迭代（Power Iteration）

直接计算 SVD 太慢（$O(n^3)$）。Spectral Normalization 用**幂迭代**近似 $\sigma_{\max}$：

```
初始化: u = 随机单位向量

每次前向传播时:
1. v = W^T u / ||W^T u||
2. u = W v / ||W v||
3. σ ≈ u^T W v

归一化:
4. W_SN = W / σ
```

**关键**：$u$ 在前向传播之间**保持**（warm start），所以每次只需要 1 次迭代就足够精确。

### 3.2 为什么 1 次迭代就够？

因为 $u$ 从上一步"继承"，已经接近真正的左奇异向量。随着训练进行，$W$ 变化缓慢，$u$ 的跟踪误差也很小。

---

## 4. 代码实现

### 4.1 PyTorch 内置

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# 方法 1: 对单个层应用
layer = nn.Linear(10, 5)
layer_sn = spectral_norm(layer)

# 方法 2: 对整个模型应用
model = nn.Sequential(
    spectral_norm(nn.Linear(2, 32)),
    nn.ReLU(),
    spectral_norm(nn.Linear(32, 16)),
    nn.ReLU(),
    spectral_norm(nn.Linear(16, 1))
)

# 正常使用
x = torch.randn(8, 2)
y = model(x)  # 自动进行谱归一化
```

### 4.2 手动实现（理解原理）

```python
class SpectralNormLinear(nn.Module):
    """
    手动实现 Spectral Normalization 的线性层
    
    每次前向传播:
    1. 用幂迭代更新 u, v
    2. 计算 σ = u^T W v
    3. 返回 (W/σ) x + b
    """
    def __init__(self, in_features, out_features, n_power_iter=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.n_power_iter = n_power_iter
        
        # 初始化 u 向量
        self.register_buffer('u', torch.randn(out_features))
        self.u.data = self.u.data / self.u.data.norm()
    
    def forward(self, x):
        W = self.linear.weight
        
        # 幂迭代
        u = self.u.clone()
        v = None
        
        for _ in range(self.n_power_iter):
            # v = W^T u / ||W^T u||
            v = torch.mv(W.t(), u)
            v = v / (v.norm() + 1e-12)
            
            # u = W v / ||W v||
            u = torch.mv(W, v)
            u = u / (u.norm() + 1e-12)
        
        # σ = u^T W v
        sigma = u @ W @ v
        
        # 更新缓冲区
        self.u.data = u.data
        
        # 归一化后的输出
        return torch.nn.functional.linear(x, W / sigma, self.bias)
    
    def get_lipschitz_bound(self):
        """返回当前谱范数（应该 ≈ 1）"""
        W = self.linear.weight
        _, S, _ = torch.linalg.svd(W)
        return S[0].item()


class SpectralNormMLP(nn.Module):
    """
    全 Spectral Normalization 的 MLP
    
    保证 Lipschitz 常数 ≤ 1
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(SpectralNormLinear(prev_dim, h_dim))
            layers.append(nn.ReLU())  # ReLU 的 Lipschitz = 1
            prev_dim = h_dim
        
        layers.append(SpectralNormLinear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def verify_lipschitz(self):
        """验证实际 Lipschitz 常数"""
        L = 1.0
        for layer in self.net:
            if isinstance(layer, SpectralNormLinear):
                L *= layer.get_lipschitz_bound()
            # ReLU 的 L = 1，不影响
        return L

# 使用
model = SpectralNormMLP(2, [32, 16], 1)
print(f"Lipschitz bound: {model.verify_lipschitz():.4f}")  # 应 ≈ 1.0

x = torch.randn(100, 2)
y = model(x)
```

### 4.3 在 CBF 控制器中的应用

```python
class SafeController(nn.Module):
    """
    带 Spectral Normalization 的安全控制器
    
    保证 Lipschitz 常数 ≤ L_max
    """
    def __init__(self, state_dim, action_dim, L_max=1.0):
        super().__init__()
        self.L_max = L_max
        
        # 用 spectral_norm 包装每层
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(state_dim, 64)),
            nn.Tanh(),  # Tanh 的 Lipschitz = 1
            spectral_norm(nn.Linear(64, 32)),
            nn.Tanh(),
            spectral_norm(nn.Linear(32, action_dim)),
        )
        
        # 输出缩放：乘以 L_max 使 Lipschitz ≤ L_max
        self.scale = L_max
    
    def forward(self, state):
        return self.net(state) * self.scale
    
    def get_lipschitz_estimate(self):
        L = 1.0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _, S, _ = torch.linalg.svd(layer.weight_orig)
                L *= S[0].item() / layer.weight_u.shape[0]  # 近似
            elif isinstance(layer, nn.Tanh):
                L *= 1.0
        return L * self.scale

# AEBS 控制器
controller = SafeController(state_dim=2, action_dim=1, L_max=2.0)
state = torch.tensor([[15.0, 10.0]])
action = controller(state)
print(f"Action: {action.item():.4f}")
```

---

## 5. 谱归一化 vs 其他方法

| 方法 | 实现 | 效果 | 训练影响 |
|------|------|------|---------|
| **Spectral Norm** | 除以 $\sigma_{\max}$ | $L \leq 1$（每层） | 轻微减慢 |
| **Weight Clipping** | $W \in [-c, c]$ | 粗糙约束 | 可能影响收敛 |
| **Gradient Penalty** | $\|\nabla f\|^2$ 惩罚 | 软约束 | 需要调参 |
| **Orthogonal Init** | $W^T W = I$ | $L = 1$（初始） | 训练后可能破坏 |

---

## 6. 注意事项

### 6.1 谱归一化不限制偏置

谱归一化只约束 $W$，不约束 $b$。因此：

$$f(x) = \frac{W}{\sigma} x + b$$

仍然可以有任意大的输出值（通过 $b$），只是**变化率**被限制。

### 6.2 输出范围控制

如果需要同时控制输出范围，可以加一个 tanh 输出层：

```python
class BoundedLipschitzNet(nn.Module):
    def __init__(self, input_dim, output_dim, L_max, y_max):
        super().__init__()
        self.L_max = L_max
        self.y_max = y_max
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            spectral_norm(nn.Linear(64, output_dim)),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x) * self.y_max  # 输出范围 [-y_max, y_max]
```

### 6.3 在 CEGIS 验证中的使用

```python
def verify_with_spectral_norm(model, safety_fn, bounds, grid_res):
    """
    利用 spectral norm 进行高效验证
    
    因为 L 已知（= 1），网格间距 δ 直接给出安全裕度
    """
    L_model = 1.0  # 由 spectral norm 保证
    L_safety = estimate_lipschitz(safety_fn, bounds)
    L_combined = L_safety * L_model
    
    delta = max((hi - lo) / grid_res for lo, hi in bounds)
    safety_margin = L_combined * delta * len(bounds)**0.5
    
    # 检查网格点
    # 如果所有网格点的 safety >= safety_margin，则全域安全
    
    return safety_margin
```

---

## 7. 相关概念

- [[Lipschitz 常数]] — Spectral Normalization 约束的目标
- [[CEGIS (反例引导合成)]] — Spectral Norm 在验证中的应用
- [[Neural Barrier Certificate]] — 使用 Spectral Norm 保证证书质量
- [[CBF (控制障碍函数)]] — 控制器的 Lipschitz 约束

---

> **参考**: 
> - Miyato et al., "Spectral Normalization for Generative Adversarial Networks," ICLR 2018
> - Gouk et al., "Regularisation of Neural Networks by Enforcing Lipschitz Continuity," 2021
