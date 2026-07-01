# Lipschitz 常数 (Lipschitz Constant)

> **一句话**：Lipschitz 常数 $L$ 衡量一个函数的**最大变化率**——它保证函数输出不会因为输入的微小变化而剧烈波动。在形式化验证和神经网络安全性中，Lipschitz 约束是保证系统**鲁棒性和可验证性**的关键工具。

---

## 1. 定义

### 1.1 Lipschitz 连续

函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是 **Lipschitz 连续**的，如果存在常数 $L \geq 0$ 使得：

$$\|f(x) - f(y)\| \leq L \|x - y\|, \quad \forall x, y$$

最小的这样的 $L$ 称为 **Lipschitz 常数**。

### 1.2 直觉

```
f(x)
 ↑        L = 10（陡）
 │       ╱
 │      ╱
 │     ╱     L = 1（平缓）
 │    ╱     ╱
 │   ╱     ╱
 ──────────────→ x
```

- $L$ 小 → 函数变化缓慢 → 输入扰动不会导致输出剧烈变化
- $L$ 大 → 函数可能变化剧烈 → 小扰动可能导致大偏差

### 1.3 一维例子

| 函数 | Lipschitz 常数 | 说明 |
|------|---------------|------|
| $f(x) = 3x$ | $L = 3$ | 线性函数，$L = $ 斜率 |
| $f(x) = x^2$ (在 $[-1, 1]$) | $L = 2$ | $L = \max |f'(x)|$ |
| $f(x) = \sin(x)$ | $L = 1$ | $|\cos(x)| \leq 1$ |
| $f(x) = |x|$ | $L = 1$ | 不可微但 Lipschitz |
| $f(x) = \sqrt{x}$ (在 $[0, 1]$) | $L = \infty$ | 在 $x=0$ 处不是 Lipschitz |

---

## 2. 在神经网络中的 Lipschitz 常数

### 2.1 各层的 Lipschitz 常数

| 层类型 | 函数 | Lipschitz 常数 |
|--------|------|---------------|
| Linear $Wx + b$ | $f(x) = Wx + b$ | $\sigma_{\max}(W)$（最大奇异值） |
| ReLU | $\max(0, x)$ | $L = 1$ |
| Tanh | $\tanh(x)$ | $L = 1$ |
| Sigmoid | $\sigma(x)$ | $L = 1/4$ |
| Softplus | $\log(1 + e^x)$ | $L = 1$ |

### 2.2 组合网络的 Lipschitz 常数

对于 $f = f_k \circ f_{k-1} \circ \cdots \circ f_1$：

$$L_f \leq \prod_{i=1}^{k} L_{f_i}$$

即各层 Lipschitz 常数的**乘积**。

**例子**：3 层 MLP = Linear → ReLU → Linear → ReLU → Linear

$$L \leq \sigma_1 \times 1 \times \sigma_2 \times 1 \times \sigma_3 = \sigma_1 \sigma_2 \sigma_3$$

---

## 3. 为什么 Lipschitz 约束重要？

### 3.1 鲁棒性保证

如果 NN 控制器 $\pi(x)$ 满足 Lipschitz 常数 $L$，那么输入扰动 $\|\delta\| \leq \epsilon$ 导致：

$$\|\pi(x + \delta) - \pi(x)\| \leq L \epsilon$$

- 小 $L$ → 即使传感器有噪声，控制量变化也不大
- 大 $L$ → 噪声可能导致控制量剧变 → 危险

### 3.2 可验证性

在 [[CEGIS (反例引导合成)]] 中，Lipschitz 常数用于**将连续验证问题离散化**：

如果在网格点 $x_i$ 上验证了安全性，并且 $L$ 已知，那么在网格间距 $\delta$ 内的任意点：

$$b(x) \geq b(x_i) - L_b \cdot \delta$$

其中 $L_b$ 是 $b \circ \pi$ 的 Lipschitz 常数。

### 3.3 收敛性

在 CBF-QP 中，如果 $\pi(x)$ 是 Lipschitz 的，则闭环系统 $\dot{x} = f(x) + g(x)\pi(x)$ 的解**存在且唯一**（由 Picard-Lindelöf 定理保证）。

---

## 4. 计算 Lipschitz 常数

### 4.1 精确计算（小网络）

```python
import torch
import torch.nn as nn

def compute_lipschitz_exact(model):
    """
    精确计算线性+ReLU 网络的 Lipschitz 常数（上界）
    
    L = Π σ_max(W_i)
    
    输入:
        model: PyTorch Sequential 模型
    输出:
        L: Lipschitz 常数上界
    """
    L = 1.0
    activation_lipschitz = {
        'ReLU': 1.0,
        'Tanh': 1.0,
        'Sigmoid': 0.25,
        'Softplus': 1.0,
        'LeakyReLU': 1.0,  # 上界
    }
    
    for layer in model:
        if isinstance(layer, nn.Linear):
            # 计算权重矩阵的最大奇异值
            _, S, _ = torch.linalg.svd(layer.weight)
            sigma_max = S[0].item()
            L *= sigma_max
        else:
            # 查找激活函数的 Lipschitz 常数
            layer_name = type(layer).__name__
            if layer_name in activation_lipschitz:
                L *= activation_lipschitz[layer_name]
    
    return L

# 测试
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

L = compute_lipschitz_exact(model)
print(f"Lipschitz 常数: {L:.4f}")
```

### 4.2 幂迭代法（大网络）

对于大矩阵，SVD 太慢。用**幂迭代**近似最大奇异值：

```python
def power_iteration(W, n_iter=20):
    """
    幂迭代法计算矩阵的最大奇异值
    
    σ_max ≈ ||Wv|| / ||v||
    
    输入:
        W: (m, n) 矩阵
        n_iter: 迭代次数
    输出:
        sigma_max: 最大奇异值近似
    """
    # 随机初始化
    v = torch.randn(W.shape[1])
    v = v / v.norm()
    
    for _ in range(n_iter):
        # u = Wv
        u = W @ v
        u = u / u.norm()
        
        # v = W'u
        v = W.T @ u
        v = v / v.norm()
    
    # σ = u'Wv
    sigma = u @ W @ v
    
    return sigma.item()

# 使用
W = torch.randn(100, 50)
sigma = power_iteration(W)
print(f"σ_max ≈ {sigma:.4f}")
# 验证
_, S, _ = torch.linalg.svd(W)
print(f"σ_max (exact) = {S[0].item():.4f}")
```

### 4.3 数值估计（黑盒方法）

```python
def estimate_lipschitz_blackbox(model, x_center, eps, n_samples=10000):
    """
    黑盒估计 Lipschitz 常数
    
    L ≈ max ||f(x1) - f(x2)|| / ||x1 - x2||
    
    输入:
        model: callable(x) -> y
        x_center: 中心点
        eps: 扰动半径
    输出:
        L_estimate: Lipschitz 常数估计
    """
    n = x_center.shape[0]
    max_ratio = 0.0
    
    for _ in range(n_samples):
        # 随机采样两个点
        x1 = x_center + eps * torch.randn(n)
        x2 = x_center + eps * torch.randn(n)
        
        y1 = model(x1)
        y2 = model(x2)
        
        ratio = (y1 - y2).norm() / (x1 - x2).norm()
        max_ratio = max(max_ratio, ratio.item())
    
    return max_ratio
```

---

## 5. Lipschitz 约束训练

### 5.1 方法概览

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| [[Spectral Normalization]] | 归一化每层权重 | 简单、无额外参数 | 限制表达能力 |
| 梯度惩罚 | 训练时加 $\|\nabla f\|$ 惩罚 | 灵活 | 只在采样点约束 |
| Weight Clipping | 限制权重范围 | 最简单 | 过于保守 |
| 正交约束 | 强制 $W^T W = I$ | 精确控制 | 太强 |

### 5.2 梯度惩罚代码

```python
def lipschitz_gradient_penalty(model, x, target_L=1.0):
    """
    梯度惩罚：鼓励 Lipschitz 常数接近 target_L
    
    penalty = E[(||∇f(x)|| - L)^2]
    """
    x.requires_grad_(True)
    y = model(x)
    
    gradients = torch.autograd.grad(
        outputs=y.sum(), inputs=x,
        create_graph=True
    )[0]
    
    grad_norm = gradients.norm(2, dim=-1)
    penalty = ((grad_norm - target_L) ** 2).mean()
    
    return penalty

# 训练循环中使用
for batch in dataloader:
    loss = criterion(model(batch), targets)
    gp = lipschitz_gradient_penalty(model, batch, target_L=1.0)
    total_loss = loss + 10.0 * gp
    total_loss.backward()
    optimizer.step()
```

---

## 6. 在安全验证中的应用

### 6.1 网格验证

```python
def grid_verify_with_lipschitz(
    safety_fn, model, 
    state_bounds, grid_size,
    L_combined=None
):
    """
    利用 Lipschitz 常数在网格上验证安全性
    
    如果在每个网格中心 c_i: safety(c_i) >= L * δ，
    则整个区域安全。
    
    输入:
        safety_fn: callable(x) -> b(x) 安全函数
        model: NN 控制器
        state_bounds: [(min, max), ...] 状态范围
        grid_size: 每维网格数
        L_combined: safety_fn ∘ model 的组合 Lipschitz 常数
    """
    n_dims = len(state_bounds)
    
    # 计算网格间距
    deltas = [(hi - lo) / grid_size for lo, hi in state_bounds]
    delta = max(deltas)
    
    if L_combined is None:
        L_combined = compute_lipschitz_exact(model)
    
    # 安全裕度需求
    required_margin = L_combined * delta * (n_dims ** 0.5)
    
    # 检查网格中心
    all_safe = True
    worst_margin = float('inf')
    
    # 生成网格中心
    centers = []
    for dim in range(n_dims):
        lo, hi = state_bounds[dim]
        d = (hi - lo) / grid_size
        centers.append(torch.linspace(lo + d/2, hi - d/2, grid_size))
    
    grid = torch.meshgrid(*centers, indexing='ij')
    points = torch.stack([g.flatten() for g in grid], dim=-1)
    
    # 批量评估
    safety_values = safety_fn(points)
    min_safety = safety_values.min().item()
    worst_margin = min_safety - required_margin
    
    all_safe = worst_margin >= 0
    
    return {
        'verified': all_safe,
        'min_safety': min_safety,
        'required_margin': required_margin,
        'worst_margin': worst_margin,
        'grid_points': len(points)
    }
```

---

## 7. Lipschitz 常数与可达性分析

在 [[Reachability Analysis]] 中，Lipschitz 常数用于**过近似**可达集：

如果 $f$ 的 Lipschitz 常数为 $L$，初始集合为 $B_r(x_0)$（半径 $r$ 的球），则：

$$f(B_r(x_0)) \subseteq B_{Lr}(f(x_0))$$

即输出的不确定半径是输入不确定半径的 $L$ 倍。

---

## 8. 相关概念

- [[Spectral Normalization]] — 在训练中约束 Lipschitz 常数的技术
- [[IBP (区间界传播)]] — 利用 Lipschitz 性质传播区间界
- [[CROWN (神经网络验证)]] — 更精确的界传播
- [[CEGIS (反例引导合成)]] — Lipschitz 常数用于网格验证
- [[Neural Barrier Certificate]] — Lipschitz 约束保证证书的有效性

---

> **参考**: 
> - Miyato et al., "Spectral Normalization for Generative Adversarial Networks," ICLR 2018
> - Fazlyab et al., "Verification of Deep Probabilistic Models," NeurIPS 2019
