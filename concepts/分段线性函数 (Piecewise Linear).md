# 分段线性函数 (Piecewise Linear, PWL)

> **一句话**：分段线性函数是由**多个线性片段拼接**而成的函数。ReLU 激活函数是最常见的例子。在神经网络验证中（[[IBP (区间界传播)]]、[[CROWN (神经网络验证)]]），理解 PWL 结构是计算输出界的基础。

---

## 1. 定义

### 1.1 形式定义

函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 是**分段线性**的，如果存在有限个凸多面体 $P_1, \ldots, P_k$ 覆盖定义域，使得在每个 $P_i$ 上：

$$f(x) = A_i x + b_i, \quad x \in P_i$$

### 1.2 ReLU 是最简单的 PWL

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} 0 & x < 0 \\ x & x \geq 0 \end{cases}$$

```
    y
    ↑     ╱ y = x
    │    ╱
    │   ╱
    │──╱─────────→ x
    │ ╱
   0
```

两个线性片段，分界点 $x = 0$。

---

## 2. 神经网络与 PWL

### 2.1 ReLU 网络是分段仿射函数

一个 ReLU 网络 $f = L_k \circ \text{ReLU} \circ L_{k-1} \circ \cdots \circ L_1$（$L_i$ 为线性层）是**分段仿射函数**：

$$f(x) = A_{\sigma} x + b_{\sigma}$$

其中 $\sigma \in \{0, 1\}^N$ 是所有 ReLU 的**激活模式**（每个 ReLU 开/关）。

### 2.2 激活模式数量

| 网络结构 | ReLU 数 $N$ | 最大激活模式数 | 实际分区数 |
|---------|-----------|-------------|---------|
| 单层 $n \to m$ | $m$ | $2^m$ | $\leq \sum_{i=0}^{n} \binom{m}{i}$ |
| 多层 | $\sum m_i$ | $2^{\sum m_i}$ | 远小于理论值 |

**例子**：2 个 ReLU，2D 输入

```
    x₂
    ↑
    │    ╲ σ = (1,1)
    │     ╲
    │ σ=(0,1) ╲ σ=(1,0)
    │─────────╲──────→ x₁
    │ σ=(0,0)  ╲
```

4 个区域，每个区域内网络是线性的。

---

## 3. 在神经网络验证中的角色

### 3.1 IBP 中的处理

在 [[IBP (区间界传播)]] 中，ReLU 的区间传播规则：

$$\text{输入}: [l, u]$$
$$\text{输出}: [\max(0, l), \max(0, u)]$$

| 情况 | 输入区间 | 输出区间 | 行为 |
|------|---------|---------|------|
| 全激活 | $l \geq 0$ | $[l, u]$ | 恒等 |
| 全关闭 | $u \leq 0$ | $[0, 0]$ | 常数 |
| 不确定 | $l < 0 < u$ | $[0, u]$ | 保守估计 |

### 3.2 CROWN 中的线性松弛

在 [[CROWN (神经网络验证)]] 中，对"不确定"状态的 ReLU 用**线性界**近似：

```
    y
    ↑     ╱ ReLU
    │    ╱
    │   ╱  ←── 上界: y ≤ u/(u-l) * (x - l)
    │  ╱
    │ ╱  ←── 下界: y ≥ 0 或 y ≥ x
    │╱
 ───┼─────────→ x
    l    0    u
```

### 3.3 α-β-CROWN 的改进

在 [[α-β-CROWN]] 中，松弛的斜率 $\alpha$ 是**可优化的参数**，通过优化使界更紧。

---

## 4. PWL 近似

### 4.1 用 PWL 近似非线性函数

任何连续函数都可以用 PWL 近似：

```python
import torch

def pwl_approximate(fn, x_min, x_max, n_segments=10):
    """
    用分段线性函数近似一个非线性函数
    
    输入:
        fn: 目标函数
        x_min, x_max: 定义域
        n_segments: 分段数
    """
    breakpoints = torch.linspace(x_min, x_max, n_segments + 1)
    values = fn(breakpoints)
    
    def pwl_fn(x):
        """分段线性插值"""
        x = torch.clamp(x, x_min, x_max)
        
        # 找到 x 所在的区间
        idx = torch.searchsorted(breakpoints, x) - 1
        idx = torch.clamp(idx, 0, n_segments - 1)
        
        # 线性插值
        x0 = breakpoints[idx]
        x1 = breakpoints[idx + 1]
        y0 = values[idx]
        y1 = values[idx + 1]
        
        t = (x - x0) / (x1 - x0)
        return y0 * (1 - t) + y1 * t
    
    return pwl_fn, breakpoints, values
```

### 4.2 在 SMT 编码中

在 [[SMT (可满足性模理论)]] 和 [[dReal]] 中，ReLU 被编码为分段线性约束：

$$y = \text{ReLU}(x) \iff (y \geq 0) \land (y \geq x) \land (y = 0 \lor y = x)$$

---

## 5. PWL 与凸分析

### 5.1 凸 PWL 函数

$$f(x) = \max_{i} (a_i^T x + b_i)$$

是凸的（最大值运算保持凸性）。

ReLU 可以看作：$\text{ReLU}(x) = \max(0, x)$。

### 5.2 PWL Lyapunov/CBF

如果用 PWL 函数作为 CBF：

$$b(x) = \min_{i} (a_i^T x + b_i)$$

每个片区上 $b$ 是线性的，[[Lie 导数]]可以直接计算。

---

## 6. 代码：精确分析 PWL 网络

```python
import torch
import itertools

def enumerate_linear_regions(model, input_bounds, input_dim):
    """
    枚举 ReLU 网络的所有线性区域
    
    输入:
        model: Sequential(Linear, ReLU, Linear, ReLU, ...)
        input_bounds: [(min, max), ...] 输入范围
    输出:
        regions: 每个区域的 (激活模式, 仿射映射)
    """
    # 收集所有 ReLU 层的位置
    relu_indices = []
    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.ReLU):
            relu_indices.append(i)
    
    n_relu_units = sum(
        model[i-1].out_features for i in relu_indices
    )
    
    # 枚举所有可能的激活模式
    patterns = list(itertools.product([0, 1], repeat=n_relu_units))
    
    valid_regions = []
    
    for pattern in patterns:
        # 检查这个激活模式是否可达
        # (需要线性规划验证)
        # 简化: 采样检查
        valid_regions.append(pattern)
    
    return valid_regions

def compute_pwl_output_bounds(model, x_lb, x_ub):
    """
    精确计算 PWL 网络的输出界
    
    方法: 在每个线性区域上计算输出的最大/最小值
    (只适用于小网络)
    """
    # 简化: 密集采样近似
    n_samples = 10000
    n_dim = x_lb.shape[0]
    
    x_samples = torch.rand(n_samples, n_dim)
    for i in range(n_dim):
        x_samples[:, i] = x_samples[:, i] * (x_ub[i] - x_lb[i]) + x_lb[i]
    
    with torch.no_grad():
        y_samples = model(x_samples)
    
    return y_samples.min(0)[0], y_samples.max(0)[0]
```

---

## 7. 相关概念

- [[IBP (区间界传播)]] — ReLU 的区间传播
- [[CROWN (神经网络验证)]] — ReLU 的线性松弛
- [[α-β-CROWN]] — 可优化松弛
- [[SMT (可满足性模理论)]] — PWL 的逻辑编码
- [[dReal]] — PWL 的非线性验证
- [[Neural Barrier Certificate]] — PWL 网络的验证

---

> **参考**: 
> - Montufar et al., "On the Number of Linear Regions of Deep Neural Networks," NeurIPS 2014
> - Raghunathan et al., "Semidefinite relaxations for certifying robustness to adversarial examples," NeurIPS 2018
