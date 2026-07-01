# IBP (区间界传播, Interval Bound Propagation)

> **一句话**：IBP 是一种**神经网络验证技术**——给定输入的范围（区间），计算神经网络输出的范围（区间）。它是 [[CROWN (神经网络验证)]] 的最简单版本。

---

## 1. 为什么需要 IBP？

在安全验证中，我们需要回答：**对于所有可能的输入 $x \in [x^L, x^U]$，神经网络 $f(x)$ 的输出范围是什么？**

例如：
- 输入图像有 $\pm 0.01$ 的扰动，网络输出的安全值范围是多少？
- 状态 $s$ 在某个网格单元内，[[SBF SBC (随机障碍函数与证书)|SBC]] 值 $B(s)$ 的范围是什么？

IBP 通过**逐层传播区间界**来高效计算输出范围。

---

## 2. 逐层传播规则

### 2.1 线性层 $y = Wx + b$

给定输入区间 $x \in [x^L, x^U]$：

$$y_i^L = \sum_j \min(W_{ij} x_j^L, W_{ij} x_j^U) + b_i$$

$$y_i^U = \sum_j \max(W_{ij} x_j^L, W_{ij} x_j^U) + b_i$$

**直觉**：
- $W_{ij} > 0$：$x_j$ 越大输出越大 → 下界用 $x_j^L$，上界用 $x_j^U$
- $W_{ij} < 0$：$x_j$ 越大输出越小 → 下界用 $x_j^U$，上界用 $x_j^L$

**代码**：

```python
def ibp_linear(x_L, x_U, W, b):
    """
    线性层 IBP 传播
    
    输入:
        x_L: (batch, in_features) 下界
        x_U: (batch, in_features) 上界
        W: (out_features, in_features) 权重
        b: (out_features,) 偏置
    """
    # 正权重和负权重分别处理
    W_pos = torch.clamp(W, min=0)  # max(W, 0)
    W_neg = torch.clamp(W, max=0)  # min(W, 0)
    
    y_L = x_L @ W_pos.T + x_U @ W_neg.T + b
    y_U = x_U @ W_pos.T + x_L @ W_neg.T + b
    
    return y_L, y_U
```

### 2.2 ReLU 激活 $y = \max(0, x)$

$$y^L = \max(0, x^L)$$
$$y^U = \max(0, x^U)$$

```python
def ibp_relu(x_L, x_U):
    return torch.clamp(x_L, min=0), torch.clamp(x_U, min=0)
```

### 2.3 Tanh 激活 $y = \tanh(x)$

$$y^L = \tanh(x^L)$$
$$y^U = \tanh(x^U)$$

（因为 $\tanh$ 单调递增）

### 2.4 Softplus 激活 $y = \ln(1 + e^x)$

$$y^L = \ln(1 + e^{x^L})$$
$$y^U = \ln(1 + e^{x^U})$$

（Softplus 也单调递增）

### 2.5 LayerNorm

LayerNorm 的处理更复杂（因为涉及均值和方差的归一化），但原理相同：分别计算均值和方差的区间，然后传播。

---

## 3. 完整 IBP 流程

```python
def ibp_forward(network, x_L, x_U):
    """
    对整个网络执行 IBP 传播
    
    输入:
        network: nn.Sequential 或 nn.Module
        x_L: 输入下界
        x_U: 输入上界
    输出:
        y_L, y_U: 输出区间
    """
    h_L, h_U = x_L, x_U
    
    for layer in network:
        if isinstance(layer, nn.Linear):
            h_L, h_U = ibp_linear(h_L, h_U, layer.weight, layer.bias)
        elif isinstance(layer, nn.ReLU):
            h_L, h_U = ibp_relu(h_L, h_U)
        elif isinstance(layer, nn.Tanh):
            h_L, h_U = torch.tanh(h_L), torch.tanh(h_U)
        elif isinstance(layer, nn.Softplus):
            h_L = torch.log1p(torch.exp(h_L))
            h_U = torch.log1p(torch.exp(h_U))
    
    return h_L, h_U
```

---

## 4. 数值例子

**网络**：$y = \text{ReLU}(Wx + b)$

$W = \begin{bmatrix} 1 & -2 \\ 3 & 1 \end{bmatrix}$, $b = \begin{bmatrix} 0.5 \\ -1 \end{bmatrix}$

输入：$x_1 \in [0, 1]$, $x_2 \in [0.5, 1.5]$

**线性层**：

$y_1^L = \min(1 \times 0, 1 \times 1) + \min(-2 \times 0.5, -2 \times 1.5) + 0.5 = 0 + (-3) + 0.5 = -2.5$

$y_1^U = \max(1 \times 0, 1 \times 1) + \max(-2 \times 0.5, -2 \times 1.5) + 0.5 = 1 + (-1) + 0.5 = 0.5$

$y_2^L = \min(3 \times 0, 3 \times 1) + \min(1 \times 0.5, 1 \times 1.5) - 1 = 0 + 0.5 - 1 = -0.5$

$y_2^U = \max(3 \times 0, 3 \times 1) + \max(1 \times 0.5, 1 \times 1.5) - 1 = 3 + 1.5 - 1 = 3.5$

**ReLU**：

$\text{ReLU}(y_1) \in [\max(0, -2.5), \max(0, 0.5)] = [0, 0.5]$

$\text{ReLU}(y_2) \in [\max(0, -0.5), \max(0, 3.5)] = [0, 3.5]$

---

## 5. IBP 的保守性

### 5.1 问题

IBP 是**逐层独立**估计的，每层假设输入区间内的值可以**任意组合**。这导致输出区间过宽（保守）。

**例子**：$y = x_1 - x_1$（同一个变量），$x_1 \in [0, 1]$

真实范围：$y = 0$（恒等于 0）

IBP 计算：$y \in [0 - 1, 1 - 0] = [-1, 1]$ ❌（过宽！）

### 5.2 改进方法

| 方法 | 精度 | 速度 | 适用场景 |
|------|------|------|---------|
| **IBP** | 低 | 快 $O(n)$ | 低维、快速筛选 |
| [[CROWN (神经网络验证)]] | 中 | 中 $O(n^2)$ | 一般验证 |
| **$\alpha$-$\beta$-CROWN** | 高 | 慢 | 精确验证 |

---

## 6. 在本项目中的应用

IBP 在三个关键环节使用：

1. **PenaltyNet 区间传播**：$p_1(s)$ 的区间 $[p_1^L, p_1^U]$
2. **SBC 网络区间传播**：$B(s)$ 的区间 $[B^L, B^U]$
3. **期望上界计算**：$\mathbb{E}[B(s')] \leq \sum_i \text{pmass}_i \cdot B^U_i$

---

## 7. 相关概念

- [[CROWN (神经网络验证)]] — 比 IBP 更精确
- [[α-β-CROWN]] — 最先进的验证器
- [[auto_LiRPA]] — IBP/CROWN 的实现库
- [[SBF SBC (随机障碍函数与证书)]] — IBP 的验证目标

---

> **参考**: Gowal et al., "On the Effectiveness of Interval Bound Propagation," arXiv 2018
