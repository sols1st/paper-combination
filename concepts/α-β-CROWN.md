# α-β-CROWN

> **一句话**：$\alpha$-$\beta$-CROWN 是目前**最先进的神经网络验证器**之一，在 CROWN 的基础上引入了 $\alpha$ 优化和 $\beta$ 约束，获得比 [[CROWN (神经网络验证)]] 更紧的界。

---

## 1. CROWN 的局限

[[CROWN (神经网络验证)]] 用固定的线性界近似激活函数。对于部分激活的 ReLU（$x^L < 0 < x^U$），CROWN 的下界使用固定斜率 $\frac{x^U}{x^U - x^L}$，但**这不是最优的**。

---

## 2. $\alpha$-CROWN 的改进

**核心思想**：让 ReLU 线性界的斜率 $\alpha$ 成为**可优化参数**。

对于部分激活的 ReLU：

$$\alpha^L x \leq \text{ReLU}(x) \leq \alpha^U x + \beta^U$$

其中 $\alpha^L \in [0, 1]$ 是可调参数（不再固定为 $\frac{x^U}{x^U - x^L}$）。

**优化目标**：选择 $\alpha^L$ 使得最终输出界最紧：

$$\max_{\alpha^L} \text{output\_lower\_bound}(\alpha^L)$$

这可以通过梯度下降高效求解。

---

## 3. $\beta$-CROWN 的改进

**核心思想**：利用已知的**中间层约束**来收紧界。

例如，如果已知某个中间层输出 $h_i \geq 0$（比如 ReLU 后的值），可以将这个约束作为额外的**$\beta$ 项**加入线性界：

$$f(x) \geq w^T x + b + \sum_i \beta_i \cdot \max(0, -h_i)$$

其中 $\beta_i \geq 0$ 是可优化参数。

---

## 4. 完整 $\alpha$-$\beta$-CROWN 流程

1. **前向传播**：计算每层的 [[IBP (区间界传播)]] 区间
2. **反向传播**：用 CROWN 方法传播线性界
3. **$\alpha$ 优化**：梯度下降优化 ReLU 的斜率参数
4. **$\beta$ 优化**：引入中间层约束的 $\beta$ 参数
5. **分支定界** (Branch-and-Bound)：对输入空间分支，递归验证

---

## 5. 与其他方法的对比

| 方法 | 精度 | 速度 | 特点 |
|------|------|------|------|
| [[IBP (区间界传播)]] | ★☆☆ | ★★★ | 最快 |
| [[CROWN (神经网络验证)]] | ★★☆ | ★★☆ | 线性近似 |
| **$\alpha$-CROWN** | ★★★ | ★★☆ | 可优化斜率 |
| **$\alpha$-$\beta$-CROWN** | ★★★★ | ★☆☆ | 最精确 |
| [[dReal]] | ★★★★★ | ★☆☆ | SMT 精确验证 |

---

## 6. 代码使用

$\alpha$-$\beta$-CROWN 通过 [[auto_LiRPA]] 库使用：

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# 包装模型
model = MyNeuralNet()
bounded_model = BoundedModule(model, torch.randn(1, 2))

# 定义输入扰动
ptb = PerturbationLpNorm(norm=float("inf"), eps=0.01)
x = torch.tensor([[1.0, 2.0]])
bounded_x = BoundedTensor(x, ptb)

# 计算 CROWN 界
lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN")

# 计算 alpha-CROWN 界（更紧）
lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN-optimized")
```

---

## 7. 相关概念

- [[CROWN (神经网络验证)]] — 基础
- [[IBP (区间界传播)]] — 更快的替代
- [[auto_LiRPA]] — 实现库
- [[dReal]] — SMT-based 替代

---

> **参考**: 
> - Xu et al., "Fast and Complete: Neural Network Verification with Tight Linear Bounds," ICLR 2021
> - Wang et al., "$\beta$-CROWN: Efficiently Certifying Neural Networks with General Specifications," NeurIPS 2021
