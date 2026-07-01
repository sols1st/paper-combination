# CROWN (神经网络验证)

> **一句话**：CROWN (Complete Robustness verification with Optimized linear bouNds) 是一种比 [[IBP (区间界传播)]] 更精确的神经网络验证方法。它用**线性函数**来近似非线性激活函数，从而获得更紧的输出界。

---

## 1. IBP 的问题

[[IBP (区间界传播)]] 在每层独立计算区间，不保留层与层之间的相关性，导致输出界过宽。

**例子**：ReLU 在区间 $[-1, 2]$ 上的 IBP 结果是 $[0, 2]$，但这丢失了"当输入 < 0 时输出 = 0"这一信息。

---

## 2. CROWN 的核心思想

**用线性函数上下界来近似每个非线性激活函数**：

对于 ReLU $y = \text{ReLU}(x)$，$x \in [x^L, x^U]$：

$$\alpha^L x + \beta^L \leq \text{ReLU}(x) \leq \alpha^U x + \beta^U$$

选择 $\alpha, \beta$ 使得线性界尽可能紧。

### 2.1 ReLU 的三种情况

**Case 1：$x^L \geq 0$（全激活）**

$\text{ReLU}(x) = x$

$\alpha^L = \alpha^U = 1$, $\beta^L = \beta^U = 0$

**Case 2：$x^U \leq 0$（全关闭）**

$\text{ReLU}(x) = 0$

$\alpha^L = \alpha^U = 0$, $\beta^L = \beta^U = 0$

**Case 3：$x^L < 0 < x^U$（部分激活）**

下界（最优三角形）：$\alpha^L = \frac{x^U}{x^U - x^L}$, $\beta^L = 0$

上界：$\alpha^U = 1$, $\beta^U = 0$（或通过面积最小化优化）

```
     ReLU(x)
      |\
    2 | \      上界
      |  \    /
    1 |   \  /
      |    \/___ 下界
    0 +----+----→ x
     -1    0    2
```

### 2.2 其他激活函数

**Tanh**：$x \in [x^L, x^U]$

下界：连接 $(x^L, \tanh(x^L))$ 和 $(x^U, \tanh(x^U))$ 的直线

上界：如果 $x^L \geq 0$ 用切线；如果跨越 0 用分段

**Softplus**：类似处理

---

## 3. 反向传播线性界

CROWN 的精髓是**从输出到输入反向传播**线性界：

设最终输出为 $f(x)$，我们想要 $f^L \leq f(x) \leq f^U$。

**Step 1**：最后一层的激活函数用线性界替代

$$\sigma(h) \approx A^L h + b^L \leq \sigma(h) \leq A^U h + b^U$$

**Step 2**：将线性界通过前面的线性层传播

$$f(x) \approx A^L (W h + b) + b^L$$

**Step 3**：继续反向传播到更前面的层

最终得到 $f(x)$ 关于输入 $x$ 的全局线性界：

$$w^L x + b^L \leq f(x) \leq w^U x + b^U$$

---

## 4. 与 IBP 的对比

| 特性 | [[IBP (区间界传播)]] | CROWN |
|------|----------------------|-------|
| **近似方式** | 区间（矩形） | 线性函数（梯形） |
| **精度** | 低（保守） | 高（紧） |
| **速度** | $O(n)$ | $O(n^2)$ |
| **保留相关性** | ❌ | ✅ |
| **使用场景** | 快速筛选 | 精确验证 |

---

## 5. 代码概念

```python
def crown_linear(x_L, x_U, W, b):
    """CROWN 对线性层的处理与 IBP 相同"""
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    y_L = x_L @ W_pos.T + x_U @ W_neg.T + b
    y_U = x_U @ W_pos.T + x_L @ W_neg.T + b
    return y_L, y_U

def crown_relu(x_L, x_U):
    """CROWN 对 ReLU 的处理"""
    # 三种情况
    # Case 1: fully active
    mask_pos = x_L >= 0
    # Case 2: fully inactive
    mask_neg = x_U <= 0
    # Case 3: partial
    mask_partial = ~mask_pos & ~mask_neg
    
    # 计算线性界参数
    alpha_L = torch.zeros_like(x_L)
    alpha_U = torch.zeros_like(x_L)
    beta_L = torch.zeros_like(x_L)
    beta_U = torch.zeros_like(x_L)
    
    alpha_L[mask_pos] = 1.0
    alpha_U[mask_pos] = 1.0
    
    alpha_L[mask_partial] = x_U[mask_partial] / (x_U[mask_partial] - x_L[mask_partial])
    alpha_U[mask_partial] = 1.0
    beta_U[mask_partial] = -x_L[mask_partial] * alpha_U[mask_partial]
    
    return alpha_L, beta_L, alpha_U, beta_U
```

---

## 6. 在本项目中的选择

本项目主要使用 [[IBP (区间界传播)]]（而非 CROWN），原因：
1. AEBS 状态空间维度低（2D），IBP 保守性可接受
2. IBP 计算更快，适合 CEGIS 循环中的频繁验证
3. [[auto_LiRPA]] 库支持切换，可以在需要时升级到 CROWN

---

## 7. 相关概念

- [[IBP (区间界传播)]] — CROWN 的基础
- [[α-β-CROWN]] — CROWN 的改进版本
- [[auto_LiRPA]] — 实现库
- [[dReal]] — 另一种验证方法（SMT-based）

---

> **参考**: Zhang et al., "Towards Certifying L-infinity Robustness using Neural Networks with L-inf Perturbations," ICML 2021
