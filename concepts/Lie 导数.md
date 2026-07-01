# Lie 导数

> **一句话**：Lie 导数衡量一个标量函数 $b(x)$ 沿着一个向量场 $f(x)$ 的变化率。它是 [[CBF (控制障碍函数)]] 和 [[HOCBF (高阶控制障碍函数)]] 的核心数学工具。

---

## 1. 直觉

想象你站在一座山上（$b(x)$ 是海拔），风从某个方向吹来（$f(x)$ 是风速向量）。Lie 导数告诉你：**如果你随风飘动，海拔会以什么速率变化？**

- $L_f b > 0$：你在上升
- $L_f b < 0$：你在下降
- $L_f b = 0$：你在等高线上移动

---

## 2. 数学定义

### 2.1 一阶 Lie 导数

函数 $b: \mathbb{R}^n \to \mathbb{R}$ 沿向量场 $f: \mathbb{R}^n \to \mathbb{R}^n$ 的 Lie 导数：

$$L_f b(x) = \nabla b(x) \cdot f(x) = \sum_{i=1}^n \frac{\partial b}{\partial x_i} f_i(x)$$

### 2.2 高阶 Lie 导数

$$L_f^2 b(x) = L_f(L_f b)(x) = \nabla(L_f b(x)) \cdot f(x)$$

$$L_f^k b(x) = \nabla(L_f^{k-1} b(x)) \cdot f(x)$$

### 2.3 沿不同向量场的 Lie 导数

$$L_g b(x) = \nabla b(x) \cdot g(x)$$

**混合 Lie 导数**：

$$L_g L_f b(x) = \nabla(L_f b(x)) \cdot g(x)$$

即先沿 $f$ 求导，再沿 $g$ 求导。

---

## 3. 在 CBF 中的应用

对于控制仿射系统 $\dot{x} = f(x) + g(x)u$：

$$\dot{b}(x) = \nabla b \cdot \dot{x} = \nabla b \cdot (f + gu) = L_f b + L_g b \cdot u$$

- $L_f b$：无控制时 $b$ 的自然变化率（漂移效应）
- $L_g b \cdot u$：控制对 $b$ 的影响

### 3.1 AEBS 系统完整计算

状态 $x = [d, v]^T$，$f = [-v, 0]^T$，$g = [0, -1]^T$

**安全函数** $b(x) = d - d_{\text{safe}} - \phi v$

$\nabla b = [1, -\phi]$

$L_f b = [1, -\phi] \cdot [-v, 0]^T = -v$

$L_g b = [1, -\phi] \cdot [0, -1]^T = \phi$

$L_f^2 b = \nabla(-v) \cdot [-v, 0]^T = [0, -1] \cdot [-v, 0]^T = 0$

$L_g L_f b = \nabla(-v) \cdot [0, -1]^T = [0, -1] \cdot [0, -1]^T = 1$

### 3.2 判断 [[相对度 (Relative Degree)]]

- $L_g b = \phi \neq 0$ → 相对度 1
- 若 $b = d - d_{\text{safe}}$（无速度项），$L_g b = 0$，$L_g L_f b = 1 \neq 0$ → 相对度 2

---

## 4. 代码实现

```python
import torch

def lie_derivative(b_fn, vector_field, x):
    """
    计算 Lie 导数 L_f b(x) = ∇b · f(x)
    
    输入:
        b_fn: callable, 标量函数 b(x)
        vector_field: callable, 向量场 f(x)
        x: tensor (batch, n), 需要 require_grad=True
    输出:
        L_f_b: tensor (batch,), Lie 导数值
    """
    x = x.clone().detach().requires_grad_(True)
    b_val = b_fn(x).sum()
    
    # 自动微分求梯度 ∇b
    grad_b = torch.autograd.grad(b_val, x, create_graph=True)[0]
    
    # 点乘 f(x)
    f_val = vector_field(x)
    L_f_b = (grad_b * f_val).sum(dim=1)
    
    return L_f_b

# 例子
def b_fn(x):
    return x[:, 0] - 6.0 - 1.0 * x[:, 1]  # d - 6 - v

def f_fn(x):
    v = x[:, 1]
    return torch.stack([-v, torch.zeros_like(v)], dim=1)

x = torch.tensor([[10.0, 2.0]])
Lf_b = lie_derivative(b_fn, f_fn, x)
print(f"L_f b = {Lf_b.item()}")  # 应输出 -2.0
```

---

## 5. 相关概念

- [[CBF (控制障碍函数)]] — Lie 导数的主要应用
- [[HOCBF (高阶控制障碍函数)]] — 需要高阶 Lie 导数
- [[相对度 (Relative Degree)]] — 由 Lie 导数判断

---

> **参考**: Khalil, "Nonlinear Systems," Chapter 11
