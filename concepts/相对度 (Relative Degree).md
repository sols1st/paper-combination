# 相对度 (Relative Degree)

> **一句话**：相对度是控制理论中的一个整数——它表示**控制输入 $u$ 需要经过多少次微分才会出现在输出中**。在 [[HOCBF (高阶控制障碍函数)]] 中，相对度决定了需要构造多少层 $\psi_i$ 函数。

---

## 1. 直觉理解

### 1.1 日常类比

想象你在控制一辆车的**位置**：

- **直接控制位置**（如机器人手臂）：$u$ 直接改变位置 → 相对度 = 1
- **控制速度来影响位置**（如汽车油门）：$u$ → 速度 → 位置，需要一次积分 → 相对度 = 2
- **控制加速度来影响位置**（如火箭推力）：$u$ → 加速度 → 速度 → 位置 → 相对度 = 3

### 1.2 图示

```
相对度 = 1:    u ──→ y（直接）
相对度 = 2:    u ──→ ẏ ──→ y
相对度 = 3:    u ──→ ÿ ──→ ẏ ──→ y
```

---

## 2. 形式定义

### 2.1 系统模型

考虑非线性系统：

$$\dot{x} = f(x) + g(x) u$$

输出函数（如 CBF）：$y = h(x)$

### 2.2 相对度的定义

输出 $h(x)$ 关于系统 $(f, g)$ 的**相对度** $m$ 是最小整数，使得：

$$L_g L_f^{m-1} h(x) \neq 0$$

其中 $L_f, L_g$ 是 [[Lie 导数]]。

等价地说：

| 微分次数 | 结果 | 是否包含 $u$ |
|---------|------|------------|
| $\dot{h} = L_f h$ | 仅含 $x$ | 否（如果 $L_g h = 0$） |
| $\ddot{h} = L_f^2 h + L_g L_f h \cdot u$ | 含 $u$ | 是（如果 $L_g L_f h \neq 0$） |

**相对度 = 第一次出现 $u$ 的微分次数**。

---

## 3. 计算方法

### 3.1 逐步求导法

对 $h(x)$ 反复求导，直到 $u$ 出现：

**第 0 步**：$h(x)$ — 不含 $u$

**第 1 步**：$\dot{h} = \nabla h \cdot \dot{x} = \nabla h \cdot (f + gu) = L_f h + L_g h \cdot u$

- 如果 $L_g h(x) \neq 0$ → 相对度 $m = 1$
- 如果 $L_g h(x) = 0$ → 继续

**第 2 步**：$\ddot{h} = \frac{d}{dt}(L_f h) = L_f^2 h + L_g L_f h \cdot u$

- 如果 $L_g L_f h(x) \neq 0$ → 相对度 $m = 2$
- 如果 $L_g L_f h(x) = 0$ → 继续

**第 $k$ 步**：$h^{(k)} = L_f^k h + L_g L_f^{k-1} h \cdot u$

- 如果 $L_g L_f^{k-1} h(x) \neq 0$ → 相对度 $m = k$

### 3.2 AEBS 例子

**系统**：

$$\dot{d} = v_e - v \quad \text{(相对距离变化率)}$$
$$\dot{v} = u \quad \text{(本车加速度)}$$

其中 $x = (d, v)$，$f(x) = \begin{pmatrix} v_e - v \\ 0 \end{pmatrix}$，$g(x) = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

**CBF**：$b(x) = d - d_{\text{safe}} - T v$（安全距离函数）

**第 1 步**：

$$\nabla b = \begin{pmatrix} 1 \\ -T \end{pmatrix}$$

$$L_g b = \nabla b \cdot g = (1)(0) + (-T)(1) = -T$$

**如果 $T \neq 0$**：$L_g b = -T \neq 0$ → 相对度 $m = 1$

**如果 $T = 0$（纯距离 CBF）**：$b(x) = d - d_{\text{safe}}$

$$L_g b = (1)(0) + (0)(1) = 0$$

需要第 2 步：

$$L_f b = v_e - v$$

$$\nabla (L_f b) = \begin{pmatrix} 0 \\ -1 \end{pmatrix}$$

$$L_g L_f b = (0)(0) + (-1)(1) = -1 \neq 0$$

→ 相对度 $m = 2$

---

## 4. 相对度与 HOCBF 的关系

在 [[HOCBF (高阶控制障碍函数)]] 中，相对度 $m$ 决定了需要构造多少层 $\psi$ 函数：

$$\psi_0 = b(x)$$
$$\psi_1 = \dot{\psi}_0 + \alpha_1(\psi_0)$$
$$\vdots$$
$$\psi_{m-1} = \dot{\psi}_{m-2} + \alpha_{m-1}(\psi_{m-2})$$

最终的安全约束在 $\psi_{m-1}$ 层：

$$L_f \psi_{m-1} + L_g \psi_{m-1} \cdot u + \alpha_m(\psi_{m-1}) \geq 0$$

因为 $L_g \psi_{m-1} \neq 0$（这正是相对度 $m$ 的含义），所以 $u$ 出现在约束中，QP 可以求解。

### 4.1 相对度 1 → 标准 CBF

$$L_f b + L_g b \cdot u + \alpha(b) \geq 0$$

$u$ 直接出现，一步到位。

### 4.2 相对度 2 → 需要 HOCBF

$$\psi_0 = b(x)$$
$$\psi_1 = \dot{b}(x) + \alpha_1(b(x)) = L_f b + \alpha_1(b)$$

安全约束：

$$L_f \psi_1 + L_g \psi_1 \cdot u + \alpha_2(\psi_1) \geq 0$$

展开后 $u$ 出现（因为 $L_g \psi_1 = L_g L_f b \neq 0$）。

---

## 5. 不同系统的典型相对度

| 系统 | 状态 | 控制 | 输出/CBF | 相对度 |
|------|------|------|---------|--------|
| 一阶积分器 | $\dot{x} = u$ | $u$ | $x$ | 1 |
| 二阶积分器 | $\ddot{x} = u$ | $u$ | $x$ | 2 |
| 二阶积分器 | $\ddot{x} = u$ | $u$ | $\dot{x}$ | 1 |
| AEBS（带 T） | $\dot{d}=v_e-v, \dot{v}=u$ | $u$ | $d - d_s - Tv$ | 1 |
| AEBS（纯距离） | 同上 | $u$ | $d - d_s$ | 2 |
| 倒立摆 | $\ddot{\theta} = f(\theta, \dot{\theta}, u)$ | $u$ | $\theta$ | 2 |
| 四旋翼位置 | $\ddddot{p} = f(\cdot, u)$ | $u$ | $p$ | 4 |

---

## 6. 代码实现

### 6.1 自动计算相对度

```python
import torch
import sympy as sp

def compute_relative_degree(f_sym, g_sym, h_sym, x_sym):
    """
    用 SymPy 符号计算相对度
    
    输入:
        f_sym: 符号向量 f(x)
        g_sym: 符号向量 g(x)
        h_sym: 符号标量 h(x) (CBF)
        x_sym: 符号向量 x
    输出:
        m: 相对度
        details: 每步的 Lie 导数
    """
    n = len(x_sym)
    details = []
    
    # 计算梯度辅助函数
    def lie_f(h):
        grad_h = sp.Matrix([sp.diff(h, xi) for xi in x_sym])
        return grad_h.dot(f_sym)
    
    def lie_g(h):
        grad_h = sp.Matrix([sp.diff(h, xi) for xi in x_sym])
        return grad_h.dot(g_sym)
    
    current_h = h_sym
    
    for m in range(1, n + 2):
        lg = lie_g(current_h)
        details.append({
            'step': m,
            'Lg_Lf^(m-1)_h': sp.simplify(lg),
            'Lf^m_h': sp.simplify(lie_f(current_h))
        })
        
        # 检查 Lg 是否非零
        if lg != 0:
            return m, details
        
        # 继续: current_h = Lf(current_h)
        current_h = lie_f(current_h)
    
    raise ValueError("无法确定相对度（可能无穷大）")

# AEBS 例子
d, v, ve, T, d_safe = sp.symbols('d v ve T d_safe')
x_sym = sp.Matrix([d, v])
f_sym = sp.Matrix([ve - v, 0])
g_sym = sp.Matrix([0, 1])

# 情况 1: b = d - d_safe - T*v
h1 = d - d_safe - T * v
m1, details1 = compute_relative_degree(f_sym, g_sym, h1, x_sym)
print(f"b = d - d_safe - T*v: 相对度 = {m1}")
# 输出: 相对度 = 1

# 情况 2: b = d - d_safe (T=0)
h2 = d - d_safe
m2, details2 = compute_relative_degree(f_sym, g_sym, h2, x_sym)
print(f"b = d - d_safe: 相对度 = {m2}")
# 输出: 相对度 = 2
```

### 6.2 PyTorch 数值计算

```python
def relative_degree_numeric(f_fn, g_fn, h_fn, x, eps=1e-5, max_order=5):
    """
    数值计算相对度 (通过有限差分近似 Lie 导数)
    
    输入:
        f_fn: callable(x) -> (n,) 漂移向量场
        g_fn: callable(x) -> (n,) 控制向量场
        h_fn: callable(x) -> scalar 输出函数
        x: (n,) 当前状态
    输出:
        m: 估计的相对度
    """
    n = x.shape[0]
    
    def numerical_gradient(fn, x, eps=1e-5):
        """数值梯度"""
        grad = torch.zeros_like(x)
        for i in range(n):
            x_plus = x.clone()
            x_plus[i] += eps
            x_minus = x.clone()
            x_minus[i] -= eps
            grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
        return grad
    
    current_fn = h_fn
    
    for m in range(1, max_order + 1):
        grad_h = numerical_gradient(current_fn, x)
        Lg_h = grad_h @ g_fn(x)
        
        if abs(Lg_h.item()) > eps:
            return m
        
        # current_fn = Lf(current_fn)
        f_val = f_fn(x)
        def new_fn(x_new, old_fn=current_fn, f=f_fn):
            grad = numerical_gradient(old_fn, x_new)
            return grad @ f(x_new)
        
        current_fn = new_fn
    
    return max_order  # 未确定

# 测试 AEBS
def f_fn(x):
    ve = 20.0
    return torch.tensor([ve - x[1].item(), 0.0])

def g_fn(x):
    return torch.tensor([0.0, 1.0])

def h_fn(x):
    d_safe = 6.0
    return x[0] - d_safe  # b = d - d_safe (纯距离)

x_test = torch.tensor([15.0, 10.0])
m = relative_degree_numeric(f_fn, g_fn, h_fn, x_test)
print(f"纯距离 CBF 的相对度: {m}")  # 2
```

---

## 7. 相对度的物理意义

### 7.1 控制延迟

相对度 $m$ 可以理解为**控制作用的"延迟"**：

- $m = 1$：控制立即影响输出
- $m = 2$：控制先影响变化率，再影响输出
- $m = 3$：控制先影响二阶变化率，再影响变化率，再影响输出

### 7.2 可控性

如果相对度**无穷大**（$L_g L_f^k h = 0$ 对所有 $k$），则控制 $u$ **无法影响**输出 $h$——系统在 $h$ 方向不可控。

### 7.3 内部动态 (Zero Dynamics)

相对度 $m < n$（$n$ 为状态维度）时，存在 $n - m$ 个**内部状态**不受 $u$ 直接影响。这些内部状态的稳定性需要单独分析。

---

## 8. 相关概念

- [[HOCBF (高阶控制障碍函数)]] — 相对度决定 HOCBF 的层数
- [[Lie 导数]] — 计算相对度的核心工具
- [[CBF (控制障碍函数)]] — 相对度 1 的特殊情况
- [[dCBF (可微控制障碍函数)]] — 自动处理不同相对度
- [[Class K 函数]] — 每层 $\psi_i$ 中使用的函数

---

> **参考**: 
> - Isidori, "Nonlinear Control Systems," Springer 1995, Chapter 5
> - Khalil, "Nonlinear Systems," Prentice Hall 2002, Chapter 13
