# 前向不变性 (Forward Invariance)

> **一句话**：前向不变性是安全控制的核心性质——如果一个系统从安全集合 $C$ 内出发，**永远留在 $C$ 内**，则 $C$ 是前向不变的。[[CBF (控制障碍函数)]] 的全部目的就是证明和保证安全集合的前向不变性。

---

## 1. 直觉理解

### 1.1 池塘类比

想象一个被栅栏围住的池塘：

```
    ╔══════════════════════╗  ← 栅栏（安全边界 ∂C）
    ║   🐟                 ║  ← 鱼（系统状态 x(t)）
    ║        🐟            ║
    ║   🐟                 ║
    ╚══════════════════════╝
    
    前向不变 = 鱼永远游不出栅栏
```

- 栅栏 = 安全边界 $\partial C$
- 池塘内部 = 安全集合 $C$
- 鱼的运动 = 系统动态 $\dot{x} = f(x, u)$

**前向不变**：鱼无论怎么游，都不会穿过栅栏。

### 1.2 更精确的说法

如果在 $t = 0$ 时鱼在池塘内（$x(0) \in C$），那么对所有 $t \geq 0$，鱼都在池塘内（$x(t) \in C$）。

---

## 2. 形式定义

### 2.1 安全集合

定义安全集合 $C$ 为函数 $h(x)$ 的**上水平集**：

$$C = \{x \in \mathbb{R}^n : h(x) \geq 0\}$$

边界：$\partial C = \{x : h(x) = 0\}$

内部：$\text{Int}(C) = \{x : h(x) > 0\}$

### 2.2 前向不变性

集合 $C$ 对于系统 $\dot{x} = f(x, u)$ 是**前向不变的**，如果：

$$x(0) \in C \implies x(t) \in C, \quad \forall t \geq 0$$

等价地：

$$h(x(0)) \geq 0 \implies h(x(t)) \geq 0, \quad \forall t \geq 0$$

### 2.3 安全性 (Safety)

**安全性** = 前向不变性 + 初始条件在安全集合内。

即：如果 $x(0) \in C$ 且 $C$ 前向不变，则系统是安全的。

---

## 3. Nagumo 定理

### 3.1 定理陈述

**Nagumo 定理**（1942）是判断前向不变性的经典工具：

集合 $C = \{x : h(x) \geq 0\}$ 是前向不变的，**当且仅当**在边界 $\partial C$ 上：

$$\nabla h(x) \cdot f(x) \geq 0, \quad \forall x \in \partial C$$

即 $L_f h(x) \geq 0$ 在边界上成立。

### 3.2 直觉

在边界 $\partial C$ 上（$h(x) = 0$）：

- $\nabla h(x)$ 是边界的**外法向量**（指向 $C$ 内部）
- $f(x)$ 是系统的**速度向量**
- $\nabla h \cdot f \geq 0$ 意味着速度向量**不指向外部**

```
    安全区域 C
    ┌──────────────────┐
    │                  │
    │     x(t) →       │  ← 速度指向内部 ✅
    │                  │
    └──────────────────┘ ∂C
    
    ∇h ↑ (指向内部)
    f  → (不指向外部)
    ∇h · f ≥ 0 ✅
```

### 3.3 为什么需要 CBF？

Nagumo 定理要求**在边界上** $\nabla h \cdot f \geq 0$。但这有一个问题：

- 只能在边界上检查，不能在内部"提前"干预
- 当系统到达边界时，可能已经太晚了

**CBF 的改进**：不仅要求边界上安全，还要求**内部也有足够的安全裕度**：

$$L_f h + L_g h \cdot u + \alpha(h(x)) \geq 0, \quad \forall x \in C$$

$\alpha(h(x))$ 项使得在接近边界时，要求更强的"向内"力。

---

## 4. CBF 如何保证前向不变性

### 4.1 定理

如果存在 [[CBF (控制障碍函数)]] $b(x)$ 使得对所有 $x \in C$：

$$\sup_u \left[ L_f b(x) + L_g b(x) \cdot u + \alpha(b(x)) \right] \geq 0$$

并且控制器 $u$ 满足：

$$L_f b(x) + L_g b(x) \cdot u + \alpha(b(x)) \geq 0$$

则集合 $C = \{x : b(x) \geq 0\}$ 是前向不变的。

### 4.2 证明思路

1. 在边界上 $b(x) = 0$，所以 $\alpha(b) = 0$
2. CBF 条件变为 $L_f b + L_g b \cdot u \geq 0$
3. 即 $\dot{b}(x) \geq 0$（$b$ 不会减小到 0 以下）
4. 由 Nagumo 定理，$C$ 是前向不变的

### 4.3 更精确的证明

当 $b(x(t)) > 0$ 时，CBF 条件给出：

$$\dot{b} \geq -\alpha(b)$$

由 [[比较引理 (Comparison Lemma)]]，$b(x(t))$ 满足：

$$b(x(t)) \geq \beta(b(x(0)), t)$$

其中 $\beta \in \mathcal{KL}$。因为 $\beta \geq 0$，所以 $b(x(t)) \geq 0$。

---

## 5. 数值例子

### 5.1 简单 1D 系统

$$\dot{x} = u, \quad C = \{x : x \geq 1\}$$

取 $h(x) = x - 1$，则 $C = \{x \geq 1\}$。

**Nagumo 条件**：在 $x = 1$ 处，$\nabla h \cdot f = 1 \cdot u = u \geq 0$。

即：在边界 $x = 1$ 上，速度必须 $\geq 0$（不能继续向左）。

**CBF 加强版**：

$$u + \gamma(x - 1) \geq 0$$

- 当 $x = 3$（远离边界）：$u \geq -2\gamma$，允许向左
- 当 $x = 1.1$（接近边界）：$u \geq -0.1\gamma$，几乎不允许向左
- 当 $x = 1.0$（在边界上）：$u \geq 0$，禁止向左

### 5.2 仿真

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_forward_invariance(x0, gamma=1.0, T=10.0, dt=0.01):
    """
    仿真 CBF 保证前向不变性
    
    系统: ẋ = u_ref + u_cbf
    CBF: b(x) = x - 1
    """
    n_steps = int(T / dt)
    x = np.zeros(n_steps)
    x[0] = x0
    u_ref_arr = np.zeros(n_steps)
    u_cbf_arr = np.zeros(n_steps)
    
    for i in range(n_steps - 1):
        # 参考控制: 向左移动（危险方向）
        u_ref = -1.0
        u_ref_arr[i] = u_ref
        
        # CBF 安全过滤
        b = x[i] - 1.0
        u_cbf = max(0, -u_ref - gamma * b)  # 补偿
        u_cbf_arr[i] = u_cbf
        
        # 总控制
        u = u_ref + u_cbf
        x[i + 1] = x[i] + u * dt
    
    return x, u_ref_arr, u_cbf_arr

# 仿真
x, u_ref, u_cbf = simulate_forward_invariance(x0=5.0)

# 画图
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(x)
axes[0].axhline(y=1.0, color='r', linestyle='--', label='Safety boundary (x=1)')
axes[0].set_ylabel('x(t)')
axes[0].legend()
axes[0].set_title('Forward Invariance: x(t) never crosses x=1')

axes[1].plot(u_ref, label='u_ref (wants to go left)')
axes[1].plot(u_cbf, label='u_cbf (safety correction)', color='g')
axes[1].set_ylabel('Control')
axes[1].set_xlabel('Time step')
axes[1].legend()
plt.tight_layout()
plt.savefig('forward_invariance_demo.png')
```

---

## 6. 前向不变性的变体

### 6.1 有限时间前向不变性

$$x(0) \in C \implies x(t) \in C, \quad \forall t \in [0, T]$$

只在有限时间 $T$ 内保证安全。适用于：
- 任务有明确终止时间
- 长期安全性难以保证

### 6.2 概率前向不变性

$$\mathbb{P}(x(t) \in C, \forall t \geq 0 \mid x(0) \in C) \geq 1 - \epsilon$$

在 [[SBF SBC (随机障碍函数与证书)]] 中使用，允许小概率违反安全约束。

### 6.3 渐近稳定性 + 前向不变性

如果 $C$ 前向不变，且存在吸引子 $x^* \in C$：

$$x(0) \in C \implies x(t) \to x^* \in C$$

这是理想情况：系统不仅安全，还能达到目标。

---

## 7. 前向不变性 vs 其他概念

| 概念 | 含义 | 关系 |
|------|------|------|
| **前向不变性** | $x(0) \in C \Rightarrow x(t) \in C$ | 基本安全性质 |
| **稳定性** | $x(t) \to x^*$ | 目标达成 |
| **可达性** | $\exists u: x(0) \to x_f$ | 能否到达目标 |
| **鲁棒不变性** | 含不确定性时的不变性 | 更强的保证 |
| **有限时间安全** | $x(t) \in C, t \in [0, T]$ | 弱化的保证 |

---

## 8. 代码实现

### 8.1 验证前向不变性（数值方法）

```python
import torch
import numpy as np

def verify_forward_invariance(
    dynamics_fn, 
    controller_fn, 
    safety_fn,
    x0_samples,
    T=10.0, 
    dt=0.01,
    tol=1e-4
):
    """
    数值验证前向不变性
    
    输入:
        dynamics_fn: callable(x, u) -> ẋ
        controller_fn: callable(x) -> u
        safety_fn: callable(x) -> h(x) (安全函数, h >= 0 为安全)
        x0_samples: (N, n) 初始状态采样
        T: 仿真时间
        dt: 时间步长
    输出:
        is_invariant: 是否前向不变
        min_h: 最小的 h(x) 值（越接近 0 越危险）
        violations: 违反安全的轨迹
    """
    n_steps = int(T / dt)
    violations = []
    min_h = float('inf')
    
    for x0 in x0_samples:
        x = x0.clone()
        h_min_traj = float('inf')
        
        for step in range(n_steps):
            h = safety_fn(x).item()
            h_min_traj = min(h_min_traj, h)
            min_h = min(min_h, h)
            
            if h < -tol:
                violations.append({
                    'x0': x0.tolist(),
                    'step': step,
                    'x': x.tolist(),
                    'h': h
                })
                break
            
            u = controller_fn(x)
            x_dot = dynamics_fn(x, u)
            x = x + x_dot * dt
        
        if h_min_traj < -tol:
            continue
    
    return {
        'is_invariant': len(violations) == 0,
        'min_h': min_h,
        'n_violations': len(violations),
        'violations': violations[:5]  # 只返回前 5 个
    }

# AEBS 测试
def aebs_dynamics(x, u):
    """ẋ = [ve - v, u]"""
    ve = 20.0
    return torch.tensor([ve - x[1].item(), u.item()])

def aebs_controller(x):
    """简单的 CBF 控制器"""
    d, v = x[0].item(), x[1].item()
    d_safe, T_gap = 6.0, 1.5
    b = d - d_safe - T_gap * v
    u_ref = 0.0  # 匀速
    if b < 2.0:
        return torch.tensor([max(-3.0, -b)])  # 减速
    return torch.tensor([u_ref])

def aebs_safety(x):
    d, v = x[0].item(), x[1].item()
    return torch.tensor(d - 6.0 - 1.5 * v)

# 随机采样初始状态
x0_samples = torch.tensor([
    [20.0, 10.0],
    [15.0, 15.0],
    [10.0, 20.0],
    [30.0, 5.0],
])

result = verify_forward_invariance(
    aebs_dynamics, aebs_controller, aebs_safety,
    x0_samples, T=20.0, dt=0.01
)
print(f"前向不变: {result['is_invariant']}")
print(f"最小安全裕度: {result['min_h']:.4f}")
```

### 8.2 用 CBF-QP 保证前向不变性

```python
def cbf_safe_filter(x, u_ref, dynamics_params, cbf_params):
    """
    CBF 安全过滤器：保证前向不变性
    
    将任意参考控制 u_ref 投影到安全集合上
    """
    d, v = x
    d_safe = cbf_params['d_safe']
    T_gap = cbf_params['T_gap']
    gamma = cbf_params['gamma']
    
    # CBF
    b = d - d_safe - T_gap * v
    
    # Lie 导数
    Lf_b = -v + dynamics_params['ve']  # 假设 ve 恒定
    Lg_b = -T_gap  # 对本车加速度 u 的导数
    # 注意: 这里 u 是减速度，符号需要仔细处理
    
    # CBF 约束: Lf_b + Lg_b * u + gamma * b >= 0
    # -v + ve + (-T_gap) * u + gamma * b >= 0
    # u <= (-v + ve + gamma * b) / T_gap
    
    u_max_safe = (-v + dynamics_params['ve'] + gamma * b) / T_gap
    
    # 投影 u_ref 到安全集合
    u_min, u_max = -3.0, 3.0
    u_safe = max(u_min, min(u_ref, u_max_safe, u_max))
    
    return u_safe
```

---

## 9. 前向不变性的常见误解

| 误解 | 正确理解 |
|------|---------|
| "前向不变 = 稳定" | 错！前向不变只保证在 $C$ 内，不保证收敛 |
| "CBF 保证绝对安全" | CBF 保证的是模型内的安全，模型不准确时可能不安全 |
| "边界上 $h=0$ 才需要关心" | CBF 要求在 $C$ 内部也满足条件，提前干预 |
| "前向不变就是永远不出事" | 只在模型正确、控制器完美执行时成立 |

---

## 10. 相关概念

- [[CBF (控制障碍函数)]] — 保证前向不变性的主要工具
- [[HOCBF (高阶控制障碍函数)]] — 高相对度系统的前向不变性
- [[Class K 函数]] — 控制前向不变性的"积极程度"
- [[比较引理 (Comparison Lemma)]] — 前向不变性证明的关键引理
- [[Nagumo 定理]] — 前向不变性的经典判定条件（本文第 3 节）
- [[SBF SBC (随机障碍函数与证书)]] — 概率版本的前向不变性

---

> **参考**: 
> - Nagumo, "Über die Lage der Integralkurven gewöhnlicher Differentialgleichungen," 1942
> - Ames et al., "Control Barrier Functions: Theory and Applications," ECC 2019
> - Blanchini, "Set Invariance in Control," Automatica 1999
