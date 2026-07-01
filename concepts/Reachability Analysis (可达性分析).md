# Reachability Analysis (可达性分析)

> **一句话**：可达性分析计算一个动态系统从初始集合出发**能够到达的所有状态的集合**。它与 [[CBF (控制障碍函数)]] 互为"对偶"——CBF 定义"不可到达"的安全区域，而可达性分析直接计算"能到达哪里"。在安全验证中，如果可达集与不安全集不相交，则系统安全。

---

## 1. 直觉理解

### 1.1 墨水滴类比

想象在一盆水中滴一滴墨水：

```
t=0:  ● (墨水初始位置)

t=1:  ◉ (墨水扩散)

t=2:  ⬤ (继续扩散)

t=3:  ██████ (覆盖更大区域)
```

- 初始集合 = 墨水初始位置
- 可达集 = 墨水在某时刻覆盖的所有区域
- 安全性 = 墨水不碰到容器壁的某部分

### 1.2 安全含义

```
状态空间
┌──────────────────────────────┐
│                              │
│   [初始集]   [可达集]        │
│      ○  ──→  ◉◉◉◉           │
│              ◉◉◉◉◉◉          │
│                              │
│         ╔══════════╗         │
│         ║ 不安全集  ║         │
│         ╚══════════╝         │
│                              │
└──────────────────────────────┘

安全: 可达集 ∩ 不安全集 = ∅
```

---

## 2. 形式定义

### 2.1 可达集

给定系统 $\dot{x} = f(x, u)$，初始集 $X_0$，控制集 $U$：

**前向可达集**（在时刻 $t$）：

$$\text{Reach}(t) = \{x(t) : x(0) \in X_0, u(\cdot) \in \mathcal{U}\}$$

**全时间可达集**：

$$\text{Reach}(\infty) = \bigcup_{t \geq 0} \text{Reach}(t)$$

### 2.2 安全性条件

系统安全 $\iff$ $\text{Reach}(\infty) \cap X_{\text{unsafe}} = \emptyset$

---

## 3. 计算方法

### 3.1 精确计算（仅限线性系统）

对于线性系统 $\dot{x} = Ax + Bu$：

$$\text{Reach}(t) = e^{At} X_0 + \int_0^t e^{A(t-s)} B \cdot U \, ds$$

可以用 Minkowski 和精确计算（当 $X_0$ 和 $U$ 是多面体时）。

### 3.2 过近似方法（通用）

精确计算通常不可行，需要**过近似**：

| 方法 | 表示 | 优点 | 缺点 |
|------|------|------|------|
| **区间** | 超矩形 | 简单 | 保守 |
| **椭球** | 椭球 | 线性系统好 | 非线性差 |
| **多面体** | 凸多面体 | 灵活 | 面数爆炸 |
| **Zonotope** | 中心+生成器 | 高效 | 对称性限制 |
| **水平集** | Level Set | 精确 | 高维困难 |
| **Taylor 模型** | 多项式+区间 | 非线性好 | 计算量大 |

### 3.3 与 Barrier Certificate 的关系

**Barrier Certificate** 不需要计算可达集！

- 可达性：显式计算所有可达状态 → 检查是否与不安全集相交
- Barrier Certificate：找到分隔函数 → 证明可达集被限制在安全区域内

**对偶关系**：

```
可达集 ⊆ {x : b(x) ≥ 0}（安全集）
⟺ Reach(∞) ∩ {x : b(x) < 0} = ∅
```

---

## 4. 代码实现

### 4.1 蒙特卡洛可达集估计

```python
import torch
import numpy as np

def monte_carlo_reachability(
    dynamics_fn,    # callable(x, u, dt) -> x_next
    x0_samples,     # (N, n) 初始状态
    u_samples_fn,   # callable() -> 随机控制序列
    T=10.0,         # 仿真时间
    dt=0.01,
    n_rollouts=1000
):
    """
    蒙特卡洛估计可达集
    
    通过大量仿真采样可达点
    
    输入:
        dynamics_fn: 系统动态
        x0_samples: 初始状态集
        u_samples_fn: 控制采样函数
    输出:
        reachable_points: (N_total, n) 可达点集
    """
    n_steps = int(T / dt)
    all_points = []
    
    for rollout in range(n_rollouts):
        # 随机选初始状态
        idx = np.random.randint(len(x0_samples))
        x = x0_samples[idx].clone()
        
        # 随机控制序列
        u_sequence = u_samples_fn(n_steps)
        
        trajectory = [x.clone()]
        
        for t in range(n_steps):
            u = u_sequence[t]
            x = dynamics_fn(x, u, dt)
            trajectory.append(x.clone())
        
        all_points.extend(trajectory)
    
    return torch.stack(all_points)

# AEBS 例子
def aebs_dynamics_mc(x, u, dt):
    """蒙特卡洛版 AEBS 动态"""
    ve = 20.0
    d_next = x[0] + (ve - x[1]) * dt
    v_next = x[1] + u * dt
    return torch.tensor([d_next, v_next])

# 初始状态
x0 = torch.tensor([
    [15.0, 10.0],
    [20.0, 15.0],
    [25.0, 5.0],
])

# 随机控制
def random_control(n_steps):
    return torch.uniform(-3, 3, (n_steps, 1))

reach_points = monte_carlo_reachability(
    aebs_dynamics_mc, x0, random_control,
    T=5.0, dt=0.01, n_rollouts=5000
)

print(f"采样了 {len(reach_points)} 个可达点")
print(f"距离范围: [{reach_points[:, 0].min():.1f}, {reach_points[:, 0].max():.1f}]")
print(f"速度范围: [{reach_points[:, 1].min():.1f}, {reach_points[:, 1].max():.1f}]")
```

### 4.2 区间过近似

```python
def interval_reachability(A, B, x0_lb, x0_ub, u_lb, u_ub, dt, n_steps):
    """
    线性系统的区间过近似可达性分析
    
    系统: x_{k+1} = A x_k + B u_k
    初始: x_0 ∈ [x0_lb, x0_ub]
    控制: u_k ∈ [u_lb, u_ub]
    
    输出: 每步的区间界 [lb_k, ub_k]
    """
    n = A.shape[0]
    lb, ub = x0_lb.clone(), x0_ub.clone()
    
    trajectory_bounds = [(lb.clone(), ub.clone())]
    
    for step in range(n_steps):
        # x_{k+1} = A x_k + B u_k
        # 区间算术:
        new_lb = torch.zeros(n)
        new_ub = torch.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if A[i, j] >= 0:
                    new_lb[i] += A[i, j] * lb[j]
                    new_ub[i] += A[i, j] * ub[j]
                else:
                    new_lb[i] += A[i, j] * ub[j]
                    new_ub[i] += A[i, j] * lb[j]
            
            for j in range(B.shape[1]):
                if B[i, j] >= 0:
                    new_lb[i] += B[i, j] * u_lb[j]
                    new_ub[i] += B[i, j] * u_ub[j]
                else:
                    new_lb[i] += B[i, j] * u_ub[j]
                    new_ub[i] += B[i, j] * u_lb[j]
        
        lb, ub = new_lb, new_ub
        trajectory_bounds.append((lb.clone(), ub.clone()))
    
    return trajectory_bounds

# 线性化 AEBS
A = torch.tensor([[0, -1], [0, 0]], dtype=torch.float32) * 0.01 + torch.eye(2)
B = torch.tensor([[0], [0.01]], dtype=torch.float32)

x0_lb = torch.tensor([15.0, 10.0])
x0_ub = torch.tensor([25.0, 15.0])
u_lb = torch.tensor([-3.0])
u_ub = torch.tensor([3.0])

bounds = interval_reachability(A, B, x0_lb, x0_ub, u_lb, u_ub, 0.01, 100)
print(f"最终可达区间: d ∈ [{bounds[-1][0][0]:.1f}, {bounds[-1][1][0]:.1f}]")
```

### 4.3 用 Barrier Certificate 验证可达性

```python
def verify_safety_via_barrier(barrier_fn, dynamics_fn, 
                               initial_set, unsafe_set,
                               state_bounds, n_samples=10000):
    """
    用障碍证书间接验证可达性
    
    如果 b(x) ≥ 0 在初始集，且 CBF 条件满足
    → 可达集 ⊆ {x: b(x) ≥ 0}
    → 如果不安全集上 b(x) < 0，则安全
    
    输入:
        barrier_fn: callable(x) -> b(x)
        dynamics_fn: 系统动态
        initial_set: 初始集采样
        unsafe_set: 不安全集采样
    """
    # 1. 检查初始集上 b >= 0
    b_init = barrier_fn(initial_set)
    init_ok = (b_init >= -1e-6).all().item()
    
    # 2. 检查不安全集上 b < 0
    b_unsafe = barrier_fn(unsafe_set)
    unsafe_ok = (b_unsafe < 0).all().item()
    
    # 3. 检查 CBF 条件（采样验证）
    x_samples = torch.cat([initial_set, 
                           torch.randn(n_samples, initial_set.shape[1]) * 10])
    u = torch.zeros(len(x_samples), 1)  # 假设的控制器
    
    # ... 检查 Lf_b + Lg_b * u + α(b) >= 0
    
    return {
        'initial_safe': init_ok,
        'unsafe_separated': unsafe_ok,
        'min_b_init': b_init.min().item(),
        'max_b_unsafe': b_unsafe.max().item()
    }
```

---

## 5. 工具

| 工具 | 语言 | 特色 |
|------|------|------|
| **SpaceEx** | MATLAB/Flow* | 混合系统 |
| **Flow*** | C++ | 非线性系统 |
| **CORA** | MATLAB | 区间/zonotope |
| **JuliaReach** | Julia | 高性能 |
| **dReach** | Python | 与 dReal 集成 |

---

## 6. 可达性 vs 其他安全方法

| 方法 | 计算什么 | 精确性 | 可扩展性 |
|------|---------|--------|---------|
| **可达性分析** | 可达集 | 过近似 | 低维 |
| **Barrier Certificate** | 安全分隔函数 | 精确 | 中维 |
| **Lyapunov** | 稳定域 | 保守 | 中维 |
| **Monte Carlo** | 可达点采样 | 不完备 | 高维 |

---

## 7. 相关概念

- [[CBF (控制障碍函数)]] — 可达性分析的对偶方法
- [[前向不变性 (Forward Invariance)]] — 可达集 ⊆ 安全集 的等价表述
- [[Neural Barrier Certificate]] — 用 NN 替代可达性计算
- [[IBP (区间界传播)]] — 神经网络的"可达性分析"
- [[CEGIS (反例引导合成)]] — 可达性反例驱动训练

---

> **参考**: 
> - Althoff et al., "Reachability Analysis of Nonlinear Systems," 2008
> - Mitchell et al., "A Time-Dependent Hamilton-Jacobi Formulation," IEEE TAC 2005
