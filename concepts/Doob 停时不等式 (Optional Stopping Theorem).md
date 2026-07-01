# Doob 停时不等式 (Doob's Optional Stopping Theorem / Maximal Inequality)

> **一句话**：Doob 停时不等式是 [[鞅理论 (Martingale Theory)]] 中最重要的工具之一——它利用 [[超鞅 (Supermartingale)]] 的性质，给出一个**随机过程达到某个阈值的概率上界**。在 [[SBF SBC (随机障碍函数与证书)]] 中，它是从超鞅条件推导出安全概率的**核心数学桥梁**。

---

## 1. 直觉理解

### 1.1 赌场故事

你带着 100 元进赌场（初始值 $X_0 = 100$），玩一个对你不利的游戏（超鞅）。

**问题**：你的钱在某个时刻达到 1000 元的概率有多大？

**Doob 的回答**：

$$\mathbb{P}\left(\max_{t \geq 0} X_t \geq 1000\right) \leq \frac{X_0}{1000} = \frac{100}{1000} = 10\%$$

即使你偶尔运气好赢了一些钱，但由于游戏对你不利，**达到高阈值的概率被初始值约束**。

### 1.2 安全翻译

在安全分析中：

- $X_t = B(s_t)$：障碍函数值
- $X_0 = B(s_0)$：初始障碍值
- 阈值 $\lambda = 1$：不安全集上 $B \geq 1$

$$\mathbb{P}(\text{ever reach unsafe}) = \mathbb{P}\left(\sup_{t} B(s_t) \geq 1\right) \leq B(s_0)$$

---

## 2. 定理陈述

### 2.1 Doob 极大不等式 (Maximal Inequality)

设 $\{X_t\}_{t \geq 0}$ 是非负 [[超鞅 (Supermartingale)]]，$\lambda > 0$。则：

$$\mathbb{P}\left(\sup_{t \geq 0} X_t \geq \lambda\right) \leq \frac{\mathbb{E}[X_0]}{\lambda}$$

### 2.2 有限时间版本

对于有限时间 $T$：

$$\mathbb{P}\left(\max_{0 \leq t \leq T} X_t \geq \lambda\right) \leq \frac{\mathbb{E}[X_0]}{\lambda}$$

### 2.3 停时版本 (Optional Stopping Theorem)

设 $\tau$ 是一个**停时**（在 $\tau$ 时刻是否停止，只依赖于到 $\tau$ 为止的信息）。

如果 $\{X_t\}$ 是超鞅，则：

$$\mathbb{E}[X_\tau] \leq \mathbb{E}[X_0]$$

**条件**：$\tau$ 几乎必然有限，且 $X_{t \wedge \tau}$ 一致可积。

---

## 3. 在安全分析中的应用

### 3.1 推导安全概率界

**步骤 1**：障碍函数 $B(s)$ 满足超鞅条件

$$\mathbb{E}[B(s_{t+1}) | s_t] \leq B(s_t)$$

**步骤 2**：不安全集 $S_u$ 上 $B(s) \geq 1$

**步骤 3**：定义停时 $\tau = \inf\{t \geq 0 : s_t \in S_u\}$（首次进入不安全集的时间）

**步骤 4**：应用 Doob 不等式

$$\mathbb{P}(\tau < \infty) = \mathbb{P}\left(\sup_t B(s_t) \geq 1\right) \leq B(s_0)$$

**步骤 5**：如果 $B(s_0) \leq \delta$，则 $\mathbb{P}(\text{ever unsafe}) \leq \delta$

### 3.2 数值例子

```
初始状态: s_0 = (d=20, v=10)
B(s_0) = 0.05

结论: P(ever reach unsafe) ≤ 0.05 = 5%
```

如果初始状态更远：

```
初始状态: s_0 = (d=30, v=5)
B(s_0) = 0.01

结论: P(ever reach unsafe) ≤ 0.01 = 1%
```

---

## 4. 证明概要

### 4.1 Doob 极大不等式的证明

**定义**：停时 $\tau = \inf\{t \geq 0 : X_t \geq \lambda\}$

**步骤 1**：由超鞅性质，$\mathbb{E}[X_{t \wedge \tau}] \leq \mathbb{E}[X_0]$

**步骤 2**：在 $\{\tau \leq t\}$ 上，$X_\tau \geq \lambda$

**步骤 3**：

$$\mathbb{E}[X_0] \geq \mathbb{E}[X_{t \wedge \tau}] \geq \mathbb{E}[X_\tau \cdot \mathbf{1}_{\tau \leq t}] + \mathbb{E}[X_t \cdot \mathbf{1}_{\tau > t}]$$

**步骤 4**：因为 $X_t \geq 0$，第二项 $\geq 0$，所以：

$$\mathbb{E}[X_0] \geq \lambda \cdot \mathbb{P}(\tau \leq t)$$

**步骤 5**：令 $t \to \infty$：

$$\mathbb{P}(\tau < \infty) \leq \frac{\mathbb{E}[X_0]}{\lambda}$$

### 4.2 图示

```
B(s_t)
 ↑
 │    B(s_0) = 0.05
 │ ●
 │  ╲  ╱╲
 │   ╲╱  ╲╱╲
 │         ╲  ╱╲
 │          ╲╱  ╲
 │                ╲  ← 超鞅：期望递减
 │───────────────── λ = 1 (不安全阈值)
 │
 └──────────────────→ t

P(max B(s_t) ≥ 1) ≤ B(s_0)/1 = 0.05
```

---

## 5. 代码实现

### 5.1 蒙特卡洛验证

```python
import torch
import numpy as np

def monte_carlo_verify_doob(
    B_fn,          # 障碍函数
    dynamics_fn,   # 随机动态
    s0,            # 初始状态
    n_rollouts=10000,
    T=100,         # 最大步数
    threshold=1.0  # 不安全阈值
):
    """
    用蒙特卡洛模拟验证 Doob 不等式
    
    比较:
    - 实际不安全概率 (MC 估计)
    - Doob 上界 B(s_0)
    
    输入:
        B_fn: callable(s) -> B(s)
        dynamics_fn: callable(s, noise) -> s'
        s0: 初始状态
    输出:
        实际概率 vs 理论上界
    """
    B_s0 = B_fn(s0.unsqueeze(0)).item()
    
    n_unsafe = 0
    max_B_values = []
    
    for rollout in range(n_rollouts):
        s = s0.clone()
        max_B = B_s0
        
        for t in range(T):
            noise = torch.randn(s.shape) * 0.1
            s = dynamics_fn(s, noise)
            B_s = B_fn(s.unsqueeze(0)).item()
            max_B = max(max_B, B_s)
        
        max_B_values.append(max_B)
        if max_B >= threshold:
            n_unsafe += 1
    
    actual_prob = n_unsafe / n_rollouts
    doob_bound = B_s0 / threshold
    
    return {
        'B(s0)': B_s0,
        'actual_probability': actual_prob,
        'doob_bound': doob_bound,
        'bound_is_valid': actual_prob <= doob_bound + 0.01,  # 允许MC误差
        'mean_max_B': np.mean(max_B_values),
        'std_max_B': np.std(max_B_values)
    }

# 测试
def simple_B(s):
    return torch.exp(-s[0] / 10.0)  # 距离越远，B越小

def simple_dynamics(s, noise):
    return s + torch.tensor([-0.5, 0.0]) + noise  # 缓慢远离危险

s0 = torch.tensor([20.0, 10.0])
result = monte_carlo_verify_doob(simple_B, simple_dynamics, s0)
print(f"B(s0) = {result['B(s0)']:.4f}")
print(f"实际概率 = {result['actual_probability']:.4f}")
print(f"Doob 上界 = {result['doob_bound']:.4f}")
print(f"上界有效 = {result['bound_is_valid']}")
```

### 5.2 训练中使用 Doob 界

```python
def doob_aware_loss(B_model, dynamics_fn, batch, 
                     unsafe_set_fn, target_prob=0.05):
    """
    考虑 Doob 界的训练损失
    
    目标: B(s_0) ≤ target_prob
    同时: 超鞅条件 + 不安全集上 B ≥ 1
    """
    s = batch
    B_s = B_model(s)
    
    # 超鞅递减
    noise = torch.randn_like(s) * 0.1
    s_next = dynamics_fn(s, noise)
    B_next = B_model(s_next)
    dec_loss = torch.relu(0.01 - (B_s - B_next)).mean()
    
    # 不安全集上 B ≥ 1
    unsafe_mask = unsafe_set_fn(s)
    if unsafe_mask.any():
        unsafe_loss = torch.relu(1.0 - B_s[unsafe_mask]).mean()
    else:
        unsafe_loss = torch.tensor(0.0)
    
    # Doob 界: 鼓励 B(s) 在安全区域尽量小
    # (使得 B(s_0) ≤ target_prob)
    safe_mask = ~unsafe_mask
    if safe_mask.any():
        small_B_loss = B_s[safe_mask].mean()  # 鼓励 B 小
    else:
        small_B_loss = torch.tensor(0.0)
    
    total_loss = 10.0 * dec_loss + 5.0 * unsafe_loss + 1.0 * small_B_loss
    
    return total_loss
```

---

## 6. Doob 不等式的变体

### 6.1 $L^p$ 极大不等式

对于 $p > 1$，如果 $\{X_t\}$ 是非负亚鞅：

$$\left\|\sup_t X_t\right\|_p \leq \frac{p}{p-1} \|X_\infty\|_p$$

这给出了**矩**的约束，而不仅仅是概率。

### 6.2 指数界

如果 $X_t$ 是鞅且增量有界 $|X_{t+1} - X_t| \leq c$，则 Azuma-Hoeffding 不等式：

$$\mathbb{P}(|X_t - X_0| \geq \lambda) \leq 2 \exp\left(-\frac{\lambda^2}{2tc^2}\right)$$

比 Doob 不等式更紧，但需要增量有界。

### 6.3 离散时间 vs 连续时间

| 版本 | 条件 | 结论 |
|------|------|------|
| 离散 Doob | $X_n$ 超鞅 | $\mathbb{P}(\max_n X_n \geq \lambda) \leq \mathbb{E}[X_0]/\lambda$ |
| 连续 Doob | $X_t$ 右连续超鞅 | $\mathbb{P}(\sup_t X_t \geq \lambda) \leq \mathbb{E}[X_0]/\lambda$ |

---

## 7. 与其他概率界方法的对比

| 方法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| **Doob 不等式** | 超鞅 | 简洁、全局 | 需要超鞅条件 |
| **Chernoff 界** | 矩母函数 | 指数衰减 | 需要 MGF 存在 |
| **Chebyshev 不等式** | 方差 | 只需二阶矩 | 较保守 |
| **Union Bound** | 概率 | 简单 | 非常保守 |

---

## 8. 相关概念

- [[鞅理论 (Martingale Theory)]] — 理论基础
- [[超鞅 (Supermartingale)]] — Doob 不等式的对象
- [[SBF SBC (随机障碍函数与证书)]] — 主要应用
- [[前向不变性 (Forward Invariance)]] — 确定性版本
- [[比较引理 (Comparison Lemma)]] — 确定性版本的"概率界"

---

> **参考**: 
> - Williams, "Probability with Martingales," Cambridge 1991, Chapter 10
> - Durrett, "Probability: Theory and Examples," Cambridge 2010, Chapter 5
