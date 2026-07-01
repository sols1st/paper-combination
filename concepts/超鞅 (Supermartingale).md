# 超鞅 (Supermartingale)

> **一句话**：超鞅是一种随机过程——它的**期望值随时间递减（或不增）**。在 [[SBF SBC (随机障碍函数与证书)]] 中，障碍函数 $B(s)$ 被设计为超鞅，从而利用 [[鞅理论 (Martingale Theory)]] 中的 Doob 不等式来**定量约束系统进入不安全状态的概率**。

---

## 1. 直觉理解

### 1.1 赌场类比

想象你在赌场玩一个**对你不利的**游戏：

- 每轮你赢或输一些钱
- 平均来看，你的钱会**越来越少**（因为赌场有优势）
- 虽然偶尔会赢，但**长期趋势**是输

你的财富就是一个**超鞅**：
- 不是确定递减（每轮可能赢）
- 而是**期望**递减（平均来看在减少）

### 1.2 安全类比

在随机安全中：
- "财富" = 障碍函数 $B(s)$
- "输钱" = $B(s)$ 在减少（好事！远离危险）
- "赢钱" = $B(s)$ 在增加（坏事！接近危险）

如果 $B(s)$ 是超鞅，则 $B(s)$ **平均在减少** → 系统**平均在远离危险**。

---

## 2. 形式定义

### 2.1 概率空间

$(\Omega, \mathcal{F}, \mathbb{P})$：概率空间

$\{\mathcal{F}_t\}_{t \geq 0}$：[[鞅理论 (Martingale Theory)]] 中的**流**（filtration，信息累积）

### 2.2 超鞅定义

随机过程 $\{X_t\}_{t \geq 0}$ 是关于流 $\{\mathcal{F}_t\}$ 的**超鞅**，如果：

1. **适应性**：$X_t$ 是 $\mathcal{F}_t$-可测的（在时刻 $t$ 已知）
2. **可积性**：$\mathbb{E}[|X_t|] < \infty$
3. **递减性**：$\mathbb{E}[X_{t+1} | \mathcal{F}_t] \leq X_t$

### 2.3 三种鞅的对比

| 类型 | 条件 | 直觉 |
|------|------|------|
| **鞅** (Martingale) | $\mathbb{E}[X_{t+1} \mid \mathcal{F}_t] = X_t$ | 公平游戏 |
| **超鞅** (Supermartingale) | $\mathbb{E}[X_{t+1} \mid \mathcal{F}_t] \leq X_t$ | 对你不利的游戏 |
| **亚鞅** (Submartingale) | $\mathbb{E}[X_{t+1} \mid \mathcal{F}_t] \geq X_t$ | 对你有利的游戏 |

---

## 3. 在随机安全中的角色

### 3.1 SBC 条件回顾

在 [[SBF SBC (随机障碍函数与证书)]] 中，安全证书 $B(s)$ 需要满足：

**条件 1**（非负性）：$B(s) \geq 0$

**条件 2**（超鞅性）：

$$\mathbb{E}[B(s') | s] \leq B(s) - \epsilon$$

其中 $s' = f(s, \pi(s), w)$ 是下一状态。

**条件 3**（不安全集高值）：$B(s) \geq 1$ 对所有 $s \in S_u$

### 3.2 为什么超鞅保证安全？

**思路**：

1. $B(s)$ 是超鞅 → $B(s)$ 的期望在减少
2. 由 Doob 不等式（见 [[鞅理论 (Martingale Theory)]]）：

$$\mathbb{P}\left(\sup_{t \geq 0} B(s_t) \geq 1\right) \leq B(s_0)$$

3. 因为不安全集上 $B \geq 1$：

$$\mathbb{P}(\text{ever reach unsafe}) \leq B(s_0)$$

4. 如果 $B(s_0) \leq \epsilon$（初始时很小），则不安全概率 $\leq \epsilon$。

### 3.3 安全概率的定量约束

| 初始 $B(s_0)$ | 不安全概率上界 | 含义 |
|---------------|-------------|------|
| 0.01 | $\leq 1\%$ | 很安全 |
| 0.1 | $\leq 10\%$ | 一般 |
| 0.5 | $\leq 50\%$ | 不太安全 |
| 1.0 | $\leq 100\%$ | 没保证 |

---

## 4. 构造超鞅

### 4.1 离散时间

给定随机系统 $s_{t+1} = f(s_t, \pi(s_t), w_t)$，构造 $B(s)$ 使得：

$$B(s_t) - \mathbb{E}[B(s_{t+1}) | s_t] \geq \epsilon > 0$$

即每步**期望递减至少 $\epsilon$**。

### 4.2 连续时间

对于 Itô 扩散过程 $ds = f(s)dt + g(s)dW$：

$B(s)$ 是超鞅当且仅当 **无穷小生成元** $\mathcal{L}B \leq 0$：

$$\mathcal{L}B = \nabla B \cdot f + \frac{1}{2} \text{tr}(g^T \nabla^2 B \cdot g) \leq 0$$

---

## 5. 代码实现

### 5.1 验证超鞅性质

```python
import torch
import torch.nn as nn

def check_supermartingale(
    B_fn,         # callable(s) -> B(s)
    dynamics_fn,      # callable(s, noise) -> s'
    state_samples,    # (N, n) 状态采样
    n_noise=100,      # 每个状态的噪声采样数
    epsilon=0.01      # 要求的最小递减量
):
    """
    验证 B(s) 是否满足超鞅条件
    
    检查: B(s) - E[B(s')|s] >= ε
    
    输入:
        B_fn: 障碍函数
        dynamics_fn: 随机动态
        state_samples: 状态采样点
        n_noise: 噪声采样数
    输出:
        is_supermartingale: 是否满足
        details: 每个状态的递减量
    """
    N = state_samples.shape[0]
    decrements = torch.zeros(N)
    
    for i in range(N):
        s = state_samples[i]
        B_s = B_fn(s.unsqueeze(0)).squeeze()
        
        # 采样下一状态
        B_next_list = []
        for _ in range(n_noise):
            noise = torch.randn(s.shape) * 0.1  # 过程噪声
            s_next = dynamics_fn(s, noise)
            B_next = B_fn(s_next.unsqueeze(0)).squeeze()
            B_next_list.append(B_next)
        
        E_B_next = torch.stack(B_next_list).mean()
        decrements[i] = B_s - E_B_next
    
    # 检查
    violations = (decrements < epsilon).sum().item()
    is_supermartingale = violations == 0
    
    return {
        'is_supermartingale': is_supermartingale,
        'min_decrement': decrements.min().item(),
        'mean_decrement': decrements.mean().item(),
        'n_violations': violations,
        'violation_rate': violations / N
    }
```

### 5.2 训练超鞅网络

```python
class SupermartingaleNetwork(nn.Module):
    """
    训练一个障碍函数 B(s) 使其满足超鞅条件
    
    B(s) = softplus(NN(s))  保证非负
    """
    def __init__(self, state_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, s):
        raw = self.net(s)
        return torch.nn.functional.softplus(raw)  # 保证 B >= 0


def supermartingale_loss(B_model, dynamics_fn, batch, 
                          unsafe_set_fn, epsilon=0.01):
    """
    超鞅训练损失
    
    Loss = L_dec + L_unsafe + L_nonneg
    
    L_dec: 鼓励 B(s) - E[B(s')] >= ε
    L_unsafe: 在不安全集上 B >= 1
    """
    s = batch
    B_s = B_model(s)
    
    # 1. 超鞅递减损失
    # 采样下一状态
    noise = torch.randn_like(s) * 0.1
    s_next = dynamics_fn(s, noise)
    B_next = B_model(s_next)
    
    decrement = B_s - B_next  # 单样本近似期望
    dec_loss = torch.relu(epsilon - decrement).mean()
    
    # 2. 不安全集损失
    unsafe_mask = unsafe_set_fn(s)
    if unsafe_mask.any():
        unsafe_loss = torch.relu(1.0 - B_s[unsafe_mask]).mean()
    else:
        unsafe_loss = torch.tensor(0.0)
    
    # 3. 总损失
    total_loss = 10.0 * dec_loss + 5.0 * unsafe_loss
    
    return total_loss, {
        'decrement_loss': dec_loss.item(),
        'unsafe_loss': unsafe_loss.item(),
        'mean_B': B_s.mean().item()
    }
```

---

## 6. 超鞅收敛定理

### 6.1 超鞅收敛定理 (Martingale Convergence Theorem)

如果 $\{X_t\}$ 是非负超鞅（$X_t \geq 0$），则：

$$X_t \to X_\infty \quad \text{a.s.}$$

即 $X_t$ **几乎必然收敛**到某个有限值 $X_\infty$。

### 6.2 安全意义

因为 $B(s_t) \geq 0$ 是超鞅，$B(s_t)$ 几乎必然收敛。这意味着：

- 系统不会永远"震荡"在安全边界附近
- 最终要么安全（$B \to 0$），要么不安全（$B \geq 1$）

---

## 7. 超鞅 vs 其他安全概念

| 概念 | 安全保证 | 类型 |
|------|---------|------|
| CBF | 确定性安全 | 确定性 |
| 超鞅 $B(s)$ | 概率安全：$\mathbb{P}(\text{unsafe}) \leq B(s_0)$ | 随机 |
| Lyapunov 函数 | 稳定性 | 确定性 |
| 随机 Lyapunov | 概率稳定性 | 随机 |

### 7.1 超鞅 vs Lyapunov

两者结构相似，但目标不同：

| | 超鞅 $B(s)$ | Lyapunov $V(s)$ |
|---|---|---|
| **目标** | 安全（不进入危险区域） | 稳定（收敛到平衡点） |
| **递减** | $\mathbb{E}[B(s')] \leq B(s)$ | $\dot{V}(s) \leq 0$ |
| **不安全** | $B(s) \geq 1$ | $V(s) = 0$ 在平衡点 |
| **保证** | 概率上界 | 渐近收敛 |

---

## 8. 相关概念

- [[鞅理论 (Martingale Theory)]] — 超鞅的理论基础
- [[SBF SBC (随机障碍函数与证书)]] — 超鞅的主要应用
- [[Doob 停时不等式 (Optional Stopping Theorem)]] — 从超鞅推出概率界
- [[前向不变性 (Forward Invariance)]] — 确定性版本的安全保证
- [[CBF (控制障碍函数)]] — 确定性版本的障碍函数

---

> **参考**: 
> - Williams, "Probability with Martingales," Cambridge 1991
> - Santoyo et al., "Barrier Certificates for Stochastic Systems," ACC 2021
