# 鞅理论 (Martingale Theory)

> **一句话**：鞅理论是概率论中研究"公平赌博"的数学框架。在形式化安全中，我们用**超鞅**性质证明系统到达危险区域的概率有上界。

---

## 1. 核心概念

### 1.1 滤子 (Filtration)

**滤子** $\{\mathcal{F}_t\}_{t \geq 0}$ 是一列递增的信息集：

$$\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \mathcal{F}_2 \subseteq \cdots$$

$\mathcal{F}_t$ 代表"到时间 $t$ 为止我们知道的所有信息"。

**直觉**：你每天看一次天气预报。$\mathcal{F}_0$ 是今天的预报，$\mathcal{F}_1$ 包含今天和明天的预报，以此类推。

### 1.2 鞅 (Martingale)

随机过程 $\{X_t\}$ 是**鞅**，如果：

1. $\mathbb{E}[|X_t|] < \infty$（可积）
2. $\mathbb{E}[X_{t+1} | \mathcal{F}_t] = X_t$（条件期望等于当前值）

**直觉**：公平赌博——平均来说不赚不赔。

### 1.3 [[Supermartingale (超鞅)]]

$\{X_t\}$ 是**超鞅**，如果：

$$\mathbb{E}[X_{t+1} | \mathcal{F}_t] \leq X_t$$

**直觉**：对你不利的赌博——平均来说在赔钱。

### 1.4 亚鞅 (Submartingale)

$$\mathbb{E}[X_{t+1} | \mathcal{F}_t] \geq X_t$$

**直觉**：对你有利的赌博——平均来说在赚钱。

---

## 2. 在安全验证中的应用

### 2.1 SBC 是超鞅

[[SBF SBC (随机障碍函数与证书)|SBC]] $B(s_t)$ 满足：

$$\mathbb{E}[B(s_{t+1}) | s_t] \leq B(s_t) - \epsilon < B(s_t)$$

因此 $B(s_t)$ 是**严格超鞅**。

**直觉**：系统的"危险度"平均在下降。从高危险度出发，到达更高危险度（不安全区域）的概率被鞅理论限制。

### 2.2 概率安全界

由 [[Doob 停时不等式 (Doob's Optional Stopping Theorem)]]：

$$\mathbb{P}(\text{到达不安全区域}) \leq \frac{B(s_0)}{B_{\min}(X_u)}$$

如果 $B(s_0) \leq 1$ 且 $B(X_u) \geq \frac{1}{1-p}$：

$$\mathbb{P}(\text{不安全}) \leq \frac{1}{1/(1-p)} = 1 - p$$

$$\mathbb{P}(\text{安全}) \geq p$$

---

## 3. 关键定理

### 3.1 Doob 极大不等式

如果 $\{X_t\}$ 是非负超鞅：

$$\mathbb{P}(\max_{0 \leq t \leq T} X_t \geq \lambda) \leq \frac{\mathbb{E}[X_0]}{\lambda}$$

**在安全中的含义**：$X_t = B(s_t)$，$\lambda = B_{\min}(X_u)$，则上式给出了到达不安全集的概率上界。

### 3.2 停时定理 (Optional Stopping)

如果 $\{X_t\}$ 是非负超鞅，$\tau$ 是停时：

$$\mathbb{E}[X_\tau] \leq \mathbb{E}[X_0]$$

（在适当条件下）

---

## 4. 直觉解释：赌博比喻

| 概念 | 赌博比喻 | 安全比喻 |
|------|---------|---------|
| $X_t$ | 你的财富 | 系统"危险度" $B(s_t)$ |
| 超鞅 | 赌博对你不利 | 危险度平均在下降 |
| 停时 $\tau$ | 你决定停止的时刻 | 系统首次进入危险区域 |
| Doob 不等式 | 从 $100$ 元出发，达到 $1000$ 元的概率 $\leq 10\%$ | 从低危险度出发，到达高危险度的概率很小 |

---

## 5. 代码：验证超鞅条件

```python
import torch

def check_supermartingale(B_net, s, dynamics_fn, noise_sampler, 
                          n_samples=16, eps=0.1):
    """
    检查 B(s_t) 是否满足超鞅条件
    
    输入:
        B_net: SBC 网络
        s: 当前状态 (batch, 2)
        dynamics_fn: 动力学函数 s_next = f(s, u)
        noise_sampler: 噪声采样器
        n_samples: 蒙特卡洛样本数
        eps: 最小递减量
    输出:
        loss: 鞅损失 (越小越好)
        violations: 违反条件的样本数
    """
    B_s = B_net(s).squeeze()  # (batch,)
    
    # 采样多个下一步状态
    B_next_list = []
    for _ in range(n_samples):
        noise = noise_sampler(s.shape[0])
        s_next = dynamics_fn(s) + noise
        B_next_list.append(B_net(s_next).squeeze())
    
    # 计算期望 E[B(s_{t+1})]
    B_next_mean = torch.stack(B_next_list, dim=0).mean(dim=0)
    
    # 鞅条件: B(s) >= E[B(s')] + eps
    # 损失: max(0, E[B(s')] - B(s) + eps)
    diff = B_next_mean - B_s + eps
    loss = torch.mean(torch.clamp(diff, min=0.0))
    
    violations = (diff > 0).sum().item()
    
    return loss, violations
```

---

## 6. 相关概念

- [[Supermartingale (超鞅)]] — SBC 的核心性质
- [[Doob 停时不等式]] — 概率界的推导工具
- [[SBF SBC (随机障碍函数与证书)]] — 鞅理论在安全中的应用
- [[Lyapunov 函数]] — 确定性版本（类比：鞅是随机的 Lyapunov）

---

> **参考**: Williams, "Probability with Martingales," Cambridge 1991
