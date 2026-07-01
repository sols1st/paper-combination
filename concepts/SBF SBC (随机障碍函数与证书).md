# SBF (随机障碍函数) / SBC (随机障碍证书)

> **一句话**：SBF/SBC 是 [[CBF (控制障碍函数)]] 在**随机系统**中的推广。它不保证系统"绝对安全"，而是保证系统"以高概率安全"。核心数学工具是 [[鞅理论 (Martingale Theory)]]。

---

## 1. 为什么需要随机版本？

现实系统中存在各种不确定性：
- 传感器噪声
- 环境扰动（风、路面摩擦变化）
- 模型误差

确定性 [[CBF (控制障碍函数)]] 无法处理这些随机性。SBF 通过概率方法给出保证：

$$\mathbb{P}(\text{系统永远不进入不安全区域}) \geq p$$

其中 $p$ 是给定的概率阈值（如 $p = 0.9$）。

---

## 2. 系统模型

**离散时间随机系统**：

$$s_{t+1} = f(s_t, u_t) + \Delta s_t$$

其中：
- $s_t \in S$：状态
- $u_t = \pi(s_t)$：控制策略
- $\Delta s_t \sim \mu$：随机扰动，独立于当前状态

**闭环动力学**：

$$s_{t+1} = \tilde{F}(s_t, z_0, \Delta s_t) = F(s_t, z_0) + \Delta s_t$$

$F(s, z_0)$ 是标称（无扰动）动力学。

---

## 3. SBC 的数学定义

**定义**：连续函数 $B: S \to \mathbb{R}_{\geq 0}$ 是**随机障碍证书 (SBC)**，如果满足以下四个条件：

| # | 条件 | 数学表达 | 直觉 |
|---|------|---------|------|
| (i) | 非负性 | $B(s) \geq 0, \forall s \in S$ | "危险度"非负 |
| (ii) | 初始集约束 | $B(s) \leq 1, \forall s \in S_0$ | 出发时"危险度"低 |
| (iii) | 不安全集约束 | $B(s) \geq \frac{1}{1-p}, \forall s \in X_u$ | 危险区域"危险度"高 |
| (iv) | 期望递减 | $B(s) \geq \mathbb{E}[B(s') | s] + \epsilon$ | "危险度"平均在下降 |

**直觉**：$B(s)$ 就像一个"危险度计"：
- 在安全区域：$B$ 值小
- 在危险区域：$B$ 值大
- 系统运行时：$B$ 平均在减小（趋向安全）

---

## 4. 核心定理

**定理**：如果存在 SBC $B(s)$ 满足上述四个条件，则：

$$\mathbb{P}_{s_0}(\text{永远安全}) \geq p, \quad \forall s_0 \in S_0$$

### 4.1 证明（完整推导）

**Step 1**：$B(s_t)$ 构成 [[Supermartingale (超鞅)]]

由条件 (iv)：$\mathbb{E}[B(s_{t+1}) | s_t] \leq B(s_t) - \epsilon < B(s_t)$

因此 $B(s_t)$ 是严格 [[Supermartingale (超鞅)]]。

**Step 2**：定义停时

$$\tau_u = \inf\{t \geq 0 : s_t \in X_u\}$$

$\tau_T = \min(\tau_u, T)$

**Step 3**：应用 [[Doob 停时不等式 (Doob's Optional Stopping Theorem)]]

$$\mathbb{E}[B(s_{\tau_T})] \leq \mathbb{E}[B(s_0)]$$

**Step 4**：分解期望

$$\mathbb{E}[B(s_{\tau_T})] \geq \frac{1}{1-p} \cdot \mathbb{P}(\tau_u \leq T) + 0 \cdot \mathbb{P}(\tau_u > T)$$

（因为到达 $X_u$ 时 $B \geq \frac{1}{1-p}$，否则 $B \geq 0$）

**Step 5**：合并

$$\frac{\mathbb{P}(\tau_u \leq T)}{1-p} \leq \mathbb{E}[B(s_0)] \leq 1$$

$$\mathbb{P}(\tau_u \leq T) \leq 1 - p$$

**Step 6**：取 $T \to \infty$

$$\mathbb{P}(\tau_u < \infty) \leq 1 - p$$

$$\mathbb{P}(\text{安全}) = 1 - \mathbb{P}(\tau_u < \infty) \geq p$$

$\blacksquare$

---

## 5. 数值例子

**AEBS 场景**：
- $S_0$: $d \in [15,16], v \in [2.5,3.0]$
- $X_u$: $d \in [5,6], v \in [0.5,3.0]$
- 目标：$p = 0.9$

**SBC 要求**：
- $B(s) \leq 1$ 在 $S_0$ 上
- $B(s) \geq \frac{1}{1-0.9} = 10$ 在 $X_u$ 上
- $B(s) - \mathbb{E}[B(s')] \geq \epsilon = 0.1$

**概率界**：

$$\mathbb{P}(\text{安全}) \geq 1 - \frac{\max_{S_0} B}{\min_{X_u} B} = 1 - \frac{1}{10} = 0.9 = 90\%$$

---

## 6. SBC 神经网络的训练

### 6.1 网络结构

```python
class SBCNetwork(nn.Module):
    """
    随机障碍证书神经网络
    
    输入: 状态 s
    输出: B(s) >= 0 (危险度)
    """
    def __init__(self, state_dim=2, hidden_dims=[16, 8]):
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
        """输出 B(s) >= 0"""
        raw = self.net(s)
        # Softplus 保证非负
        return F.softplus(raw)
```

### 6.2 训练损失

```python
def sbc_loss(B_net, s_batch, s_next_batch, 
             s0_batch, su_batch, 
             eps=0.1, gamma=1.0):
    """
    SBC 训练损失
    
    输入:
        B_net: SBC 网络
        s_batch: 随机状态 (batch, 2)
        s_next_batch: 下一步状态（含噪声）(batch, 2)
        s0_batch: 初始集采样 (batch0, 2)
        su_batch: 不安全集采样 (batchu, 2)
    """
    # 1. 鞅递减损失
    B_s = B_net(s_batch)
    B_next = B_net(s_next_batch)
    # max(0, E[B(s')] - gamma*B(s) + eps)
    dec_loss = torch.mean(F.relu(B_next - gamma * B_s + eps))
    
    # 2. 区域约束损失
    B_s0 = B_net(s0_batch)
    B_su = B_net(su_batch)
    # B(s0) <= 1
    region_s0 = torch.mean(F.relu(B_s0 - 1.0))
    # B(su) >= 1/(1-p)
    region_su = torch.mean(F.relu(10.0 - B_su))
    region_loss = region_s0 + region_su
    
    # 3. Lipschitz 正则化（可选）
    # lip_loss = ...
    
    total = 1000 * dec_loss + region_loss
    return total
```

---

## 7. SBF 与 SBC 的区别

| 术语 | 全称 | 用途 |
|------|------|------|
| **SBF** | Stochastic Barrier Function | 泛称，随机障碍函数 |
| **SBC** | Stochastic Barrier Certificate | 满足特定条件的 SBF，可直接给出概率保证 |
| **SCBF** | Stochastic Control Barrier Function | 带控制的随机障碍函数，每步施加控制约束 |

在本项目中主要使用 **SBC** 这一术语（来自 SafePVC 论文）。

---

## 8. 与 CBF 的对比

| 特性 | [[CBF (控制障碍函数)]] | SBC |
|------|------------------------|-----|
| **安全保证** | 确定性（绝对安全） | 概率性（以概率 $p$ 安全） |
| **系统类型** | 确定性 | 随机 |
| **核心条件** | $\dot{b} \geq -\alpha(b)$ | $\mathbb{E}[B(s')] \leq B(s) - \epsilon$ |
| **数学工具** | 微分方程 | [[鞅理论 (Martingale Theory)]] |
| **适用场景** | 模型精确已知 | 存在随机扰动 |

---

## 9. 相关概念

- [[CBF (控制障碍函数)]] — 确定性版本
- [[鞅理论 (Martingale Theory)]] — SBC 的数学基础
- [[Supermartingale (超鞅)]] — $B(s_t)$ 的性质
- [[Doob 停时不等式]] — 概率界的推导工具
- [[CEGIS (反例引导合成)]] — SBC 的训练框架
- [[IBP (区间界传播)]] — SBC 的验证方法
- [[Lipschitz 常数]] — 验证中的关键参数

---

> **参考论文**: 
> - Prajna et al., "Stochastic Barrier Certificates," 2007
> - SafePVC (DAC 2026) — 本项目使用的 SBC 框架
