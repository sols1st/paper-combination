# PPO (Proximal Policy Optimization)

> **一句话**：PPO 是一种**策略梯度强化学习算法**——它通过限制每次策略更新的幅度（使用 clipping 或 KL 惩罚），在保证训练稳定性的同时高效地学习控制策略。在本项目中，PPO 用于**预训练参考控制器** $\pi_{\text{ref}}$。

---

## 1. 为什么需要 PPO？

### 1.1 策略梯度的问题

原始策略梯度算法（REINFORCE）：

$$\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]$$

**问题**：更新步长太大时，策略可能崩溃（performance collapse）。

### 1.2 TRPO 的解决

TRPO（Trust Region Policy Optimization）限制 KL 散度：

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A(s,a)\right] \quad \text{s.t.} \quad \text{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$$

**问题**：需要计算二阶导数（共轭梯度法），实现复杂。

### 1.3 PPO 的简洁方案

PPO 用简单的 **clipping** 替代 KL 约束，效果相当但实现简单得多。

---

## 2. 数学原理

### 2.1 重要性比率

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

- $r_t = 1$：新旧策略相同
- $r_t > 1$：新策略更倾向于这个动作
- $r_t < 1$：新策略不太倾向这个动作

### 2.2 PPO-Clip 目标函数

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中 $\epsilon$ 是 clip 范围（通常 0.1~0.2）。

### 2.3 Clipping 的直觉

**当优势 $\hat{A}_t > 0$**（这个动作好）：

```
目标函数
 ↑
 │     clip 后
 │    ┌─────
 │   ╱
 │  ╱ ← r * A（无 clip）
 │ ╱
 ─┼─────────→ r_t
  1-ε  1  1+ε
```

- 鼓励增大 $r_t$（更倾向好动作），但最多到 $1+\epsilon$

**当优势 $\hat{A}_t < 0$**（这个动作差）：

```
目标函数
 ↑
 │
 ─┼─────────→ r_t
  1-ε  1  1+ε
 │ ╲
 │  ╲ ← r * A
 │    └───── clip 后
```

- 鼓励减小 $r_t$（更不倾向差动作），但最多到 $1-\epsilon$

---

## 3. 完整算法

### 3.1 伪代码

```
for each iteration:
    1. 用当前策略 π_θ 收集 T 步轨迹数据
    2. 计算优势估计 Â_t (用 GAE)
    3. for each epoch K (通常 3-10):
        for each minibatch:
            4. 计算 r_t(θ) = π_θ / π_old
            5. 计算 PPO-Clip 损失
            6. 梯度上升更新 θ
    7. θ_old = θ
```

### 3.2 GAE (Generalized Advantage Estimation)

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

- $\gamma$：折扣因子（通常 0.99）
- $\lambda$：GAE 参数（通常 0.95）

---

## 4. 代码实现

### 4.1 PPO Agent

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    """Actor-Critic 网络"""
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        
        # 策略网络 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        
        # 值函数网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        # 动作标准差（可学习）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.actor(state)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist
    
    def value(self, state):
        return self.critic(state).squeeze(-1)


class PPOAgent:
    """
    PPO 训练代理
    """
    def __init__(self, state_dim, action_dim, 
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, k_epochs=10):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = []
    
    def select_action(self, state):
        """选择动作"""
        with torch.no_grad():
            dist = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def compute_gae(self, rewards, values, next_value, done):
        """
        计算 GAE 优势估计
        
        输入:
            rewards: (T,) 奖励序列
            values: (T,) 值函数估计
            next_value: 最后一步的值函数
            done: (T,) 是否结束
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - done[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages)
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self):
        """PPO 更新"""
        # 准备数据
        states = torch.stack([d['state'] for d in self.buffer])
        actions = torch.stack([d['action'] for d in self.buffer])
        old_log_probs = torch.stack([d['log_prob'] for d in self.buffer])
        rewards = [d['reward'] for d in self.buffer]
        dones = [d['done'] for d in self.buffer]
        
        with torch.no_grad():
            values = self.policy.value(states)
            next_value = self.policy.value(states[-1:])
        
        advantages = self.compute_gae(rewards, values, next_value.item(), 
                                       torch.tensor(dones))
        
        # PPO 更新
        for epoch in range(self.k_epochs):
            # 计算当前策略的 log_prob
            dist = self.policy(states)
            new_log_probs = dist.log_prob(actions).sum(-1)
            
            # 重要性比率
            ratio = (new_log_probs - old_log_probs).exp()
            
            # PPO-Clip 损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 值函数损失
            new_values = self.policy.value(states)
            critic_loss = ((new_values - (advantages + values)) ** 2).mean()
            
            # 熵正则化（鼓励探索）
            entropy = dist.entropy().sum(-1).mean()
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.buffer = []
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }
```

### 4.2 AEBS 奖励设计

```python
def aebs_reward(state, action, next_state, params):
    """
    AEBS 场景的奖励函数
    
    奖励 = -碰撞惩罚 - 急刹惩罚 - 距离奖励
    
    输入:
        state: (d, v) 相对距离, 本车速度
        action: u 本车加速度
        next_state: (d', v')
    """
    d, v = state
    d_next, v_next = next_state
    
    d_safe = params.get('d_safe', 6.0)
    v_desired = params.get('v_desired', 20.0)
    
    reward = 0.0
    
    # 1. 碰撞惩罚
    if d_next < d_safe:
        reward -= 100.0  # 大惩罚
    
    # 2. 速度跟踪
    reward -= 0.1 * (v - v_desired) ** 2
    
    # 3. 控制量惩罚（避免急刹）
    reward -= 0.01 * action ** 2
    
    # 4. 安全距离奖励
    if d_next > d_safe + 5:
        reward += 1.0
    
    # 5. 接近危险区惩罚
    if d_next < d_safe + 3:
        reward -= 5.0 * (d_safe + 3 - d_next)
    
    return reward
```

### 4.3 训练循环

```python
def train_ppo_aebs(env, n_iterations=1000, steps_per_iter=2048):
    """
    用 PPO 训练 AEBS 参考控制器
    """
    agent = PPOAgent(state_dim=2, action_dim=1)
    
    for iteration in range(n_iterations):
        state = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_iter):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob = agent.select_action(state_tensor)
            action_np = action.numpy()
            
            next_state, done = env.step(action_np)
            reward = aebs_reward(state, action_np, next_state, env.params)
            
            agent.buffer.append({
                'state': state_tensor,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'done': float(done)
            })
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = env.reset()
        
        # PPO 更新
        metrics = agent.update()
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1}: "
                  f"actor_loss={metrics['actor_loss']:.4f}, "
                  f"critic_loss={metrics['critic_loss']:.4f}")
    
    return agent.policy.actor  # 返回策略网络作为参考控制器
```

---

## 5. PPO 超参数指南

| 超参数 | 典型值 | 说明 |
|--------|-------|------|
| $\gamma$ (discount) | 0.99 | 未来奖励的折扣 |
| $\lambda$ (GAE) | 0.95 | 偏差-方差折中 |
| $\epsilon$ (clip) | 0.1~0.2 | 更新步长限制 |
| $K$ (epochs) | 3~10 | 每批数据的训练轮数 |
| lr (学习率) | 3e-4 | 常用线性衰减 |
| batch_size | 64~256 | minibatch 大小 |
| entropy_coef | 0.01 | 探索激励 |
| value_coef | 0.5 | 值函数损失权重 |
| max_grad_norm | 0.5 | 梯度裁剪 |

---

## 6. PPO 在本项目中的角色

```
PPO 训练参考控制器 π_ref
         │
         ▼
π_ref(x) → 参考控制 u_ref
         │
         ▼
CBF/BarrierNet 安全过滤
         │
         ▼
u* = safe_filter(u_ref) → 安全控制
```

1. **预训练阶段**：用 PPO 训练 $\pi_{\text{ref}}$（不需要安全约束，只需完成任务）
2. **安全过滤阶段**：用 CBF-QP 或 BarrierNet 将 $u_{\text{ref}}$ 投影到安全集合
3. **联合训练阶段**（可选）：端到端微调 $\pi_{\text{ref}}$ + 安全层

---

## 7. 与其他 RL 算法的对比

| 算法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| **PPO** | On-policy | 稳定、简单 | 样本效率低 |
| **SAC** | Off-policy | 样本效率高 | 超参数敏感 |
| **TD3** | Off-policy | 连续动作好 | 需要经验回放 |
| **A2C** | On-policy | 简单 | 方差大 |
| **DDPG** | Off-policy | 连续动作 | 不稳定 |

---

## 8. 相关概念

- [[BarrierNet]] — PPO 训练的参考控制器被 BarrierNet 安全过滤
- [[CBF (控制障碍函数)]] — 安全过滤器
- [[CEGIS (反例引导合成)]] — PPO 策略的验证
- [[Lipschitz 常数]] — PPO 策略的鲁棒性

---

> **参考**: 
> - Schulman et al., "Proximal Policy Optimization Algorithms," arXiv 2017
> - Engstrom et al., "Implementation Matters in Deep Policy Gradients," ICLR 2020
