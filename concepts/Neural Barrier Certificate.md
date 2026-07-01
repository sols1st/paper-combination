# Neural Barrier Certificate (神经障碍证书)

> **一句话**：Neural Barrier Certificate 是用**神经网络来参数化障碍函数**（如 [[CBF (控制障碍函数)]]、[[SBF SBC (随机障碍函数与证书)]]），然后用 [[CEGIS (反例引导合成)]] 框架训练并验证。它结合了神经网络的表达能力和形式化验证的严格安全保证。

---

## 1. 为什么用神经网络做障碍函数？

### 1.1 传统方法的局限

传统障碍函数通常是**多项式或简单函数**：

$$b(x) = x^T P x + q^T x + r$$

**问题**：表达能力有限，对复杂系统可能找不到合适的 $b$。

### 1.2 神经网络的优势

$$b(x) = \text{NN}(x; \theta)$$

- 可以表达任意连续函数（万能逼近定理）
- 适合高维状态空间
- 可以通过梯度下降高效训练

### 1.3 挑战

- 训练后需要**形式化验证**（神经网络可能有过拟合/鲁棒性问题）
- 验证代价高（需要 [[dReal]]、[[IBP (区间界传播)]]、[[CROWN (神经网络验证)]] 等工具）

---

## 2. 不同类型的神经障碍证书

### 2.1 Neural CBF (确定性安全)

$$b_\theta(x) = \text{NN}(x)$$

需要满足：
1. $b_\theta(x) \geq 0$ 在安全集 $S$
2. $b_\theta(x) < 0$ 在不安全集 $S_u$
3. $L_f b_\theta + L_g b_\theta \cdot u + \alpha(b_\theta) \geq 0$

### 2.2 Neural SBC (概率安全)

$$B_\theta(s) = \text{softplus}(\text{NN}(s))$$

需要满足：
1. $B_\theta(s) \geq 0$（非负）
2. $\mathbb{E}[B_\theta(s')] \leq B_\theta(s) - \epsilon$（[[超鞅 (Supermartingale)]]）
3. $B_\theta(s) \geq 1$ 在不安全集

### 2.3 Neural Lyapunov (稳定性)

$$V_\theta(x) = \text{NN}(x)^2 + \epsilon \|x\|^2$$

需要满足：
1. $V_\theta(x) > 0$ 对 $x \neq 0$
2. $\dot{V}_\theta(x) < 0$ 对 $x \neq 0$

---

## 3. 训练方法

### 3.1 纯 CEGIS

```python
class NeuralBarrierCertificate:
    """
    神经障碍证书的 CEGIS 训练
    """
    def __init__(self, state_dim, barrier_type='cbf'):
        self.barrier_type = barrier_type
        
        # 障碍函数网络
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.dataset = []
        self.counterexamples = []
    
    def barrier_fn(self, x):
        """计算障碍函数值"""
        raw = self.model(x)
        
        if self.barrier_type == 'sbc':
            return torch.nn.functional.softplus(raw)  # 非负
        elif self.barrier_type == 'lyapunov':
            return raw ** 2 + 0.01 * (x ** 2).sum(-1, keepdim=True)
        else:
            return raw  # CBF 可正可负
    
    def train_step(self, dynamics_fn, safety_spec, 
                    n_epochs=10, lr=1e-3):
        """CEGIS 训练步骤"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if not self.dataset:
            self.dataset = self._sample_initial(1000)
        
        for epoch in range(n_epochs):
            batch = torch.stack(self.dataset)
            
            loss = self._compute_loss(batch, dynamics_fn, safety_spec)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def verify(self, dynamics_fn, safety_spec, state_bounds):
        """形式化验证"""
        if self.barrier_type == 'cbf':
            return self._verify_cbf(dynamics_fn, safety_spec, state_bounds)
        elif self.barrier_type == 'sbc':
            return self._verify_sbc(dynamics_fn, safety_spec, state_bounds)
    
    def _compute_loss(self, batch, dynamics_fn, spec):
        """计算训练损失"""
        x = batch
        b = self.barrier_fn(x)
        
        loss = torch.tensor(0.0)
        
        if self.barrier_type == 'cbf':
            # 安全集上 b >= 0
            safe_mask = spec.safe_set_fn(x)
            if safe_mask.any():
                loss += torch.relu(-b[safe_mask]).mean()
            
            # 不安全集上 b < 0
            unsafe_mask = spec.unsafe_set_fn(x)
            if unsafe_mask.any():
                loss += torch.relu(b[unsafe_mask]).mean()
            
            # CBF 条件
            u = spec.controller_fn(x)
            Lf_b, Lg_b = compute_lie_derivatives(
                self.barrier_fn, dynamics_fn, x
            )
            cbf_val = Lf_b + Lg_b * u + b  # α(r) = r
            loss += 10.0 * torch.relu(-cbf_val).mean()
        
        elif self.barrier_type == 'sbc':
            # 非负性（已由 softplus 保证）
            
            # 超鞅递减
            noise = torch.randn_like(x) * 0.1
            x_next = dynamics_fn(x, noise)
            b_next = self.barrier_fn(x_next)
            dec_loss = torch.relu(0.01 - (b - b_next)).mean()
            loss += 10.0 * dec_loss
            
            # 不安全集上 B >= 1
            unsafe_mask = spec.unsafe_set_fn(x)
            if unsafe_mask.any():
                loss += torch.relu(1.0 - b[unsafe_mask]).mean()
        
        return loss
    
    def synthesize(self, dynamics_fn, safety_spec, state_bounds, 
                    max_iter=50):
        """完整 CEGIS 循环"""
        for i in range(max_iter):
            # 训练
            self.train_step(dynamics_fn, safety_spec, n_epochs=20)
            
            # 验证
            result = self.verify(dynamics_fn, safety_spec, state_bounds)
            
            if result['verified']:
                print(f"✅ 第 {i} 次迭代验证通过!")
                return self.model
            else:
                ce = result['counterexample']
                self.counterexamples.append(ce)
                self.dataset.append(ce)
                print(f"❌ 第 {i} 次: 反例 {ce.tolist()}")
        
        raise RuntimeError("CEGIS 未收敛")
```

### 3.2 网络结构设计

```python
class BarrierNetDesign(nn.Module):
    """
    精心设计的障碍函数网络
    
    关键: 
    1. 使用 Tanh/Softplus 等平滑激活（Lie 导数需要连续可微）
    2. 输出层保证所需符号约束
    3. 可选 spectral normalization
    """
    def __init__(self, state_dim, barrier_type='cbf'):
        super().__init__()
        self.barrier_type = barrier_type
        
        # 主干网络
        from torch.nn.utils import spectral_norm
        
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(state_dim, 64)),
            nn.Tanh(),  # 平滑！
            spectral_norm(nn.Linear(64, 64)),
            nn.Tanh(),
            spectral_norm(nn.Linear(64, 32)),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        raw = self.net(x)
        
        if self.barrier_type == 'sbc':
            # 非负: softplus 或 square
            return torch.nn.functional.softplus(raw) + 1e-3
        
        elif self.barrier_type == 'lyapunov':
            # 正定: 平方 + 正则项
            return raw ** 2 + 0.01 * (x ** 2).sum(-1, keepdim=True)
        
        else:
            # CBF: 可正可负
            return raw
```

---

## 4. 验证方法

### 4.1 用 IBP 验证

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def verify_barrier_ibp(barrier_net, state_bounds, property_type='cbf'):
    """
    用 IBP 验证神经障碍证书
    
    方法: 将状态空间分为网格，用 IBP 计算每个网格上 b(x) 的界
    """
    dummy = torch.randn(1, len(state_bounds))
    bounded_net = BoundedModule(barrier_net, dummy)
    
    # 网格划分
    n_grid = 10
    all_verified = True
    worst_bound = float('inf')
    
    for grid_cell in generate_grid(state_bounds, n_grid):
        lb, ub = grid_cell
        center = (lb + ub) / 2
        eps = (ub - lb).max().item() / 2
        
        ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
        bounded_x = BoundedTensor(center.unsqueeze(0), ptb)
        
        b_lb, b_ub = bounded_net.compute_bounds(
            x=(bounded_x,), method="IBP"
        )
        
        if property_type == 'sbc':
            # 需要 B >= 0
            if b_lb.item() < 0:
                all_verified = False
                worst_bound = min(worst_bound, b_lb.item())
    
    return all_verified, worst_bound
```

### 4.2 用 dReal 精确验证

```python
from dreal import *

def verify_barrier_dreal(barrier_net, dynamics, safety_spec, 
                          state_bounds, delta=0.001):
    """
    用 dReal 精确验证神经障碍证书
    """
    # 声明状态变量
    n = len(state_bounds)
    x_vars = [Variable(f"x_{i}") for i in range(n)]
    
    # 编码神经网络为 dReal 公式
    b_expr = encode_nn_for_dreal(barrier_net, x_vars)
    
    # 验证 CBF 条件
    # ∃x ∈ S: Lf_b + Lg_b * u + α(b) < 0
    violation_formula = build_violation_formula(
        b_expr, x_vars, dynamics, safety_spec
    )
    
    # 添加状态范围约束
    constraints = [violation_formula]
    for i, (lo, hi) in enumerate(state_bounds):
        constraints.append(x_vars[i] >= lo)
        constraints.append(x_vars[i] <= hi)
    
    result = CheckSat(And(*constraints), delta)
    
    if result:
        x_ce = [result[v].mid() for v in x_vars]
        return {'verified': False, 'counterexample': x_ce}
    else:
        return {'verified': True}
```

---

## 5. 与传统障碍函数的对比

| 特性 | 多项式障碍函数 | 神经障碍证书 |
|------|-------------|------------|
| **表达能力** | 有限（多项式） | 任意连续函数 |
| **维度** | 低维（< 5） | 高维可处理 |
| **训练** | SOS 优化 | 梯度下降 |
| **验证** | SOS（半定规划） | IBP/CROWN/dReal |
| **可解释性** | 高 | 低 |
| **实现难度** | 中 | 中 |

---

## 6. 实际应用案例

### 6.1 AEBS 安全证书

```python
# 定义 AEBS 安全规范
class AEBSSafetySpec:
    def __init__(self):
        self.d_safe = 6.0
        self.T_gap = 1.5
    
    def safe_set_fn(self, x):
        """安全集: d > d_safe + T_gap * v"""
        d, v = x[:, 0], x[:, 1]
        return (d - self.d_safe - self.T_gap * v) > 0
    
    def unsafe_set_fn(self, x):
        """不安全集: d < d_safe"""
        d = x[:, 0]
        return d < self.d_safe
    
    def controller_fn(self, x):
        """待验证的控制器"""
        return trained_controller(x)

# 训练神经障碍证书
nbc = NeuralBarrierCertificate(state_dim=2, barrier_type='cbf')
spec = AEBSSafetySpec()

safe_model = nbc.synthesize(
    dynamics_fn=aebs_dynamics,
    safety_spec=spec,
    state_bounds=[(5, 50), (0, 30)],
    max_iter=50
)
```

---

## 7. 常见问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CEGIS 不收敛 | 网络太小或太大 | 调整网络结构 |
| 验证太慢 | 网络层数多 | 用 IBP 替代 dReal |
| 反例在边界 | 采样不均匀 | 边界加强采样 |
| 过拟合 | 训练集太小 | 数据增强 |
| Lie 导数不准 | 激活函数不平滑 | 用 Tanh 替代 ReLU |

---

## 8. 相关概念

- [[CBF (控制障碍函数)]] — 神经 CBF 的理论基础
- [[SBF SBC (随机障碍函数与证书)]] — 神经 SBC
- [[CEGIS (反例引导合成)]] — 训练框架
- [[dReal]] — 精确验证
- [[IBP (区间界传播)]] — 快速验证
- [[CROWN (神经网络验证)]] — 精确验证
- [[Spectral Normalization]] — 网络正则化
- [[Lipschitz 常数]] — 鲁棒性保证
- [[前向不变性 (Forward Invariance)]] — 安全性质
- [[超鞅 (Supermartingale)]] — 概率安全

---

> **参考**: 
> - Abate et al., "FARKAS: Deep Flywheel Control with Neural Barrier Certificates," 2022
> - Zhao et al., "Neural Barrier Certificates for Stochastic Safety," 2023
> - Dai et al., "Safe Reinforcement Learning with Neural Barrier Certificates," 2021
