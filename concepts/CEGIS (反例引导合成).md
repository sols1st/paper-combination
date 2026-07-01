# CEGIS (反例引导归纳合成)

> **一句话**：CEGIS 是一种"训练-验证-反馈"的**迭代合成框架**——先训练一个候选程序/网络，再用形式化方法验证它。如果验证失败，将反例加入训练集重新训练，直到验证通过。它是本项目中合成安全神经网络控制器的核心方法。

---

## 1. 直觉理解

### 1.1 考试类比

想象你在准备一场考试：

```
循环:
1. 【学习】做练习题，学会解题方法
2. 【模考】老师出一份新卷子测试你
3. 【反馈】
   - 如果通过 → 恭喜！你真的学会了
   - 如果失败 → 把错题加入练习题集，回到步骤 1
```

CEGIS 就是这个过程的自动化版本：
- **学习** = 训练神经网络
- **模考** = 用 SMT/IBP 验证安全性
- **错题** = 反例（违反安全条件的状态）

---

## 2. 形式化框架

### 2.1 基本结构

```
┌─────────────────────────────────────────┐
│              CEGIS Loop                  │
│                                         │
│  ┌───────────┐      ┌──────────────┐    │
│  │  Learner  │ ───→ │  Verifier    │    │
│  │  (训练器)  │      │  (验证器)     │    │
│  └───────────┘      └──────────────┘    │
│       ↑                     │           │
│       │                     │           │
│       │  反例 (counterexample)           │
│       └─────────────────────┘           │
│                                         │
└─────────────────────────────────────────┘
```

### 2.2 组件

| 组件 | 角色 | 本项目中的实现 |
|------|------|--------------|
| **Learner** | 训练候选网络 | PyTorch + PPO/梯度下降 |
| **Verifier** | 验证安全属性 | [[dReal]] / [[IBP (区间界传播)]] / [[CROWN (神经网络验证)]] |
| **Dataset** | 训练数据 | 初始采样 + 反例集合 |
| **Property** | 要验证的性质 | CBF/SBC 条件 |

### 2.3 算法

```
输入:
  - 状态空间 S
  - 安全属性 φ(x)（如 CBF 条件）
  - 初始训练集 D = {(x_1), ..., (x_N)}

输出:
  - 满足 φ 的网络 π*

算法:
  D = initial_samples(S)
  
  while True:
    # 1. 训练
    π = train(D, loss_fn)
    
    # 2. 验证
    result = verify(π, φ, S)
    
    if result == "safe":
      return π  # 验证通过！
    else:
      # 3. 提取反例
      x_ce = result.counterexample
      D = D ∪ {x_ce}  # 加入训练集
```

---

## 3. 在安全控制器合成中的应用

### 3.1 CBF 控制器合成

**目标**：找到神经网络 $\pi(x)$ 使得 [[CBF (控制障碍函数)]] 条件全局满足：

$$L_f b(x) + L_g b(x) \cdot \pi(x) + \alpha(b(x)) \geq 0, \quad \forall x \in S$$

**CEGIS 流程**：

```python
def cegis_cbf(dynamics, cbf, state_bounds, max_iter=100):
    """
    CEGIS 合成 CBF 安全控制器
    
    dynamics: (f_fn, g_fn)
    cbf: (b_fn, alpha_fn)
    state_bounds: [(min, max), ...]
    """
    dataset = sample_initial(state_bounds, n=1000)
    model = make_controller_net()
    
    for iteration in range(max_iter):
        # 1. 训练: 最小化 CBF 违反
        train(model, dataset, cbf_loss)
        
        # 2. 验证: 是否存在 x 使 CBF 条件不满足？
        # 即: ∃x ∈ S: Lf_b + Lg_b * π(x) + α(b) < 0
        result = verify_cbf_condition(
            model, dynamics, cbf, state_bounds
        )
        
        if result == "safe":
            print(f"✅ 在第 {iteration} 次迭代验证通过！")
            return model
        else:
            # 3. 反例
            x_ce = result.counterexample
            dataset.append(x_ce)
            print(f"❌ 第 {iteration} 次: 反例 x = {x_ce}")
    
    raise RuntimeError("CEGIS 未收敛")
```

### 3.2 SBC 合成

对于 [[SBF SBC (随机障碍函数与证书)]]，CEGIS 验证的是鞅递减条件：

$$\forall s: B(s) - \mathbb{E}[B(s')] \geq \epsilon$$

---

## 4. 训练器 (Learner) 详解

### 4.1 CBF 损失函数

```python
def cbf_loss(model, batch, dynamics, cbf):
    """
    CBF 训练损失
    
    目标: Lf_b + Lg_b * π(x) + α(b) >= 0
    
    损失: max(0, -(Lf_b + Lg_b * π(x) + α(b)))
    """
    x = batch  # (N, n)
    
    # 计算 CBF 值
    b = cbf.b_fn(x)
    
    # 计算 Lie 导数
    Lf_b, Lg_b = compute_lie_derivatives(cbf.b_fn, dynamics, x)
    
    # 控制器输出
    u = model(x)
    
    # CBF 条件
    cbf_condition = Lf_b + Lg_b * u + cbf.alpha_fn(b)
    
    # 损失: 违反量
    violation = torch.relu(-cbf_condition)
    loss = violation.mean()
    
    return loss
```

### 4.2 多种损失组合

```python
def combined_loss(model, batch, dynamics, cbf, u_ref_fn=None):
    """
    组合损失:
    1. CBF 安全损失
    2. 参考跟踪损失 (可选)
    3. 控制量正则化
    4. 边界条件损失
    """
    x = batch
    u = model(x)
    
    # 1. CBF 安全损失
    b = cbf.b_fn(x)
    Lf_b, Lg_b = compute_lie_derivatives(cbf.b_fn, dynamics, x)
    cbf_val = Lf_b + Lg_b * u + cbf.alpha_fn(b)
    safety_loss = torch.relu(-cbf_val).mean()
    
    # 2. 参考跟踪
    if u_ref_fn is not None:
        u_ref = u_ref_fn(x)
        tracking_loss = ((u - u_ref) ** 2).mean()
    else:
        tracking_loss = 0
    
    # 3. 控制量正则化
    control_loss = (u ** 2).mean()
    
    # 4. 边界条件: 在安全边界上，u 应该指向安全方向
    boundary_mask = (b.abs() < 0.1)
    if boundary_mask.any():
        boundary_loss = torch.relu(-cbf_val[boundary_mask]).mean()
    else:
        boundary_loss = 0
    
    total_loss = (
        10.0 * safety_loss +      # 安全最重要
        1.0 * tracking_loss +      # 参考跟踪
        0.1 * control_loss +       # 控制量小
        5.0 * boundary_loss        # 边界行为
    )
    
    return total_loss
```

---

## 5. 验证器 (Verifier) 详解

### 5.1 使用 dReal

```python
from dreal import *

def verify_with_dreal(model, dynamics, cbf, state_bounds, delta=0.001):
    """
    用 dReal 验证 CBF 条件
    
    检查: ∃x ∈ S: Lf_b(x) + Lg_b(x) * π(x) + α(b(x)) < 0
    
    如果 unsat → 安全
    如果 δ-sat → 不安全，返回反例
    """
    # 声明变量
    x_vars = [Variable(f"x_{i}") for i in range(len(state_bounds))]
    
    # 编码神经网络
    u_expr = encode_nn_for_dreal(model, x_vars)
    
    # 编码 CBF 条件
    b_expr = encode_cbf_for_dreal(cbf, x_vars)
    Lf_b_expr = encode_lie_f_for_dreal(cbf, dynamics, x_vars)
    Lg_b_expr = encode_lie_g_for_dreal(cbf, dynamics, x_vars)
    
    # 违反条件: CBF 条件 < 0
    violation = Lf_b_expr + Lg_b_expr * u_expr + b_expr < 0
    
    # 添加状态范围约束
    constraints = [violation]
    for i, (lo, hi) in enumerate(state_bounds):
        constraints.append(x_vars[i] >= lo)
        constraints.append(x_vars[i] <= hi)
    
    formula = And(*constraints)
    result = CheckSat(formula, delta)
    
    if result:
        # 找到反例
        x_ce = [result[v].mid() for v in x_vars]
        return {"status": "unsafe", "counterexample": x_ce}
    else:
        return {"status": "safe"}
```

### 5.2 使用 IBP（更快但不精确）

```python
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def verify_with_ibp(model, dynamics, cbf, state_bounds):
    """
    用 IBP 验证 CBF 条件
    
    计算 CBF 条件在整个状态区间上的下界
    如果下界 >= 0 → 安全
    否则 → 不确定（可能保守）
    """
    # 包装模型
    dummy = torch.randn(1, len(state_bounds))
    bounded_model = BoundedModule(model, dummy)
    
    # 计算区间中心
    center = torch.tensor([(lo + hi) / 2 for lo, hi in state_bounds])
    eps = max((hi - lo) / 2 for lo, hi in state_bounds)
    
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_x = BoundedTensor(center.unsqueeze(0), ptb)
    
    # 计算输出界
    lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="IBP")
    
    # 如果控制量的下界满足 CBF 条件...
    # （需要组合 dynamics 和 cbf 一起计算）
    
    return lb, ub
```

---

## 6. 反例处理策略

### 6.1 直接加入训练集

```python
dataset.append(counterexample)
```

### 6.2 反例邻域扩充

```python
def augment_counterexample(x_ce, radius=0.1, n_samples=10):
    """在反例附近采样更多点"""
    augmented = [x_ce]
    for _ in range(n_samples):
        noise = torch.randn_like(x_ce) * radius
        augmented.append(x_ce + noise)
    return augmented
```

### 6.3 加权训练

```python
def weighted_loss(model, batch, weights):
    """给反例更高的权重"""
    losses = per_sample_loss(model, batch)
    return (losses * weights).mean()

# 反例权重 = 10, 普通样本权重 = 1
weights = torch.ones(len(dataset))
weights[counterexample_indices] = 10.0
```

---

## 7. 收敛性分析

### 7.1 CEGIS 何时收敛？

| 条件 | 说明 |
|------|------|
| **有限假设空间** | 如果候选网络结构有限，CEGIS 保证终止 |
| **连续假设空间** | 神经网络有无穷多参数，不保证收敛 |
| **凸问题** | 如果损失和属性都是凸的，收敛更快 |
| **好的验证器** | 验证器产生的反例越"有信息量"，收敛越快 |

### 7.2 实践建议

```python
# CEGIS 的实用技巧

# 1. 设定最大迭代
max_cegis_iters = 50

# 2. 监控收敛
cegis_history = {
    'n_counterexamples': [],
    'worst_violation': [],
    'verification_time': []
}

# 3. 如果验证太慢，切换到近似验证
if verification_time > 60:  # 超过 1 分钟
    verifier = IBPVerifier()  # 更快但不精确
else:
    verifier = dRealVerifier()  # 精确但慢

# 4. 并行验证多个候选
candidates = [train(seed=i) for i in range(4)]
for c in candidates:
    if verify(c) == "safe":
        return c
```

---

## 8. CEGIS 变体

| 变体 | 全称 | 特色 |
|------|------|------|
| **CEGIS** | Counter-Example Guided Inductive Synthesis | 基础版本 |
| **CEGAR** | Counter-Example Guided Abstraction Refinement | 用于模型检测 |
| **NN-CEGIS** | Neural Network CEGIS | 专门用于神经网络 |
| **$\alpha$-$\beta$-CEGIS** | — | 用 α-β-CROWN 作为验证器 |
| **Probabilistic CEGIS** | — | 概率版本，允许小概率违反 |

---

## 9. 完整代码示例

```python
import torch
import torch.nn as nn

class CEGISSynthesizer:
    """
    CEGIS 安全控制器合成器
    """
    def __init__(self, state_dim, action_dim, state_bounds, 
                 dynamics_fn, cbf_fn):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_bounds = state_bounds
        self.dynamics_fn = dynamics_fn
        self.cbf_fn = cbf_fn
        
        # 控制器网络
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )
        
        # 训练集
        self.dataset = self._initial_sample(1000)
        self.counterexamples = []
    
    def _initial_sample(self, n):
        """均匀采样初始训练集"""
        samples = []
        for _ in range(n):
            x = torch.tensor([
                torch.empty(1).uniform_(lo, hi).item()
                for lo, hi in self.state_bounds
            ])
            samples.append(x)
        return samples
    
    def train_step(self, n_epochs=10, batch_size=64, lr=1e-3):
        """训练一步"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            # 随机打乱
            indices = torch.randperm(len(self.dataset))
            
            for i in range(0, len(self.dataset), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = torch.stack([self.dataset[j] for j in batch_indices])
                
                # CBF 损失
                x = batch
                u = self.model(x)
                b, Lf_b, Lg_b = self.cbf_fn(x, self.dynamics_fn)
                alpha_b = b  # 简化: α(r) = r
                
                cbf_val = Lf_b + Lg_b * u + alpha_b
                loss = torch.relu(-cbf_val).mean()
                
                # 控制量正则化
                loss += 0.01 * (u ** 2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def verify(self):
        """验证（简化版：密集采样检查）"""
        # 密集采样
        n_test = 10000
        test_samples = self._initial_sample(n_test)
        test_batch = torch.stack(test_samples)
        
        u = self.model(test_batch)
        b, Lf_b, Lg_b = self.cbf_fn(test_batch, self.dynamics_fn)
        cbf_val = Lf_b + Lg_b * u + b
        
        # 找到最差违反
        min_val, min_idx = cbf_val.min(0)
        
        if min_val >= 0:
            return {"status": "safe"}
        else:
            x_ce = test_batch[min_idx]
            return {"status": "unsafe", "counterexample": x_ce, 
                    "violation": min_val.item()}
    
    def synthesize(self, max_iter=50):
        """完整的 CEGIS 循环"""
        for i in range(max_iter):
            print(f"\n=== CEGIS Iteration {i+1}/{max_iter} ===")
            
            # 训练
            self.train_step(n_epochs=20)
            
            # 验证
            result = self.verify()
            
            if result["status"] == "safe":
                print(f"✅ 验证通过! 共 {len(self.counterexamples)} 个反例")
                return self.model
            else:
                x_ce = result["counterexample"]
                self.counterexamples.append(x_ce)
                self.dataset.append(x_ce)
                
                # 反例邻域扩充
                for _ in range(5):
                    noise = torch.randn_like(x_ce) * 0.1
                    self.dataset.append(x_ce + noise)
                
                print(f"❌ 反例: x={x_ce.tolist()}, "
                      f"violation={result['violation']:.4f}")
        
        raise RuntimeError(f"CEGIS 未在 {max_iter} 次迭代内收敛")

# 使用
synth = CEGISSynthesizer(
    state_dim=2, action_dim=1,
    state_bounds=[(5, 30), (0, 25)],
    dynamics_fn=my_dynamics,
    cbf_fn=my_cbf
)
safe_controller = synth.synthesize()
```

---

## 10. 相关概念

- [[dReal]] — 精确验证器
- [[IBP (区间界传播)]] — 快速近似验证器
- [[CROWN (神经网络验证)]] — 更精确的近似验证器
- [[α-β-CROWN]] — 最精确的验证器
- [[CBF (控制障碍函数)]] — CEGIS 验证的安全属性
- [[SBF SBC (随机障碍函数与证书)]] — 概率安全属性
- [[Neural Barrier Certificate]] — CEGIS 合成的产物
- [[SMT (可满足性模理论)]] — 验证器的理论基础
- [[Lipschitz 常数]] — CEGIS 中用于网格验证

---

> **参考**: 
> - Solar-Lezama et al., "Combinatorial Sketching for Finite Programs," ASPLOS 2006
> - Abate et al., "Farkas: Automated Synthesis of Barrier Certificates," CAV 2018
> - Dai et al., "Bridging Machine Learning and Formal Methods," ICML 2021
