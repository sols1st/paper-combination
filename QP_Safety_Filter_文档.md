# QP-based CBF 安全滤波器实现文档

## 1. 概述

本文档记录了在 SafePVC（Provably Probabilistic Safe Controller Synthesis）框架基础上，新增的 **QP（二次规划）安全滤波器** 的实现细节、数学原理和使用方法。

SafePVC 框架通过离线训练 **随机障碍证书（Stochastic Barrier Certificate, SBC）** 来提供概率安全保证。但其训练好的神经网络控制器在运行时是一个纯前向传播网络，没有实时的安全约束机制。

本次新增的 QP 安全滤波器在 NN 控制器之后加入一个 **运行时安全层**，基于 **控制障碍函数（Control Barrier Function, CBF）** 和 **二次规划（Quadratic Programming）** 对 NN 控制器的输出进行最小化修正，确保每一步控制动作都满足安全约束。

---

## 2. 文件结构

```
Aebs/QP/
├── __init__.py                 # 包初始化，导出 CBFQPSafetyFilter
├── qp_safety_filter.py         # 核心：QP 安全滤波器类
├── simulate_with_qp.py         # 仿真脚本：对比纯 NN 与 NN+QP 的轨迹
├── visualize_qp.py             # 可视化脚本：CBF 安全区域、干预热力图等
└── results/                    # 输出目录（自动创建）
    ├── qp_safety_comparison.png
    ├── cbf_safe_region_stopping.png
    ├── cbf_landscape.png
    └── ...
```

---

## 3. 数学原理

### 3.1 系统动力学（AEBS）

CARLA 紧急制动系统的离散时间动力学模型：

$$d_{k+1} = d_k - v_k \cdot \Delta t$$

$$v_{k+1} = v_k - a_k \cdot \Delta t$$

其中：
- 状态 $s = [d_{\text{norm}}, v]$：归一化距离和速度
- 动作 $u = a$：加速度，$a \in [-3.0, 3.0]$ m/s²
- $\Delta t = 0.05$ s：控制周期
- $d_{\text{norm}} = d / \sigma$：$\sigma$ 为训练数据的距离标准差

动力学关于控制输入 $u$ 是**仿射的（affine）**：

$$s_{k+1} = f(s_k) + g(s_k) \cdot u_k$$

其中 $\partial f / \partial u = [0, -\Delta t]^\top$。

### 3.2 控制障碍函数（CBF）

控制障碍函数 $h(s)$ 定义安全集：

$$\mathcal{C} = \{ s \in S : h(s) \geq 0 \}$$

$h(s) \geq 0$ 表示状态安全，$h(s) < 0$ 表示状态不安全。

#### 模式一：制动距离 CBF（推荐，相对阶 2）

$$h(s) = (d_{\text{real}} - d_{\text{unsafe}}) - \frac{v^2}{2 \cdot a_{\max}}$$

其中：
- $d_{\text{real}} = d_{\text{norm}} \cdot \sigma$：真实距离（米）
- $d_{\text{unsafe}}$：不安全区域上界的真实距离（米）
- $a_{\max} = 3.0$ m/s²：最大制动减速度
- $v^2 / (2 a_{\max})$：当前速度下的最小制动距离

**物理含义**：$h(s) \geq 0$ 意味着车辆有足够距离在到达不安全区域之前停下来。

梯度（关于状态 $s = [d_{\text{norm}}, v]$）：

$$\frac{\partial h}{\partial d_{\text{norm}}} = \sigma, \quad \frac{\partial h}{\partial v} = -\frac{v}{a_{\max}}$$

**关键特性**：$\partial h / \partial v \neq 0$（当 $v > 0$），使得 QP 可以通过加速度直接执行安全约束。这是**相对阶 2** 的 CBF（加速度影响速度，速度影响距离）。

#### 模式二：几何 CBF（相对阶 1）

$$h(s) = (d_{\text{norm}} - d_{\text{unsafe,norm}}) \cdot \sigma$$

梯度：$\nabla h = [\sigma, 0]$

**局限性**：$\partial h / \partial v = 0$，加速度无法在单步内直接影响距离。QP 约束变为关于 $u$ 的常量，只能在违反时触发紧急制动。

#### 模式三：学习型 CBF（使用 SafePVC 训练的 Barrier Certificate）

$$h(s) = B_{\text{threshold}} - B(s)$$

其中：
- $B(s)$：SafePVC 训练的障碍证书网络（L-Net）
- $B_{\text{threshold}} = \frac{1}{1-p}$：定理 2.2 中的阈值（$p$ 为目标安全概率）

梯度通过 PyTorch autograd 自动计算。

### 3.3 QP 问题形式化

在每一个控制时刻，求解以下二次规划：

$$\min_{u} \quad \frac{1}{2} \| u - u_{\text{nn}} \|^2$$

$$\text{s.t.} \quad h(f(s, u)) \geq (1 - \alpha) \cdot h(s) \quad \text{(CBF 约束)}$$

$$u_{\min} \leq u \leq u_{\max} \quad \text{(动作边界)}$$

其中：
- $u_{\text{nn}} = \pi(o)$：NN 控制器的输出
- $\alpha \in (0, 1]$：CBF 松弛参数，$\alpha$ 越小安全约束越严格

#### 线性化 CBF 约束

由于动力学关于 $u$ 是仿射的，对 $h$ 在零动作下一阶展开：

$$h(f(s, u)) \approx h(s_{\text{frozen}}) + \nabla h^\top \cdot \frac{\partial f}{\partial u} \cdot u$$

其中 $s_{\text{frozen}} = f(s, 0)$（零动作下的下一状态）。

定义：

$$H = \nabla h^\top \cdot \frac{\partial f}{\partial u} = \frac{\partial h}{\partial v} \cdot (-\Delta t)$$

$$c = (1 - \alpha) \cdot h(s) - h(s_{\text{frozen}})$$

则 CBF 约束化为线性不等式：

$$H \cdot u \geq c$$

### 3.4 解析求解（1D 动作空间）

由于 AEBS 的动作空间是一维的（$u \in \mathbb{R}$），QP 有**解析解**，无需外部求解器：

**步骤 1**：计算约束要求的最小/最大控制量

$$u_{\text{cbf}} = \frac{c}{H}$$

**步骤 2**：根据 $H$ 的符号确定可行域

| $H$ 的符号 | 约束方向 | 可行域 |
|:----------:|:--------:|:------:|
| $H > 0$ | $u \geq u_{\text{cbf}}$ | $[\max(u_{\min}, u_{\text{cbf}}),\; u_{\max}]$ |
| $H < 0$ | $u \leq u_{\text{cbf}}$ | $[u_{\min},\; \min(u_{\max}, u_{\text{cbf}})]$ |
| $H \approx 0$ | 与 $u$ 无关 | 检查 $c \leq 0$ 是否成立 |

**步骤 3**：将 $u_{\text{nn}}$ 投影到可行域

$$u^* = \text{clip}(u_{\text{nn}},\; u_{\text{lower}},\; u_{\text{upper}})$$

**步骤 4**：不可行时的回退策略

若可行域为空（$u_{\text{lower}} > u_{\text{upper}}$），采用**紧急制动** $u^* = u_{\max} = 3.0$。

---

## 4. 核心类设计

### `CBFQPSafetyFilter`

```python
class CBFQPSafetyFilter:
    def __init__(self, barrier_model, env, alpha=0.5,
                 cbf_mode="learned", reach_prob=0.95,
                 device=None, a_max=3.0):
        """
        Args:
            barrier_model: 训练好的 L-Net（None 时使用几何 CBF）
            env: Aebs 环境对象
            alpha: CBF 松弛参数 (0 < α ≤ 1)
            cbf_mode: 'learned' | 'geometric' | 'stopping_distance'
            reach_prob: 目标安全概率 p
            a_max: 最大制动减速度（仅 stopping_distance 模式）
        """

    def compute_cbf_value(self, s):
        """计算 h(s)，返回 [B, 1] 张量"""

    def compute_cbf_gradient(self, s):
        """计算 ∇h(s)，返回 [B, 2] 张量"""

    def solve_qp(self, s, u_nn):
        """解析求解 QP，返回 (u_safe, intervened_mask)"""

    def shield(self, s, u_nn):
        """主入口：返回 (u_safe, stats_dict)"""

    def get_cbf_boundary(self, n_points=100):
        """计算 CBF 等高线数据，用于可视化"""

    def reset_stats(self):
        """重置干预统计"""
```

### 设计要点

1. **解析求解器**：1D 动作空间无需外部 QP 求解器（如 cvxopt、OSQP），纯 PyTorch 实现
2. **批处理计算**：所有操作支持 batch 维度，可并行处理多个状态
3. **双输入兼容**：`shield()` 同时接受 torch.Tensor 和 numpy.ndarray
4. **干预统计**：自动记录干预次数和干预率
5. **不可行回退**：QP 不可行时自动触发紧急制动（$u = u_{\max}$）

---

## 5. 仿真脚本说明

### `simulate_with_qp.py`

运行 100 条并行轨迹，对比三种控制策略：

| 策略 | 描述 |
|------|------|
| NN (no filter) | 纯神经网络控制器，无安全滤波 |
| NN + QP (stopping) | NN 控制器 + 制动距离 CBF 的 QP 滤波器 |
| NN + QP (geometric) | NN 控制器 + 几何 CBF 的 QP 滤波器 |

**输出指标**：
- **Safety Rate**：不进入不安全区域的轨迹比例
- **Intervention Rate**：QP 修改 NN 动作的时间步比例
- **Avg Episode Length**：平均轨迹长度

**运行方式**：
```bash
cd artical-F122
python -m Aebs.QP.simulate_with_qp
```

### `visualize_qp.py`

生成以下可视化图表：

| 图表 | 描述 |
|------|------|
| CBF 安全区域 | $h(s)$ 等高线图，绿色=安全，红色=不安全 |
| QP 干预热力图 | 状态空间中 QP 触发干预的区域 |
| 动作修正量图 | $\Delta u = u_{\text{safe}} - u_{\text{nn}}$ 的空间分布 |
| Barrier Certificate | $B(s)$ 的景观图（需加载训练好的 L-Net） |
| $\alpha$ 参数对比 | 不同松弛参数下的 CBF 安全区域对比 |

**运行方式**：
```bash
python -m Aebs.QP.visualize_qp
```

---

## 6. 与 SafePVC 的关系

```
┌─────────────────────────────────────────────────────────────┐
│                     SafePVC 离线训练                         │
│                                                              │
│  cGAN → MLP 蒸馏 → PPO 预训练 → SBC 学习 + CEGIS 迭代       │
│                                                              │
│  输出：                                                      │
│    - π (训练好的 NN 控制器, AebsEnd2EndNet)                   │
│    - B(s) (训练好的障碍证书, L-Net/MLP)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  QP 安全滤波器（在线运行时）                   │
│                                                              │
│  每个控制时刻：                                               │
│    1. π 输出 u_nn                                            │
│    2. 计算 h(s) 和 ∇h(s)（使用 B(s) 或几何 CBF）             │
│    3. 求解 QP → u_safe                                       │
│    4. 执行 u_safe                                            │
│                                                              │
│  提供运行时安全保证，弥补离线验证的局限性                       │
└─────────────────────────────────────────────────────────────┘
```

### 互补性分析

| 维度 | SafePVC (SBC) | QP 安全滤波器 (CBF) |
|------|---------------|---------------------|
| **保证类型** | 概率安全下界（无限时间范围） | 逐步安全约束（瞬时） |
| **计算方式** | 离线训练 + 形式化验证 | 在线实时优化 |
| **计算开销** | 训练耗时，推理为纯前向传播 | 每步需解 QP（解析解极快） |
| **保守性** | 依赖 IBP 验证精度 | 依赖 CBF 设计和线性化精度 |
| **适用场景** | 需要正式安全证书 | 需要运行时安全保障 |

### 使用训练好的 L-Net 作为 CBF

SafePVC 训练的 L-Net（barrier certificate）可以直接用作 QP 的 CBF：

```python
# 加载训练好的 L-Net
from Aebs.VT.utils import MLP
barrier_model = MLP([2, 16, 8, 1], activation="tanh", square_output=True)
barrier_model.load_state_dict(torch.load("./Aebs/Aebs_l_model.pth"))

# 创建 QP 滤波器
qp_filter = CBFQPSafetyFilter(
    barrier_model=barrier_model,
    env=env,
    alpha=0.5,
    cbf_mode="learned",    # 使用学习型 CBF
    reach_prob=0.95,       # 对应 B_threshold = 1/(1-0.95) = 20
)
```

---

## 7. 参数说明

### 关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|:------:|------|----------|
| `alpha` | 0.5 | CBF 松弛参数 | 减小→更安全但更频繁干预；增大→更宽松 |
| `a_max` | 3.0 | 最大制动减速度 (m/s²) | 设为系统的最大可用制动能力 |
| `cbf_mode` | "learned" | CBF 类型 | 推荐 "stopping_distance" |
| `reach_prob` | 0.95 | 目标安全概率 | 与 SafePVC 的 reach_prob 保持一致 |

### α 参数的影响

$\alpha$ 控制 CBF 约束的严格程度：

- $\alpha \to 0$：要求 $h(s_{k+1}) \geq h(s_k)$，即安全裕度不能减小 → 最保守
- $\alpha \to 1$：要求 $h(s_{k+1}) \geq 0$，即只需保持安全 → 最宽松
- $\alpha = 0.5$：平衡安全与性能的经验值

$$h(s_{k+1}) \geq (1 - \alpha) \cdot h(s_k)$$

---

## 8. 测试验证结果

### 单元测试（制动距离 CBF，$\alpha=0.5$, $a_{\max}=3.0$）

| 场景 | 真实距离 | 速度 | $h$ (m) | $u_{\text{nn}}$ | $u_{\text{safe}}$ | 干预 |
|------|:--------:|:----:|:-------:|:---------------:|:-----------------:|:----:|
| 初始状态 | 15.5m | 2.8 m/s | +8.19 | -0.50 | -0.50 | ❌ |
| 中间状态 | 10.0m | 1.5 m/s | +3.63 | 0.00 | 0.00 | ❌ |
| 较近状态 | 8.0m | 2.0 m/s | +1.33 | -0.50 | -0.50 | ❌ |
| 接近危险 | 6.5m | 2.5 m/s | -0.54 | -0.50 | +3.00 | ✅ |
| 不安全区 | 5.5m | 2.0 m/s | -1.17 | -1.00 | +3.00 | ✅ |

**验证逻辑**：
- 初始状态：距离裕度 9.5m，制动距离 1.31m，$h = 8.19 > 0$ → 无需干预 ✓
- 接近危险：距离裕度 0.5m，制动距离 1.04m，$h = -0.54 < 0$ → QP 强制最大制动 ✓

### 模式对比（$s = [d=6.5\text{m}, v=2.5\text{m/s}]$）

| CBF 模式 | $h$ (m) | $\partial h / \partial v$ | 干预行为 |
|----------|:-------:|:-------------------------:|----------|
| geometric | +0.50 | 0.000 | 被动干预（H≈0，回退紧急制动） |
| stopping_distance | -0.54 | -2.917 | 主动干预（通过 QP 计算最小修正） |

制动距离 CBF 能够**提前感知速度风险**，在距离裕度不足时主动调整加速度；几何 CBF 只在距离已进入危险区后才被动触发。

---

## 9. 运行指南

### 前置条件

确保已安装 SafePVC 所需的全部依赖（参考 `environment.yml`），以及 `gymnasium`：

```bash
pip install gymnasium
```

### 使用训练好的 SafePVC 模型

1. 确认模型文件存在：
   - `Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth`（观测模型）
   - `Aebs/controller/state_net_trained.pth`（状态估计网络）
   - `Aebs/controller/best_model/best_model.zip`（PPO 控制器）
   - `Aebs/Aebs_l_model.pth`（障碍证书，可选）

2. 运行仿真：
   ```bash
   cd artical-F122
   python -m Aebs.QP.simulate_with_qp
   ```

3. 运行可视化：
   ```bash
   python -m Aebs.QP.visualize_qp
   ```

### 在代码中使用

```python
from Aebs.QP import CBFQPSafetyFilter
from Aebs.system.env import Aebs

env = Aebs(0.05)

# 创建 QP 滤波器
qp = CBFQPSafetyFilter(
    barrier_model=None,       # 或传入训练好的 L-Net
    env=env,
    alpha=0.5,
    cbf_mode="stopping_distance",
    a_max=3.0,
    device="cuda",
)

# 在每个控制时刻
u_safe, stats = qp.shield(state, u_nn)
# u_safe: 安全动作
# stats:  {'intervention_rate': 0.15, 'num_intervened': 3, ...}
```

---

## 10. 参考文献

1. Ames, A. D., et al. "Control barrier functions: Theory and applications." *ECC*, 2019.
2. Xiao, W., et al. "Differentiable control barrier functions for vision-based end-to-end autonomous driving." *arXiv:2203.02401*, 2022.
3. Abdi, H., et al. "Safe control using vision-based control barrier function (V-CBF)." *ICRA*, 2023.
4. Lechner, M., et al. "Stability verification in stochastic control systems via neural network supermartingales." *AAAI*, 2022.
