# HOCBF (高阶控制障碍函数)

> **一句话**：HOCBF 是 [[CBF (控制障碍函数)]] 的扩展，用于处理**高 [[相对度 (Relative Degree)]]** 的系统——即控制输入 $u$ 不直接出现在安全函数 $b(x)$ 的一阶导数中，而需要多次求导才出现。

---

## 1. 为什么需要 HOCBF？

### 1.1 标准 CBF 的限制

标准 CBF 要求控制 $u$ **直接出现在 $b$ 的一阶导数中**：

$$\dot{b}(x) = L_f b(x) + L_g b(x) u$$

当 $L_g b(x) = 0$（控制不直接影响 $b$ 的变化率）时，标准 CBF 无法工作。

### 1.2 常见的高相对度场景

**例子 1：车辆跟车**

安全约束：$b(x) = d - d_{\text{safe}}$（距离约束）

动力学：$\dot{d} = -v$, $\dot{v} = -a$（$a$ 是控制输入——加速度）

$\dot{b} = -v$（不含 $a$！）—— **相对度 = 2**

需要对 $b$ 求两次导数，$a$ 才显式出现：

$\ddot{b} = -\dot{v} = a$ ✅

**例子 2：车辆车道保持**

安全约束：$b(x) = d_{\text{lane}} - |y|$（横向偏移约束）

如果控制是转向角速率（$\dot{\delta}$），需要多次求导：

$b \to \dot{b} \to \ddot{b} \to \dddot{b}$ —— **相对度 = 3**

---

## 2. 数学定义

### 2.1 [[相对度 (Relative Degree)]]

**定义**：函数 $b(x)$ 对系统 $\dot{x} = f(x) + g(x)u$ 的相对度 $m$ 是最小的正整数，使得控制 $u$ 在 $b$ 的第 $m$ 阶时间导数中**显式出现**。

- 相对度 1：$L_g b(x) \neq 0$ → 标准 CBF 可用
- 相对度 2：$L_g b = 0$ 但 $L_g L_f b \neq 0$ → 需要 HOCBF
- 相对度 $m$：前 $m-1$ 次 Lie 导数不含 $u$，第 $m$ 次才含

### 2.2 序列函数构造

HOCBF 的核心是构造一个**递推序列** $\psi_0, \psi_1, \ldots, \psi_m$：

$$\psi_0(x) := b(x)$$

$$\psi_i(x) := \dot{\psi}_{i-1}(x) + \alpha_i(\psi_{i-1}(x)), \quad i = 1, \ldots, m$$

其中每个 $\alpha_i$ 是一个 $(m-i)$ 阶可微的 [[Class K 函数]]。

**逐层展开**：

$\psi_0 = b$

$\psi_1 = \dot{b} + \alpha_1(b) = L_f b + \alpha_1(b)$

$\psi_2 = \dot{\psi}_1 + \alpha_2(\psi_1) = L_f^2 b + L_g L_f b \cdot u + \alpha_1'(b) L_f b + \alpha_2(\psi_1)$

到第 $m$ 层时，$u$ 显式出现在 $\psi_m$ 中。

### 2.3 HOCBF 约束

$$\sup_{u \in U} \left[ L_f^m b(x) + L_g L_f^{m-1} b(x) \cdot u + O(b(x)) + \alpha_m(\psi_{m-1}(x)) \right] \geq 0$$

其中 $O(b(x))$ 是中间项的总和：

$$O(b(x)) = \sum_{i=1}^{m-1} L_f^i (\alpha_{m-i} \circ \psi_{m-i-1})(x)$$

### 2.4 对应的安全集序列

$$C_i = \{x \in \mathbb{R}^n : \psi_{i-1}(x) \geq 0\}, \quad i = 1, \ldots, m$$

HOCBF 保证 $C_1 \cap C_2 \cap \cdots \cap C_m$ 是 [[前向不变性 (Forward Invariance)|前向不变的]]。

---

## 3. 直觉解释：递归保证机制

HOCBF 的安全性保证是**递归的**：

```
ψ_m ≥ 0 → ψ_{m-1} ≥ 0 → ... → ψ_1 ≥ 0 → ψ_0 = b ≥ 0 (安全!)
```

每一层的保证机制：

- $\psi_m \geq 0$ 意味着 $\dot{\psi}_{m-1} + \alpha_m(\psi_{m-1}) \geq 0$
- 这等价于说 $\psi_{m-1}$ 满足一个标准 CBF 条件
- 因此 $\psi_{m-1} \geq 0$ 被保证
- 递推下去，最终 $b(x) \geq 0$ 被保证

**类比**：多米诺骨牌
- $\psi_m \geq 0$ 是第一块骨牌倒下
- 每块骨牌倒下都会推倒下一块
- 最后一块骨牌是 $b(x) \geq 0$（安全！）

---

## 4. 具体例子：AEBS 系统 (相对度 2)

### 4.1 系统设定

状态：$x = [d, v]^T$

动力学：$\dot{d} = -v$, $\dot{v} = -a$（$u = a$ 是加速度）

安全约束：$b(x) = d - d_{\text{safe}}$

### 4.2 验证相对度

$L_g b = \nabla b \cdot g = [1, 0] \cdot [0, -1]^T = 0$

$L_g L_f b = \nabla(L_f b) \cdot g = \nabla(-v) \cdot [0, -1]^T = [0, -1] \cdot [0, -1]^T = 1 \neq 0$

→ **相对度 $m = 2$**

### 4.3 构造序列函数

选择 $\alpha_1(r) = c_1 r$, $\alpha_2(r) = c_2 r$（线性 Class K）

**第 0 层**：$\psi_0 = b = d - d_{\text{safe}}$

**第 1 层**：$\psi_1 = \dot{b} + c_1 b = -v + c_1(d - d_{\text{safe}})$

**第 2 层**（需要 $u$ 出现）：

$\psi_2 = \dot{\psi}_1 + c_2 \psi_1$

$= \frac{d}{dt}[-v + c_1(d - d_{\text{safe}})] + c_2[-v + c_1(d - d_{\text{safe}})]$

$= [-\dot{v} + c_1 \dot{d}] + c_2[-v + c_1(d - d_{\text{safe}})]$

$= [a + c_1(-v)] + c_2[-v + c_1(d - d_{\text{safe}})]$

$= a - c_1 v - c_2 v + c_1 c_2 (d - d_{\text{safe}})$

$= a - (c_1 + c_2)v + c_1 c_2 (d - d_{\text{safe}})$

### 4.4 HOCBF 约束

$$\psi_2 \geq 0 \implies a \geq (c_1 + c_2)v - c_1 c_2 (d - d_{\text{safe}})$$

### 4.5 数值例子

设 $d_{\text{safe}} = 6$, $c_1 = 1$, $c_2 = 1$

当前状态：$d = 10$, $v = 2$

$\psi_0 = 10 - 6 = 4$

$\psi_1 = -2 + 1 \times 4 = 2$

约束：$a \geq (1+1) \times 2 - 1 \times 1 \times 4 = 0$

即 $a \geq 0$（不减速也可以，因为还比较远）

如果 $d = 7$, $v = 2$：

$\psi_0 = 1$

$\psi_1 = -2 + 1 = -1$ ❌（$\psi_1 < 0$，已经违反了中间约束！）

约束：$a \geq 4 - 1 = 3$（需要最大刹车 $a = 3$）

---

## 5. 代码实现

```python
import torch

class HOCBF:
    """
    High Order Control Barrier Function
    
    处理相对度 m=2 的系统
    安全约束: b(s) = d - d_safe >= 0
    """
    def __init__(self, d_safe=6.0, c1=1.0, c2=1.0,
                 u_min=-3.0, u_max=3.0):
        self.d_safe = d_safe
        self.c1 = c1
        self.c2 = c2
        self.u_min = u_min
        self.u_max = u_max
    
    def compute_psi(self, s):
        """
        计算序列函数 ψ_0, ψ_1
        
        输入:
            s: tensor (batch, 2), [d, v]
        输出:
            psi0, psi1: tensor (batch,)
        """
        d = s[:, 0]
        v = s[:, 1]
        
        psi0 = d - self.d_safe
        psi1 = -v + self.c1 * psi0
        
        return psi0, psi1
    
    def compute_constraint(self, s):
        """
        计算 HOCBF 约束的下界
        
        ψ_2 = a - (c1+c2)*v + c1*c2*(d-d_safe) >= 0
        → a >= (c1+c2)*v - c1*c2*(d-d_safe)
        
        返回: u_min_hocbf (CBF 要求的最小控制)
        """
        d = s[:, 0]
        v = s[:, 1]
        
        u_min_hocbf = ((self.c1 + self.c2) * v 
                       - self.c1 * self.c2 * (d - self.d_safe))
        
        return u_min_hocbf
    
    def safety_filter(self, s, u_ref):
        """HOCBF 安全过滤器"""
        u_min_hocbf = self.compute_constraint(s)
        
        # 同时考虑 HOCBF 约束和控制界限
        u_lower = torch.max(u_min_hocbf, 
                           torch.tensor(self.u_min))
        
        u_safe = torch.clamp(u_ref, min=u_lower, max=self.u_max)
        return u_safe
```

---

## 6. 与 [[dCBF (可微控制障碍函数)]] 的关系

传统 HOCBF 的 $\alpha_i$ 是**固定的**，这导致保守性：

- $\alpha_i$ 太大 → 过度保守（提前很远就开始刹车）
- $\alpha_i$ 太小 → 约束太弱（可能来不及刹车）

**dCBF 的创新**：将 $\alpha_i$ 替换为 $p_i(z) \cdot \alpha_i(\cdot)$，其中 $p_i(z)$ 是**可学习的环境依赖函数**。

$$\psi_i = \dot{\psi}_{i-1} + p_i(z) \alpha_i(\psi_{i-1})$$

当场景安全时 $p_i$ 大（放松约束），当场景危险时 $p_i$ 小（收紧约束）。

---

## 7. 参数选择指南

| 参数 | 影响 | 选择建议 |
|------|------|---------|
| $c_1, c_2$ | 越大越保守，越早干预 | 从 1.0 开始调整 |
| $\alpha$ 类型 | 线性最简单，非线性更灵活 | 线性 $\alpha(r) = cr$ 最常用 |
| 相对度 $m$ | 由系统和 $b(x)$ 决定 | 先计算 $L_g b$ 和 $L_g L_f b$ |

---

## 8. 相关概念

- [[CBF (控制障碍函数)]] — HOCBF 的基础
- [[相对度 (Relative Degree)]] — 决定是否需要 HOCBF
- [[Lie 导数]] — 计算相对度的工具
- [[Class K 函数]] — $\alpha_i$ 的选择
- [[dCBF (可微控制障碍函数)]] — HOCBF 的可学习版本
- [[前向不变性 (Forward Invariance)]] — HOCBF 保证的性质
- [[QP (二次规划)]] — HOCBF 的实际使用方式
- [[比较引理 (Comparison Lemma)]] — HOCBF 安全性的数学基础

---

> **参考论文**: 
> - Wei Xiao, Calin Belta, "High Order Control Barrier Functions," IEEE TAC 2022
