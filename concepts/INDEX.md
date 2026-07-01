# 形式化安全控制 — 概念笔记索引

> 本目录包含形式化安全控制领域的原子化笔记，使用 Obsidian 双链 `[[ ]]` 互相关联。
> 建议按「推荐阅读顺序」从上到下阅读。

---

## 📚 推荐阅读顺序

### 第一层：核心安全概念（先读这些）

| # | 笔记 | 一句话 |
|---|------|--------|
| 1 | [[CBF (控制障碍函数)]] | 安全过滤器：保证系统不进入危险区域 |
| 2 | [[前向不变性 (Forward Invariance)]] | 安全集合的核心性质：进去了就出不来 |
| 3 | [[Lyapunov 稳定性]] | CBF 的"对偶"：保证系统收敛到目标 |

### 第二层：数学基础（CBF 的数学工具）

| # | 笔记 | 一句话 |
|---|------|--------|
| 4 | [[Lie 导数]] | CBF 条件中的核心计算工具 |
| 5 | [[Class K 函数]] | 控制"多积极地"保持安全 |
| 6 | [[相对度 (Relative Degree)]] | 控制输入需要几次微分才出现在输出中 |
| 7 | [[比较引理 (Comparison Lemma)]] | 从微分不等式推出安全性 |

### 第三层：CBF 扩展

| # | 笔记 | 一句话 |
|---|------|--------|
| 8 | [[HOCBF (高阶控制障碍函数)]] | 处理高相对度系统的 CBF |
| 9 | [[dCBF (可微控制障碍函数)]] | 可学习、自适应的 CBF |
| 10 | [[CLF (控制 Lyapunov 函数)]] | 安全 + 稳定的联合控制 |

### 第四层：优化与求解

| # | 笔记 | 一句话 |
|---|------|--------|
| 11 | [[凸优化基础 (Convex Optimization)]] | LP/QP/SDP 的统一框架 |
| 12 | [[QP (二次规划)]] | CBF 控制器的求解核心 |
| 13 | [[KKT 条件]] | 带约束优化的最优性条件 |
| 14 | [[可微 QP (Differentiable QP)]] | 让 QP 可反向传播 |
| 15 | [[对偶理论与拉格朗日松弛]] | 硬约束 → 软约束的数学基础 |

### 第五层：概率安全

| # | 笔记 | 一句话 |
|---|------|--------|
| 16 | [[鞅理论 (Martingale Theory)]] | 随机过程的基础理论 |
| 17 | [[超鞅 (Supermartingale)]] | 期望递减的随机过程 |
| 18 | [[Doob 停时不等式 (Optional Stopping Theorem)]] | 从超鞅推出安全概率界 |
| 19 | [[SBF SBC (随机障碍函数与证书)]] | CBF 的随机版本 |

### 第六层：神经网络验证

| # | 笔记 | 一句话 |
|---|------|--------|
| 20 | [[分段线性函数 (Piecewise Linear)]] | ReLU 网络的结构特性 |
| 21 | [[IBP (区间界传播)]] | 最快的神经网络输出界计算 |
| 22 | [[CROWN (神经网络验证)]] | 更精确的线性松弛方法 |
| 23 | [[α-β-CROWN]] | 最精确的可优化验证方法 |
| 24 | [[auto_LiRPA]] | IBP/CROWN 的 Python 实现库 |

### 第七层：形式化验证工具

| # | 笔记 | 一句话 |
|---|------|--------|
| 25 | [[SMT (可满足性模理论)]] | 逻辑公式的自动推理 |
| 26 | [[dReal]] | 非线性 SMT 求解器 |
| 27 | [[Reachability Analysis (可达性分析)]] | 计算系统能到达哪些状态 |

### 第八层：训练与合成框架

| # | 笔记 | 一句话 |
|---|------|--------|
| 28 | [[CEGIS (反例引导合成)]] | 训练-验证-反馈的迭代框架 |
| 29 | [[Neural Barrier Certificate]] | 用神经网络做障碍函数 |
| 30 | [[PPO (Proximal Policy Optimization)]] | 预训练参考控制器 |
| 31 | [[Lipschitz 常数]] | 约束网络变化率 |
| 32 | [[Spectral Normalization]] | 训练时约束 Lipschitz |

### 第九层：高级方法

| # | 笔记 | 一句话 |
|---|------|--------|
| 33 | [[BarrierNet]] | 端到端可训练的安全控制器 |
| 34 | [[SOS (Sum-of-Squares)]] | 多项式优化的 SDP 方法 |
| 35 | [[几何规划与多项式规划 (GP & PP)]] | 特殊优化问题 |

---

## 🗺️ 概念关系图

```
                    ┌─────────────────────────────────┐
                    │        安全控制核心              │
                    │                                 │
                    │  CBF ←→ 前向不变性 ←→ Lyapunov  │
                    │   │         ↑           ↑       │
                    │   │    比较引理      CLF         │
                    │   │                             │
                    └───┼─────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐
      │  数学工具  │ │  优化方法  │ │ 概率安全  │
      │          │ │          │ │          │
      │ Lie导数  │ │ 凸优化    │ │ 鞅理论   │
      │ Class K  │ │ QP       │ │ 超鞅     │
      │ 相对度   │ │ KKT      │ │ Doob     │
      │          │ │ 可微QP   │ │ SBF/SBC  │
      │          │ │ 对偶理论  │ │          │
      └──────────┘ └──────────┘ └──────────┘
            │           │           │
            ▼           ▼           │
      ┌──────────┐ ┌──────────┐    │
      │  CBF扩展  │ │ 训练框架  │    │
      │          │ │          │    │
      │ HOCBF   │ │ CEGIS    │    │
      │ dCBF    │ │ PPO      │    │
      │ CLF+CBF │ │ Neural BC│    │
      └──────────┘ │ BarrierNet│   │
                   └──────────┘    │
            │           │          │
            ▼           ▼          ▼
      ┌──────────────────────────────────────┐
      │           验证工具                    │
      │                                      │
      │  IBP → CROWN → α-β-CROWN → auto_LiRPA│
      │  SMT → dReal                         │
      │  SOS (SDP-based)                     │
      │  Lipschitz / Spectral Normalization  │
      │  Reachability Analysis               │
      └──────────────────────────────────────┘
```

---

## 📁 文件列表

共 **35** 篇笔记。

```
concepts/
├── CBF (控制障碍函数).md
├── HOCBF (高阶控制障碍函数).md
├── dCBF (可微控制障碍函数).md
├── CLF (控制 Lyapunov 函数).md
├── Lie 导数.md
├── Class K 函数.md
├── 相对度 (Relative Degree).md
├── 前向不变性 (Forward Invariance).md
├── 比较引理 (Comparison Lemma).md
├── Lyapunov 稳定性.md
├── 凸优化基础 (Convex Optimization).md
├── QP (二次规划).md
├── KKT 条件.md
├── 可微 QP (Differentiable QP).md
├── 对偶理论与拉格朗日松弛.md
├── 鞅理论 (Martingale Theory).md
├── 超鞅 (Supermartingale).md
├── Doob 停时不等式 (Optional Stopping Theorem).md
├── SBF SBC (随机障碍函数与证书).md
├── 分段线性函数 (Piecewise Linear).md
├── IBP (区间界传播).md
├── CROWN (神经网络验证).md
├── α-β-CROWN.md
├── auto_LiRPA.md
├── SMT (可满足性模理论).md
├── dReal.md
├── Reachability Analysis (可达性分析).md
├── CEGIS (反例引导合成).md
├── Neural Barrier Certificate.md
├── PPO (Proximal Policy Optimization).md
├── Lipschitz 常数.md
├── Spectral Normalization.md
├── BarrierNet.md
├── SOS (Sum-of-Squares).md
├── 几何规划与多项式规划 (GP & PP).md
└── INDEX.md  (本文件)
```

---

> 笔记使用 Obsidian 格式，所有概念通过 `[[ ]]` 双向链接互联。
> 在 Obsidian 中打开此目录后，可以使用 **Graph View** 查看概念关系图。
