# 实验v3: 解耦双重安全架构 — 完整设计文档

> **核心思路**: QP只在推理时作为安全盾，不参与训练；SBC值调制CBF参数；彻底解决v2中QP干扰SBC的问题

---

## 目录

1. [问题背景: v2为什么失败](#1-问题背景)
2. [v3核心设计理念](#2-核心设计)
3. [网络结构与数据流](#3-网络结构)
4. [SBC调制CBF参数机制](#4-sbc调制)
5. [四种对比配置](#5-四种配置)
6. [代码架构](#6-代码架构)
7. [与原始代码的关系](#7-与原始代码的关系)

---

## 1. 问题背景: v2为什么失败

### 1.1 v2架构回顾

```
v2 (训练时QP):
  训练: state → VCLS → controller → QP层 → u* → 动力学 → SBC验证
                                              ↑
                                        QP在训练循环中!
  问题: QP层的非线性增加了闭环Lipschitz常数
        → SBC验证的K值增大 → 验证更保守
        → 形式化概率界从95.8%降至87.9% (-8%)
```

### 1.2 根因分析

| 问题 | 机制 |
|------|------|
| Lipschitz增大 | QP层 $u^* = \text{QP}(q,p,s)$ 增加闭环非线性，$L_{QP} > 1$ |
| CBF vs SBC冲突 | CBF追求$b(s_{t+1})\geq 0$，SBC追求$\mathbb{E}[B(s_{t+1})] \leq B(s_t)$ |
| 梯度噪声 | qpth隐式微分在约束边界处数值不稳定 |
| 反例减少 | QP阻止不安全行为→SBC收集的反例减少→CEGIS效率下降 |

**核心洞察**: QP的运行时安全价值和SBC的形式化验证价值应该**解耦**。

---

## 2. v3核心设计理念

### 2.1 解耦原则

```
           训练阶段                    推理阶段
    ┌──────────────────┐       ┌──────────────────┐
    │  Baseline 训练    │       │  QP 安全盾        │
    │  (无 QP 干扰!)    │       │  (不影响 SBC!)    │
    │                  │       │                  │
    │  Controller ←→ SBC│       │  u_ref → QP → u* │
    │  互相精化        │       │           ↑       │
    │                  │       │      SBC B(s)     │
    │  → 95%+ 形式化界 │       │     调制 p 参数    │
    └──────────────────┘       └──────────────────┘
```

### 2.2 关键优势

| 维度 | v2 | v3 |
|------|-----|-----|
| SBC训练质量 | 受QP干扰 | **不受影响 (95%+)** |
| 运行时安全 | ✓ | ✓ |
| SBC→CBF协同 | ✗ | **✓ B(s)调制p** |
| 训练速度 | +25% | **同baseline** |
| 模块化 | 耦合 | **解耦 (可独立替换)** |

---

## 3. 网络结构与数据流

### 3.1 完整系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SafePVC + QP Shield (v3)                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  感知模块 (frozen, 来自 artical-F122 原始代码)                 │   │
│  │                                                              │   │
│  │  state s=(d,v)                                               │   │
│  │     │                                                        │   │
│  │     ├──→ gen_net(z, d) ──→ img (32×32 灰度图)               │   │
│  │     │         │                                              │   │
│  │     │         └──→ state_net(img) ──→ state_est              │   │
│  │     │                              │                         │   │
│  │     │    v ────────────────────────┼──→ [state_est, v]       │   │
│  │     │                              │        │                │   │
│  └─────┼──────────────────────────────┼────────┼────────────────┘   │
│        │                              │        │                    │
│  ┌─────┼──────────────────────────────┼────────┼────────────────┐   │
│  │     │      控制器模块              │        │                │   │
│  │     │                              ▼        │                │   │
│  │     │   controller_net([state_est, v])       │                │   │
│  │     │     = mlp_extractor + action_net       │                │   │
│  │     │     (来自 PPO 预训练, frozen)          │                │   │
│  │     │         │                              │                │   │
│  │     │         ▼                              │                │   │
│  │     │      u_ref (参考加速度)                │                │   │
│  │     │         │                              │                │   │
│  └─────┼─────────┼──────────────────────────────┼────────────────┘   │
│        │         │                              │                    │
│  ┌─────┼─────────┼──────────────────────────────┼────────────────┐   │
│  │     │   QP 安全盾 (★ v3 新增, 仅推理时使用)  │                │   │
│  │     │         │                              │                │   │
│  │     │    u_ref│                              │                │   │
│  │     │         ▼                              │                │   │
│  │     │  ┌──────────────────────────┐          │                │   │
│  │     │  │  SBC B(s) → p = f(B(s)) │ ← trained SBC             │   │
│  │     │  │  CBF约束: -t·u ≤ -v+p·b │                          │   │
│  │     │  │  QP: min ½(u-u_ref)²    │                          │   │
│  │     │  │  → u_safe               │                          │   │
│  │     │  └──────────────────────────┘          │                │   │
│  │     │         │                              │                │   │
│  │     │         ▼                              │                │   │
│  │     │      u_safe → 动力学 → s_next          │                │   │
│  └─────┼────────────────────────────────────────┼────────────────┘   │
│        │                                        │                    │
│  ┌─────┼────────────────────────────────────────┼────────────────┐   │
│  │  SBC 模块 (训练时使用, 推理时用于调制 p)     │                │   │
│  │     │                                        │                │   │
│  │     s ──→ MLP[2,16,8,1](tanh+softplus) ──→ B(s) ≥ 0           │   │
│  │                                                              │   │
│  │  SBC 训练: martingale_loss + lip_loss + region_loss          │   │
│  │  ★ 训练时 controller 使用 DIRECT 模式 (无 QP)                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 训练阶段数据流

```
每个训练迭代:

1. 采样状态 s_t ∈ S (离散网格100×100)
2. 加扰动: s_t_pert = s_t + δ · random([-0.5, 0.5])
3. 感知: o_t = gen_net(z₀, s_t_pert[:,0])  [frozen]
4. 状态估计: ŝ_t = state_net(o_t)  [frozen]
5. ★ 控制器: u_t = controller_net([ŝ_t, v_t])  [DIRECT模式, 无QP!]
6. 动力学: s_{t+1} = f(s_t, u_t) + noise
7. SBC: B(s_t), B(s_{t+1})
8. SBC损失: martingale + lip + region
9. 控制器损失: martingale + lip + MSE(teacher)

★ 关键: 训练全程无QP参与, SBC看到的是"干净"的闭环系统
```

### 3.3 推理阶段数据流

```
每个时间步:

1. 感知: o_t = gen_net(z₀, s_t) → state_net → ŝ_t
2. 控制器: u_ref = controller_net([ŝ_t, v_t])  [参考控制]
3. ★ SBC调制: p = f(B(s_t))  [CBF参数]
4. ★ QP安全盾:
   min  ½(u - u_ref)²
   s.t. CBF: -t_gap·u ≤ -v + p·(d - v·t_gap)
        -3 ≤ u ≤ 3
   → u_safe
5. 执行: u_safe → 动力学 → s_{t+1}
6. 记录: u_safe, u_ref, p, B(s), intervened, margin
```

---

## 4. SBC调制CBF参数机制

### 4.1 调制公式

$$p(s) = p_{min} + (p_{max} - p_{min}) \cdot \sigma\left(\frac{B(s) - B_{thresh}}{T}\right)$$

| 参数 | 值 | 含义 |
|------|-----|------|
| $p_{min}$ | 0.1 | 非常安全时的CBF参数 (松弛) |
| $p_{max}$ | 4.0 | 接近不安全时的CBF参数 (收紧) |
| $B_{thresh}$ | $1/(1-p_{target})$ | SBC安全阈值 |
| $p_{target}$ | 0.95 | 目标安全概率 |
| $T$ | 0.5 | 温度参数 (过渡平滑度) |

### 4.2 物理意义

```
B(s) << B_thresh (非常安全):
  → sigmoid ≈ 0 → p ≈ 0.1
  → CBF约束极松 → 控制器几乎完全自由

B(s) ≈ B_thresh (临界安全):
  → sigmoid ≈ 0.5 → p ≈ 2.05
  → CBF约束适中 → 适度限制控制器

B(s) >> B_thresh (危险):
  → sigmoid ≈ 1 → p ≈ 4.0
  → CBF约束极紧 → 强制推回安全区域
```

### 4.3 v3首轮实验结果暴露的问题

**首轮测试中 V3A/V3B 的 QP 盾没有激活 (干预率0%)**

根因: 测试时使用的SBC是随机初始化的(未训练)，$B(s) \approx 0.5$ (softplus输出)，远小于 $B_{thresh}=20.0$。

$$
\frac{0.5 - 20.0}{0.5} = -39 \implies \sigma(-39) \approx 0 \implies p \approx 0.1
$$

$p=0.1$时CBF约束极松($-1.5u \leq -v + 0.1b$) → QP总是能找到满足约束的解→从不干预

**解决方案** (待实施):
1. 使用训练好的SBC (从v2 baseline实验)
2. 或降低 $B_{thresh}$ 到合理值 (如 2.0)
3. 或使用不同的调制策略 (如基于barrier值而非SBC值)

---

## 5. 四种对比配置

| 代号 | QP训练 | QP推理 | p参数来源 | SBC形式化界 | 运行时安全 |
|------|--------|--------|----------|-----------|----------|
| **B (Baseline)** | ✗ | ✗ | N/A | **95.8%** | ✗ |
| **V2 (训练QP)** | ✓ | ✓ | NN学习 | 87.9% | ✓ |
| **V3A (推理QP固定p)** | ✗ | ✓ | 固定 p=2.0 | **95.8%** | ✓ |
| **V3B (推理QP SBC调制)** | ✗ | ✓ | SBC调制 p=f(B(s)) | **95.8%** | ✓ (更强) |

---

## 6. 代码架构

### 6.1 目录结构

```
paper-combination/
├── src/                              # ★ 共享实验代码
│   ├── models/
│   │   ├── cbf_constraints.py        # CBF约束数学实现
│   │   ├── qp_controller.py          # v2风格 QP控制器
│   │   └── sbc_modulated_qp.py       # v3 SBC调制QP盾
│   ├── training/
│   │   ├── qp_trainer.py             # QP增强训练器
│   │   └── qp_loop.py                # 主训练循环
│   ├── eval/
│   │   ├── compare_shields.py        # 4种配置对比评估
│   │   └── collect_qp_benefits.py    # QP收益数据收集
│   └── utils/
│       └── qp_solver.py              # QP求解器封装
├── 实验v2/                            # v2文档+结果
│   ├── docs/                         # 实验方案+结果报告
│   ├── configs/                      # 实验配置
│   └── results/                      # 实验数据
├── 实验v3/                            # v3文档+结果
│   ├── docs/                         # 实验设计+结果报告
│   ├── configs/                      # 实验配置
│   └── results/                      # 实验数据
└── artical-F122/                     # ★ 原始代码, 未修改
```

### 6.2 对原始代码的改动

**零修改!** `artical-F122/` 中所有文件保持不变。

新代码通过 import 使用原始模块:
```python
# src/ 中的代码导入原始模块
from Aebs.system.env import Aebs        # 环境定义
from Aebs.VT.utils import MLP           # SBC网络
from Combined_network.model import ...   # 端到端网络
from cGAN.taxi_models_and_data import ... # 生成器
```

---

## 7. 与原始代码的关系

### 7.1 原始 SafePVC (artical-F122) 做了什么

```
artical-F122/Aebs/VT/loop.py 主循环:
  1. 加载 gen_net (cGAN生成器, frozen)
  2. 加载 state_net (状态估计器, frozen)
  3. 加载 PPO controller (RL预训练策略)
  4. 构建 VCLS = gen_net ∘ state_net ∘ controller
  5. 循环:
     a. 训练 SBC B(s) (martingale loss)
     b. 验证 SBC 条件 (IBP on 100×100 grid)
     c. 计算概率下界 p
     d. 训练 controller (反例驱动)
```

### 7.2 v2 改了什么

```
v2 修改 (src/training/qp_trainer.py):
  controller_net 替换为 QPAebsController:
    原始: controller_net → u (直接输出)
    v2:   shared_backbone → q_head (参考控制)
                          → p_head (CBF参数)
                          → QP层 → u* (安全控制)
  
  新增损失项: L_CBF (CBF约束违反惩罚)
  新增超参数: λ_cbf (CBF损失权重)
```

### 7.3 v3 改了什么

```
v3 修改 (src/models/sbc_modulated_qp.py):
  ★ 训练阶段: 与Baseline完全相同 (无QP)
  ★ 推理阶段: 在controller之后插入QP安全盾
    原始: controller → u → 执行
    v3:   controller → u_ref → QP盾 → u_safe → 执行
                              ↑
                         SBC B(s) 调制 p
  
  QP盾独立于训练: 可以"插拔"到任何训练好的controller上
```
