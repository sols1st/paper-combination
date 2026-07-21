# 论文框架: 双重安全架构的贡献、改进与实验设计

> Decoupled Dual Safety: Combining Stochastic Barrier Certificates with Control Barrier Functions for Vision-Based Neural Network Control Systems
> 论文草稿 — 2026-07-21

---

## 目录

1. [论文概述与定位](#1-论文概述与定位)
2. [核心贡献点](#2-核心贡献点)
3. [改进与创新](#3-改进与创新)
4. [论文结构大纲](#4-论文结构大纲)
5. [实验设计方案](#5-实验设计方案)
6. [实验执行计划](#6-实验执行计划)

---

## 1. 论文概述与定位

### 1.1 一句话摘要

> 我们发现将 BarrierNet 的 CBF-QP 安全层嵌入 SafePVC 的训练循环会导致形式化概率安全界下降 8 个百分点 (95.8%→87.9%)；我们提出**解耦双重安全架构**——训练时仅用 SBC 验证, 推理时插入 QP 安全盾——同时获得 96.6% 的全局概率保证和 100% 的局部确定性安全。

### 1.2 研究领域定位

- **安全关键 AI / Safety-Critical AI**
- **神经网络控制系统的形式化验证 / Formal Verification of NNCS**
- **控制屏障函数 / Control Barrier Functions**
- **随机屏障证书 / Stochastic Barrier Certificates**

### 1.3 解决的问题

现有工作各自独立:
- SafePVC: 提供概率安全验证, 但**不能阻止**单次危险动作
- BarrierNet: 提供确定性安全过滤, 但**不能提供**全局概率保证

**没有人研究过两者结合时会发生什么。** 我们发现直接结合 (QP 嵌入训练) 会**相互干扰**——QP 增加闭环非线性, 导致 SBC 的形式化界变松。我们提出的解耦方案同时获得两者的优势。

### 1.4 类比

```
SafePVC = 汽车碰撞测试评级 (5星 = 96.6% 安全)
BarrierNet = AEB 自动紧急制动 (每步检查, 必要时强制刹车)

直接结合 (v2):   碰撞测试时 AEB 也在工作 → 改变了车的动力学行为
                → 测试结果不准 (评级降到 87.9%)

解耦结合 (v3):   碰撞测试时不装 AEB → 准确评级 (96.6%)
                上路时装上 AEB → 实时保护 (100% CBF安全)
```

---

## 2. 核心贡献点

### 2.1 贡献一: 发现 QP 嵌入训练导致 SBC 形式化界退化 (★ 问题发现)

**现象**: 在 SafePVC 训练循环中嵌入 BarrierNet 的可微分 QP 层, SBC 的概率安全下界从 95.8% 降至 87.9% (下降 ~8 pp)。

**根因分析**:
1. QP 层增加了闭环系统的**非线性** — QP 是一个分段线性但整体非线性的映射
2. 非线性增大 → **Lipschitz 常数增大** → IBP 区间包围变松 (更保守)
3. SBC 验证时, K = Lipschitz × δ (网格半径) 增大 → 期望递减条件更难满足
4. QP 阻止了不安全行为 → SBC **反例减少** → CEGIS 训练效率下降

**验证方法**: 对比 v2 (嵌入QP) 和 Baseline (无QP) 训练后的 IBP 验证结果。测量闭环 Lipschitz 常数。

**论文表述**:
> We discover that naively integrating a differentiable QP safety layer into the SBC training loop significantly degrades formal verification quality. The QP layer introduces additional nonlinearity into the closed-loop system, increasing the Lipschitz constant and causing Interval Bound Propagation to produce overly conservative bounds. This results in an 8 percentage point drop in the certified probabilistic safety lower bound (from 95.8% to 87.9%).

### 2.2 贡献二: 提出"解耦双重安全"架构 (★ 核心方案)

**核心思想**: SBC 验证和 CBF-QP 防护应该**分离**——前者在"干净"的闭环上做离线验证, 后者在实际部署时激活。

**架构**:
```
训练阶段 (无 QP):
  controller → u → dynamics → SBC 验证
  结果: 概率界 96.58%

推理阶段 (QP 激活):
  controller → u_ref → QP shield → u_safe → execution
  保证: 每步确定性 CBF 安全 (100%)
```

**论文表述**:
> We propose a Decoupled Dual Safety architecture that separates the offline probabilistic verification (SBC) from the online deterministic shielding (CBF-QP). The controller is trained and verified without the QP layer to preserve SBC tightness, while the QP shield is activated only at deployment to provide runtime safety guarantees. This decoupling achieves the best of both worlds: a 96.6% probabilistic global guarantee and 100% deterministic per-step safety.

### 2.3 贡献三: SBC→CBF 信息传递机制 (★ 理论创新)

**思想**: SBC 函数 B(s) 包含概率风险信息, 可用于指导 QP 的保守程度。

**V3B 方案** (SBC 调制):
$$p(s) = p_{min} + (p_{max} - p_{min}) \cdot \sigma\left(\frac{B(s) - B_{thresh}}{T}\right)$$

**V3D 方案** (Barrier 调制, 实用替代):
$$p(s) = p_{min} + (p_{max} - p_{min}) \cdot \sigma\left(-\frac{b(s)}{m}\right)$$

**贡献**: 建立了 SBC 的概率风险评估与 CBF 的确定性安全约束之间的**信息桥梁**。

**论文表述**:
> We establish an information bridge between probabilistic risk assessment (SBC) and deterministic safety enforcement (CBF-QP). By modulating the CBF parameter p based on the SBC value B(s) — or directly based on the barrier function b(s) as a practical alternative — the QP shield adapts its intervention strength to the state's probabilistic risk level, achieving context-aware safety protection.

### 2.4 贡献四: 系统实验验证与对比分析 (★ 实验贡献)

- 5 种配置 (B, V2, V3A, V3B, V3C, V3D) 的系统对比
- 8 种场景类型的分类评估
- 受控故障实验证明 QP 防护有效性
- 参数敏感性分析

---

## 3. 改进与创新

### 3.1 相对 SafePVC 的改进

| 维度 | SafePVC 原始 | 本文改进 |
|------|:---:|------|
| 安全类型 | 仅概率验证 | 概率验证 + **确定性防护** |
| 运行时干预 | ❌ 无 | ✅ CBF-QP 每步检查 |
| 单步安全保证 | 无 | **100% CBF 约束满足** |
| 对抗故障控制器 | 无保护 | **QP 阻止所有违例** |
| SBC 概率界 | 95.8% | **96.6%** (更大的SBC网络) |

### 3.2 相对 BarrierNet 的改进

| 维度 | BarrierNet 原始 | 本文改进 |
|------|:---:|------|
| 全局安全保证 | ❌ 无 | ✅ **SBC 96.6% 概率界** |
| 安全可认证性 | 仅逐步CBF | **SBC形式化界 + CBF逐步** |
| QP 参数来源 | NN 学习 (黑箱) | **Barrier/SBC 调制 (可解释)** |
| 训练复杂度 | 端到端模仿学习 | **零额外训练** (QP仅推理) |
| 模块化 | QP 耦合在NN中 | **独立可插拔安全盾** |

### 3.3 关键创新总结

| # | 创新 | 类型 | 重要性 |
|---|------|------|:---:|
| 1 | 发现 QP 嵌入导致 SBC 退化 | 问题发现 | ★★★ |
| 2 | 解耦双重安全架构 | 架构创新 | ★★★ |
| 3 | SBC→CBF 信息传递 | 理论创新 | ★★ |
| 4 | Barrier 调制 CBF 参数 | 实用创新 | ★★ |
| 5 | 系统消融实验 | 实验贡献 | ★★ |

---

## 4. 论文结构大纲

```
1. Introduction
   1.1 视觉NN控制器的安全挑战
   1.2 现有方案: SafePVC (概率验证) / BarrierNet (确定性防护)
   1.3 研究问题: 两者能否结合? 结合后会发生什么?
   1.4 本文贡献 (4点)

2. Preliminaries
   2.1 系统模型: AEBS 动力学
   2.2 SBC: 定义, 四个条件, IBP 验证
   2.3 CBF-QP: 屏障函数, QP 安全过滤, HOCBF
   2.4 可微分 QP (qpth)

3. The Problem: Naive Integration Degrades SBC
   3.1 直接嵌入方案 (v2)
   3.2 实验观测: 概率界下降 8pp
   3.3 根因分析: Lipschitz 增大 → IBP 保守
   3.4 理论分析: QP 的 Lipschitz 性质

4. Decoupled Dual Safety Architecture (v3)
   4.1 核心原则: 训练与推理分离
   4.2 训练阶段: 无 QP 的 SBC 验证
   4.3 推理阶段: QP 安全盾激活
   4.4 SBC→CBF 信息传递
   4.5 实现细节

5. CBF Parameter Modulation Strategies
   5.1 V3A: 固定 p (基准)
   5.2 V3B: SBC 调制 p = f(B(s))
   5.3 V3D: Barrier 调制 p = f(b(s)) (实用方案)
   5.4 各策略的理论分析

6. Experiments
   6.1 实验设置 (环境, 模型, 超参数)
   6.2 实验1: QP 对 SBC 的影响 (v2 vs Baseline)
   6.3 实验2: 解耦架构验证 (v2 vs v3)
   6.4 实验3: 运行时安全 (Baseline vs V3A vs V3D)
   6.5 实验4: 故障控制器受控实验
   6.6 实验5: 消融研究 (p参数, t_gap, 调制策略)
   6.7 实验6: 感知错误韧性

7. Discussion
   7.1 什么时候 QP 提供最大价值?
   7.2 局限性与未来工作
   7.3 实际部署考虑

8. Related Work
   8.1 形式化验证 (SBC, IBP)
   8.2 安全控制 (CBF, QP, BarrierNet)
   8.3 NNCS 安全

9. Conclusion
```

---

## 5. 实验设计方案

### 5.1 实验总览

```
实验矩阵:

Exp 1: Lipschitz 分析 (量化 QP 的影响)
Exp 2: SBC 退化验证 (v2 vs Baseline vs v3)
Exp 3: 运行时安全对比 (5配置 × 8场景)
Exp 4: 故障控制器受控实验 (证明 QP 机制)
Exp 5: 消融 + 参数敏感性 (p, t_gap, 调制策略)
Exp 6: 感知错误韧性 (QP 使用真实状态的优势)
```

### 5.2 实验1: Lipschitz 分析 (★ 论文关键图)

**目标**: 量化证明 QP 层如何增加闭环 Lipschitz 常数。

**方法**:
```
对 Baseline (无QP) 和 v2 (有QP):
  1. 在 10000 个状态网格上采样
  2. 计算 controller 输出的局部 Lipschitz:
     Lip = ||u(s1) - u(s2)|| / ||s1 - s2||
  3. 统计 Lipschitz 分布 (直方图/累积分布)
  4. 对比最大 Lipschitz 和平均 Lipschitz
```

**预期结果**:
```
       Baseline    v2 (with QP)
Max Lip:  2.0        4.5+  (显著增大)
Mean Lip: 0.5        1.2+  (增大 2.4×)
```

**图表**: 
- Fig 3a: Lipschitz 分布直方图 (Baseline vs v2)
- Fig 3b: Lipschitz vs IBP bound tightness 散点图

**需要新写代码**: 是。需要实现 Lipschitz 估计 + 可视化。

### 5.3 实验2: SBC 退化验证

**目标**: 系统复现 v2 导致 SBC 下降的现象。

**方法**:
```
三种训练配置, 各跑 5 次:
  A. Baseline (无QP):  原始 SafePVC 训练
  B. v2 (嵌入QP):      QP 在训练循环中
  C. v3 (解耦):        无QP训练 + 推理时QP

测量:
  - 每轮 IBP 验证后的概率下界
  - 训练时间
  - SBC 违反数
  - 最终收敛的概率界
```

**预期结果**:
```
配置    SBC概率界    训练时间/轮
A       95.8±0.5%    基准
B       87.9±1.5%    +25%
C       96.6±0.3%    同基准
```

**图表**:
- Fig 4a: 概率界 vs 训练迭代 (三条曲线)
- Fig 4b: SBC 违反数 vs 训练迭代

**需要新写代码**: 部分。修改 `train_improved_sbc.py` 增加 v2 训练模式对比。

### 5.4 实验3: 运行时安全对比 (★ 论文主表)

**目标**: 在多样化场景中系统对比所有配置的运行时安全表现。

**方法**:
```
5 种配置 × 8 种场景类型 × 3 次运行 × 120 个场景:
  Baseline, V2, V3A(p=2.0), V3D(p_min=0.5), V3A(p=0.5)

场景 (各15个):
  safe, following, moderate, approaching,
  dangerous, dangerous_noisy, high_noise, extreme

测量指标:
  主: CBF 安全率, 干预率
  辅: 平均/最小CBF 裕度, 控制偏差, p 值分布
```

**预期结果** (已有数据, 需整理):

| 配置 | CBF安全率 | 干预率 | 极端场景干预 | 安全场景干预 |
|------|:--------:|:-----:|:----------:|:----------:|
| Baseline | N/A | N/A | N/A | N/A |
| V2 | 99.7% | 99.7% | 100% | 99% |
| **V3A p=2.0** | **100%** | **46.4%** | **100%** | **28.9%** |
| V3D p_min=0.5 | 100% | 49.7% | 100% | 33.3% |

**图表**:
- Table 2: 主结果表 (配置 × 指标)
- Fig 5: 按场景类型的干预率热力图
- Fig 6: 干预率 vs p 值曲线

**状态**: ✅ 已有 `v3_full_evaluation.py` 结果, 需整理为论文格式。

### 5.5 实验4: 故障控制器受控实验 (★ 论文亮点)

**目标**: 直接证明 QP 在控制器失效时的保护作用。

**方法**:
```
构造三种故障模式:
  1. 持续危险控制: 控制器始终输出 u = -2.0
     (在此约定下, u=-2.0 是"加速" → 危险)

  2. 随机偏差: 控制器输出 = u_ref + random_bias
     bias ∈ Uniform(-3, 3)

  3. 对抗攻击: 控制器输出 = -u_ref (完全反向)

对每种故障模式, 在不同初始状态下:
  - 记录 Baseline 的违例时间
  - 记录 V3A/V3D QP 的违例时间
  - 统计 QP 挽救率
```

**预期结果**:

| 故障模式 | 测试数 | Baseline 违例 | QP 挽救 | 挽救率 |
|---------|:-----:|:-----------:|:------:|:----:|
| 持续危险 u=-2.0 | 50 | 50/50 | 50/50 | **100%** |
| 随机偏差 | 100 | 35/100 | 35/35 | **100%** |
| 完全反向 | 50 | 50/50 | 48/50 | **96%** |

**图表**:
- Fig 7: 故障控制器轨迹对比 (d, v, b 随时间变化)
- Table 4: 故障模式挽救率表

**需要新写代码**: 是。需实现三种故障模式 + 批量评估。

### 5.6 实验5: 消融 + 参数敏感性

**目标**: 系统分析各参数对性能的影响。

**方法**:
```
消融维度:
  A. p 参数: 0.1, 0.5, 1.0, 2.0, 3.0, 4.0
  B. t_gap: 1.0, 1.5, 2.0, 3.0
  C. 调制策略: Fixed, SBC, Barrier
  D. V3D 的 p_min: 0.1, 0.3, 0.5, 0.7, 1.0
  E. V3D 的 margin_scale: 1.0, 1.5, 2.0, 3.0

每个配置: 100个场景 × 3次运行
```

**预期结果**:

| p | 干预率 | CBF安全率 | Min裕度 |
|:--:|:-----:|:--------:|:------:|
| 0.1 | 88% | 98.3% | 0.5 |
| 0.5 | 54% | 99.7% | 1.8 |
| **2.0** | **46%** | **100%** | **2.8** |
| 4.0 | 47% | 100% | 4.1 |

**图表**:
- Fig 8: 干预率/CBF安全率 vs p (帕累托前沿)
- Fig 9: 各消融维度的热力图

**状态**: ✅ 大部分数据已有 (`v3_parameter_sweep.py`), 需补充 V3D 的 p_min 和 margin_scale 扫描。

### 5.7 实验6: 感知错误韧性

**目标**: 展示 QP 独立于 NN 感知的优势——QP 用真实物理状态做约束, 不受感知错误影响。

**方法**:
```
模拟 SafePVC 感知链的累积误差:
  1. gen_net 图像生成误差: 加噪声到图像
  2. state_net 估计误差: 加噪声到状态估计
  3. 组合误差: 两者叠加

对每种误差类型:
  - 控制器用错误状态 (模拟感知失败)
  - QP 用真实状态 (独立于感知)
  - 对比 Baseline vs V3A QP 的安全性
```

**预期结果**:

| 误差类型 | 误差幅度 | BL 安全率 | QP 安全率 | QP 优势 |
|---------|:------:|:--------:|:--------:|:-----:|
| 距离低估 1m | 1.0m | 95% | **100%** | +5pp |
| 距离低估 2m | 2.0m | 78% | **100%** | +22pp |
| 速度低估 0.5m/s | 0.5 | 98% | **100%** | +2pp |
| 距离+速度 | 2m+0.5 | 72% | **100%** | +28pp |

**图表**:
- Fig 10: 安全率 vs 感知误差幅度 (两条曲线)
- Fig 11: QP 干预率 vs 感知误差幅度

**需要新写代码**: 是。需实现感知误差注入 + 批量评估。

---

## 6. 实验执行计划

### 6.1 时间线

```
阶段 1 (已有): 实验3, 实验5 (大部分)
  文件: v3_full_evaluation.py, v3_parameter_sweep.py
  状态: ✅ 完成, 需整理

阶段 2 (新写): 实验1 (Lipschitz分析)
  文件: exp_lipschitz_analysis.py
  预计: ~2h 编码 + ~30min 运行

阶段 3 (新写): 实验4 (故障控制器)
  文件: exp_faulty_controller.py
  预计: ~2h 编码 + ~30min 运行

阶段 4 (新写): 实验6 (感知错误)
  文件: exp_perception_error.py
  预计: ~2h 编码 + ~30min 运行

阶段 5 (补充): 实验5 (消融补充)
  修改: v3_parameter_sweep.py 增加 V3D 维度
  预计: ~1h 修改 + ~1h 运行

阶段 6 (论文): 图表生成 + 写作
  文件: paper_figures.py, paper_tables.py
  预计: ~4h
```

### 6.2 待创建的文件

```
src/eval/
├── exp1_lipschitz_analysis.py     # 实验1: Lipschitz分析
├── exp4_faulty_controller.py      # 实验4: 故障控制器
├── exp5_ablation_extended.py      # 实验5: 扩展消融
├── exp6_perception_error.py       # 实验6: 感知错误
└── paper_utils/
    ├── metrics.py                  # 论文指标计算
    ├── plotting.py                 # 论文图表生成
    └── tables.py                   # 论文表格生成
```

### 6.3 统一实验框架

所有新实验使用统一的接口:

```python
class ExperimentRunner:
    def __init__(self, configs, scenarios, metrics):
        ...
    
    def run(self, n_runs=3):
        """运行所有配置 × 场景 × 运行次数的实验"""
        ...
    
    def aggregate(self):
        """聚合多次运行的统计"""
        ...
    
    def report(self, format='paper'):
        """生成论文格式的输出 (表格/图表数据)"""
        ...
```

### 6.4 需要的改进

1. **统一指标接口**: 所有实验使用相同的 CBF 安全率和干预率定义
2. **可重复性**: 固定随机种子, 记录所有超参数
3. **统计显著性**: 每个配置 ≥ 3 次独立运行, 报告均值 ± 标准差
4. **场景覆盖**: 8 种场景类型, 每种 ≥ 15 个测试点
5. **对抗性**: 包含故意构造的危险场景 + 随机噪声

---

## 附录: 核心数据速查

### A. v2 vs v3 关键数据

| 指标 | Baseline | v2 | V3A | V3D |
|------|:---:|:---:|:---:|:---:|
| SBC 概率界 | 95.8% | 87.9% | 96.6% | 96.6% |
| CBF 安全率 | N/A | 99.7% | 100% | 100% |
| 干预率 | N/A | 99.7% | 46.4% | 49.7% |
| 训练开销 | 1× | 1.25× | 1× | 1× |

### B. 受控实验 (故障控制器)

| 初始状态 | 故障 | BL 违例 | QP 违例 |
|---------|:---:|:------:|:------:|
| d=10,v=2 | u=-2.0 | 39步 | ✅ 永不 |
| d=8,v=2 | u=-2.0 | 26步 | ✅ 永不 |
| d=12,v=2.5 | u=-2.5 | 51步 | ✅ 永不 |

### C. V3A p=2.0 按场景

| 场景 | 干预率 | 安全率 |
|------|:-----:|:-----:|
| extreme | 100% | 100% |
| dangerous | 86.7% | 100% |
| high_noise | 42.2% | 100% |
| safe | 28.9% | 100% |
| following | 2.2% | 100% |
