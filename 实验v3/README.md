# 实验v3: 解耦双重安全 — QP推理时安全盾

> **核心思路**: QP只在推理时作为安全盾，SBC/Barrier值调制CBF参数，彻底解决v2中QP干扰SBC训练的问题

## 实验结果 (2026-07-20)

| 指标 | Baseline | v2 (训练QP) | **V3A (固定p=2.0)** ✨ | **V3D (Barrier调制)** |
|------|----------|------------|----------------------|---------------------|
| SBC概率界 | 96.58% | 87.9% | **96.58%** | **96.58%** |
| CBF安全率 | N/A | 99.7% | **✓ 100%** | **✓ 100%** |
| QP干预率 | N/A | 99.7% | **46.4%** | 49.7% |
| 自适应能力 | ✗ | NN学习 | ✗ (手动p) | **✓ (自动)** |

**🥇 最优: V3A固定p=2.0** — 100%安全, 最低干预率, 最简单
**🥈 自适应: V3D Barrier调制p_min=0.5** — 100%安全, 自动适应状态

## v2 vs v3

| 维度 | v2 (训练时QP) | v3 (推理时QP) |
|------|-------------|-------------|
| SBC形式化界 | 87.9% | **96.6% (+8.7pp)** |
| 运行时安全 | ✓ | ✓ |
| SBC→CBF协同 | ✗ | ✓ (Barrier调制) |
| 训练速度 | 慢25% | 同baseline |

## 五种配置

| 代号 | QP训练 | QP推理 | p参数来源 | 状态 |
|------|:---:|:---:|------|:---:|
| **B** | ✗ | ✗ | N/A | 基准 |
| **V2** | ✓ | ✓ | NN学习 | ❌ |
| **V3A** | ✗ | ✓ | 固定 p=2.0 | ✅ **最优** |
| **V3B** | ✗ | ✓ | SBC调制 | ❌ |
| **V3C** | ✗ | ✓ | 训练p-network | △ |
| **V3D** | ✗ | ✓ | Barrier调制 | ✅ **自适应** |

## 文档

| 文档 | 内容 | 适合 |
|------|------|------|
| **[📖 完整技术文档](docs/master_document.md)** | 从原理到实验的全貌 | ★ 推荐首读 |
| [SBC 深度解析](docs/sbc_deep_dive.md) | SafePVC 原理/训练/验证/数据流 | 理解 SBC |
| [BarrierNet 深度解析](docs/barriernet_deep_dive.md) | CBF-QP 原理/架构/数据流 | 理解 QP |
| [项目结构](docs/project_structure.md) | 代码文件/依赖/数据流 | 了解代码 |
| [QP 价值验证](docs/qp_shield_benefit_report.md) | 四项实验证明 QP 价值 | 理解 QP 收益 |
| [详细实验数据](docs/v3_improved_results.md) | 完整参数/场景/对比数据 | 查阅数据 |

## 运行

```bash
cd /root/paper-combination/artical-F122

# 训练改进SBC (96.58%概率界)
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/train_improved_sbc.py

# 完整评估
PYTHONPATH=/root/paper-combination:/root/paper-combination/artical-F122 \
python /root/paper-combination/src/eval/v3_full_evaluation.py
```

## 关键概念速查

| 概念 | 一句话解释 |
|------|----------|
| **SBC** | 函数 B(s) — 给状态打分, 通过IBP验证导出"96.58%概率安全" |
| **CBF** | 约束 — 保证每步控制 u 满足安全条件 |
| **QP** | 优化 — 在满足CBF约束下找最接近原始的控制 |
| **干预率** | QP修改了控制器输出的步骤占比 (越低越好) |
| **CBF安全率** | CBF约束被满足的步骤占比 (越高越好) |
