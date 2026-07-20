# 实验v3: 解耦双重安全 — QP推理时安全盾

> **核心思路**: QP只在推理时作为安全盾，SBC/Barrier值调制CBF参数，彻底解决v2中QP干扰SBC训练的问题

## 实验结果 (2026-07-20)

| 指标 | Baseline | v2 (训练QP) | **V3A (固定p=2.0)** ✨ | **V3D (Barrier调制)** |
|------|----------|------------|----------------------|---------------------|
| SBC概率界 | 96.58% | 87.9% | **96.58%** | **96.58%** |
| 运行时CBF安全 | ✗ | ✓ | **✓ 100%** | **✓ 100%** |
| QP干预率 | N/A | 99.7% | **46.4%** | 49.7% |
| 控制质量损失 | — | — | +3.3% | -3.0% |
| 自适应能力 | ✗ | NN学习 | ✗ (手动p) | **✓ (自动)** |

**🥇 最优: V3A固定p=2.0** — 100%安全, 最低干预率, 最简单
**🥈 自适应: V3D Barrier调制p_min=0.5** — 100%安全, 自动适应状态

## v2 vs v3 对比

| 维度 | v2 (训练时QP) | v3 (推理时QP) |
|------|-------------|-------------|
| QP在训练中 | ✓ (干扰SBC) | ✗ (不影响SBC) |
| SBC形式化界 | 87.9% | **96.6% (+8.7pp)** |
| 运行时安全 | ✓ | ✓ |
| SBC→CBF协同 | ✗ | ✓ (Barrier调制) |
| 训练速度 | 慢25% | 同baseline |

> **干预率 (Intervention Rate)**: QP安全盾修改控制器输出的时间步比例。PPO控制器先输出参考加速度 u_ref，QP检查其是否满足CBF安全约束；若不满足则求解QP找到最近的安全控制 u_safe。|u_safe - u_ref| > 0.01 即计为一次"干预"。越低说明控制器本身越安全/QP越不打扰。

## 五种对比配置

| 代号 | QP训练 | QP推理 | p参数来源 | 状态 |
|------|--------|--------|----------|------|
| **B** | ✗ | ✗ | N/A | ✅ 基准 |
| **V2** | ✓ | ✓ | NN学习 | ❌ SBC降8% |
| **V3A** | ✗ | ✓ | 固定 p=2.0 | ✅ **最优** |
| **V3B** | ✗ | ✓ | SBC调制 | ❌ SBC不校准 |
| **V3C** | ✗ | ✓ | 训练p-network | △ 待改进 |
| **V3D** | ✗ | ✓ | Barrier调制 | ✅ **自适应** |

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

## 文档

- [完整实验结果 (最新)](docs/v3_improved_results.md) — ★ 推荐阅读
- [首轮实验结果](docs/complete_results.md) — v2+v3对比
- [参数扫描结果](docs/sweep_results.md) — 26配置×2运行
- [实验设计文档](docs/experiment_design.md) — 架构细节
- [实验计划](docs/experimental_plan.md) — 原始计划
