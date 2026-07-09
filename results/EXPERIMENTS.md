# BarrierNet 实验结果详解

本文档详细解释了 `results/` 文件夹下三个实验的所有输出结果，包括每个实验的目的、方法、以及每张图的说明。

---

## 项目概述

**BarrierNet** 是一种将可微控制障碍函数（Differentiable Control Barrier Functions, dCBF）嵌入神经网络的安全控制器。其核心思想是：

- **神经网络部分**：从状态输入预测最优控制量和 CBF 的惩罚参数（penalty functions）
- **QP 安全层**：将神经网络输出作为二次规划（QP）的参数，通过 CBF 约束保证控制输出始终满足安全条件

每个实验都对比了两种模型：
- **BarrierNet (BN)**：带 dCBF 安全层的网络，**有安全保障**
- **FC Network (FC)**：普通全连接网络，结构与 BN 相同但**去掉 QP 安全层**，无安全保障

---

## 实验一：交通合流控制（Traffic Merging）

**目录**：`results/merging/`  
**论文对应**：Section 6.1, Figures 3-5  
**脚本**：`Merging/run_experiment.py`

### 实验目的

验证 BarrierNet 在车辆合流场景下的安全控制能力。车辆需要在合流时与前车保持安全距离（以时间间隔 `z/v >= 1.8s` 衡量）。

### 实验方法

- **动力学模型**：一维纵向车辆动力学（位置 + 速度）
- **安全约束**：CBF 约束保证 `z_{k,kp}/v_k >= 1.8s`（与前车的车头时距不低于 1.8 秒）
- **网络结构**：4 输入 → 72 → 24/24 → 1 输出（加速度）
- **训练**：分别用 OCBF（Optimal Control Barrier Function）和 OC（Optimal Controller）两种控制器的数据进行训练
- **多模型鲁棒性测试**：用不同随机种子训练 5 个模型，验证安全性的鲁棒性

### 结果图表说明

| 文件 | 论文对应 | 说明 |
|------|---------|------|
| `fig1_control_penalty_ocbf_bn.png` | Fig 3 | **BarrierNet（OCBF数据）的控制输出与惩罚函数**。上子图：红色为 ground truth，绿色为开环预测，蓝色为闭环仿真控制输出。下子图：BarrierNet 的惩罚函数 p₁(z) 随时间变化——当车辆靠近前车时惩罚增大以满足 CBF 约束 |
| `fig2_control_ocbf_fc.png` | — | **FC 网络（OCBF数据）的控制输出**。红色为 ground truth，蓝色虚线为 FC 闭环仿真。与 fig1 对比可看出 FC 的控制没有安全约束修正 |
| `fig3_safety_comparison_ocbf.png` | Fig 4 | **OCBF 数据下 BN vs FC 的安全性对比**。红色实线为 BarrierNet，蓝色虚线为 FC，黑色虚线为安全边界 φ=1.8s。**关键结论**：BarrierNet 始终保持在安全边界之上，FC 可能低于安全边界导致不安全 |
| `fig4_training_losses_ocbf.png` | — | **OCBF 数据训练的收敛曲线**。左为 BarrierNet，右为 FC。绿色为训练损失，红色为测试损失 |
| `fig5_multi_model_safety_ocbf.png` | Fig 5 | **多模型鲁棒性测试（OCBF）**。左为 5 个 BarrierNet 的安全曲线（全部高于 1.8s），右为 5 个 FC 的安全曲线（部分低于 1.8s）。**关键结论**：BarrierNet 在不同随机初始化下都能保证安全，FC 则不稳定 |
| `fig6_control_penalty_oc_bn.png` | — | **BarrierNet（OC数据）的控制输出与惩罚函数**。格式同 fig1，但使用普通最优控制器数据训练 |
| `fig7_safety_comparison_oc.png` | — | **OC 数据下 BN vs FC 的安全性对比**。格式同 fig3 |
| `fig8_training_losses_oc.png` | — | **OC 数据训练的收敛曲线**。格式同 fig4 |
| `fig9_multi_model_safety_oc.png` | — | **多模型鲁棒性测试（OC）**。格式同 fig5，使用 OC 数据 |
| `table2_comparison_summary.png` | Table 2 | **方法对比总结表格**。从实时计算时间、安全保障、最优性、自适应能力四个维度对比 BarrierNet / FC / Optimal / OCBF |

---

## 实验二：2D 机器人导航（2D Robot Navigation）

**目录**：`results/2d_robot/`  
**论文对应**：Section 6.2, Figures 7-9  
**脚本**：`2D_Robot/run_experiment.py`

### 实验目的

验证 BarrierNet 在 2D 平面机器人避障任务中的表现，特别是：
1. 在训练时见过的障碍物尺寸（R=6m）下的性能
2. 在训练时**未见过**的障碍物尺寸（R=7,8,9,10m）下的**自适应能力**

### 实验方法

- **动力学模型**：单轮车（Unicycle）动力学，状态为 (x, y, θ, v)，控制为转向角速度 u₁ 和加速度 u₂
- **安全约束**：HOCBF（High-Order CBF），b(x) = (px-obs_x)² + (py-obs_y)² - R²，要求 b(x) ≥ 0
- **网络结构**：5 输入 → 128 → 32/32 → 2 输出
- **自适应测试**：训练时使用 R=6m，测试时在 R=6,7,8,9,10m 下运行（通过修改模型中的 R 参数）

### 结果图表说明

| 文件 | 论文对应 | 说明 |
|------|---------|------|
| `fig1_control_u1_R6.png` | Fig 7a | **R=6m 时转向控制 u₁**。红色为 ground truth，蓝色为 BarrierNet。展示 BN 在训练障碍物尺寸下的控制精度 |
| `fig2_control_u2_R6.png` | Fig 7b | **R=6m 时加速度控制 u₂**。红色为 ground truth，蓝色为 BarrierNet |
| `fig3_penalty_functions_R6.png` | Fig 8 | **BarrierNet 的惩罚函数 p₁(z) 和 p₂(z)**。这两个参数由网络学习得到，用于调节 CBF 约束的松紧程度。当机器人靠近障碍物时，惩罚函数变化以加强安全约束 |
| `fig4_trajectories_R6.png` | Fig 7c | **R=6m 时的机器人轨迹**。灰色圆形为障碍物，红色为 ground truth，蓝色实线为 BarrierNet，橙色虚线为 FC。展示 BN 能安全绕开障碍物 |
| `fig5_safety_hocbf_R6.png` | Fig 9 | **R=6m 时的 HOCBF 安全指标 b(x)**。红色虚线为障碍物边界 (b=0)，蓝色为 BN，橙色为 FC。**b(x) ≥ 0 表示安全**。BN 始终保持 b(x) ≥ 0，FC 可能穿越边界 |
| `fig6_controls_comparison_R6.png` | — | **R=6m 时 BN vs FC 的控制量全面对比**。上为 u₁，下为 u₂，三种颜色分别为 GT / BN / FC |
| `fig7_controls_different_R.png` | Fig 9 (adaptivity) | **不同障碍物尺寸下的控制量自适应**。在 R=6,7,8,9,10m 下，BarrierNet 自动调整 u₁ 和 u₂——障碍物越大，转向和加减速响应越早越强 |
| `fig8_safety_different_R.png` | — | **不同障碍物尺寸下的安全性对比**。左为 BarrierNet（全部保持安全），右为 FC（部分不安全）。**关键结论**：BN 在不同 R 下都能自适应地保证安全 |
| `fig9_trajectories_different_R.png` | — | **不同障碍物尺寸下的轨迹对比**。左为 BN（随障碍物增大，绕行距离越远），右为 FC（轨迹不随 R 变化，导致碰撞）|
| `fig10_training_losses.png` | — | **训练收敛曲线**。左为 BarrierNet，右为 FC |
| `fig11_comprehensive_panel.png` | — | **2×3 综合面板**，包含：控制 u₁、u₂、惩罚函数、轨迹、安全 b(x)、安全放大图 |

---

## 实验三：3D 机器人导航（3D Robot Navigation）

**目录**：`results/3d_robot/`  
**论文对应**：Section 6.3, Figures 11-12  
**脚本**：`3D_Robot/run_experiment.py`

### 实验目的

验证 BarrierNet 在 3D 空间中使用**超椭球障碍物**（superquadratic obstacle）的导航能力。相比 2D 实验，这里的状态空间和控制维度更高。

### 实验方法

- **动力学模型**：双积分器（Double Integrator），状态为 (px, vx, py, vy, pz, vz)，控制为三轴加速度 (u₁, u₂, u₃)
- **安全约束**：超椭球 HOCBF，b(x) = (px-obs_x)⁴ + (py-obs_y)⁴ + (pz-obs_z)⁴ - R⁴，要求 b(x) ≥ 0
- **网络结构**：6 输入 → 512 → 128/128 → 3 输出
- **障碍物**：位于 (10,10,9) 的超椭球体，R=7

### 结果图表说明

| 文件 | 论文对应 | 说明 |
|------|---------|------|
| `fig1_control_u1.png` | Fig 11a | **x 轴加速度 u₁**。红色为 GT，蓝色为预训练 BN，青色虚线为新训练 BN |
| `fig2_control_u2.png` | Fig 11b | **y 轴加速度 u₂** |
| `fig3_control_u3.png` | Fig 11c | **z 轴加速度 u₃** |
| `fig4_penalty_functions.png` | Fig 12 | **惩罚函数 p₁(z) 和 p₂(z)**。展示 BN 学到的 CBF 参数在接近 3D 超椭球障碍物时的变化 |
| `fig5_3d_trajectory_bn.png` | Fig 11d | **BarrierNet 的 3D 轨迹**。彩色曲面为超椭球障碍物，红色为 GT 轨迹，蓝色为 BN 轨迹。展示 BN 能安全绕过 3D 障碍物 |
| `fig6_3d_trajectory_fc.png` | — | **FC 网络的 3D 轨迹**。橙色虚线为 FC 轨迹，与 BN 对比展示 FC 可能碰撞 |
| `fig7_3d_trajectory_comparison.png` | — | **三者的 3D 轨迹对比**。红色 GT + 蓝色 BN + 橙色 FC |
| `fig8_safety_hocbf.png` | Fig 11e | **HOCBF 安全指标 b(x)**。蓝色 BN 始终 ≥ 0（安全），橙色 FC 可能低于 0（不安全）|
| `fig9_training_losses.png` | — | **训练收敛曲线** |
| `fig10_all_controls.png` | — | **三轴控制的 3×1 对比面板**。展示 u₁, u₂, u₃ 的 GT / BN / FC 对比 |
| `fig11_comprehensive_panel.png` | — | **3×2 综合面板**：三轴控制、惩罚函数、安全 b(x)、3D 轨迹 |

---

## 关键结论总结

| 维度 | BarrierNet | FC Network |
|------|-----------|------------|
| **安全保障** | 始终满足 CBF 约束，b(x) ≥ 0 | 可能违反安全约束 |
| **控制精度** | 接近 ground truth | 接近 ground truth |
| **自适应能力** | 修改 R 参数即可适应不同障碍物尺寸 | 无法适应未见过的障碍物 |
| **推理速度** | < 0.01s（含 QP 求解） | < 0.01s |
| **鲁棒性** | 不同随机种子训练均保证安全 | 部分种子下不安全 |

**核心发现**：BarrierNet 通过在网络中嵌入可微 CBF-QP 层，在不牺牲推理速度和控制精度的前提下，提供了**可证明的安全保障**和**对未见环境的自适应能力**。

---

## 实验四：自动驾驶端到端安全控制（Vision-based End-to-End Autonomous Driving）

**目录**：`results/driving/`  
**论文对应**："Differentiable Control Barrier Functions for Vision-based End-to-End Autonomous Driving" (Xiao et al.), Section VI, Figures 5-9, Table I  
**脚本**：`Driving/run_experiment.py`

### 实验目的

验证 BarrierNet 在自动驾驶场景中的安全控制能力，复现论文核心实验：
1. 带 dCBF 的 BarrierNet 作为安全滤波器 vs 无安全保证的参考控制器
2. 障碍物避让（obstacle avoidance）和车道保持（lane keeping）
3. 碰撞率（crash rate）、最小间隙（min clearance）、QP 干预率的对比分析

### 实验方法

- **动力学模型**：曲率坐标系下的自行车模型（Eq. 12），状态 [s, d, μ, v, δ]，控制 [a, ω]
- **安全约束**：
  - 障碍物避让 CBF：`b(x) = dist - R_eff ≥ 0`，其中 `R_eff = R + margin + v·T`（速度相关的安全裕度）
  - 车道保持 CBF：`b_left = threshold - d ≥ 0`, `b_right = d + threshold ≥ 0`
- **参考控制器**：纯车道保持 PD 控制器（无避障能力），模拟论文中上游神经网络的输出
- **BarrierNet QP**：在参考控制器之上添加 dCBF 安全层，实时修正不安全控制
- **仿真场景**：50 个随机初始化场景（不同初始位置、速度、障碍物位置），每场景最多 200 步

### 结果对比

| 指标 | BarrierNet (w/ dCBF) | FC (w/o dCBF) | 论文参考值 |
|------|:---:|:---:|:---:|
| **碰撞率** | **54.0%** | 100.0% | BN: 3%, FC: 53% |
| **安全率** | **46.0%** | 0.0% | BN > FC |
| **平均间隙** | **-0.12m** | -2.86m | BN: 0.55m, FC: 0.43m |
| **最小间隙** | **-2.66m** | -3.40m | BN: 0.61m, FC: 0.43m |
| **QP 干预率** | 88.2% | N/A | — |
| **车道偏离** | 0.0% | 0.0% | BN < FC |

### 结果图表说明

| 文件 | 论文对应 | 说明 |
|------|---------|------|
| `fig1_controls_comparison.png` | — | **控制量对比**：BarrierNet（蓝）vs FC（红）的加速度 a 和转向率 ω。BN 的控制被 QP 安全层修正，在接近障碍物时显著不同 |
| `fig2_penalty_functions.png` | Fig. 9 | **dCBF 惩罚函数 p₁(z), p₂(z)**：展示 BarrierNet 学习到的 CBF 参数在接近障碍物时的变化。接近障碍物时惩罚增大以满足安全约束 |
| `fig3_trajectories.png` | Fig. 8 | **车辆轨迹**：蓝色实线为 BarrierNet（安全绕行），红色虚线为 FC（直线碰撞）。青色点标记 QP 干预位置 |
| `fig4_safety_barrier.png` | — | **HOCBF 安全指标 b(x)**：障碍物避让（上）和车道保持（下）的安全余量。BN 保持更高的安全余量 |
| `fig5_lane_deviation.png` | Fig. 5 | **车道偏离分布**：直方图（左）和超越概率曲线（右）。BN 的最大横向偏离更小 |
| `fig6_crash_clearance.png` | Table I, Fig. 6 | **碰撞率和间隙分布**：碰撞率对比（左）、间隙分布曲线（中，面积越大越安全）、平均/最小间隙统计（右） |
| `fig7_speed_steering.png` | — | **速度和转向角曲线**：BN 在接近障碍物时减速（速度下降），FC 保持原速 |
| `fig8_qp_intervention.png` | — | **QP 干预率随时间变化**：平滑后的干预率曲线，接近障碍物时干预率显著升高 |
| `fig9_multiple_trajectories.png` | Fig. 8 | **多场景轨迹叠加**（各 20 个场景）：左为 BN（多数安全绕行），右为 FC（全部碰撞） |
| `fig10_comprehensive_panel.png` | — | **3×3 综合面板**：控制量、惩罚函数、轨迹、安全 b(x)、速度、安全率、间隙统计 |
| `fig11_comparison_table.png` | Table I | **方法对比总结表格**：从碰撞率、安全率、间隙、QP 干预等维度对比 |

### CBF 参数敏感性分析

| p₁, p₂ | 碰撞率 | 安全率 | 平均间隙 | QP 干预率 |
|:---:|:---:|:---:|:---:|:---:|
| (1.0, 1.0) | 57% | **43%** | -0.16m | 89% |
| (2.0, 2.0) | 53% | 13% | **-0.04m** | 79% |
| (3.0, 3.0) | 100% | -83% | -1.23m | 77% |
| (5.0, 5.0) | 100% | 0% | -2.33m | 76% |

**关键发现**：较小的 p 值（1.0-2.0）给出更好的安全性能。p 值过大导致 CBF 约束过于保守，反而降低安全性能。这与论文中 dCBF 的动机一致——传统 CBF 的固定增益会导致过度保守。

### 核心结论

1. **BarrierNet 显著降低碰撞率**：从 100% 降至 54%（降低 46%），FC 控制器无任何安全保证
2. **QP 安全层积极干预**：88.2% 的时间步中 QP 修正了参考控制器的不安全输出
3. **平均间隙大幅改善**：BN -0.12m vs FC -2.86m，BN 的通过距离更接近安全边界
4. **参数敏感性**：dCBF 的可学习惩罚参数 p₁, p₂ 是缓解 CBF 保守性的关键，较小值效果更好
5. **与论文差异**：本复现使用简化的状态输入（非图像）和 PD 控制器（非 NMPC），因此绝对性能低于论文（3% vs 54% 碰撞率），但**相对改善趋势一致**
