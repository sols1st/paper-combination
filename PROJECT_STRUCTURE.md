# artical-F122 项目结构说明

> **项目名称：** SafePVC — Provably Probabilistic Safe Controller Synthesis
>
> **核心目标：** 为神经网络控制器提供可证明的概率安全保证，以 CARLA 自动驾驶紧急制动（AEBS）为基准场景。

---

## 总体架构

```
CARLA 仿真器 → 图像数据 → cGAN 观测模型 → 状态估计 → PPO 控制器 → SafePVC 形式化验证 (CEGIS) → QP 安全滤波器
```

---

## 目录总览

```
artical-F122/
├── Aebs/                   # 核心实验模块：CARLA 紧急制动场景全流程
├── cGAN/                   # 条件生成对抗网络（cGAN）通用训练框架
├── Combined_network/       # 端到端组合策略网络定义
├── auto_LiRPA/             # 神经网络界限计算库（α,β-CROWN 核心引擎）
├── runs/                   # 实验运行日志
├── README.md               # 项目总览与使用说明
├── means.md                # 复现指南（中文）
├── experiment_results_report.md  # 实验结果分析报告（中文）
├── environment.yml         # Conda 环境配置（UTF-16 编码）
├── environment_fixed.yml   # Conda 环境配置（UTF-8 修复版）
├── info.log                # 8 次实验运行的摘要日志
└── .gitignore
```

---

## 1. `Aebs/` — 核心实验模块（Automatic Emergency Braking System）

实现 SafePVC 框架的完整流水线，从 CARLA 数据采集到形式化验证再到 QP 安全滤波。

**场景定义：** 车辆在距离 `d`（5–16 m）、速度 `v`（0–3 m/s）下接近障碍物，需制动避免进入不安全区域（d ∈ [5,6] m 且 v > 0.5 m/s）。状态 `s = [d_norm, v]`，动作 `a ∈ [-3, 3] m/s²`。

```
Aebs/
├── carla_data/          # CARLA 仿真器原始数据
├── connect/             # 数据预处理流水线
├── data/                # 处理后的训练数据集
├── cGAN/                # AEBS 场景的观测模型训练
├── controller/          # PPO 控制器 + 状态估计网络训练
├── system/              # 环境定义、端到端仿真、扰动估计
├── VT/                  # SafePVC CEGIS 验证-训练主循环
└── QP/                  # 基于 CBF-QP 的运行时安全滤波器
```

### 1.1 `carla_data/` — CARLA 原始数据

| 文件 | 说明 |
|------|------|
| `labels.csv` | CSV 文件，映射图像文件名到真实距离（米） |

存储从 CARLA 仿真器直接采集的原始截图和标签。

### 1.2 `connect/` — 数据预处理

| 文件 | 说明 |
|------|------|
| `genGANData.py` | 连接 CARLA，生成 Tesla Model 3，在 400 个距离点（5–16 m）拍摄截图，保存为 PNG + CSV |
| `downSample.py` | 将 640×640 原始图像降采样到 32×32 灰度图，打包为 HDF5（`data/Downsampled.h5`） |
| `view.py` | 可视化处理后的 HDF5 数据 |

### 1.3 `data/` — 训练数据集

| 文件 | 说明 |
|------|------|
| `Downsampled.h5` | HDF5 格式，包含 `X_train`（32×32 灰度图）和 `y_train`（距离标签），是所有下游训练的核心数据源 |

### 1.4 `cGAN/` — 观测模型训练

| 文件/目录 | 说明 |
|-----------|------|
| `train_gans.py` | 训练卷积 cGAN（AebsGConv + AebsDConvSpectral），以距离为条件生成合成 CARLA 图像，使用 LSGAN 损失 |
| `train_mlp.py` | 训练 MLP 生成器（监督模式，MSE 损失）：`(z, distance) → image`，作为端到端控制器的**观测模型（gen_net）** |
| `view.py` | 对比可视化：原始 CARLA 图像 vs 降采样图像 vs GAN 生成图像 |
| `LS_BS16_LR1e-04/` | cGAN 训练输出（模型检查点、各 epoch 生成图像） |
| `mlp_supervised_ld4/` | 监督 MLP 生成器权重（`mlp_supervised.pth`），即流水线中使用的 gen_net |

### 1.5 `controller/` — 控制器与状态估计

| 文件/目录 | 说明 |
|-----------|------|
| `Controller_train.py` | 使用 `stable_baselines3` 的 PPO 算法，在 `AebsEnv` 中训练 200K 步，学习 `[d_norm, v] → acceleration` |
| `StateEstimate_train.py` | 训练**状态估计网络**（SubNet）：将 gen_net 输出的图像映射回距离标量，使用 MSE + Lipschitz 谱正则化 |
| `state_net_trained.pth` | 训练好的状态估计网络权重 |
| `ppo_aebs_controller.zip` | 最终 PPO 模型 |
| `best_model/` | 最佳 PPO 检查点 |
| `logs/` | 每 50K 步的训练检查点 |

### 1.6 `system/` — 环境定义与仿真

| 文件 | 说明 |
|------|------|
| `env.py` | 定义两个环境类：**`Aebs`**（SafePVC 验证用的核心系统模型，含动力学、初始集、不安全集、噪声模型）和 **`AebsEnv`**（Gymnasium 封装，供 PPO 训练用，含奖励函数） |
| `combined.py` | 使用完整端到端网络（gen_net → state_net → PPO）运行 100 条轨迹的批量仿真，绘制 d-v 相图 |
| `estimate.py` | 估计端到端控制器的**扰动模型**：在 5000 个随机状态 × 10000 个随机隐向量 z 下，测量状态偏移量 (Δs, Δv) |

### 1.7 `VT/` — SafePVC 验证与训练（CEGIS 主循环）

这是项目的**核心创新模块**，实现 Counter-Example Guided Inductive Synthesis 循环。

| 文件 | 说明 |
|------|------|
| `utils.py` | 工具函数 + **L-Net**（障碍证书 B(s)）的 MLP 类，支持 tanh 激活和 softplus 输出（保证 B(s) ≥ 0），含 `martingale_loss()` |
| `train.py` | **`VTLearner`** — 训练组件：`train_step_l`（训练 L-Net 障碍证书，含鞅递减损失 + Lipschitz 正则化）、`train_step_p`（训练 P-Net 控制器，含 PPO 教师蒸馏）、`train_step_joint`（联合训练） |
| `verify.py` | **`VTVerifier`** — 验证组件：状态空间网格离散化 → IBP 计算 B(s) 的认证界限 → 检查超鞅递减条件 E[B(s_next)] < B(s) → 计算概率安全下界 P(safe) ≥ 1 - 1/B_threshold → 返回反例 |
| `loop.py` | **CEGIS 主循环**：交替执行 ① 训练 L-Net 10 个 epoch → ② IBP 验证 → ③ 若通过则计算安全概率 → ④ 若未通过则用反例重训 P-Net。最多 100 次迭代或 1 小时超时 |

### 1.8 `QP/` — CBF-QP 运行时安全滤波器

项目的**最新增模块**，基于控制障碍函数（CBF）和二次规划（QP）的运行时安全滤波。

| 文件 | 说明 |
|------|------|
| `__init__.py` | 导出 `CBFQPSafetyFilter` |
| `qp_safety_filter.py` | 核心类，求解 `min ½‖u - u_nn‖² s.t. h(f(s,u)) ≥ (1-α)·h(s)`。三种 CBF 模式：**learned**（学习到的 B(s)）、**geometric**（纯距离）、**stopping_distance**（考虑速度，推荐）。1D 动作空间有解析解 |
| `simulate_with_qp.py` | 100 条轨迹对比仿真：纯 NN / NN+QP(stopping) / NN+QP(geometric) / NN+QP(learned) |
| `visualize_qp.py` | 可视化：CBF 安全区域等高线、QP 干预热力图、动作调制图、障碍证书景观图 |
| `QP_Safety_Filter_文档.md` | 完整中文文档：CBF 理论、QP 公式、解析解推导、API 设计、参数调优 |
| `实验报告.md` | 实验结果：三种策略均达 100% 安全，QP 滤波器使轨迹长度缩短 61% |

---

## 2. `cGAN/` — 条件 GAN 通用训练框架

可复用的 PyTorch cGAN 训练框架，支持多个应用域（Taxi 和 AEBS）。

| 文件 | 说明 |
|------|------|
| `spectral_norm.py` | **谱归一化**层（DenseSN、ConvSN），基于幂迭代法计算最大奇异值，稳定 GAN 训练 |
| `cGAN_common.py` | 核心框架：**Settings** 配置容器、4 种损失函数（DCGAN / LSGAN / WGAN-GP / Hinge）、正交正则化、训练循环、图像网格可视化 |
| `taxi_models_and_data.py` | 域特定模型和数据加载：**生成器**（TaxiGConv、TaxiGMLP、AebsGConv、AebsMLPGenerator）、**判别器**（TaxiDConvSpectral、TaxiDMLP、AebsDConvSpectral）、HDF5 数据加载函数 |

**设计特点：** 通过 `Settings` 对象自由组合生成器、判别器、损失函数和数据源，高度可组合。

---

## 3. `Combined_network/` — 端到端组合策略网络

定义两个仿真域的端到端网络架构，核心思想：冻结预训练的生成模型，只训练控制器头部。

| 文件 | 说明 |
|------|------|
| `model.py` | 所有网络架构定义 |

**X-Plane 11 域（飞行控制）：**
- `ControllBN` — 全连接控制器（BatchNorm + ReLU + Tanh），`128 → 32 → 1`
- `End2EndNet` — 冻结 gen_net + 可训练 ControllBN：`z, ny → gen_net → flatten → ControllBN → phi`

**AEBS 域（紧急制动）：**
- `SubNet` — 特征提取器（LayerNorm + ReLU），`1024 → 256 → 64 → 1`
- `CombinedPolicyNetwork` — 两阶段策略网络（mlp_extractor + action_net），对标 Stable-Baselines3 的 actor 架构
- `AebsEnd2EndNet` — 完整端到端模型：`z, s → split(d,v) → gen_net → SubNet → concat(state, v) → CombinedPolicyNetwork → acc`

---

## 4. `auto_LiRPA/` — 神经网络界限计算库

**auto_LiRPA**（Automatic Linear Relaxation based Perturbation Analysis），版本 0.6.0，BSD 3-Clause 许可证。这是 **α,β-CROWN** 神经网络验证器的核心界限计算引擎（VNN-COMP 2021–2024 冠军），由 UIUC、UCLA 等机构开发。

```
auto_LiRPA/
├── auto_LiRPA/           # 核心 Python 子包
│   ├── operators/        # 所有支持的神经网络算子的界限实现
│   └── cuda/             # 自定义 CUDA 核函数
├── examples/             # 示例（simple / vision / language / sequence）
├── doc/                  # Sphinx 文档
├── tests/                # 30+ 测试文件
└── setup.py              # 安装脚本
```

### 4.1 `auto_LiRPA/` 核心子包

| 模块 | 说明 |
|------|------|
| `bound_general.py` | **BoundedModule** — 核心类，封装任意 nn.Module，解析计算图，调度界限计算 |
| `backward_bound.py` | **反向 LiRPA**（CROWN）界限传播，含批量处理和 Patches 表示 |
| `forward_bound.py` | **前向 LiRPA** 界限传播 |
| `interval_bound.py` | **区间界限传播（IBP）** — 最基础的界限方法 |
| `optimized_bounds.py` | **α-CROWN** 优化界限 — 通过梯度优化收紧松弛参数 |
| `beta_crown.py` | **β-CROWN** 分裂约束 + GenBaB 分支定界 |
| `bounded_tensor.py` | BoundedTensor / BoundedParameter — 携带扰动信息的张量 |
| `perturbations.py` | 扰动规格：Lp 范数、L0 范数、同义词替换 |
| `parse_graph.py` | 通过 ONNX 符号执行解析 PyTorch 模型为内部图表示 |
| `patches.py` | Patches 类 — CNN 界限的内存高效表示 |
| `output_constraints.py` | **INVPROP** 算法 — 利用输出约束收紧界限 |
| `jacobian.py` | Jacobian 界限计算，支持局部 Lipschitz 常数 |
| `eps_scheduler.py` | 认证训练的 ε 调度策略 |
| `wrapper.py` | 认证鲁棒性损失封装器 |

### 4.2 `auto_LiRPA/operators/` — 界限算子

覆盖所有主流神经网络操作的界限实现（ReLU、Tanh、GeLU、Sigmoid、Conv、Linear、BatchNorm、LayerNorm、Softmax、LSTM、Sin/Cos 等 30+ 算子）。

### 4.3 `examples/` — 示例

| 子目录 | 说明 |
|--------|------|
| `simple/` | 基础教程：toy.py（手动权重网络）、invprop.py（前像过近似）、mip_lp_solver.py（精确验证） |
| `vision/` | 视觉模型：MNIST/CIFAR/TinyImageNet/ImageNet 认证训练与验证、自定义算子、Jacobian 界限 |
| `language/` | NLP 模型：Transformer/LSTM 认证鲁棒训练 |
| `sequence/` | 序列数据：LSTM 认证训练 |

### 4.4 `tests/` — 测试套件

30+ pytest 测试文件，覆盖各类模型、算子、激活函数、Patches 表示等，含回归测试参考数据。

---

## 5. `runs/` — 实验运行日志

| 文件 | 大小 | 说明 |
|------|------|------|
| `run_01.log` | ~238 KB | Run 1 详细日志（100 次迭代），结果 UNSAFE |
| `run_02.log` | ~2.4 MB | Run 2 详细日志（1000 次迭代），长时间训练仍 UNSAFE |
| `run_03.log` | ~243 KB | Run 3 详细日志（100 次迭代），**成功运行**，峰值安全概率 95.691% |

---

## 6. 根目录文件

| 文件 | 说明 |
|------|------|
| `README.md` | 项目总览、安装指南、仓库结构、端到端使用流程 |
| `means.md` | 复现指南（中文）：环境搭建、三阶段实验流程、输出解读、常见问题排查 |
| `experiment_results_report.md` | 实验分析报告（中文）：8 次运行结果、参数配置、逐次分析、调优建议 |
| `environment.yml` | Conda 环境配置（`vt`）：Python 3.8 + PyTorch 2.3.1 + CUDA 12.1 + stable-baselines3 等（⚠️ UTF-16 编码） |
| `environment_fixed.yml` | 同上内容的 UTF-8 修复版本 |
| `info.log` | 8 次实验的机器可读摘要：Run 1–7 UNSAFE，Run 8 成功（max_reach_prob = 95.7%） |
| `.gitignore` | 忽略 `__pycache__/` 和 `info.log` |

---

## 关键实验结果

- 共 8 次独立运行，仅 **Run 8** 成功获得 ≥ 95% 的概率安全下界（峰值 95.691%）
- 成功率 12.5%，方差较高
- QP 安全滤波器可将轨迹长度缩短 61%（272 → 105 步），三种 CBF 策略均达 100% 安全
- 主要调优方向：降低 P-Net 学习率、增大 Lipschitz 系数、增加迭代次数、L-Net 预训练
