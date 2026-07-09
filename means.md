 SafePVC 复现指南：基于视觉的神经网络控制系统概率安全合成

  1. 环境配置
  本项目依赖 PyTorch 进行深度学习训练，以及 auto_LiRPA 进行形式化验证。

  基础环境安装
  建议使用 Conda 创建独立环境：

   1 # 创建并激活环境
   2 conda env create -f environment.yml
   3 conda activate vt
   4
   5 # 安装 auto_LiRPA (核心验证后端)
   6 cd auto_LiRPA
   7 pip install -e .
   8 cd ..

  注意：由于包含验证计算，建议使用支持 CUDA 的 NVIDIA GPU。在 macOS 上运行可能需要将代码中的 .cuda() 或 device='cuda' 修改为 device='cpu' 或
  device='mps'。

  ---

  2. 核心实验流程
  复现工作分为三个阶段：感知模型蒸馏、控制器预训练、以及最关键的 SafePVC 验证与加固循环。

  阶段一：感知模型近似 (Observation Model)
  论文为了使高维图像可验证，将 cGAN 蒸馏为紧凑的 MLP。
   - 预训练模型：项目中已提供 Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth。
   - 手动重新训练（可选）：

   1   # 训练 cGAN
   2   python Aebs/cGAN/train_gans.py
   ### 做了
   3   # 蒸馏为 MLP
   4   python Aebs/cGAN/train_mlp.py
   ### 做了

  阶段二：控制器预训练 (RL Pre-training)
  基于感知模型估计的状态，使用 PPO 算法训练初始控制器。
   - 预训练权重：Aebs/controller/state_net_trained.pth。
   - 说明：此步骤提供了系统的初始策略 $\pi_0$。

  阶段三：SafePVC 主循环 (验证、反例提取与加固)
  这是论文的核心（Algorithm 1），通过随机障碍证书（SBC）指导控制器加固。

   1 # 在根目录下执行
   2 python -m Aebs.VT.loop
  该脚本会自动执行以下循环：
   1. 验证：调用 auto_LiRPA 检查当前的 SBC 是否满足安全条件。
   2. 反例收集：如果验证不通过，提取导致安全冲突的状态点（Counterexamples）。
   3. 交替训练：
       * 更新 SBC 网络 以获得更紧的安全概率下界。
       * 更新 控制器网络 以规避验证发现的反例。

  ---

  3. 实验结果查看与分析

  终端输出
  在运行 Aebs.VT.loop 时，重点关注：
   - Verification Result: 提示条件是否满足。
   - Safety Lower Bound: 当前证明的安全概率下界（如 $0.921$ 表示 $92.1\%$ 安全）。
   - iters: 迭代次数，展示系统收敛的速度。

  可视化
  运行 Aebs/VT/verify.py 可以查看安全边界的分布：
   - 它会生成类似论文 Fig. 3 的等高线图。
   - 蓝/绿色区域代表 SBC 证书确认的安全区域，红色点代表反例。

  ---

  4. 常见问题
   - 路径报错：部分脚本可能含有硬编码路径，运行前请确保在项目根目录下执行。
   - 内存/显存不足：验证过程（Interval Bound Propagation）非常耗内存，若崩溃可尝试减小 Aebs/VT/loop.py 中的 batch_size。