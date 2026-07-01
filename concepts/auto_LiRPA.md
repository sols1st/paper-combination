# auto_LiRPA

> **一句话**：auto_LiRPA 是一个 **Python 库**，用于自动计算神经网络的输出界（上下界）。它支持 [[IBP (区间界传播)]]、[[CROWN (神经网络验证)]]、[[α-β-CROWN]] 等多种验证方法，是本项目的核心依赖之一。

---

## 1. 概述

**LiRPA** = Linear Relaxation-based Perturbation Analysis

auto_LiRPA 将任何 PyTorch 模型包装为**有界模型**，可以计算输出在输入扰动下的范围。

**GitHub**: https://github.com/Verified-Intelligence/auto_LiRPA

---

## 2. 安装

```bash
pip install auto-lirpa
# 或从源码安装
git clone https://github.com/Verified-Intelligence/auto_LiRPA.git
cd auto_LiRPA
pip install -e .
```

---

## 3. 基本使用

### 3.1 包装模型

```python
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# 1. 定义普通 PyTorch 模型
model = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

# 2. 包装为有界模型
dummy_input = torch.randn(1, 2)
bounded_model = BoundedModule(model, dummy_input)
```

### 3.2 计算输出界

```python
# 3. 定义输入和扰动
x = torch.tensor([[1.0, 2.0]])
eps = 0.1  # L-infinity 扰动半径
ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
bounded_x = BoundedTensor(x, ptb)

# 4. 计算 IBP 界（最快，最保守）
lb_ibp, ub_ibp = bounded_model.compute_bounds(
    x=(bounded_x,), method="IBP"
)
print(f"IBP: [{lb_ibp.item():.4f}, {ub_ibp.item():.4f}]")

# 5. 计算 CROWN 界（更紧）
lb_crown, ub_crown = bounded_model.compute_bounds(
    x=(bounded_x,), method="CROWN"
)
print(f"CROWN: [{lb_crown.item():.4f}, {ub_crown.item():.4f}]")

# 6. 计算 IBP+CROWN 混合界
lb_mix, ub_mix = bounded_model.compute_bounds(
    x=(bounded_x,), method="IBP+CROWN"
)
```

### 3.3 区间输入（非扰动）

如果输入本身就是一个区间（而非中心点+扰动）：

```python
from auto_LiRPA import PerturbationLpNorm

# 输入区间: x_1 in [0.9, 1.1], x_2 in [1.9, 2.1]
x_L = torch.tensor([[0.9, 1.9]])
x_U = torch.tensor([[1.1, 2.1]])
x_center = (x_L + x_U) / 2
eps = (x_U - x_L).max().item() / 2

ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
bounded_x = BoundedTensor(x_center, ptb)

lb, ub = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN")
```

---

## 4. 在本项目中的使用

```python
# Aebs/VT/verify.py 中使用 auto_LiRPA 进行 IBP 验证

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

class VTVerifier:
    def __init__(self, l_model, env, ...):
        # 将 SBC 网络包装为有界模型
        self.l_ibp = BoundedModule(l_model, torch.randn(1, 2))
    
    def compute_bounds_on_set(self, grid_lb, grid_ub):
        """
        计算 SBC 在网格区间上的界
        
        输入:
            grid_lb: (batch, 2) 区间下界
            grid_ub: (batch, 2) 区间上界
        """
        center = (grid_lb + grid_ub) / 2
        eps = (grid_ub - grid_lb).max().item() / 2
        ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
        bounded_x = BoundedTensor(center, ptb)
        
        lb, ub = self.l_ibp.compute_bounds(
            x=(bounded_x,), method="IBP"
        )
        
        return lb.min().item(), ub.max().item()
```

---

## 5. 支持的操作

| 操作 | IBP | CROWN | 备注 |
|------|-----|-------|------|
| Linear | ✅ | ✅ | 完整支持 |
| ReLU | ✅ | ✅ | 完整支持 |
| Tanh | ✅ | ✅ | 完整支持 |
| Softplus | ✅ | ✅ | 完整支持 |
| LayerNorm | ⚠️ | ⚠️ | 部分支持 |
| Conv2d | ✅ | ✅ | 完整支持 |
| 自定义操作 | 需注册 | 需注册 | 使用 `@Bound.register` |

---

## 6. 性能优化

```python
# 使用 CUDA 加速
bounded_model = bounded_model.cuda()
bounded_x = bounded_x.cuda()

# 批处理
x_batch = torch.randn(1024, 2)
bounded_x_batch = BoundedTensor(x_batch, ptb)
lb, ub = bounded_model.compute_bounds(
    x=(bounded_x_batch,), method="IBP"
)
```

---

## 7. 相关概念

- [[IBP (区间界传播)]] — 基础验证方法
- [[CROWN (神经网络验证)]] — 更精确的方法
- [[α-β-CROWN]] — 最精确的方法
- [[dReal]] — 替代验证工具（SMT-based）

---

> **文档**: https://auto-lirpa.readthedocs.io/
