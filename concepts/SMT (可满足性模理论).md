# SMT (可满足性模理论)

> **一句话**：SMT 是一种**自动推理技术**，判定一个逻辑公式在某个数学理论（如线性算术、位向量）下是否存在使公式为真的变量赋值。它是 [[dReal]]、Z3 等求解器的核心。

---

## 1. 基本概念

### 1.1 SAT vs SMT

**SAT (布尔可满足性)**：给定一个布尔公式（如 $(A \lor B) \land (\neg A \lor C)$），是否存在 $A, B, C$ 的真值赋值使公式为真？

**SMT**：SAT 的扩展——变量不限于布尔，可以是**整数、实数、数组**等，公式可以包含**算术运算、比较**等。

### 1.2 SMT 公式示例

$$\exists x \in \mathbb{R}, y \in \mathbb{R}: x + y = 3 \land x - y = 1 \land x > 0$$

答案：**sat**（$x=2, y=1$）

$$\exists x \in \mathbb{R}: x^2 < 0$$

答案：**unsat**（平方不可能为负）

---

## 2. 常见 SMT 理论

| 理论 | 缩写 | 支持的操作 | 复杂度 |
|------|------|----------|--------|
| 线性实数算术 | LRA | $+, -, \times \text{常数}, \leq, \geq$ | P |
| 线性整数算术 | LIA | 同上 + 整数 | NP-complete |
| 非线性实数算术 | NRA | $+, -, \times, \div, \sin, \exp$ | 不可判定 |
| 位向量 | BV | 位运算 | NP-complete |
| 数组 | AX | 读写数组 | 可判定 |

---

## 3. 常见 SMT 求解器

| 求解器 | 开发者 | 特色 | Python 接口 |
|--------|--------|------|------------|
| **Z3** | Microsoft | 最通用 | `pip install z3-solver` |
| **CVC5** | Stanford/UIowa | 多理论 | `pip install cvc5` |
| **dReal** | CMU/UCSD | 非线性实数 | `pip install dreal` |
| **Yices** | SRI | 快速 | `pip install yices` |

### 3.1 Z3 快速入门

```python
from z3 import *

# 声明变量
x = Real('x')
y = Real('y')

# 构造约束
solver = Solver()
solver.add(x + y == 3)
solver.add(x - y == 1)
solver.add(x > 0)

# 求解
if solver.check() == sat:
    m = solver.model()
    print(f"x = {m[x]}, y = {m[y]}")
else:
    print("unsat")
```

---

## 4. 在安全验证中的应用

### 4.1 验证神经网络属性

**问题**：给定神经网络 $f: \mathbb{R}^n \to \mathbb{R}^m$，验证：

$$\forall x \in [x^L, x^U]: f(x) \geq 0$$

**SMT 编码**：
1. 将 ReLU 编码为分段线性约束：$y = \max(0, x) \iff (y \geq 0) \land (y \geq x) \land (y = 0 \lor y = x)$
2. 将线性层编码为等式约束
3. 添加输入范围约束和输出条件约束
4. 求解是否存在违反条件的输入

### 4.2 在 CEGIS 中的角色

在 [[CEGIS (反例引导合成)]] 中，SMT 求解器作为**验证器**：

```
while True:
    # 1. 训练器: 训练候选网络
    network = train(training_data)
    
    # 2. 验证器: SMT 求解
    result = smt_verify(network, property)
    
    if result == "unsat":
        print("验证通过!")
        break
    else:
        # 3. 反例: 添加到训练集
        counterexample = result.get_model()
        training_data.append(counterexample)
```

---

## 5. 与区间方法的对比

| 特性 | [[IBP (区间界传播)]] / [[CROWN (神经网络验证)]] | SMT |
|------|----------------------------------------------|-----|
| **方法** | 区间传播 | 逻辑推理 |
| **精确性** | 保守 | 精确 |
| **速度** | 快 | 慢（指数级） |
| **输出** | 界 | sat/unsat + 模型 |
| **可扩展性** | 大网络 | 小网络 |

---

## 6. 相关概念

- [[dReal]] — 非线性 SMT 求解器
- [[CEGIS (反例引导合成)]] — SMT 在训练中的应用
- [[IBP (区间界传播)]] — 替代验证方法

---

> **参考**: Barrett et al., "Satisfiability Modulo Theories," Handbook of Satisfiability, 2021
