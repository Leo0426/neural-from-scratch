# 模块 05：激活函数（Activation Functions）

## 为什么需要激活函数

若无非线性激活函数，多层神经网络等价于单层线性变换，无法拟合复杂函数。

## 实际作用（用途）

- 为网络引入非线性，使模型能够学习图像边缘、文本语义、用户偏好等复杂模式。
- 不同激活函数直接影响训练稳定性和速度：ReLU 常用于 CNN/MLP，GELU 常用于 Transformer。
- 输出层激活决定任务形式：Sigmoid 常用于二分类，Softmax 常用于多分类，线性输出常用于回归。
- 激活函数选择会影响梯度传播，是解决梯度消失、Dying ReLU 等训练问题的重要手段。

## 代码思路

本章代码围绕“函数曲线”和“梯度曲线”组织：

1. 先分别实现 Sigmoid、Tanh 的导数、ReLU、ReLU 导数和 GELU，保持每个函数都能接收 NumPy 向量。
2. Sigmoid 和 Tanh 使用解析导数，ReLU 使用分段导数，GELU 用数值差分近似导数。
3. 构造一组连续输入 `xs`，一次性计算不同激活函数在整个区间上的输出。
4. 验证部分用 PyTorch autograd 对照 Sigmoid 导数，说明手写导数和自动求导一致。
5. 可视化同时画输出和梯度，帮助观察饱和区、线性区和梯度消失现象。

## 常见激活函数

### Sigmoid

```
σ(x) = 1 / (1 + e⁻ˣ)
σ'(x) = σ(x)(1 - σ(x))
```

- 输出范围：(0, 1)
- 问题：深层网络中梯度消失（饱和区梯度趋近于 0）

### Tanh

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

- 输出范围：(-1, 1)，零中心化
- 仍有梯度消失问题

### ReLU

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0 else 0
```

- 计算简单，缓解梯度消失
- 问题：Dying ReLU（负值区域梯度为 0）

### GELU

```
GELU(x) = x · Φ(x)
```

其中 Φ(x) 为标准正态 CDF。GELU 在 BERT / GPT 等 Transformer 模型中广泛使用。

## 梯度消失问题

网络层数增加时，Sigmoid/Tanh 的饱和区使梯度指数衰减，底层参数几乎无法更新。
ReLU 通过线性区梯度恒为 1 来缓解这一问题。

## 参考论文

- He, K. et al. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.*
