# 模块 09：Attention 机制

## 核心概念

Attention 让模型在处理一个 token 时，可以动态选择序列中其他 token 的信息。它的关键不是“记住全部历史”，而是用相似度计算出一组权重，再按权重汇总上下文。

## 实际作用（用途）

- 让模型按需聚合上下文，是机器翻译、文本摘要、问答系统、代码补全和多模态理解的关键机制。
- 在长文本中，Attention 可以直接连接相距很远的 token，缓解 RNN 逐步传递信息造成的长距离依赖问题。
- Attention 权重可用于分析模型关注了哪些输入位置，帮助调试翻译对齐、摘要依据或检索增强结果。
- Multi-Head Attention 允许模型同时学习多种关系，例如语法依赖、实体指代、局部搭配和全局主题。

## 代码思路

本章代码直接实现 Scaled Dot-Product Attention 的数据流：

1. `softmax` 先做最大值平移，保证指数运算更稳定。
2. `scaled_dot_product_attention` 用 `QK^T` 计算 token 之间的匹配分数，再除以 `sqrt(d_k)` 控制数值尺度。
3. 如果传入 mask，就把不可见位置替换成极小值，使它们在 softmax 后几乎没有权重。
4. 注意力权重乘以 `V` 得到上下文表示，返回 `context` 和 `weights`，便于同时看结果和注意力分布。
5. 示例中手工构造 Q/K/V，先看普通 attention，再看 causal mask 对权重矩阵的影响。

## Q / K / V

给定输入序列 `X`，通常通过三组线性变换得到：

```
Q = XWq    # Query：当前 token 想找什么
K = XWk    # Key：每个 token 提供什么索引
V = XWv    # Value：每个 token 提供什么内容
```

Query 与 Key 的点积表示匹配程度，Value 是最终被加权求和的信息。

## Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QKᵀ / √dk) V
```

- `dk` 是 Key 的维度
- 除以 `√dk` 可以避免点积随维度增大而过大，防止 softmax 过早饱和
- softmax 后的矩阵每一行和为 1，表示当前位置对所有位置的注意力分布

## Mask

在序列建模中常见两类 mask：

- **padding mask**：忽略补齐 token
- **causal mask**：禁止当前位置看到未来 token，用于自回归生成

## Multi-Head Attention

单头 Attention 只在一个表示子空间里计算相似度。Multi-Head Attention 把表示拆成多个 head，并行学习不同关系：

```
headᵢ = Attention(QWqᵢ, KWkᵢ, VWvᵢ)
MHA(Q, K, V) = concat(head₁, ..., headₕ) Wo
```

多个 head 可以分别关注局部词序、长距离依赖、语法结构或语义关联。

## 参考论文

- Vaswani, A. et al. (2017). *Attention Is All You Need.*
