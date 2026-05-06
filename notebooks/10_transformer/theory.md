# 模块 10：Transformer

## 核心概念

Transformer 用 Attention 替代循环结构，让序列中任意两个位置可以直接交互。它的基本单元由 Self-Attention、前馈网络、残差连接和 LayerNorm 组成。

## 实际作用（用途）

- 是现代大语言模型、机器翻译、文本摘要、代码生成、语音识别和视觉 Transformer 的核心架构。
- 并行处理序列，训练效率高，适合在大规模语料和硬件加速器上扩展。
- Encoder 适合理解任务，例如文本分类、语义检索和信息抽取；Decoder 适合生成任务，例如对话、写作和代码补全。
- 通过 Attention、残差连接、LayerNorm 和 FeedForward 的组合，Transformer 能在深层网络中稳定学习复杂上下文关系。

## 代码思路

本章代码把 Transformer Encoder Block 拆成几个可独立理解的函数：

1. `positional_encoding` 生成正弦/余弦位置编码，并加到 token 表示上，让模型获得顺序信息。
2. `layer_norm` 对每个 token 的特征维度做归一化，稳定残差连接后的数值分布。
3. `self_attention` 把输入投影成 Q/K/V，计算注意力权重，再汇总 Value 得到上下文表示。
4. `feed_forward` 是逐 token 的两层 MLP，用于在 attention 之后进一步变换每个位置的表示。
5. 主流程按 `Self-Attention -> 残差 + LayerNorm -> FFN -> 残差 + LayerNorm` 组合成最小 Encoder Block，并额外演示 causal mask 的效果。

## 位置编码

Self-Attention 本身不包含顺序信息，因此需要给 token 表示加入位置编码。经典正弦位置编码为：

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

它让模型在不新增可训练参数的情况下感知绝对位置和相对距离。

## Encoder Block

```
x1 = LayerNorm(x + MultiHeadSelfAttention(x))
y  = LayerNorm(x1 + FeedForward(x1))
```

其中 FeedForward 通常是两层 MLP：

```
FFN(x) = max(0, xW1 + b1) W2 + b2
```

残差连接让梯度更容易穿过深层网络，LayerNorm 稳定每个 token 的特征分布。

## Decoder Block

Decoder 相比 Encoder 多了 causal mask 和 cross-attention：

```
masked self-attention → encoder-decoder attention → feed-forward
```

causal mask 保证第 `t` 个 token 只能依赖 `0..t` 的历史信息，不会偷看未来答案。

## Transformer 与 LLM

现代大语言模型通常以 Decoder-only Transformer 为主体：输入 token 经过嵌入、位置编码和多层 masked self-attention 后，预测下一个 token 的概率分布。

## 参考论文

- Vaswani, A. et al. (2017). *Attention Is All You Need.*
