# neural-from-scratch

本项目按 README 定义的 10 个课程模块组织。每个模块包含理论说明与 Notebook，核心实现只依赖 NumPy，并通过 PyTorch 数值测试验证关键公式。

## 学习顺序

01 感知器 → 02 线性回归 → 03 梯度下降 → 04 反向传播 → 05 激活函数 → 06 MLP → 07 CNN → 08 RNN/LSTM → 09 Attention → 10 Transformer

## 本地验证

```bash
uv sync --all-groups
uv run pytest tests/ -v
```
