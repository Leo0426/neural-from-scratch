# neural-from-scratch

> 个人学习路径和经验：从零手写神经网络。每个模块包含理论推导、NumPy 实现、数值验证、可视化和论文导读。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX%20Ready-black.svg)](https://github.com/ml-explore/mlx)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 项目简介

`neural-from-scratch` 是一个面向入门者的神经网络学习项目。

它的目标不是把深度学习框架再包装一层，而是把神经网络中的关键组件逐个拆开：先理解数学和数据流，再用 NumPy 写出最小可运行实现，最后用测试和可视化确认直觉是否正确。

- **理论**：用直白语言解释核心概念，再给出必要数学推导
- **实现**：用 NumPy 手写前向传播、反向传播和训练循环
- **验证**：对比 PyTorch / Keras 输出，确保关键结果数值正确
- **可视化**：为核心算法提供交互式动画，帮助建立直觉
- **迁移**：部分模块附 MLX 进阶实现，展示 NumPy 思维到 Apple Silicon 原生框架的迁移方式

学完本项目，你将能够：

1. 从零实现一个多层神经网络，理解每一行代码的含义
2. 读懂主流深度学习论文中的数学符号
3. 自主选用经典网络架构解决实际问题
4. 理解 Attention、Transformer 与现代 LLM 的核心计算机制，可以帮助你建立对上下文、注意力、推理与工具调用的工程直觉，从而更好地设计和实现 Agent、Skill 与 Harness 系统。

---

## 课程模块

| 模块 | 主题 | 代码实现 | 可视化 | MLX 进阶 | 参考论文 |
|------|------|----------|--------|----------|----------|
| 01 | 感知器 | 逻辑门 / 线性分类器 | 决策边界动态绘制 | - | Rosenblatt 1958 |
| 02 | 线性回归 | 正规方程 vs 梯度下降 | 拟合过程逐帧动画 | - | Andrew Ng ML |
| 03 | 梯度下降 | SGD / Momentum / Adam | 3D 损失曲面 + 轨迹对比 | ✓ MLX 等价实现对比 | Kingma & Ba 2014 |
| 04 | 反向传播 | 链式法则，逐层梯度打印 | 计算图 + 梯度流动动画 | ✓ `mlx.core.grad` 对比 | Rumelhart et al. 1986 |
| 05 | 激活函数 | Sigmoid / ReLU / GELU | 函数图像 + 梯度消失对比 | - | He et al. 2015 |
| 06 | 多层网络 | MLP，MNIST 分类 | 神经元激活热力图 | ✓ `mlx.nn.Linear` 重写 | LeCun et al. 1989 |
| 07 | 卷积神经网络 | 手写卷积 + LeNet/ResNet | 卷积核滑动 + 特征图 | ✓ `mlx.nn.Conv2d` 对比 | He et al. 2016 |
| 08 | RNN / LSTM | LSTM cell，序列预测 | 时间步展开 + 门控机制 | - | Hochreiter & Schmidhuber 1997 |
| 09 | Attention 机制 | Scaled Dot-Product / Multi-Head Attention | Q/K/V 相似度矩阵 + 注意力权重热力图 | - | Vaswani et al. 2017 |
| 10 | Transformer | Encoder block / Positional Encoding / Causal Mask | 位置编码 + Causal Mask 矩阵 | - | Vaswani et al. 2017 |

模块 09 重点拆解 Attention 的矩阵计算，模块 10 在此基础上组合最小 Transformer block，为理解现代 LLM 架构打基础。

每个模块的 Notebook 遵循统一结构：

```
理论背景 → 数学推导 → 代码思路 → NumPy 手写实现 → 结果验证 → 可视化 → [MLX 进阶]
```

标注 `[MLX 进阶]` 的模块（03 / 04 / 06 / 07）会在 NumPy 实现之后附加一个进阶 cell，展示等价的 MLX 写法。NumPy 实现始终是教学核心，MLX cell 仅作迁移参考，可独立跳过。

---

## 项目结构

```
neural-from-scratch/
├── README.md
├── pyproject.toml
├── uv.lock
│
├── notebooks/                   # 核心学习内容（Jupyter Notebook）
│   ├── 01_perceptron/
│   │   ├── theory.md            # 理论说明
│   │   └── notebook.ipynb       # 代码实现
│   ├── 02_linear_regression/
│   ├── 03_gradient_descent/
│   │   ├── theory.md
│   │   ├── notebook.ipynb       # NumPy 实现 + Keras 对比
│   │   └── mlx_advanced.ipynb   # [进阶] MLX 等价实现
│   ├── 04_backpropagation/
│   │   ├── theory.md
│   │   ├── notebook.ipynb
│   │   └── mlx_advanced.ipynb   # [进阶] mlx.core.grad 对比
│   ├── 05_activation_functions/
│   ├── 06_mlp/
│   │   ├── theory.md
│   │   ├── notebook.ipynb
│   │   └── mlx_advanced.ipynb   # [进阶] mlx.nn.Linear 重写
│   ├── 07_cnn/
│   │   ├── theory.md
│   │   ├── notebook.ipynb
│   │   └── mlx_advanced.ipynb   # [进阶] mlx.nn.Conv2d 对比
│   ├── 08_rnn_lstm/
│   ├── 09_attention/
│   │   ├── theory.md
│   │   └── notebook.ipynb       # Scaled Dot-Product Attention + mask
│   └── 10_transformer/
│       ├── theory.md
│       └── notebook.ipynb       # 位置编码 + Encoder block
│
├── visualizations/              # 交互式 Web 可视化（D3.js / Plotly）
│   ├── gradient_descent/        # 损失曲面 3D 动画
│   ├── backprop/                # 计算图梯度流动
│   ├── activation/              # 激活函数图像对比
│   ├── cnn/                     # 卷积核滑动动画
│   ├── attention/               # Attention 权重热力图
│   └── transformer/             # 位置编码与 causal mask
│
├── tests/                       # 数值正确性单元测试
│   └── test_*.py                # 断言手写结果与 PyTorch autograd 误差 < 1e-5
│
└── docs/                        # MkDocs Material 文档站源文件
```

---

## 快速开始

### 环境要求

- Python 3.12+
- uv
- Jupyter Notebook 或 JupyterLab

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/neural-from-scratch.git
cd neural-from-scratch

# 安装依赖
uv sync --all-groups
```

核心依赖在 `pyproject.toml` 中统一管理：

```
numpy>=1.24
matplotlib>=3.7
jupyter>=1.0
jupyterlab>=3.0
keras>=2.12
torch>=2.0          # 仅用于数值验证对比
plotly>=5.14
ipywidgets>=8.0
```

### Apple Silicon (M 系列) 专项配置

如果你使用 M1/M2/M3/M4 芯片，推荐额外安装 MLX 以运行进阶模块：

```bash
# 添加 tensorflow-macos（替代标准 tensorflow）
uv add tensorflow-macos tensorflow-metal

# 添加 MLX（运行进阶模块 03 / 04 / 06 / 07 需要）
uv add mlx
```

> 不在 M 系列芯片上运行也完全没有问题。所有 `mlx_advanced.ipynb` 均为可选内容，不影响主线学习。

### 启动 Notebook

```bash
uv run jupyter lab
```

打开 `notebooks/01_perceptron/notebook.ipynb` 开始第一个模块。

### 运行测试

```bash
# 运行所有数值正确性测试
pytest tests/ -v
```

### Docker 可视化镜像

```bash
# 构建可视化镜像，默认生成 neural-from-scratch-visualizations:latest
./scripts/docker-build.sh

# 分别构建 AMD64 / ARM64 镜像
ARCH=amd64 ./scripts/docker-build.sh
ARCH=arm64 ./scripts/docker-build.sh

# 启动可视化服务，访问 http://localhost:8080/
./scripts/docker-run.sh

# 使用 Docker Compose 启动
./scripts/docker-compose-up.sh

# 指定架构运行
ARCH=arm64 ./scripts/docker-run.sh
ARCH=amd64 ./scripts/docker-run.sh
```

可通过环境变量覆盖默认配置：

```bash
IMAGE_TAG=dev ./scripts/docker-build.sh
IMAGE_TAG=dev VIS_PORT=8090 ./scripts/docker-run.sh
VIS_PORT=8090 DETACH=1 ./scripts/docker-compose-up.sh
```

在 Apple Silicon 上优先运行 `ARCH=arm64` 构建出的镜像。若需要运行 AMD64 镜像，请使用 `ARCH=amd64 ./scripts/docker-run.sh`，脚本会自动加上 `--platform linux/amd64`。

Docker Compose 也支持相同的架构参数：

```bash
ARCH=arm64 ./scripts/docker-compose-up.sh
ARCH=amd64 ./scripts/docker-compose-up.sh
```

双架构镜像 manifest 需要推送到镜像仓库：

```bash
ARCH=all PUSH=1 IMAGE_NAME=your-registry/neural-from-scratch-visualizations ./scripts/docker-build.sh
```

---

## 学习路径建议

```
入门路径（应用为主）
  01 感知器 → 02 线性回归 → 03 梯度下降 → 04 反向传播 → 06 多层网络
  完成后可直接使用 Keras 搭建 07 CNN / 08 LSTM 解决实际问题

深入路径（研究为主）
  完整按序学习 01 → 10，重点精读每个模块的参考论文
  并验证每一步数学推导的正确性

MLX 进阶路径（Apple Silicon 用户）
  完成主线模块后，按序学习：
  03 mlx_advanced → 04 mlx_advanced → 06 mlx_advanced → 07 mlx_advanced
  目标：理解 NumPy 实现与 MLX 实现的 API 映射关系，
        感受统一内存架构在中小模型上的性能优势
```

---

## 可视化预览

| 模块 | 可视化内容 |
|------|-----------|
| 梯度下降 | 三种优化器在鞍点曲面上的轨迹对比（SGD 困在鞍点，Adam 穿越） |
| 反向传播 | 计算图节点高亮 + 梯度数值实时流动 |
| 卷积操作 | 卷积核在输入特征图上逐步滑动的动画 |
| 激活函数 | 函数曲线 + 导数曲线 + 深层网络梯度消失的对比图 |
| Attention | Q/K/V 相似度矩阵 + 注意力权重热力图 |
| Transformer | 正弦位置编码 + Decoder causal mask |

运行可视化：

```bash
# 在项目根目录下启动本地服务，确保页面能读取同目录 fixture.json
cd visualizations
python -m http.server 8080
```

然后访问：

- 统一入口：`http://localhost:8080/`

---

## 参考资料

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Christopher Olah
- [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Andrej Karpathy
- [The spelled-out intro to neural networks](https://www.youtube.com/watch?v=VMj-3S1tku0) — Andrej Karpathy
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville
- [MLX Documentation](https://ml-explore.github.io/mlx/) — Apple Machine Learning Research
- [MLX GitHub](https://github.com/ml-explore/mlx) — ml-explore/mlx

---

## License

MIT License. 详见 [LICENSE](LICENSE)。

---

> 如果这个项目对你有帮助，欢迎 Star。
