# MTS (Multivariate Time Series) 异常检测项目

这是一个基于 LightTS 模型的多元时间序列异常检测项目。它利用 EasyTSAD 框架进行数据处理、模型训练、评估和结果可视化。

## 项目结构

```
.trae/
Results/                # 存储模型运行结果、评估数据和图表
├── Evals/              # 评估结果
├── Plots/              # 绘图结果
└── RunTime/            # 运行时数据
datasets/               # 存放时间序列数据集
├── MTS/                # 多元时间序列数据集
└── SMD_structure.txt   # 数据集结构说明
lightTS.py              # LightTS 模型的核心实现
old_runMTS.py           # 旧的运行脚本（备份）
runMTS.py               # 主要的运行脚本，包含模型训练和评估逻辑
requirements.txt        # 项目依赖库列表
environment.txt         # 运行环境信息，例如 Python 版本
README.md               # 项目说明文件
```

## 安装

在运行项目之前，请确保您的系统已安装 Python 3.x。然后，您可以通过以下步骤安装所有必要的依赖库：

1. **克隆项目仓库**：
   ```bash
   git clone <您的项目仓库地址>
   cd MTS
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

## 运行

项目的主要入口是 `runMTS.py` 文件。您可以直接运行此文件来执行模型的训练和异常检测任务：

```bash
python runMTS.py
```

### 可配置的超参数

在 `runMTS.py` 文件中，您可以通过修改 `gctrl.run_exps` 方法的 `hparams` 参数来调整模型的超参数，例如：

- `lr` (学习率)
- `epochs` (训练轮次)
- `chunk_sizes` (模型处理时间序列的块大小)

此外，`ModelConfig` 类中的 `d_model` 和 `dropout` 也是重要的超参数。

### 预处理方法

目前，`runMTS.py` 中默认使用的预处理方法是 "z-score"。EasyTSAD 框架还支持 "raw" (不进行预处理) 和 "min-max" (最小-最大归一化) 预处理方法。您可以通过修改 `gctrl.run_exps` 方法中的 `preprocess` 参数来切换预处理方式，例如：

```python
gctrl.run_exps(
    # ... 其他参数
    preprocess="min-max", # 或 "raw", "z-score"
)
```

## 结果

模型运行完成后，所有的评估结果、运行时数据和生成的图表都将保存在 `Results/` 目录下。

## 环境配置

当前项目的 Python 版本信息记录在 `environment.txt` 文件中。

## 许可证

[根据您的项目选择合适的许可证，例如 MIT, Apache 2.0 等]