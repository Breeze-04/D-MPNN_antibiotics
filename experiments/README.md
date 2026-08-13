# Chemprop D-MPNN 回归实验

本目录提供两个基于 Chemprop 的分子性质回归实验：

- `baseline.py`：D-MPNN + MeanAggregation + RegressionFFN；
- `attention.py`：D-MPNN + AttentiveAggregation + RegressionFFN。

两个脚本采用相同的数据划分、模型维度和训练流程，主要用于比较普通平均聚合与注意力聚合。示例任务以 SMILES 为分子输入，以连续数值为回归目标。

> 以下命令使用 PowerShell 语法。反引号 `` ` `` 表示命令换行，反引号后不能再添加空格。

## 1. 环境安装与 CLI 入门

### 1.1 创建 Conda 环境

建议使用 Python 3.11：

```powershell
conda create -n chemprop-dev python=3.11
conda activate chemprop-dev
```

### 1.2 安装当前 Chemprop 源码

进入包含 `pyproject.toml` 和 `chemprop/` 源码目录的 Chemprop 仓库根目录，然后执行：

```powershell
python -m pip install -e .
```

其中：

- `python -m pip` 表示使用当前 Conda 环境中的 Python 调用 pip；
- `.` 表示安装当前目录中的 Chemprop 项目；
- `-e` 表示 editable（可编辑）安装，Python 会直接引用本地 Chemprop 源码；
- 安装后既可以在 Python 中执行 `import chemprop`，也会注册 `chemprop` 命令行入口。

可以使用下面的命令检查安装：

```powershell
python -c "import chemprop, torch, rdkit, lightning; print(chemprop.__version__)"
chemprop --help
```

### 1.3 什么是 CLI

CLI 是 Command-Line Interface，即命令行界面。Chemprop 的官方 CLI 形式是：

```powershell
chemprop train --help
chemprop predict --help
```

本目录中的实验脚本也采用 CLI 参数形式，只是入口由 `chemprop train` 变成了：

```powershell
python experiments/baseline.py [参数]
python experiments/attention.py [参数]
```

这种方式的优点是每次实验的输入、输出和超参数都能完整地保留在命令中，便于复现和比较。

## 2. Baseline 与 Attention 训练脚本

### 2.1 代码来源

`baseline.py` 和 `attention.py` 是本项目编写的实验入口脚本，不是 Chemprop 官方 CLI 文件的直接复制。脚本通过：

```python
from chemprop import data, featurizers, models, nn
```

调用 Chemprop 提供的官方数据处理、分子图特征化和神经网络组件，包括：

- `MoleculeDatapoint`、数据划分和 DataLoader；
- `SimpleMoleculeMolGraphFeaturizer`；
- `BondMessagePassing`；
- `MeanAggregation` 或 `AttentiveAggregation`；
- `RegressionFFN`；
- `models.MPNN`。

训练循环、早停、checkpoint 和 CSV 日志由 Lightning 管理。

### 2.2 `baseline.py`

Baseline 架构为：

```text
SMILES
  │
  ▼
Molecular graph featurization
  │
  ▼
BondMessagePassing (D-MPNN)
  │
  ▼
MeanAggregation
  │
  ▼
RegressionFFN (MLP)
  │
  ▼
Continuous prediction
```

主要逻辑是：

1. 从 CSV 读取 SMILES 列和目标列；
2. 固定随机种子划分训练集、验证集和测试集；
3. 只使用训练集拟合目标值标准化；
4. 使用 D-MPNN 学习原子表示；
5. 对一个分子内的原子表示取平均，得到分子表示；
6. 使用 RegressionFFN 输出连续回归值；
7. 根据 validation loss 选择最佳 checkpoint；
8. 在测试集计算 RMSE、MAE 和 R²，并保存逐分子预测。

使用 Chemprop 自带的 lipophilicity 数据运行 5 epoch smoke test：

```powershell
python experiments/baseline.py `
  --data-path tests/data/regression/mol/mol.csv `
  --smiles-column smiles `
  --target-columns lipo `
  --split-type RANDOM `
  --epochs 5 `
  --patience 5 `
  --num-workers 0 `
  --accelerator cpu `
  --output-dir experiments/outputs/baseline_smoke
```

这里的 `lipo` 是用于验证代码能否正常运行的回归标签，不是 MIC 数据。

### 2.3 `attention.py`

Attention 架构为：

```text
SMILES
  │
  ▼
Molecular graph featurization
  │
  ▼
BondMessagePassing (D-MPNN)
  │
  ▼
AttentiveAggregation
  │
  ├── Linear(atom representation → scalar score)
  ├── molecule-wise normalization
  └── weighted sum of atom representations
  │
  ▼
RegressionFFN (MLP)
  │
  ▼
Continuous prediction
```

该脚本与 Baseline 保持相同的数据处理和训练逻辑，只把：

```text
MeanAggregation
```

替换为 Chemprop 官方提供的：

```text
chemprop.nn.AttentiveAggregation
```

当前注意力层为单头注意力聚合：它为每个原子学习一个标量权重，然后进行加权求和；它不是多头自注意力，也不是 Transformer。

运行：

```powershell
python experiments/attention.py `
  --data-path tests/data/regression/mol/mol.csv `
  --smiles-column smiles `
  --target-columns lipo `
  --split-type RANDOM `
  --epochs 5 `
  --patience 5 `
  --num-workers 0 `
  --accelerator cpu `
  --output-dir experiments/outputs/attention_smoke
```

## 3. 绘图脚本

三个绘图脚本都直接生成透明背景的 SVG 矢量图。

### 3.1 `plot_baseline_diagnostics.py`

读取 Baseline 的 `test_predictions.csv`，在同一张图中绘制：

- 左图：测试集真实值与预测值散点图，并给出理想预测线 `y = x`；
- 右图：残差与真实值散点图，其中残差为 `predicted - observed`；
- 图中同时显示测试集 RMSE、MAE 和 R²。

```powershell
python experiments/plot_baseline_diagnostics.py `
  --predictions experiments/outputs/baseline_smoke/test_predictions.csv `
  --output experiments/outputs/baseline_smoke/baseline_diagnostics.svg
```

### 3.2 `plot_attention_diagnostics.py`

读取 Attention 的 `test_predictions.csv`，绘制与 Baseline 相同的诊断图：

- 左图：真实值与预测值散点图；
- 右图：残差与真实值散点图；
- 同时显示 RMSE、MAE 和 R²。

```powershell
python experiments/plot_attention_diagnostics.py `
  --predictions experiments/outputs/attention_smoke/test_predictions.csv `
  --output experiments/outputs/attention_smoke/attention_diagnostics.svg
```

### 3.3 `plot_model_comparison.py`

同时读取 Baseline 和 Attention 的训练结果，生成一张组间比较图，包含：

- Validation R² 随 epoch 的变化；
- Training loss 和 Validation loss 随 epoch 的变化；
- 测试集 R²、RMSE 和 MAE 的并列比较；
- 两个模型各自的最佳 validation-loss epoch。

脚本会检查两次实验的主要训练配置和 `splits.csv` 是否一致，并自动寻找与 `metrics.json` 相匹配的 Lightning 日志，避免把旧日志用于比较。

```powershell
python experiments/plot_model_comparison.py `
  --baseline-dir experiments/outputs/baseline_smoke `
  --attention-dir experiments/outputs/attention_smoke `
  --output experiments/outputs/model_comparison.svg
```

## 4. `outputs/` 训练输出结构

以 Baseline 为例：

```text
experiments/outputs/baseline_smoke/
├── best_model.pt
├── config.json
├── metrics.json
├── splits.csv
├── test_predictions.csv
├── checkpoints/
│   ├── best-epoch=XX-val_loss=X.XXXX.ckpt
│   └── last.ckpt
└── logs/
    └── version_0/
        ├── hparams.yaml
        └── metrics.csv
```

各文件含义如下：

| 文件或目录 | 内容 | 主要用途 |
|---|---|---|
| `best_model.pt` | Chemprop 格式的最佳模型 | 后续独立预测与模型部署 |
| `config.json` | 命令行参数、模型架构、软件版本和最佳 checkpoint 路径 | 记录并复现实验条件 |
| `metrics.json` | 测试集 RMSE、MAE、R²和有效样本数 | 汇报最终模型性能 |
| `splits.csv` | 每条原始数据的 `source_row` 和 train/validation/test 归属 | 固定并核对数据划分 |
| `test_predictions.csv` | 测试集原始信息、真实标签与 `pred_<target>` 预测列 | 逐分子检查预测、排名和绘图 |
| `checkpoints/best-*.ckpt` | validation loss 最低时保存的 Lightning checkpoint | 测试前重新载入最佳模型 |
| `checkpoints/last.ckpt` | 训练停止时最后一轮的 checkpoint | 恢复或检查最后训练状态 |
| `logs/version_N/hparams.yaml` | Lightning 保存的模型超参数 | 日志追踪 |
| `logs/version_N/metrics.csv` | 每个 epoch 的 train loss、validation loss、validation 指标及测试指标 | 绘制训练曲线和分析收敛情况 |

再次使用同一个输出目录运行时，Lightning 通常会新增：

```text
logs/version_1/
logs/version_2/
...
```

脚本不会自动清空旧目录。因此正式实验应为每一次运行使用明确的输出目录，或者在运行前由实验者自行确认和整理旧结果，避免混用不同训练记录。
