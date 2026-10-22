[English](README.md) | **简体中文**

# Boston 房价预测：决策树回归

这是一个基于 Boston Housing 数据集的决策树回归项目。仓库在保留原始课程实验 Notebook 和文档的基础上，补充了可复现的 Python 工程、命令行训练/预测入口、自动剪枝调参、模型评估、可视化产物与测试。

> 数据集中的 `MEDV` 表示自住房屋价格中位数，单位为千美元。本项目用于机器学习教学与复现实验，不建议将模型用于真实房产估值或决策。

## 项目能力

- 从仓库自带的 `boston_housing_data.csv` 读取数据，不再依赖新版 scikit-learn 已移除的 `load_boston()`。
- 使用训练集生成代价复杂度剪枝参数（`ccp_alpha`），通过 K 折交叉验证自动选择参数。
- 输出 R²、MAE、RMSE、树深度和叶节点数量。
- 保存训练模型、交叉验证结果、特征重要性图和决策树图。
- 支持对新的 CSV 数据批量预测。
- 固定随机种子并提供自动化测试，便于复现与继续开发。

## 项目结构

```text
.
├── src/house_price/          # 数据处理、训练、产物保存和 CLI
├── tests/                    # pytest 自动化测试
├── examples/sample_input.csv # 预测输入示例
├── boston_housing_data.csv   # Boston Housing 数据集（506 行）
├── iris.csv                  # Iris 数据集（原实验材料）
├── *.ipynb                   # 原始决策树实验 Notebook
├── *.docx                    # 原始实验说明文档
└── pyproject.toml            # 项目与依赖配置
```

## 环境要求

- Python 3.10 或更高版本
- Windows、macOS 或 Linux

推荐使用 [uv](https://docs.astral.sh/uv/)，也可以使用常规的 `pip`。

### 使用 uv

```bash
uv sync --extra dev --extra notebook
```

### 使用 pip

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -e ".[dev,notebook]"
```

如果只运行命令行项目、不需要打开 Notebook，可以省略 `notebook` 可选依赖：

```bash
python -m pip install -e ".[dev]"
```

## 快速开始

### 1. 训练与评估

使用 uv 时，在仓库根目录运行：

```bash
uv run house-price train
```

如果已经激活通过 pip 创建的虚拟环境，则可以直接运行：

```bash
house-price train
# 或
python -m house_price train
```

默认使用 80% 数据训练、20% 数据测试和 5 折交叉验证。运行结果会在终端显示，并写入 `artifacts/`：

```text
artifacts/
├── model.joblib             # 训练后的模型
├── metrics.json             # 测试集指标
├── cv_results.csv           # 每个 ccp_alpha 的交叉验证结果
├── feature_importance.png   # 特征重要性
└── decision_tree.png        # 剪枝后决策树（前五层）
```

常用参数示例：

```bash
uv run house-price train \
  --data boston_housing_data.csv \
  --output-dir artifacts \
  --test-size 0.2 \
  --cv-folds 5 \
  --random-state 42
```

PowerShell 可以写成一行，或把行尾的 `\` 换成反引号。

### 2. 批量预测

输入 CSV 必须包含 13 个特征列，列顺序不限；仓库提供了示例文件：

```bash
uv run house-price predict \
  --model artifacts/model.joblib \
  --input examples/sample_input.csv \
  --output predictions.csv
```

输出文件会保留输入特征，并新增 `PREDICTED_MEDV` 列。模型文件基于 Python pickle 机制，请只加载可信来源生成的 `.joblib` 文件。

## 数据字段

| 字段 | 含义 |
| --- | --- |
| `CRIM` | 城镇人均犯罪率 |
| `ZN` | 大面积住宅用地占比 |
| `INDUS` | 非零售商业用地占比 |
| `CHAS` | 是否临近 Charles River |
| `NOX` | 一氧化氮浓度 |
| `RM` | 每套住宅平均房间数 |
| `AGE` | 1940 年前建成的自住单位占比 |
| `DIS` | 到波士顿就业中心的加权距离 |
| `RAD` | 放射状公路可达性指数 |
| `TAX` | 每万美元房产税率 |
| `PTRATIO` | 城镇师生比 |
| `B` | 数据集定义的人口统计变换特征 |
| `LSTAT` | 数据集定义的低社会经济地位人口占比 |
| `MEDV` | 自住房屋价格中位数（目标值，千美元） |

原始 CSV 把 `PTRATIO` 拼写成了 `PIRATIO`；加载器会自动兼容，两种写法都可识别。

## 开发与验证

运行测试：

```bash
uv run pytest
```

运行代码检查：

```bash
uv run ruff check src tests
```

## 原始实验材料

- `BostonTree.ipynb`：Boston 房价决策树回归实验（旧版 Notebook）。
- `Dtree-iris.ipynb`、`Dtree2-for-iris.ipynb`：Iris 分类边界实验。
- `决策树分类和回归可视化.ipynb`：分类树与回归树可视化。
- `决策树实例-CART算法.ipynb`：手写 CART 分类算法示例。

历史版本的 `BostonTree.ipynb` 使用了已移除的 `sklearn.datasets.load_boston`，仓库中的版本已经改为读取本地 CSV。新的、可复现的主流程位于 `src/house_price/`，建议优先使用上面的 CLI。

## 已知限制

- 数据集样本较少，仅适合教学演示和基线实验。
- 单棵决策树容易产生较高方差；本项目使用剪枝缓解该问题，但不代表最佳预测效果。
- Boston Housing 数据集包含一个有争议的人口统计派生特征，且数据年代较早。展示结果时应说明数据背景，避免把它作为公平、现代的住房估值基准。



## 作者与许可证

作者/维护者：[https://github.com/OOOHGOD](https://github.com/OOOHGOD)。

原创贡献采用 [MIT 许可证](LICENSE)，允许商用和修改；分发软件副本或实质部分时须保留版权与许可声明。第三方代码、数据、模型和媒体素材保留各自权利，详见 [LICENSE_SCOPE.md](LICENSE_SCOPE.md)。
