<div align="center">

# Boston Housing Regression

A reproducible Decision Tree baseline, from training to prediction.

**English** · [简体中文](README.zh-CN.md)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square) ![scikit-learn](https://img.shields.io/badge/scikit--learn-334155?style=flat-square) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square) [![MIT License](https://img.shields.io/badge/License-MIT-2563EB?style=flat-square)](LICENSE)

</div>

<p align="center">
<a href="#overview">Overview</a> · <a href="#install">Install</a> · <a href="#repository-guide">Repository guide</a> · <a href="#author--license">Author &amp; License</a>
</p>

---

> **Project focus** · Machine Learning


<details>
<summary>Full contents</summary>

- [Overview](#overview)
- [Features](#features)
- [Install](#install)
- [Train and evaluate](#train-and-evaluate)
- [Batch prediction](#batch-prediction)
- [Repository guide](#repository-guide)
- [Development and limitations](#development-and-limitations)
- [Author & License](#author--license)

</details>

## Overview

A reproducible decision-tree regression baseline with a Python package, training/prediction CLI, cross-validation, cost-complexity pruning, and saved evaluation artifacts. Original decision-tree and Iris experiment notebooks are retained.


## Features

- Read the checked-in CSV without the removed `load_boston()` API.
- Select `ccp_alpha` through K-fold cross-validation on training data.
- Report R², MAE, RMSE, tree depth, and leaf count.
- Save a model, cross-validation results, feature importances, and a tree plot.
- Predict from a CSV containing the required feature columns.

The current packaged workflow uses a decision tree, rather than the model comparison mentioned in the previous repository description.

## Install

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install -e ".[dev,notebook]"
```

Alternatively, use `uv sync --extra dev --extra notebook`. The notebook extra is optional for CLI-only use.

## Train and evaluate

```bash
house-price train
house-price train --data boston_housing_data.csv --output-dir artifacts --test-size 0.2 --cv-folds 5 --random-state 42
```

Run either command. The defaults are an 80/20 train/test split, five-fold cross-validation, and random state 42. Equivalent entry points are `python -m house_price train` and `uv run house-price train`.

Outputs under `artifacts/`: `model.joblib`, `metrics.json`, `cv_results.csv`, `feature_importance.png`, and `decision_tree.png`.

## Batch prediction

```bash
house-price predict --model artifacts/model.joblib --input examples/sample_input.csv --output predictions.csv
```

Input needs 13 feature columns in any order; output adds `PREDICTED_MEDV`. The target `MEDV` is measured in thousands of dollars. The loader accepts both `PTRATIO` and the source CSV's spelling `PIRATIO`. Only load trusted joblib files.

## Repository guide

`src/house_price/` contains data/model/CLI code; `tests/` contains tests. [Sample input](examples/sample_input.csv) demonstrates prediction data. The CSV files, notebooks, and DOCX documents retain the original coursework context.

## Development and limitations

```bash
pytest
ruff check src tests
```

The small historical dataset is intended for teaching and baseline experiments, not present-day valuation. It includes a controversial demographic-derived feature. Pruning does not resolve its age, representativeness, or fairness limitations. No new benchmark results are claimed here.

## Author & License

Author / maintainer: [OOOHGOD](https://github.com/OOOHGOD) · Original contributions: [MIT License](LICENSE)

Commercial use and modification are permitted with preservation of copyright and license notices in copies or substantial portions. Third-party code, datasets, models, and media retain their own terms; see [license scope](LICENSE_SCOPE.md).
