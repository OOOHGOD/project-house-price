from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class TrainingResult:
    model: DecisionTreeRegressor
    metrics: dict[str, float | int]
    cv_results: pd.DataFrame


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> TrainingResult:
    """Train a pruned decision tree and evaluate it on a held-out test set."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    if cv_folds > len(x_train):
        raise ValueError("cv_folds cannot exceed the number of training rows")

    baseline = DecisionTreeRegressor(random_state=random_state)
    pruning_path = baseline.cost_complexity_pruning_path(x_train, y_train)
    alphas = np.unique(pruning_path.ccp_alphas)
    # A fully grown tree can expose hundreds of nearly identical alphas. Sampling
    # the path keeps the CLI responsive while still covering the complete range.
    if len(alphas) > 50:
        alpha_indices = np.linspace(0, len(alphas) - 1, num=50, dtype=int)
        alphas = np.unique(alphas[alpha_indices])

    search = GridSearchCV(
        estimator=baseline,
        param_grid={"ccp_alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    predictions = model.predict(x_test)

    metrics: dict[str, float | int] = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "best_ccp_alpha": float(model.ccp_alpha),
        "tree_depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": random_state,
    }
    cv_results = pd.DataFrame(
        {
            "ccp_alpha": search.cv_results_["param_ccp_alpha"].astype(float),
            "mean_cv_rmse": -search.cv_results_["mean_test_score"],
            "std_cv_rmse": search.cv_results_["std_test_score"],
        }
    ).sort_values("ccp_alpha")
    return TrainingResult(model=model, metrics=metrics, cv_results=cv_results)


def save_artifacts(
    result: TrainingResult,
    feature_names: list[str],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist the trained model, metrics, CV results, and diagnostic charts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": destination / "model.joblib",
        "metrics": destination / "metrics.json",
        "cv_results": destination / "cv_results.csv",
        "feature_importance": destination / "feature_importance.png",
        "tree": destination / "decision_tree.png",
    }

    joblib.dump(result.model, paths["model"])
    paths["metrics"].write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result.cv_results.to_csv(paths["cv_results"], index=False)

    order = np.argsort(result.model.feature_importances_)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(np.array(feature_names)[order], result.model.feature_importances_[order])
    axis.set(title="Decision-tree feature importance", xlabel="Importance")
    figure.tight_layout()
    figure.savefig(paths["feature_importance"], dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(22, 10))
    plot_tree(
        result.model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        precision=2,
        max_depth=4,
        ax=axis,
    )
    axis.set_title("Pruned decision tree (first five levels)")
    figure.tight_layout()
    figure.savefig(paths["tree"], dpi=160)
    plt.close(figure)
    return paths


def load_model(path: str | Path) -> Any:
    """Load a model created by this project."""
    return joblib.load(path)
