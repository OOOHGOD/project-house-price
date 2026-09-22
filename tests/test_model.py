from pathlib import Path

import pandas as pd

from house_price.data import FEATURE_COLUMNS, load_training_data
from house_price.model import load_model, save_artifacts, train_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_train_and_save_artifacts(tmp_path: Path) -> None:
    features, target = load_training_data(PROJECT_ROOT / "boston_housing_data.csv")
    result = train_model(features, target, cv_folds=3)
    paths = save_artifacts(result, list(FEATURE_COLUMNS), tmp_path)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert {"r2", "mae", "rmse", "best_ccp_alpha"}.issubset(result.metrics)

    restored = load_model(paths["model"])
    predictions = restored.predict(features.head(2))
    assert len(predictions) == 2
    assert isinstance(result.cv_results, pd.DataFrame)
