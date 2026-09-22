from pathlib import Path

import pandas as pd
import pytest

from house_price.data import FEATURE_COLUMNS, load_prediction_data, load_training_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_training_data_accepts_original_column_typo() -> None:
    features, target = load_training_data(PROJECT_ROOT / "boston_housing_data.csv")

    assert features.shape == (506, 13)
    assert target.shape == (506,)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert not features.isna().any().any()


def test_prediction_data_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"CRIM": [0.1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_prediction_data(path)
