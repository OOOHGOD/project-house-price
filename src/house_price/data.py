from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "MEDV"
FEATURE_COLUMNS = (
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
)


def _normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names while remaining compatible with the original CSV."""
    normalised = data.copy()
    normalised.columns = [str(column).strip().upper() for column in normalised.columns]
    # The source file contains this historical typo. Accept it without editing the dataset.
    if "PIRATIO" in normalised.columns and "PTRATIO" not in normalised.columns:
        normalised = normalised.rename(columns={"PIRATIO": "PTRATIO"})
    return normalised


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate a labelled Boston housing CSV file."""
    data = _normalise_columns(pd.read_csv(path))
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    selected = data[[*FEATURE_COLUMNS, TARGET_COLUMN]].apply(pd.to_numeric, errors="raise")
    if selected.isna().any().any():
        raise ValueError("Dataset contains missing values in required columns")
    return selected.loc[:, list(FEATURE_COLUMNS)], selected[TARGET_COLUMN]


def load_prediction_data(path: str | Path) -> pd.DataFrame:
    """Load and validate an unlabelled feature CSV file."""
    data = _normalise_columns(pd.read_csv(path))
    missing = sorted(set(FEATURE_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    features = data.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any():
        raise ValueError("Input contains missing values in required columns")
    return features
