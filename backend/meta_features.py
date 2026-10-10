"""Meta-feature extraction: turn a dataset into a numeric "fingerprint".

Contract: see INTEGRATION_CONTRACT.md, section 2.2 and 3.
STATUS: stub. Returns correctly shaped placeholder values. Replace the body of
extract_meta_features with real computations, one key at a time.
"""
from __future__ import annotations

import pandas as pd

META_FEATURE_KEYS = [
    "n_rows",
    "n_cols",
    "n_numeric",
    "n_categorical",
    "categorical_ratio",
    "pct_missing",
    "pct_rows_with_missing",
    "mean_cardinality",
    "class_imbalance_ratio",
    "mean_skewness",
    "outlier_fraction",
    "mean_abs_correlation",
    "task",
]


def extract_meta_features(df: pd.DataFrame, target: str, task: str) -> dict:
    """Return the meta-feature dict defined in the contract (section 2.2).

    Args:
        df: dataset including the target column.
        target: name of the target column.
        task: "classification" or "regression".
    """
    features = df.drop(columns=[target])
    numeric = features.select_dtypes(include="number")
    categorical = features.drop(columns=numeric.columns)

    # TODO: replace these placeholders with real computations.
    return {
        "n_rows": int(len(df)),
        "n_cols": int(features.shape[1]),
        "n_numeric": int(numeric.shape[1]),
        "n_categorical": int(categorical.shape[1]),
        "categorical_ratio": 0.0,
        "pct_missing": 0.0,
        "pct_rows_with_missing": 0.0,
        "mean_cardinality": None,
        "class_imbalance_ratio": None if task == "regression" else 1.0,
        "mean_skewness": 0.0,
        "outlier_fraction": 0.0,
        "mean_abs_correlation": 0.0,
        "task": task,
    }
