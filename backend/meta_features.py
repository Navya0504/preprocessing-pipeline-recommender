"""Meta-feature extraction: turn a dataset into a numeric "fingerprint".

Contract: see INTEGRATION_CONTRACT.md, section 2.2 and 3.
STATUS: real implementation. Every key is computed from the data.
Values that do not apply (for example imbalance for regression) are None.
"""
from __future__ import annotations

import numpy as np
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


def _split_columns(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature columns into numeric and categorical (everything non-numeric, bool included)."""
    numeric = features.select_dtypes(include="number")
    categorical = features.drop(columns=numeric.columns)
    return numeric, categorical


def _pct_missing(features: pd.DataFrame) -> float:
    """Percent of all feature cells that are missing (0 to 100)."""
    if features.size == 0:
        return 0.0
    return float(features.isna().sum().sum() / features.size * 100)


def _pct_rows_with_missing(features: pd.DataFrame) -> float:
    """Percent of rows that have at least one missing feature value (0 to 100)."""
    if len(features) == 0:
        return 0.0
    return float(features.isna().any(axis=1).mean() * 100)


def _mean_cardinality(categorical: pd.DataFrame) -> float | None:
    """Mean number of distinct values across categorical columns (None if there are none)."""
    if categorical.shape[1] == 0:
        return None
    return float(categorical.nunique(dropna=True).mean())


def _class_imbalance_ratio(target_values: pd.Series) -> float | None:
    """Minority class count divided by majority class count (0 to 1; 1 means balanced)."""
    counts = target_values.dropna().value_counts()
    if len(counts) < 2:
        return None
    return float(counts.min() / counts.max())


def _mean_skewness(numeric: pd.DataFrame) -> float | None:
    """Mean absolute skewness over numeric columns (columns with too little variation are skipped)."""
    if numeric.shape[1] == 0:
        return None
    skews = numeric.skew().abs().dropna()
    if skews.empty:
        return None
    return float(skews.mean())


def _outlier_fraction(numeric: pd.DataFrame) -> float | None:
    """Fraction of numeric values outside 1.5 x IQR, over all non-missing numeric values (0 to 1)."""
    if numeric.shape[1] == 0:
        return None
    total_outliers = 0
    total_values = 0
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        total_outliers += int(((values < low) | (values > high)).sum())
        total_values += len(values)
    if total_values == 0:
        return None
    return float(total_outliers / total_values)


def _mean_abs_correlation(numeric: pd.DataFrame) -> float | None:
    """Mean absolute pairwise correlation between numeric columns (None if fewer than 2)."""
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr().abs().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[~np.isnan(upper)]
    if upper.size == 0:
        return None
    return float(upper.mean())


def extract_meta_features(df: pd.DataFrame, target: str, task: str) -> dict:
    """Return the meta-feature dict defined in the contract (section 2.2).

    Args:
        df: dataset including the target column.
        target: name of the target column.
        task: "classification" or "regression".
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")

    features = df.drop(columns=[target])
    numeric, categorical = _split_columns(features)
    n_cols = int(features.shape[1])

    return {
        "n_rows": int(len(df)),
        "n_cols": n_cols,
        "n_numeric": int(numeric.shape[1]),
        "n_categorical": int(categorical.shape[1]),
        "categorical_ratio": float(categorical.shape[1] / n_cols) if n_cols else 0.0,
        "pct_missing": _pct_missing(features),
        "pct_rows_with_missing": _pct_rows_with_missing(features),
        "mean_cardinality": _mean_cardinality(categorical),
        "class_imbalance_ratio": _class_imbalance_ratio(df[target]) if task == "classification" else None,
        "mean_skewness": _mean_skewness(numeric),
        "outlier_fraction": _outlier_fraction(numeric),
        "mean_abs_correlation": _mean_abs_correlation(numeric),
        "task": task,
    }
