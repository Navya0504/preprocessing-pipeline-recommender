"""Candidate preprocessing pipelines.

Contract: see INTEGRATION_CONTRACT.md, section 2.3 and 3.
STATUS: stub. generate_candidates returns a tiny fixed list; build_pipeline
returns a pass-through pipeline. Replace with the real search space.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

IMPUTERS = ["mean", "median", "most_frequent", "knn", "drop_rows"]
ENCODERS = ["onehot", "ordinal", "target"]
SCALERS = ["none", "standard", "minmax", "robust"]
OUTLIER_METHODS = ["none", "iqr_clip", "isolation_forest"]
IMBALANCE_METHODS = ["none", "class_weight", "smote"]  # classification only


def generate_candidates(meta_features: dict) -> list[dict]:
    """Return candidate pipeline specs (contract 2.3) that suit this dataset."""
    # TODO: build the full grid, then prune combinations that do not apply
    # (for example imbalance handling for regression, imputers when nothing is missing).
    is_classification = meta_features.get("task") == "classification"
    return [
        {
            "imputer": "median",
            "encoder": "onehot",
            "scaler": "standard",
            "outlier": "none",
            "imbalance": "none",
        },
        {
            "imputer": "knn",
            "encoder": "ordinal",
            "scaler": "robust",
            "outlier": "iqr_clip",
            "imbalance": "class_weight" if is_classification else "none",
        },
    ]


def build_pipeline(
    spec: dict, numeric_cols: list[str], categorical_cols: list[str], task: str
) -> Pipeline:
    """Turn a spec into a fit-ready preprocessing pipeline (no model)."""
    # TODO: build a ColumnTransformer from the spec using sklearn / imblearn.
    return Pipeline([("passthrough", "passthrough")])
