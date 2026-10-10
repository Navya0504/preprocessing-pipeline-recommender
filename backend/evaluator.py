"""Multi-objective evaluation of pipelines.

Contract: see INTEGRATION_CONTRACT.md, section 2.4, 2.5 and 3.
STATUS: stub. Returns correctly shaped placeholder scores. Replace with real
cross-validation using a fixed reference model and a fixed random seed.
"""
from __future__ import annotations

import pandas as pd

from backend.meta_features import META_FEATURE_KEYS

SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"]
SCORE_KEYS = ["performance", "efficiency", "simplicity", "robustness", "explainability"]
RAW_KEYS = ["raw_cv_mean", "raw_cv_std", "raw_fit_seconds"]


def evaluate_pipeline(
    spec: dict,
    df: pd.DataFrame,
    target: str,
    task: str,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Return the five 0..1 scores (higher is better) plus raw_* values."""
    # TODO: cross-validate spec on df. No test data may leak into preprocessing.
    scores = {key: 0.5 for key in SCORE_KEYS}
    scores.update({"raw_cv_mean": 0.0, "raw_cv_std": 0.0, "raw_fit_seconds": 0.0})
    return scores


def evaluate_many(
    specs: list[dict], df: pd.DataFrame, target: str, task: str
) -> pd.DataFrame:
    """Evaluate every spec and return one row per spec (spec columns + scores)."""
    rows = []
    for spec in specs:
        row = dict(spec)
        row.update(evaluate_pipeline(spec, df, target, task))
        rows.append(row)
    return pd.DataFrame(rows, columns=SPEC_KEYS + SCORE_KEYS + RAW_KEYS)


def build_experience_db(
    datasets: dict[str, tuple[pd.DataFrame, str, str]], out_path: str
) -> None:
    """Run all candidates on all datasets and write the experience database."""
    # TODO: for each dataset: extract meta-features, generate candidates,
    # evaluate them, and append rows. Columns follow contract section 2.5.
    columns = ["dataset_id"] + META_FEATURE_KEYS + SPEC_KEYS + SCORE_KEYS + RAW_KEYS
    pd.DataFrame(columns=columns).to_csv(out_path, index=False)
