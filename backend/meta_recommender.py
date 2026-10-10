"""Meta-learner: recommend pipelines from similar past datasets.

Contract: see INTEGRATION_CONTRACT.md, section 3.
STATUS: stub. Returns the first k rows / a default spec. Replace with a real
nearest-neighbour search over scaled meta-features.
"""
from __future__ import annotations

import pandas as pd

SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"]
DEFAULT_SPEC = {
    "imputer": "median",
    "encoder": "onehot",
    "scaler": "standard",
    "outlier": "none",
    "imbalance": "none",
}


def find_similar_datasets(
    meta_features: dict, experience_db: pd.DataFrame, k: int = 5
) -> pd.DataFrame:
    """Return the k most similar past datasets with a `distance` column."""
    # TODO: scale meta-features, compute distances, return the k nearest datasets.
    similar = experience_db.head(k).copy()
    similar["distance"] = 0.0
    return similar


def recommend(
    meta_features: dict, experience_db: pd.DataFrame, k: int = 5
) -> list[dict]:
    """Return ranked candidate pipeline specs learned from similar datasets."""
    # TODO: rank specs by how well they did on the neighbours.
    neighbours = find_similar_datasets(meta_features, experience_db, k)
    if neighbours.empty:
        return [dict(DEFAULT_SPEC)]
    return [
        {key: row[key] for key in SPEC_KEYS}
        for _, row in neighbours.drop_duplicates(subset=SPEC_KEYS).iterrows()
    ]
