"""Plain-English explanations of why a pipeline was recommended.

Contract: see INTEGRATION_CONTRACT.md, section 3.
STATUS: stub. The interpretability table values are placeholders to refine and
justify in the paper. explain() returns correctly shaped placeholder text.
"""
from __future__ import annotations

import pandas as pd

# Higher = easier for a human to understand. TODO: justify these values.
INTERPRETABILITY: dict[str, dict[str, float]] = {
    "imputer": {"mean": 1.0, "median": 1.0, "most_frequent": 1.0, "knn": 0.6, "drop_rows": 0.9},
    "encoder": {"onehot": 1.0, "ordinal": 0.8, "target": 0.5},
    "scaler": {"none": 1.0, "standard": 0.9, "minmax": 0.9, "robust": 0.8},
    "outlier": {"none": 1.0, "iqr_clip": 0.8, "isolation_forest": 0.4},
    "imbalance": {"none": 1.0, "class_weight": 0.8, "smote": 0.5},
}


def explainability_score(spec: dict) -> float:
    """Return 0..1 (higher = easier for a human to understand)."""
    values = [
        INTERPRETABILITY[step][choice]
        for step, choice in spec.items()
        if step in INTERPRETABILITY and choice in INTERPRETABILITY[step]
    ]
    return sum(values) / len(values) if values else 0.0


def explain(
    meta_features: dict,
    chosen_spec: dict,
    scores: dict,
    neighbours: pd.DataFrame,
    front: pd.DataFrame,
) -> dict:
    """Return {"summary": str, "reasons": list[str], "tradeoffs": list[str]}."""
    # TODO: write reasons from the dataset traits (missingness, imbalance, outliers),
    # the neighbours' results, and what was given up compared with the rest of the front.
    return {
        "summary": "Placeholder explanation.",
        "reasons": ["TODO: explain which dataset traits drove this choice."],
        "tradeoffs": ["TODO: explain what was traded off against other pipelines."],
    }
