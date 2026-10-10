"""Pareto-front selection across the five objectives (higher is always better).

Contract: see INTEGRATION_CONTRACT.md, section 3.
STATUS: working implementation. Safe to use as is; improve later if needed.
"""
from __future__ import annotations

import pandas as pd

SCORE_KEYS = ["performance", "efficiency", "simplicity", "robustness", "explainability"]


def pareto_front(results: pd.DataFrame) -> pd.DataFrame:
    """Return only the non-dominated rows across the five score columns.

    A row is dominated if another row is at least as good on every score and
    strictly better on at least one.
    """
    if results.empty:
        return results.copy()
    values = results[SCORE_KEYS].to_numpy()
    keep = []
    for i, row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i != j and (other >= row).all() and (other > row).any():
                dominated = True
                break
        keep.append(not dominated)
    return results[keep].reset_index(drop=True)


def select_best(front: pd.DataFrame, weights: dict[str, float] | None = None) -> dict:
    """Pick one pipeline from the front by weighted score (equal weights by default)."""
    if front.empty:
        raise ValueError("The Pareto front is empty; nothing to select.")
    if weights is None:
        weights = {key: 1.0 for key in SCORE_KEYS}
    total = sum(weights.get(key, 0.0) for key in SCORE_KEYS)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    weighted = sum(front[key] * weights.get(key, 0.0) for key in SCORE_KEYS) / total
    return front.loc[weighted.idxmax()].to_dict()
