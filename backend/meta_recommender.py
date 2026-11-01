"""Meta-learner: recommend pipelines from similar past datasets.

Contract: see INTEGRATION_CONTRACT.md, section 3.
STATUS: real implementation.

How it works
    1. Every past dataset in the experience database has a "fingerprint" (its meta-features).
    2. Fingerprints are put on a common scale and compared with the new dataset's
       fingerprint. The k closest past datasets are its "neighbours".
    3. For each pipeline, the scores it earned on the neighbours are averaged, giving
       more weight to closer neighbours. This is the predicted score. Performance,
       efficiency and robustness are first made RELATIVE within each past dataset
       (0 = worst pipeline there, 1 = best; 0.5 for all if they are within 0.02 of each other, a tie), because an accuracy of 0.9 on one dataset
       and an R^2 of 0.2 on another cannot be averaged directly.
    4. The predicted scores go through Pareto selection, so the recommendations are the
       pipelines nothing else beats on all five goals, ranked by weighted average.

The experience database must have the columns of contract section 2.5:
dataset_id, the meta-feature keys, the five spec keys and the five score keys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.pareto_selector import SCORE_KEYS, pareto_front
from backend.pipeline_space import generate_candidates

SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"]
DEFAULT_SPEC = {
    "imputer": "median",
    "encoder": "onehot",
    "scaler": "standard",
    "outlier": "none",
    "imbalance": "none",
}

# Meta-features compared when looking for similar datasets (task is handled separately).
NUMERIC_META = [
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
]
# Counts and heavy-tailed values are compared on a log scale, so 100 vs 1,000 rows
# counts as much as 10,000 vs 100,000.
LOG_META = {"n_rows", "n_cols", "n_numeric", "n_categorical", "mean_cardinality", "mean_skewness", "pct_missing"}

TASK_PENALTY = 1.0  # added to the distance when a past dataset has a different task
WEIGHT_SOFTENER = 0.1  # neighbour weight = 1 / (distance + this)
# Scores whose size depends on the dataset; simplicity and explainability depend only on the pipeline.
RELATIVE_SCORES = ["performance", "efficiency", "robustness"]
# If every pipeline on a dataset scores within this of each other (0..1 scale), the differences
# are noise (very easy data, or timings too small to matter), so they are treated as a tie.
TIE_SPREAD = 0.02


# ------------------------------------------------------------------ fingerprints
def _check_columns(experience_db: pd.DataFrame) -> None:
    needed = ["dataset_id", "task"] + NUMERIC_META + SPEC_KEYS + SCORE_KEYS
    missing = [column for column in needed if column not in experience_db.columns]
    if missing:
        raise ValueError(f"experience database is missing columns: {missing}")


def _transform(table: pd.DataFrame) -> np.ndarray:
    """Meta-feature table -> float matrix (log scale where needed, unknown = NaN)."""
    values = table[NUMERIC_META].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    for column, name in enumerate(NUMERIC_META):
        if name in LOG_META:
            values[:, column] = np.log1p(np.clip(values[:, column], 0, None))
    return values


def _fingerprints(experience_db: pd.DataFrame) -> pd.DataFrame:
    return experience_db.drop_duplicates(subset="dataset_id")[["dataset_id", "task"] + NUMERIC_META].reset_index(drop=True)


def find_similar_datasets(
    meta_features: dict,
    experience_db: pd.DataFrame,
    k: int = 5,
    exclude_dataset_id: str | None = None,
) -> pd.DataFrame:
    """Return the k most similar past datasets with a `distance` column (smallest first).

    exclude_dataset_id leaves one dataset out, for leave-one-dataset-out testing.
    """
    _check_columns(experience_db)
    prints = _fingerprints(experience_db)
    if exclude_dataset_id is not None:
        prints = prints[prints["dataset_id"] != exclude_dataset_id].reset_index(drop=True)
    if prints.empty:
        return prints.assign(distance=pd.Series(dtype=float))

    past = _transform(prints)
    query = _transform(pd.DataFrame([{name: meta_features.get(name) for name in NUMERIC_META}]))

    with np.errstate(all="ignore"):
        centre = np.nanmean(past, axis=0)
        spread = np.nanstd(past, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)

    # Unknown values (NaN) become 0, meaning "average", so they neither help nor hurt.
    past = np.nan_to_num((past - centre) / spread, nan=0.0)
    query = np.nan_to_num((query - centre) / spread, nan=0.0)

    distance = np.sqrt(((past - query) ** 2).mean(axis=1))
    distance = distance + np.where(prints["task"].to_numpy() == meta_features.get("task"), 0.0, TASK_PENALTY)

    result = prints.assign(distance=distance).sort_values("distance", kind="stable").head(k)
    return result.reset_index(drop=True)


# -------------------------------------------------------------------- predictions
def _spec_key(row) -> tuple:
    return tuple(row[key] for key in SPEC_KEYS)


def predict_pipeline_scores(
    meta_features: dict,
    experience_db: pd.DataFrame,
    k: int = 5,
    candidates: list[dict] | None = None,
    exclude_dataset_id: str | None = None,
    relative: bool = True,
) -> pd.DataFrame:
    """Predict the five scores of each pipeline on the new dataset.

    With relative=True (default), performance, efficiency and robustness are rescaled
    within each past dataset to 0..1 before averaging (0 = worst pipeline tried there,
    1 = best), so the prediction says "how much better than the others", not an absolute score.

    Returns one row per pipeline: spec columns, five predicted scores, and
    `n_neighbours` (how many neighbours had results for it). Only pipelines that
    suit this dataset (generate_candidates, or `candidates` if given) and that at
    least one neighbour has tried are returned.
    """
    columns = SPEC_KEYS + SCORE_KEYS + ["n_neighbours"]
    neighbours = find_similar_datasets(meta_features, experience_db, k, exclude_dataset_id)
    if neighbours.empty:
        return pd.DataFrame(columns=columns)

    allowed = {_spec_key(spec) for spec in (candidates if candidates is not None else generate_candidates(meta_features))}
    weights = 1.0 / (neighbours.set_index("dataset_id")["distance"] + WEIGHT_SOFTENER)

    rows = experience_db[experience_db["dataset_id"].isin(weights.index)].copy()
    rows = rows[[tuple(r) in allowed for r in rows[SPEC_KEYS].itertuples(index=False, name=None)]]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["_weight"] = rows["dataset_id"].map(weights)
    if relative:
        for score in RELATIVE_SCORES:
            low = rows.groupby("dataset_id")[score].transform("min")
            high = rows.groupby("dataset_id")[score].transform("max")
            span = (high - low).where((high - low) >= TIE_SPREAD)
            rows[score] = ((rows[score] - low) / span).fillna(0.5)

    records = []
    for spec_values, group in rows.groupby(SPEC_KEYS, sort=False):
        total = group["_weight"].sum()
        record = dict(zip(SPEC_KEYS, spec_values))
        for score in SCORE_KEYS:
            record[score] = float((group[score] * group["_weight"]).sum() / total)
        record["n_neighbours"] = int(group["dataset_id"].nunique())
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def _utility(table: pd.DataFrame, weights: dict[str, float] | None) -> pd.Series:
    weights = weights or {key: 1.0 for key in SCORE_KEYS}
    total = sum(weights.get(key, 0.0) for key in SCORE_KEYS)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return sum(table[key] * weights.get(key, 0.0) for key in SCORE_KEYS) / total


def recommend(
    meta_features: dict,
    experience_db: pd.DataFrame,
    k: int = 5,
    n: int = 5,
    weights: dict[str, float] | None = None,
    candidates: list[dict] | None = None,
    exclude_dataset_id: str | None = None,
    relative: bool = True,
    min_gain: float = 0.0,
) -> list[dict]:
    """Return up to n ranked pipeline specs learned from similar datasets.

    min_gain (0 = off): a pipeline other than the default is only kept if its predicted
    performance beats the default pipeline's predicted performance by at least this much
    (on the 0..1 relative scale). It stops the recommender adding steps for no real gain.
    Ignored when the default pipeline has no prediction.

    The Pareto front of the predicted scores comes first, ordered by weighted average
    (equal weights unless `weights` is given). If the front has fewer than n pipelines,
    the next best by weighted average follow. With nothing to learn from, the default
    pipeline is returned.
    """
    predicted = predict_pipeline_scores(meta_features, experience_db, k, candidates, exclude_dataset_id, relative)
    if predicted.empty:
        return [dict(DEFAULT_SPEC)]

    if min_gain > 0:
        is_default = predicted.apply(lambda row: _spec_key(row) == _spec_key(DEFAULT_SPEC), axis=1)
        if is_default.any():
            floor = float(predicted.loc[is_default, "performance"].iloc[0]) + min_gain
            predicted = predicted[is_default | (predicted["performance"] >= floor)].reset_index(drop=True)

    predicted = predicted.assign(_utility=_utility(predicted, weights))
    front_keys = {_spec_key(row) for _, row in pareto_front(predicted).iterrows()}
    on_front = predicted.apply(lambda row: _spec_key(row) in front_keys, axis=1)

    ranked = pd.concat(
        [
            predicted[on_front].sort_values("_utility", ascending=False, kind="stable"),
            predicted[~on_front].sort_values("_utility", ascending=False, kind="stable"),
        ]
    ).head(n)
    return [{key: row[key] for key in SPEC_KEYS} for _, row in ranked.iterrows()]