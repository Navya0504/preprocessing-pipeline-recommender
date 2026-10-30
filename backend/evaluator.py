"""Multi-objective evaluation of pipelines.

Contract: see INTEGRATION_CONTRACT.md, section 2.4, 2.5 and 3.
STATUS: real implementation.

THE FIVE OBJECTIVES (every score is between 0 and 1, higher is better)
    performance    mean 5-fold cross-validated score of ONE fixed reference model
                   (balanced accuracy for classification, R^2 for regression),
                   clipped to [0, 1].
    efficiency     1 / (1 + s), where s = mean seconds to fit the whole pipeline
                   (preprocessing + model) on one training fold.
    simplicity     1 - (active steps / 5). A step is active unless it is "none".
                   Imputer and encoder are always active, so the best possible
                   simplicity is 0.6.
    robustness     1 / (1 + 10 * std), where std = standard deviation of the
                   fold scores. Steady results across folds score near 1.
    explainability mean human-understandability rating (0..1) of the chosen
                   methods, from the table in explainer.py.

Fairness rules: the reference model, the random seed and the folds are fixed, and
all preprocessing is fitted inside each training fold only, so no test data leaks.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from backend.explainer import explainability_score
from backend.meta_features import META_FEATURE_KEYS, extract_meta_features
from backend.pipeline_space import build_pipeline, generate_candidates, uses_class_weight, with_model

SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"]
SCORE_KEYS = ["performance", "efficiency", "simplicity", "robustness", "explainability"]
RAW_KEYS = ["raw_cv_mean", "raw_cv_std", "raw_fit_seconds"]

ROBUSTNESS_STRENGTH = 10.0  # how fast robustness drops as fold-to-fold spread grows


# ------------------------------------------------------ the five score formulas
def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def performance_score(cv_mean: float) -> float:
    """Mean cross-validated score, clipped to 0..1 (negative R^2 counts as 0)."""
    return _clip01(cv_mean)


def efficiency_score(fit_seconds: float) -> float:
    """1 / (1 + seconds): 0 s gives 1.0, 1 s gives 0.5, 9 s gives 0.1."""
    return _clip01(1.0 / (1.0 + max(0.0, fit_seconds)))


def simplicity_score(spec: dict) -> float:
    """1 - (active steps / 5), where a step set to 'none' is not active."""
    active = sum(1 for step in SPEC_KEYS if spec.get(step) != "none")
    return _clip01(1.0 - active / len(SPEC_KEYS))


def robustness_score(cv_std: float) -> float:
    """1 / (1 + 10 * std of fold scores): zero spread gives 1.0."""
    return _clip01(1.0 / (1.0 + ROBUSTNESS_STRENGTH * max(0.0, cv_std)))


# ------------------------------------------------------------------- internals
def _reference_model(task: str, spec: dict, random_state: int):
    """The one fixed model used to judge every pipeline. A scale-sensitive linear
    model is used on purpose, so that preprocessing choices visibly matter."""
    if task == "classification":
        weight = "balanced" if uses_class_weight(spec) else None
        return LogisticRegression(max_iter=1000, class_weight=weight, random_state=random_state)
    return Ridge(random_state=random_state)


def _failed_result(spec: dict) -> dict:
    """Result for a pipeline that could not be evaluated: worst scores, NaN raw values."""
    result = {key: 0.0 for key in SCORE_KEYS}
    result["simplicity"] = simplicity_score(spec)
    result["explainability"] = _clip01(explainability_score(spec))
    result.update({"raw_cv_mean": float("nan"), "raw_cv_std": float("nan"), "raw_fit_seconds": float("nan")})
    return result


def evaluate_pipeline(
    spec: dict,
    df: pd.DataFrame,
    target: str,
    task: str,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Return the five 0..1 scores (higher is better) plus raw_* values.

    A pipeline that cannot run on this dataset gets zeros for performance,
    efficiency and robustness, and NaN raw values, so it never reaches the Pareto front.
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")
    data = df.dropna(subset=[target])
    X = data.drop(columns=[target])
    y = data[target]
    numeric_cols = list(X.select_dtypes(include="number").columns)
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    try:
        if task == "classification":
            n_splits = min(cv, int(y.value_counts().min()))
            if y.nunique() < 2 or n_splits < 2:
                return _failed_result(spec)
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scoring = "balanced_accuracy"
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            scoring = "r2"

        pipeline = with_model(
            build_pipeline(spec, numeric_cols, categorical_cols, task),
            _reference_model(task, spec, random_state),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outcome = cross_validate(
                pipeline, X, y, cv=splitter, scoring=scoring, error_score="raise", n_jobs=1
            )
    except Exception:
        return _failed_result(spec)

    fold_scores = np.asarray(outcome["test_score"], dtype=float)
    cv_mean = float(fold_scores.mean())
    cv_std = float(fold_scores.std())
    fit_seconds = float(np.mean(outcome["fit_time"]))
    if not np.isfinite(cv_mean):
        return _failed_result(spec)

    return {
        "performance": performance_score(cv_mean),
        "efficiency": efficiency_score(fit_seconds),
        "simplicity": simplicity_score(spec),
        "robustness": robustness_score(cv_std),
        "explainability": _clip01(explainability_score(spec)),
        "raw_cv_mean": cv_mean,
        "raw_cv_std": cv_std,
        "raw_fit_seconds": fit_seconds,
    }


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
    datasets: dict[str, tuple[pd.DataFrame, str, str]], out_path: str, verbose: bool = False
) -> None:
    """Run all candidates on all datasets and write the experience database.

    datasets maps dataset_id -> (dataframe, target column, task).
    Output columns follow contract section 2.5:
    dataset_id + meta-features + spec + scores + raw values, one row per (dataset, pipeline).
    """
    columns = ["dataset_id"] + META_FEATURE_KEYS + SPEC_KEYS + SCORE_KEYS + RAW_KEYS
    tables = []
    for number, (dataset_id, (df, target, task)) in enumerate(datasets.items(), start=1):
        meta = extract_meta_features(df, target, task)
        table = evaluate_many(generate_candidates(meta), df, target, task)
        table.insert(0, "dataset_id", dataset_id)
        for key in META_FEATURE_KEYS:
            table[key] = meta[key]
        tables.append(table[columns])
        if verbose:
            print(f"[{number}/{len(datasets)}] {dataset_id}: {len(table)} pipelines evaluated")
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
    result.to_csv(out_path, index=False)