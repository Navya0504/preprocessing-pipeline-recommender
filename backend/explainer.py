"""Plain-English explanations of why a pipeline was recommended.

Contract: see INTEGRATION_CONTRACT.md, section 3.
STATUS: real implementation. The INTERPRETABILITY ratings are still a judgement call
(1.0 = anyone can follow it, lower = harder to explain); keep them stable once experiments start.

explain() returns {"summary": str, "reasons": list[str], "tradeoffs": list[str]}:
    summary    one sentence naming every step of the chosen pipeline in everyday words
    reasons    why each step suits THIS dataset (its measured traits), plus which similar
               past datasets the advice is based on
    tradeoffs  what was given up compared with the other best-trade-off pipelines
"""
from __future__ import annotations

import math

import pandas as pd

# Higher = easier for a human to understand.
INTERPRETABILITY: dict[str, dict[str, float]] = {
    "imputer": {"mean": 1.0, "median": 1.0, "most_frequent": 1.0, "knn": 0.6, "drop_rows": 0.9},
    "encoder": {"onehot": 1.0, "ordinal": 0.8, "target": 0.5},
    "scaler": {"none": 1.0, "standard": 0.9, "minmax": 0.9, "robust": 0.8},
    "outlier": {"none": 1.0, "iqr_clip": 0.8, "isolation_forest": 0.4},
    "imbalance": {"none": 1.0, "class_weight": 0.8, "smote": 0.5},
}

SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"]
SCORE_KEYS = ["performance", "efficiency", "simplicity", "robustness", "explainability"]

GOAL_WORDS = {
    "performance": "accuracy",
    "efficiency": "speed",
    "simplicity": "simplicity",
    "robustness": "stability",
    "explainability": "explainability",
}
STEP_WORDS = {"imputer": "gap filling", "encoder": "text encoding", "scaler": "scaling",
              "outlier": "outlier handling", "imbalance": "class balancing"}

SUMMARY_PHRASES = {
    "imputer": {
        "mean": "fill gaps with the average (mean)",
        "median": "fill gaps with the middle value (median)",
        "most_frequent": "fill gaps with the most common value",
        "knn": "fill gaps using the most similar rows (KNN)",
        "drop_rows": "remove rows that have gaps",
    },
    "encoder": {
        "onehot": "turn text columns into yes/no columns (one-hot)",
        "ordinal": "turn text categories into numbers (ordinal)",
        "target": "replace each text category with its average outcome (target encoding)",
    },
    "scaler": {
        "none": "leave number scales unchanged",
        "standard": "standardise numbers (centre on 0, spread of 1)",
        "minmax": "squeeze numbers into the range 0 to 1 (min-max)",
        "robust": "scale numbers using the median and quartiles (robust scaling)",
    },
    "outlier": {
        "none": "leave extreme values alone",
        "iqr_clip": "cap extreme values (IQR clipping)",
        "isolation_forest": "remove the most unusual rows (isolation forest)",
    },
    "imbalance": {
        "none": "no class balancing",
        "class_weight": "give the rare class more weight when training",
        "smote": "create extra examples of the rare class (SMOTE)",
    },
}

CLOSE_ENOUGH = 0.03  # a score within this of the best counts as "among the best"


def explainability_score(spec: dict) -> float:
    """Return 0..1 (higher = easier for a human to understand)."""
    values = [
        INTERPRETABILITY[step][choice]
        for step, choice in spec.items()
        if step in INTERPRETABILITY and choice in INTERPRETABILITY[step]
    ]
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------- helpers
def _num(value) -> float | None:
    """A float, or None when the value is missing / not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _summary(spec: dict, is_classification: bool) -> str:
    steps = ["imputer", "encoder", "scaler", "outlier"] + (["imbalance"] if is_classification else [])
    parts = [SUMMARY_PHRASES[step].get(spec.get(step), f"{STEP_WORDS[step]}: {spec.get(step)}") for step in steps]
    return "Recommended pipeline: " + "; ".join(parts) + "."


def _imputer_reason(spec: dict, meta: dict) -> str:
    missing, rows = _num(meta.get("pct_missing")), _num(meta.get("pct_rows_with_missing"))
    if not missing:
        return "No values are missing, so the gap-filling method has almost no effect on this dataset."
    opening = f"{missing:.0f}% of the values are missing" + (f" (in {rows:.0f}% of the rows)." if rows is not None else ".")
    why = {
        "drop_rows": "Only some rows are affected, so removing them loses little data and avoids guessing.",
        "knn": "KNN fills each gap from the most similar rows, which works well when gaps are spread across many columns.",
        "median": "The median is not thrown off by extreme values, so it is a safe way to fill gaps.",
        "mean": "The mean is simple and works well when the values are fairly evenly spread.",
        "most_frequent": "The most common value is a safe fill that also works for text categories.",
    }
    return f"{opening} {why.get(spec['imputer'], '')}".strip()


def _encoder_reason(spec: dict, meta: dict) -> str:
    count = _num(meta.get("n_categorical"))
    if count is not None and count == 0:
        return "There are no text columns, so no encoding is needed."
    cardinality = _num(meta.get("mean_cardinality"))
    opening = ""
    if count is not None and cardinality is not None:
        opening = f"There are {count:.0f} text column(s) with about {cardinality:.0f} categories each on average. "
    why = {
        "onehot": "One-hot encoding keeps every category separate and is the easiest to understand (a column with many categories keeps only its 20 most common).",
        "ordinal": "Ordinal encoding keeps the table narrow, which helps when columns have many categories, but it implies an order that may not exist.",
        "target": "Target encoding replaces each category with its average outcome. It handles many categories well, is calculated so it does not leak the answer, but is harder to explain.",
    }
    return (opening + why.get(spec["encoder"], "")).strip()


def _scaler_reason(spec: dict, meta: dict) -> str:
    count = _num(meta.get("n_numeric"))
    if count is not None and count == 0:
        return "There are no numeric columns, so no scaling is needed."
    extreme = _num(meta.get("outlier_fraction"))
    why = {
        "none": "Scaling was skipped to keep the pipeline simple and easy to follow.",
        "standard": "Numbers sit on different scales, so standardising puts them on equal footing.",
        "minmax": "Min-max scaling keeps every number between 0 and 1, which is easy to read.",
        "robust": (
            f"About {extreme * 100:.0f}% of numeric values are extreme, so robust scaling (median and quartiles) is less thrown off by them."
            if extreme
            else "Robust scaling uses the median and quartiles, so extreme values pull on it less."
        ),
    }
    return why.get(spec["scaler"], "")


def _outlier_reason(spec: dict, meta: dict) -> str:
    count = _num(meta.get("n_numeric"))
    if count is not None and count == 0:
        return "There are no numeric columns, so there are no extreme values to handle."
    extreme = _num(meta.get("outlier_fraction"))
    share = f"{extreme * 100:.1f}%" if extreme is not None else "some"
    if spec["outlier"] == "none":
        if not extreme:
            return "No extreme values were found, so none needed handling."
        return f"Only {share} of numeric values are extreme, so no special handling was added, which keeps the pipeline simple."
    if spec["outlier"] == "iqr_clip":
        return f"{share} of numeric values sit far from the rest. Capping them limits their pull while keeping every row."
    return (f"{share} of numeric values sit far from the rest. The most unusual rows (about 5%) are left out "
            "during training only. This is powerful but harder to explain.")


def _imbalance_reason(spec: dict, meta: dict) -> str | None:
    ratio = _num(meta.get("class_imbalance_ratio"))
    if ratio is None:
        return None
    size = f"The smallest class is {ratio * 100:.0f}% the size of the largest."
    if spec["imbalance"] == "none":
        if ratio >= 0.8:
            return f"{size} Classes are fairly balanced, so no balancing was needed."
        return f"{size} No balancing was chosen, which keeps the pipeline simple."
    if spec["imbalance"] == "class_weight":
        return f"{size} Giving the smaller class more weight during training helps without changing your data."
    return f"{size} SMOTE creates extra examples of the smaller class during training only, so the model sees both classes more evenly."


def _neighbour_reason(neighbours: pd.DataFrame | None) -> str | None:
    if neighbours is None or neighbours.empty or "dataset_id" not in neighbours.columns:
        return None
    names = ", ".join(str(name) for name in neighbours["dataset_id"].head(3))
    return f"This advice is based on the most similar past datasets: {names} (closest first)."


def _differences(chosen: dict, other: pd.Series) -> str:
    changed = [f"{STEP_WORDS[step]} = {other[step]}" for step in SPEC_KEYS if step in other and other[step] != chosen.get(step)]
    return ", ".join(changed) if changed else "the same steps"


def _tradeoffs(chosen: dict, scores: dict, front: pd.DataFrame | None) -> list[str]:
    if not scores or any(_num(scores.get(goal)) is None for goal in SCORE_KEYS):
        return ["Scores for the chosen pipeline were not provided, so trade-offs cannot be shown."]
    if front is None or front.empty or not set(SCORE_KEYS) <= set(front.columns):
        return ["No alternative pipelines were available to compare against."]

    notes, strong = [], []
    for goal in SCORE_KEYS:
        mine = float(scores[goal])
        best_row = front.loc[front[goal].idxmax()]
        best = float(best_row[goal])
        if best - mine > CLOSE_ENOUGH:
            notes.append(
                f"Gave up some {GOAL_WORDS[goal]}: this pipeline scores {mine:.2f}, while another top-trade-off pipeline "
                f"({_differences(chosen, best_row)}) scores {best:.2f}."
            )
        else:
            strong.append(GOAL_WORDS[goal])
    if strong:
        notes.insert(0, "Among the best of the options on: " + ", ".join(strong) + ".")
    notes.append(
        f"{len(front)} pipeline(s) were not beaten on all five goals; this one was picked by the weighted average of the five scores."
    )
    return notes


# -------------------------------------------------------------------------- main
def explain(
    meta_features: dict,
    chosen_spec: dict,
    scores: dict,
    neighbours: pd.DataFrame,
    front: pd.DataFrame,
) -> dict:
    """Return {"summary": str, "reasons": list[str], "tradeoffs": list[str]}."""
    is_classification = meta_features.get("task", "classification") == "classification"

    reasons = [
        _imputer_reason(chosen_spec, meta_features),
        _encoder_reason(chosen_spec, meta_features),
        _scaler_reason(chosen_spec, meta_features),
        _outlier_reason(chosen_spec, meta_features),
    ]
    if is_classification:
        reasons.append(_imbalance_reason(chosen_spec, meta_features))
    reasons.append(_neighbour_reason(neighbours))

    return {
        "summary": _summary(chosen_spec, is_classification),
        "reasons": [reason for reason in reasons if reason],
        "tradeoffs": _tradeoffs(chosen_spec, scores or {}, front),
    }