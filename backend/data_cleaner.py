"""Apply a recommended pipeline to a dataset and return the cleaned table.

Contract: see INTEGRATION_CONTRACT.md (cleaned-dataset download).
STATUS: real implementation.

    clean_dataset(df, target, spec, task, add_synthetic_rows=False) -> DataFrame

What the result is:
    - every feature column, filled, encoded, capped and scaled as the spec says
      (text columns become number columns, e.g. "city_Delhi"; names are kept readable)
    - the target column last, unchanged
    - rows removed by the spec (drop_rows, isolation_forest) are gone
    - SMOTE creates invented rows. They are NOT real data, so by default they are left out.
      Pass add_synthetic_rows=True to include them.
Rows with a missing target and columns that are completely empty are always removed,
exactly as the evaluator does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.pipeline_space import build_pipeline, validate_spec


def _readable(names) -> list[str]:
    """'numeric__age' -> 'age', 'categorical__city_Delhi' -> 'city_Delhi'; keeps names unique."""
    result, seen = [], {}
    for name in names:
        name = str(name).split("__", 1)[-1]
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return result


def clean_dataset(
    df: pd.DataFrame,
    target: str,
    spec: dict,
    task: str,
    add_synthetic_rows: bool = False,
) -> pd.DataFrame:
    """Return the dataset after applying the pipeline (see the module docstring)."""
    validate_spec(spec, task)
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in the dataset")

    data = df.dropna(subset=[target])
    X = data.drop(columns=[target]).dropna(axis=1, how="all")
    y = data[target]
    numeric_cols = list(X.select_dtypes(include="number").columns)
    categorical_cols = [column for column in X.columns if column not in numeric_cols]
    if not numeric_cols and not categorical_cols:
        raise ValueError("the dataset has no usable feature columns")

    used = dict(spec)
    if used["imbalance"] == "smote" and not add_synthetic_rows:
        used["imbalance"] = "none"

    pipeline = build_pipeline(used, numeric_cols, categorical_cols, task)
    cleaned, new_target = pipeline.fit_resample(X, y)

    names = _readable(pipeline.named_steps["prep"].get_feature_names_out())
    table = pd.DataFrame(np.asarray(cleaned), columns=names)
    table[target] = np.asarray(new_target)
    return table