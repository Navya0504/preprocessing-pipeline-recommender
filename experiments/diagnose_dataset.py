"""Show why a dataset's pipelines fail. Prints the target's health and the REAL error.

Run from the project root, using the file name without .csv:
    python experiments/diagnose_dataset.py car_prices
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from build_experience_db import DEFAULT_SPEC, load_dataset  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate  # noqa: E402

from backend.evaluator import _reference_model  # noqa: E402
from backend.pipeline_space import build_pipeline, with_model  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python experiments/diagnose_dataset.py <dataset name without .csv>")
    name = sys.argv[1]
    data_dir = ROOT / "experiments" / "data"
    sheet = pd.read_csv(data_dir / "datasets.csv")
    rows = sheet[sheet["file"].str.replace(".csv", "", regex=False) == name]
    if rows.empty:
        sys.exit(f"'{name}' is not in datasets.csv")
    row = rows.iloc[0]
    df = load_dataset(data_dir, row, 2000)
    target, task = row["target"], row["task"]

    print(f"Dataset {name}: {len(df)} rows, target '{target}', task {task}")
    y = df[target]
    print(f"Target type: {y.dtype}")
    print(f"Target values present: {int(y.notna().sum())} of {len(y)}")
    print(f"Target examples: {y.dropna().head(5).tolist()}")
    print("\nColumn types:")
    print(df.dtypes.astype(str).to_string())

    data = df.dropna(subset=[target])
    if data.empty:
        print("\nPROBLEM: every target value is missing. This file has no answers to learn from.")
        return
    X, y = data.drop(columns=[target]), data[target]
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    print(f"\nRunning the default pipeline once: {DEFAULT_SPEC}")
    try:
        pipe = with_model(build_pipeline(DEFAULT_SPEC, numeric, categorical, task), _reference_model(task, DEFAULT_SPEC, 42))
        splitter = StratifiedKFold(3, shuffle=True, random_state=42) if task == "classification" else KFold(3, shuffle=True, random_state=42)
        scoring = "balanced_accuracy" if task == "classification" else "r2"
        print("Fold scores:", cross_validate(pipe, X, y, cv=splitter, scoring=scoring, error_score="raise")["test_score"])
    except Exception:
        print("\nTHE REAL ERROR:\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()