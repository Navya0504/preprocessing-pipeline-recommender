"""Print every CSV in experiments/data/: rows, columns, column names, and the share of missing values.

Run from the project root:
    python experiments/show_headers.py
Paste the output so the new datasets can be added to datasets.csv.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

for path in sorted(DATA.glob("*.csv")):
    if path.name == "datasets.csv":
        continue
    try:
        df = pd.read_csv(path, encoding_errors="replace", low_memory=False, nrows=20000)
    except Exception as error:
        print(f"\n{path.name}: CANNOT READ ({error})")
        continue
    missing = df.isna().mean().mean() * 100
    print(f"\n{path.name}: {len(df)} rows (first 20000), {df.shape[1]} columns, {missing:.1f}% missing")
    print("  " + " | ".join(map(str, df.columns)))