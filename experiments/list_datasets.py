"""List every CSV in experiments/data with its size and columns.

Run from the project root:   python experiments/list_datasets.py
Then paste the output into the chat so the datasets.csv file can be written.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

files = sorted(DATA_DIR.glob("*.csv"))
files = [f for f in files if f.name != "datasets.csv"]
if not files:
    print(f"No CSV files found in {DATA_DIR}")

for path in files:
    try:
        df = pd.read_csv(path, nrows=50000, encoding_errors="replace")
    except Exception as error:
        print(f"\n{path.name}: could not be read ({error})")
        continue
    print(f"\n{path.name}  |  {len(df)} rows  |  {df.shape[1]} columns")
    print("  columns:", ", ".join(map(str, df.columns)))