import os
import pandas as pd
from build_experience_db import tidy

meta = pd.read_csv(os.path.join("data", "datasets.csv"))
for _, r in meta.iterrows():
    df = pd.read_csv(os.path.join("data", r["file"]), sep=None,
                     engine="python", encoding_errors="replace")
    t = tidy(df)
    to_numeric = [c for c in df.columns
                  if df[c].dtype == "object" and pd.api.types.is_numeric_dtype(t[c])]
    new_gaps = int(t.isna().sum().sum() - df.isna().sum().sum())
    if to_numeric or new_gaps:
        print(r["file"], "| text->numeric:", to_numeric, "| new gaps:", new_gaps)
        