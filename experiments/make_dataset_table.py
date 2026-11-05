import os
import pandas as pd

DATA_DIR = r"C:\Project\preprocessing-pipeline-recommender\experiments\data"

meta = pd.read_csv(os.path.join(DATA_DIR, "datasets.csv"))
rows = []

for _, r in meta.iterrows():
    path = os.path.join(DATA_DIR, str(r["file"]))
    out = {"dataset": r["file"], "task": r["task"], "target": r["target"]}
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")

        drops = []
        if pd.notna(r.get("drop")):
            drops = [
                c.strip()
                for c in str(r["drop"]).replace("|", ";").replace(",", ";").split(";")
                if c.strip()
            ]
        df = df.drop(columns=[c for c in drops if c in df.columns])

        df = df.dropna(subset=[r["target"]])
        df = df.dropna(axis=1, how="all")

        X = df.drop(columns=[r["target"]])
        num = X.select_dtypes("number").shape[1]

        out["rows"] = len(df)
        out["features"] = X.shape[1]
        out["numeric"] = num
        out["categorical"] = X.shape[1] - num
        out["missing_%"] = round(X.isna().sum().sum() / X.size * 100, 2)

        if str(r["task"]).lower().startswith("class"):
            vc = df[r["target"]].value_counts()
            out["classes"] = len(vc)
            out["imbalance_ratio"] = round(vc.max() / vc.min(), 2)
    except Exception as e:
        out["error"] = str(e)[:80]
    rows.append(out)

table = pd.DataFrame(rows)
table.to_csv(os.path.join(DATA_DIR, "..", "dataset_table.csv"), index=False)
table.to_excel(os.path.join(DATA_DIR, "..", "dataset_table.xlsx"), index=False)
print(table.to_string())