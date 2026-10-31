"""Does the recommender actually learn? Leave-one-dataset-out check on results.csv.

Run from the project root:
    python experiments/leave_one_out.py

For each dataset: hide it, let the recommender learn from the other datasets only,
then see how good its top pick really was on the hidden dataset. Only pipelines that
were tested on the hidden dataset can be judged, so the recommender is limited to those.

"Utility" = the average of the five scores (equal weights). Higher is better.
Percentile = share of the hidden dataset's tested pipelines that the top pick beats.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.meta_features import META_FEATURE_KEYS  # noqa: E402
from backend.meta_recommender import DEFAULT_SPEC, SCORE_KEYS, SPEC_KEYS, recommend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "experiments" / "results.csv"))
    parser.add_argument("--k", type=int, default=3, help="how many similar datasets to learn from")
    parser.add_argument("--n", type=int, default=5, help="how many pipelines the recommender returns")
    args = parser.parse_args()

    table = pd.read_csv(args.results)
    table["utility"] = table[SCORE_KEYS].mean(axis=1)
    report = []

    for dataset_id in table["dataset_id"].unique():
        held = table[table["dataset_id"] == dataset_id].reset_index(drop=True)
        others = table[table["dataset_id"] != dataset_id]
        if others["dataset_id"].nunique() == 0:
            continue
        meta = held.iloc[0][META_FEATURE_KEYS].to_dict()
        tested = held[SPEC_KEYS].drop_duplicates().to_dict("records")

        picks = recommend(meta, others, k=args.k, n=args.n, candidates=tested)
        merged = pd.DataFrame(picks).merge(held, on=SPEC_KEYS, how="left")
        if merged["utility"].isna().all():
            print(f"{dataset_id}: no overlap with the recommendation, skipped")
            continue
        top1 = float(merged["utility"].iloc[0])
        default = held[(held[SPEC_KEYS] == pd.Series(DEFAULT_SPEC)).all(axis=1)]["utility"]
        report.append({
            "dataset": dataset_id,
            "top pick": round(top1, 3),
            f"top {args.n} avg": round(float(merged["utility"].mean()), 3),
            "default": round(float(default.iloc[0]), 3) if len(default) else float("nan"),
            "all avg": round(float(held["utility"].mean()), 3),
            "best": round(float(held["utility"].max()), 3),
            "percentile": round(float((held["utility"] < top1).mean() * 100), 0),
        })

    result = pd.DataFrame(report)
    print(result.to_string(index=False))
    if not result.empty:
        print(f"\nAverage over {len(result)} datasets:")
        print(f"  top pick utility   {result['top pick'].mean():.3f}")
        print(f"  all pipelines avg  {result['all avg'].mean():.3f}   (what random choice would give)")
        print(f"  default pipeline   {result['default'].mean():.3f}")
        print(f"  best possible      {result['best'].mean():.3f}")
        print(f"  top pick beats on average {result['percentile'].mean():.0f}% of the tested pipelines")


if __name__ == "__main__":
    main()