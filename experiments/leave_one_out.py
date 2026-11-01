"""Does the recommender actually learn? Leave-one-dataset-out check on results.csv.

Run from the project root:
    python experiments/leave_one_out.py

For each dataset: hide it, let the recommender learn from the other datasets only,
then see how good its top pick really was on the hidden dataset. Only pipelines that
were tested on the hidden dataset can be judged, so the recommender is limited to those.

"Utility" = the weighted average of the five scores. Higher is better.
Default is equal weights; use --focus performance to judge on accuracy alone.
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


def leave_one_out(
    table: pd.DataFrame,
    focus: str = "equal",
    k: int = 3,
    n: int = 5,
    relative: bool = True,
    min_gain: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per dataset: how good the recommender's top pick was on that hidden dataset.

    Columns: dataset, top_pick, top_n_avg, default, random (average of all tested
    pipelines), best, percentile. Values are not rounded.
    """
    table = table.copy()
    weights = {"performance": 1.0} if focus == "performance" else None
    table["utility"] = table["performance"] if focus == "performance" else table[SCORE_KEYS].mean(axis=1)
    rows = []

    for dataset_id in table["dataset_id"].unique():
        held = table[table["dataset_id"] == dataset_id].reset_index(drop=True)
        others = table[table["dataset_id"] != dataset_id]
        if others["dataset_id"].nunique() == 0:
            continue
        meta = held.iloc[0][META_FEATURE_KEYS].to_dict()
        tested = held[SPEC_KEYS].drop_duplicates().to_dict("records")

        picks = recommend(meta, others, k=k, n=n, weights=weights, candidates=tested, relative=relative, min_gain=min_gain)
        merged = pd.DataFrame(picks).merge(held, on=SPEC_KEYS, how="left")
        if merged["utility"].isna().all():
            if verbose:
                print(f"{dataset_id}: no overlap with the recommendation, skipped")
            continue
        top1 = float(merged["utility"].iloc[0])
        default = held[(held[SPEC_KEYS] == pd.Series(DEFAULT_SPEC)).all(axis=1)]["utility"]
        rows.append({
            "dataset": dataset_id,
            "top_pick": top1,
            "top_n_avg": float(merged["utility"].mean()),
            "default": float(default.iloc[0]) if len(default) else float("nan"),
            "random": float(held["utility"].mean()),
            "best": float(held["utility"].max()),
            "percentile": float((held["utility"] < top1).mean() * 100),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "experiments" / "results.csv"))
    parser.add_argument("--k", type=int, default=3, help="how many similar datasets to learn from")
    parser.add_argument("--n", type=int, default=5, help="how many pipelines the recommender returns")
    parser.add_argument("--focus", choices=["equal", "performance"], default="equal",
                        help="equal = all five goals count the same; performance = accuracy only")
    parser.add_argument("--min-gain", type=float, default=0.0, help="only add steps beyond the default if predicted accuracy gain is at least this")
    parser.add_argument("--absolute", action="store_true",
                        help="average raw scores across datasets instead of relative ones (older behaviour)")
    args = parser.parse_args()

    result = leave_one_out(pd.read_csv(args.results), args.focus, args.k, args.n, not args.absolute, args.min_gain, verbose=True)
    print(f"Judged on: {'performance only' if args.focus == 'performance' else 'all five goals, equal weights'}\n")
    shown = result.rename(columns={"top_pick": "top pick", "top_n_avg": f"top {args.n} avg", "random": "all avg"}).round(3)
    shown["percentile"] = shown["percentile"].round(0)
    print(shown.to_string(index=False))
    if not result.empty:
        print(f"\nAverage over {len(result)} datasets:")
        print(f"  top pick utility   {result['top_pick'].mean():.3f}")
        print(f"  all pipelines avg  {result['random'].mean():.3f}   (what random choice would give)")
        print(f"  default pipeline   {result['default'].mean():.3f}")
        print(f"  best possible      {result['best'].mean():.3f}")
        print(f"  top pick beats on average {result['percentile'].mean():.0f}% of the tested pipelines")


if __name__ == "__main__":
    main()