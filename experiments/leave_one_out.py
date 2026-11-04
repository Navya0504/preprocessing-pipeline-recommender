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
from backend.meta_recommender import (  # noqa: E402
    DEFAULT_SPEC, RELATIVE_SCORES, SCORE_KEYS, SPEC_KEYS, TIE_SPREAD, accuracy_only, recommend,
)


def global_best_spec(others: pd.DataFrame, tested: list[dict], weights: dict | None, relative: bool) -> dict | None:
    """The single pipeline with the best average result over ALL other datasets (no dataset similarity).

    This is the "one fixed recipe for everybody" baseline: if the recommender cannot beat it,
    looking at what a new dataset is like adds nothing. Uses the same relative scoring as the recommender.
    """
    allowed = {tuple(spec[key] for key in SPEC_KEYS) for spec in tested}
    rows = others[[tuple(r) in allowed for r in others[SPEC_KEYS].itertuples(index=False, name=None)]].copy()
    if rows.empty:
        return None
    for score in (RELATIVE_SCORES if relative else []):
        low = rows.groupby("dataset_id")[score].transform("min")
        high = rows.groupby("dataset_id")[score].transform("max")
        span = (high - low).where((high - low) >= TIE_SPREAD)
        rows[score] = ((rows[score] - low) / span).fillna(0.5)
    w = weights or {key: 1.0 for key in SCORE_KEYS}
    rows["_u"] = sum(rows[key] * w.get(key, 0.0) for key in SCORE_KEYS)
    best = rows.groupby(SPEC_KEYS)["_u"].mean().idxmax()
    return dict(zip(SPEC_KEYS, best))


def leave_one_out(
    table: pd.DataFrame,
    focus: str = "equal",
    k: int = 3,
    n: int = 5,
    relative: bool | None = None,
    min_gain: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per dataset: how good the recommender's top pick was on that hidden dataset.

    Columns: dataset, global_best (one fixed pipeline for all), top_pick, top_n_avg, default, random (average of all tested
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
        fixed = global_best_spec(others, tested, weights, accuracy_only(weights) if relative is None else relative)
        fixed_row = held[(held[SPEC_KEYS] == pd.Series(fixed)).all(axis=1)]["utility"] if fixed else []
        rows.append({
            "dataset": dataset_id,
            "global_best": float(fixed_row.iloc[0]) if len(fixed_row) else float("nan"),
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
    scoring = parser.add_mutually_exclusive_group()
    scoring.add_argument("--absolute", action="store_true", help="force raw scores averaged across datasets")
    scoring.add_argument("--relative", action="store_true", help="force relative (per-dataset 0..1) scores")
    args = parser.parse_args()

    result = leave_one_out(pd.read_csv(args.results), args.focus, args.k, args.n, True if args.relative else (False if args.absolute else None), args.min_gain, verbose=True)
    print(f"Judged on: {'performance only' if args.focus == 'performance' else 'all five goals, equal weights'}\n")
    shown = result.rename(columns={"top_pick": "top pick", "top_n_avg": f"top {args.n} avg", "random": "all avg"}).round(3)
    shown["percentile"] = shown["percentile"].round(0)
    print(shown.to_string(index=False))
    if not result.empty:
        print(f"\nAverage over {len(result)} datasets:")
        print(f"  top pick utility   {result['top_pick'].mean():.3f}")
        print(f"  all pipelines avg  {result['random'].mean():.3f}   (what random choice would give)")
        print(f"  default pipeline   {result['default'].mean():.3f}")
        print(f"  one fixed best     {result['global_best'].mean():.3f}   (best average pipeline over the other datasets)")
        print(f"  best possible      {result['best'].mean():.3f}")
        print(f"  top pick beats on average {result['percentile'].mean():.0f}% of the tested pipelines")


if __name__ == "__main__":
    main()