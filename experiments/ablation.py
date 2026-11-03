"""Which design choices matter? Switch one feature of the recommender off at a time.

Run from the project root:
    python experiments/ablation.py

Every variant is judged with leave-one-dataset-out on the same datasets, k fixed at 3,
and compared with the full recommender, dataset by dataset (wins/ties/losses and a paired test).
Variants (all decided in advance):
    full                 the recommender as it is (relative scores only when accuracy alone counts)
    always relative      per-dataset 0..1 scores for every goal setting
    always raw           raw scores averaged across datasets for every goal setting
    no tie rule          do not treat near-equal scores on a dataset as a tie (TIE_SPREAD = 0)
    no task penalty      past datasets of a different task count as equally similar (TASK_PENALTY = 0)
    equal neighbours     every neighbour counts the same, however close (huge WEIGHT_SOFTENER)
A variant that is WORSE than full shows that the feature helps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import backend.meta_recommender as mr  # noqa: E402
from compare import line, summarise  # noqa: E402
from leave_one_out import leave_one_out  # noqa: E402

VARIANTS = {
    "full": {},
    "always relative scores": {"relative": True},
    "always raw scores": {"relative": False},
    "no tie rule": {"TIE_SPREAD": 0.0},
    "no task penalty": {"TASK_PENALTY": 0.0},
    "equal neighbours": {"WEIGHT_SOFTENER": 1e6},
}


def run_variant(table: pd.DataFrame, settings: dict, focus: str, k: int, n: int) -> pd.DataFrame:
    constants = {key: value for key, value in settings.items() if key != "relative"}
    saved = {key: getattr(mr, key) for key in constants}
    try:
        for key, value in constants.items():
            setattr(mr, key, value)
        return leave_one_out(table, focus=focus, k=k, n=n, relative=settings.get("relative"))
    finally:
        for key, value in saved.items():
            setattr(mr, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "experiments" / "results.csv"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--first", type=int, default=0, help="report only on the first N datasets listed in datasets.csv")
    parser.add_argument("--last", type=int, default=0, help="report only on the last N datasets listed in datasets.csv")
    parser.add_argument("--tie", type=float, default=0.005)
    args = parser.parse_args()

    table = pd.read_csv(args.results)
    keep = None
    if args.first or args.last:
        names = [name.replace(".csv", "") for name in pd.read_csv(ROOT / "experiments" / "data" / "datasets.csv")["file"]]
        keep = set(names[:args.first] if args.first else names[-args.last:])
        print(f"Reporting only on {len(keep)} datasets ({'first' if args.first else 'last'} {len(keep)} in datasets.csv).")
    print(f"Results: {len(table)} rows, {table['dataset_id'].nunique()} datasets. k = {args.k}.\n")
    for focus, name in [("performance", "ACCURACY ONLY"), ("equal", "ALL FIVE GOALS (equal weights)")]:
        print(f"=== Judged on: {name} ===")
        outcomes = {label: run_variant(table, settings, focus, args.k, args.n) for label, settings in VARIANTS.items()}
        if keep is not None:
            outcomes = {label: res[res["dataset"].isin(keep)] for label, res in outcomes.items()}
        full = outcomes["full"].set_index("dataset")["top_pick"]
        print(f"  {'full recommender':<30} mean top pick {full.mean():.3f}")
        for label, result in outcomes.items():
            if label == "full":
                continue
            picks = result.set_index("dataset")["top_pick"]
            stats = summarise((picks - full).dropna(), args.tie)  # variant minus full: negative = the feature helps
            print(f"  {label:<30} mean top pick {picks.mean():.3f}")
            print(line("   vs full (variant - full)", stats))
        print()
    print("Read the 'vs full' lines as: negative mean diff = the variant is worse, so the removed feature helps.")


if __name__ == "__main__":
    main()