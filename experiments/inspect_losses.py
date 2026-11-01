"""Why does the recommender lose to the default? Shows the picked pipeline vs the default on each losing dataset.

Run from the project root:
    python experiments/inspect_losses.py

For every dataset where the top pick scores clearly below the default (equal weights, margin --margin),
it prints the pipeline the recommender chose and the five REAL scores of the pick and of the default.
The last lines count which choices the recommender picks most often on the losing datasets.
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
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.03)
    args = parser.parse_args()

    table = pd.read_csv(args.results)
    table["utility"] = table[SCORE_KEYS].mean(axis=1)
    short = {"performance": "perf", "efficiency": "eff", "simplicity": "simp", "robustness": "rob", "explainability": "expl"}
    picked_specs = []

    for dataset_id in table["dataset_id"].unique():
        held = table[table["dataset_id"] == dataset_id].reset_index(drop=True)
        others = table[table["dataset_id"] != dataset_id]
        meta = held.iloc[0][META_FEATURE_KEYS].to_dict()
        tested = held[SPEC_KEYS].drop_duplicates().to_dict("records")
        pick = recommend(meta, others, k=args.k, n=1, candidates=tested)[0]
        mine = held[(held[SPEC_KEYS] == pd.Series(pick)).all(axis=1)]
        default = held[(held[SPEC_KEYS] == pd.Series(DEFAULT_SPEC)).all(axis=1)]
        if mine.empty or default.empty:
            continue
        mine, default = mine.iloc[0], default.iloc[0]
        if default["utility"] - mine["utility"] <= args.margin:
            continue
        picked_specs.append(pick)
        print(f"\n{dataset_id}  (task: {meta['task']}, utility pick {mine['utility']:.3f} vs default {default['utility']:.3f})")
        print("  picked :", " / ".join(pick[key] for key in SPEC_KEYS))
        print("           " + "  ".join(f"{short[s]} {mine[s]:.2f}" for s in SCORE_KEYS))
        print("  default:", " / ".join(DEFAULT_SPEC[key] for key in SPEC_KEYS))
        print("           " + "  ".join(f"{short[s]} {default[s]:.2f}" for s in SCORE_KEYS))

    if picked_specs:
        print(f"\nChoices made on the {len(picked_specs)} losing datasets:")
        frame = pd.DataFrame(picked_specs)
        for key in SPEC_KEYS:
            counts = frame[key].value_counts()
            print(f"  {key:<10} " + ", ".join(f"{name} x{count}" for name, count in counts.items()))
    else:
        print("No dataset where the pick loses to the default by that margin.")


if __name__ == "__main__":
    main()