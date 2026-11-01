"""Is the recommender really better, or just lucky? Wins/ties/losses, confidence interval and a significance test.

Run from the project root:
    python experiments/compare.py

For each way of judging (accuracy only, all five goals) it compares the recommender's top pick
with (a) the default pipeline and (b) a random pipeline, dataset by dataset, using
leave-one-dataset-out. It reports:
    wins / ties / losses     datasets where the recommender is better / about equal / worse
    mean difference + 95% CI how much better on average (bootstrap, fixed seed)
    Wilcoxon p-value         paired test across datasets; below 0.05 is usually called significant
Then it shows how the result changes with k (number of similar datasets learned from).

A "tie" means the difference is smaller than --tie (default 0.005). Datasets where both sides are
equal are left out of the Wilcoxon test, as is standard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from leave_one_out import leave_one_out  # noqa: E402

SEED = 42
FOCUS_NAMES = {"performance": "ACCURACY ONLY", "equal": "ALL FIVE GOALS (equal weights)"}


def summarise(differences: pd.Series, tie: float) -> dict:
    diff = differences.dropna().to_numpy(dtype=float)
    if len(diff) == 0:
        return {}
    rng = np.random.default_rng(SEED)
    boots = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    nonzero = diff[diff != 0]
    p_value = float("nan")
    if len(nonzero) >= 6:
        p_value = float(wilcoxon(nonzero).pvalue)
    return {
        "n": len(diff),
        "wins": int((diff > tie).sum()),
        "ties": int((np.abs(diff) <= tie).sum()),
        "losses": int((diff < -tie).sum()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p": p_value,
    }


def line(label: str, s: dict) -> str:
    if not s:
        return f"  {label:<26} no data"
    p = "n/a (too few differences)" if np.isnan(s["p"]) else f"{s['p']:.3f}"
    return (f"  {label:<26} wins {s['wins']:>2} | ties {s['ties']:>2} | losses {s['losses']:>2} | "
            f"mean diff {s['mean_diff']:+.3f}  (95% CI {s['ci_low']:+.3f} to {s['ci_high']:+.3f}) | Wilcoxon p = {p}")


def verdict(s: dict) -> str:
    if not s or np.isnan(s["p"]):
        return "not enough data to say"
    if s["p"] < 0.05 and s["mean_diff"] > 0:
        return "clearly better (statistically significant)"
    if s["p"] < 0.05 and s["mean_diff"] < 0:
        return "clearly WORSE (statistically significant)"
    return "no clear difference yet (could be luck)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "experiments" / "results.csv"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--tie", type=float, default=0.005)
    parser.add_argument("--sweep", default="1,2,3,5,8", help="values of k to try, comma separated")
    args = parser.parse_args()

    table = pd.read_csv(args.results)
    print(f"Results: {len(table)} rows, {table['dataset_id'].nunique()} datasets. Recommender learns from k={args.k} similar datasets.\n")

    for focus in ["performance", "equal"]:
        result = leave_one_out(table, focus=focus, k=args.k, n=args.n)
        print(f"=== Judged on: {FOCUS_NAMES[focus]} ({len(result)} datasets) ===")
        vs_default = summarise(result["top_pick"] - result["default"], args.tie)
        vs_random = summarise(result["top_pick"] - result["random"], args.tie)
        print(line("recommender vs default", vs_default))
        print(f"    -> {verdict(vs_default)}")
        print(line("recommender vs random", vs_random))
        print(f"    -> {verdict(vs_random)}")
        print()

    print("=== Minimum-gain rule: only add steps beyond the default if predicted accuracy gain >= X (k fixed) ===")
    print("Decided before looking: X in 0, 0.1, 0.2, 0.3. All are shown; none is picked on these results.")
    rows = []
    for gain in [0.0, 0.1, 0.2, 0.3]:
        row = {"min_gain": gain}
        for focus in ["performance", "equal"]:
            result = leave_one_out(table, focus=focus, k=args.k, n=args.n, min_gain=gain)
            diff = result["top_pick"] - result["default"]
            stats = summarise(diff, args.tie)
            row[f"{focus}: top pick"] = round(result["top_pick"].mean(), 3)
            row[f"{focus}: W/T/L vs default"] = f"{stats['wins']}/{stats['ties']}/{stats['losses']}"
            row[f"{focus}: p"] = "n/a" if np.isnan(stats["p"]) else round(stats["p"], 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    ks = [int(x) for x in args.sweep.split(",") if x.strip()]
    print("=== How many similar datasets to learn from (k)? Mean utility of the top pick ===")
    rows = []
    for k in ks:
        row = {"k": k}
        for focus in ["performance", "equal"]:
            result = leave_one_out(table, focus=focus, k=k, n=args.n)
            row[f"{focus}: top pick"] = round(result["top_pick"].mean(), 3)
            row[f"{focus}: default"] = round(result["default"].mean(), 3)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nNote: choosing the best k on the same datasets is slightly optimistic. Treat this table as a sensitivity check, "
          "and keep the k you chose before looking (default 3) for any reported result.")


if __name__ == "__main__":
    main()