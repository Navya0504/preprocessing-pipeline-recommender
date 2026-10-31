"""Build the experience database: run candidate pipelines on every dataset.

Run from the project root:
    python experiments/build_experience_db.py

It reads experiments/data/datasets.csv (columns: file,target,task,drop) and writes
experiments/results.csv, one row per (dataset, pipeline). Results are saved after
each dataset, and datasets already in results.csv are skipped, so you can stop
(Ctrl+C) and run it again to continue.

Options:
    --max-rows 2000        use at most this many rows per dataset (random sample, fixed seed)
    --max-candidates 60    evaluate this many pipelines per dataset (random sample of the
                           pruned grid, fixed seed; the default pipeline is always included)
    --only NAME            run just one dataset (file name without .csv), for a quick test
    --redo                 with --only: delete that dataset's old rows and run it again
    --fresh                ignore an existing results.csv and start over
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.evaluator import RAW_KEYS, SCORE_KEYS, SPEC_KEYS, evaluate_pipeline  # noqa: E402
from backend.meta_features import META_FEATURE_KEYS, extract_meta_features  # noqa: E402
from backend.pipeline_space import generate_candidates  # noqa: E402

COLUMNS = ["dataset_id"] + META_FEATURE_KEYS + SPEC_KEYS + SCORE_KEYS + RAW_KEYS
DEFAULT_SPEC = {"imputer": "median", "encoder": "onehot", "scaler": "standard", "outlier": "none", "imbalance": "none"}
SEED = 42


def load_dataset(data_dir: Path, row: pd.Series, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(data_dir / row["file"], encoding_errors="replace", low_memory=False)
    drop_text = row["drop"] if pd.notna(row.get("drop")) else ""
    drop = [name for name in str(drop_text).split("|") if name]
    missing = [name for name in drop + [row["target"]] if name not in df.columns]
    if missing:
        raise ValueError(f"columns not found in {row['file']}: {missing}")
    df = df.drop(columns=drop)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED).reset_index(drop=True)
    return df


def pick_candidates(meta: dict, max_candidates: int) -> list[dict]:
    grid = generate_candidates(meta)
    if len(grid) <= max_candidates:
        return grid
    chosen = random.Random(SEED).sample(grid, max_candidates)
    if DEFAULT_SPEC in grid and DEFAULT_SPEC not in chosen:
        chosen[0] = DEFAULT_SPEC
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "experiments" / "data"))
    parser.add_argument("--out", default=str(ROOT / "experiments" / "results.csv"))
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--only", default=None)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    data_dir, out_path = Path(args.data_dir), Path(args.out)
    sheet = pd.read_csv(data_dir / "datasets.csv")
    if args.only:
        sheet = sheet[sheet["file"].str.replace(".csv", "", regex=False) == args.only]
        if sheet.empty:
            sys.exit(f"no dataset named '{args.only}' in datasets.csv")

    done = pd.DataFrame(columns=COLUMNS)
    if out_path.exists() and not args.fresh:
        done = pd.read_csv(out_path)
    if args.redo and args.only:
        done = done[done["dataset_id"] != args.only].reset_index(drop=True)
    finished = set(done["dataset_id"])

    for number, (_, row) in enumerate(sheet.iterrows(), start=1):
        dataset_id = row["file"].replace(".csv", "")
        if dataset_id in finished:
            print(f"[{number}/{len(sheet)}] {dataset_id}: already done, skipping")
            continue
        try:
            df = load_dataset(data_dir, row, args.max_rows)
            meta = extract_meta_features(df, row["target"], row["task"])
        except Exception as error:
            print(f"[{number}/{len(sheet)}] {dataset_id}: SKIPPED ({error})")
            continue

        specs = pick_candidates(meta, args.max_candidates)
        print(f"[{number}/{len(sheet)}] {dataset_id}: {len(df)} rows, {len(specs)} pipelines")
        started, rows = time.time(), []
        for i, spec in enumerate(specs, start=1):
            result = evaluate_pipeline(spec, df, row["target"], row["task"])
            rows.append({"dataset_id": dataset_id, **meta, **spec, **result})
            if i % 10 == 0 or i == len(specs):
                print(f"    {i}/{len(specs)} pipelines  ({time.time() - started:.0f}s)")
        table = pd.DataFrame(rows, columns=COLUMNS)
        failed = int(table["raw_cv_mean"].isna().sum())
        best = table["performance"].max()
        print(f"    done in {time.time() - started:.0f}s | failed pipelines: {failed} | best performance: {best:.3f}")

        done = table if done.empty else pd.concat([done, table], ignore_index=True)
        done.to_csv(out_path, index=False)  # saved after every dataset

    print(f"\nSaved {len(done)} rows for {done['dataset_id'].nunique()} datasets to {out_path}")


if __name__ == "__main__":
    main()