"""Value tests for meta_recommender.py, using a small hand-made experience database."""
import numpy as np
import pandas as pd
import pytest

from backend.meta_features import META_FEATURE_KEYS
from backend.meta_recommender import (
    DEFAULT_SPEC,
    SCORE_KEYS,
    SPEC_KEYS,
    find_similar_datasets,
    predict_pipeline_scores,
    recommend,
)

SPEC_A = dict(imputer="knn", encoder="onehot", scaler="robust", outlier="iqr_clip", imbalance="none")
SPEC_B = dict(imputer="median", encoder="onehot", scaler="standard", outlier="none", imbalance="none")
SPEC_C = dict(imputer="mean", encoder="ordinal", scaler="minmax", outlier="none", imbalance="none")


def meta(**changes):
    base = dict(
        n_rows=1000, n_cols=10, n_numeric=6, n_categorical=4, categorical_ratio=0.4,
        pct_missing=10.0, pct_rows_with_missing=30.0, mean_cardinality=8.0,
        class_imbalance_ratio=0.5, mean_skewness=1.0, outlier_fraction=0.05,
        mean_abs_correlation=0.3, task="classification",
    )
    base.update(changes)
    return base


def scores(performance, efficiency=0.5, simplicity=0.5, robustness=0.5, explainability=0.5):
    return dict(performance=performance, efficiency=efficiency, simplicity=simplicity,
                robustness=robustness, explainability=explainability)


def experience_table(entries):
    """entries: list of (dataset_id, meta dict, [(spec, scores), ...])"""
    rows = []
    for dataset_id, m, results in entries:
        for spec, sc in results:
            rows.append({"dataset_id": dataset_id, **m, **spec, **sc, "raw_cv_mean": 0.0, "raw_cv_std": 0.0, "raw_fit_seconds": 0.0})
    return pd.DataFrame(rows)


@pytest.fixture
def db():
    messy = meta(pct_missing=40.0, pct_rows_with_missing=90.0, n_rows=500)
    clean = meta(pct_missing=0.0, pct_rows_with_missing=0.0, n_rows=50000, outlier_fraction=0.0)
    regression = meta(task="regression", class_imbalance_ratio=None)
    return experience_table([
        ("messy", messy, [(SPEC_A, scores(0.9)), (SPEC_B, scores(0.4)), (SPEC_C, scores(0.5))]),
        ("clean", clean, [(SPEC_A, scores(0.3)), (SPEC_B, scores(0.9)), (SPEC_C, scores(0.5))]),
        ("reg", regression, [(SPEC_A, scores(0.2)), (SPEC_B, scores(0.2)), (SPEC_C, scores(0.95))]),
    ])


# ------------------------------------------------------------- similar datasets
def test_experience_table_has_contract_columns(db):
    assert set(META_FEATURE_KEYS) <= set(db.columns)


def test_nearest_dataset_is_the_one_with_matching_traits(db):
    query = meta(pct_missing=35.0, pct_rows_with_missing=85.0, n_rows=800)
    similar = find_similar_datasets(query, db, k=3)
    assert similar.iloc[0]["dataset_id"] == "messy"
    assert similar["distance"].is_monotonic_increasing
    assert "distance" in similar.columns and len(similar) == 3


def test_identical_fingerprint_has_distance_zero(db):
    messy = db[db["dataset_id"] == "messy"].iloc[0]
    similar = find_similar_datasets({name: messy[name] for name in META_FEATURE_KEYS}, db, k=1)
    assert similar.iloc[0]["dataset_id"] == "messy"
    assert similar.iloc[0]["distance"] == pytest.approx(0.0, abs=1e-9)


def test_different_task_is_pushed_away(db):
    query = meta(task="regression", class_imbalance_ratio=None)
    assert find_similar_datasets(query, db, k=1).iloc[0]["dataset_id"] == "reg"


def test_k_limits_results_and_exclude_works(db):
    assert len(find_similar_datasets(meta(), db, k=2)) == 2
    without = find_similar_datasets(meta(), db, k=3, exclude_dataset_id="messy")
    assert "messy" not in without["dataset_id"].tolist()


def test_unknown_meta_features_do_not_crash(db):
    query = meta(mean_cardinality=None, class_imbalance_ratio=None, mean_skewness=np.nan)
    assert len(find_similar_datasets(query, db, k=3)) == 3


def test_missing_columns_give_clear_error():
    with pytest.raises(ValueError):
        find_similar_datasets(meta(), pd.DataFrame({"dataset_id": ["a"]}), k=1)


# ------------------------------------------------------------------ predictions
def test_predicted_scores_lean_towards_nearer_dataset(db):
    query = meta(pct_missing=40.0, pct_rows_with_missing=90.0, n_rows=500)
    predicted = predict_pipeline_scores(query, db, k=2)
    perf = predicted.set_index(SPEC_KEYS)["performance"]
    a = perf.loc[tuple(SPEC_A[key] for key in SPEC_KEYS)]
    # SPEC_A was the best on the nearest (messy) dataset and the worst on the other: prediction leans to the best
    assert 0.5 < a <= 1.0
    assert set(predicted.columns) >= set(SPEC_KEYS + SCORE_KEYS + ["n_neighbours"])
    assert (predicted["n_neighbours"] == 2).all()


def test_only_pipelines_valid_for_the_dataset_are_predicted(db):
    predicted = predict_pipeline_scores(meta(task="regression", class_imbalance_ratio=None), db, k=3)
    assert (predicted["imbalance"] == "none").all()
    restricted = predict_pipeline_scores(meta(), db, k=3, candidates=[SPEC_B])
    assert len(restricted) == 1 and restricted.iloc[0]["imputer"] == "median"


# --------------------------------------------------------------- recommendations
def test_messy_dataset_gets_the_pipeline_that_worked_on_messy_data(db):
    query = meta(pct_missing=40.0, pct_rows_with_missing=90.0, n_rows=500)
    assert recommend(query, db, k=1, n=3)[0] == SPEC_A


def test_clean_dataset_gets_the_pipeline_that_worked_on_clean_data(db):
    query = meta(pct_missing=0.0, pct_rows_with_missing=0.0, n_rows=50000, outlier_fraction=0.0)
    assert recommend(query, db, k=1, n=3)[0] == SPEC_B


def test_recommendations_are_valid_specs_and_limited_by_n(db):
    result = recommend(meta(), db, k=3, n=2)
    assert len(result) <= 2
    assert all(set(spec) == set(SPEC_KEYS) for spec in result)


def test_weights_change_the_ranking():
    m = meta()
    db = experience_table([("d", m, [
        (SPEC_A, scores(0.9, efficiency=0.1, simplicity=0.1, robustness=0.1, explainability=0.1)),
        (SPEC_B, scores(0.2, efficiency=0.9, simplicity=0.9, robustness=0.9, explainability=0.9)),
    ])])
    assert recommend(m, db, k=1, n=2, weights={"performance": 1.0})[0] == SPEC_A
    assert recommend(m, db, k=1, n=2, weights={"efficiency": 1.0})[0] == SPEC_B


def test_dominated_pipeline_is_ranked_after_the_pareto_front():
    m = meta()
    db = experience_table([("d", m, [
        (SPEC_A, scores(0.9, 0.9, 0.9, 0.9, 0.9)),
        (SPEC_B, scores(0.1, 0.1, 0.1, 0.1, 0.1)),  # beaten on every goal
        (SPEC_C, scores(0.95, 0.2, 0.9, 0.9, 0.9)),  # trade-off, stays on the front
    ])])
    result = recommend(m, db, k=1, n=3, weights={"performance": 1.0})
    assert result[-1] == SPEC_B
    assert SPEC_A in result[:2] and SPEC_C in result[:2]


def test_empty_experience_returns_default(db):
    assert recommend(meta(), db.iloc[0:0], k=3) == [DEFAULT_SPEC]
    assert recommend(meta(), db, k=3, exclude_dataset_id="messy", candidates=[dict(SPEC_B, scaler="none")]) == [DEFAULT_SPEC]


def test_scores_are_compared_within_each_dataset_not_across_them():
    # On "easy", every pipeline scores about 0.9 and SPEC_A is the best of them.
    # On "hard", every pipeline scores about 0.2 and SPEC_A is again the best of them.
    # Averaging raw scores would call SPEC_A "0.55"; relative scores correctly say "best".
    easy, hard = meta(n_rows=900), meta(n_rows=950)
    db = experience_table([
        ("easy", easy, [(SPEC_A, scores(0.92)), (SPEC_B, scores(0.88)), (SPEC_C, scores(0.89))]),
        ("hard", hard, [(SPEC_A, scores(0.25)), (SPEC_B, scores(0.15)), (SPEC_C, scores(0.18))]),
    ])
    relative = predict_pipeline_scores(meta(), db, k=2).set_index(SPEC_KEYS)["performance"]
    raw = predict_pipeline_scores(meta(), db, k=2, relative=False).set_index(SPEC_KEYS)["performance"]
    key = tuple(SPEC_A[name] for name in SPEC_KEYS)
    assert relative.loc[key] == pytest.approx(1.0)
    assert raw.loc[key] < 0.7
    assert recommend(meta(), db, k=2, n=1, weights={"performance": 1.0})[0] == SPEC_A


def test_relative_scores_stay_between_zero_and_one(db):
    predicted = predict_pipeline_scores(meta(), db, k=3)
    assert predicted[SCORE_KEYS].min().min() >= 0.0 and predicted[SCORE_KEYS].max().max() <= 1.0


def test_near_identical_scores_on_a_dataset_count_as_a_tie():
    # On "easy", every pipeline scores 0.998 to 0.999: the differences are noise, not information.
    easy = meta(n_rows=900)
    db = experience_table([("easy", easy, [(SPEC_A, scores(0.999)), (SPEC_B, scores(0.998)), (SPEC_C, scores(0.9985))])])
    predicted = predict_pipeline_scores(meta(), db, k=1)
    assert (predicted["performance"] == 0.5).all()
    # Real differences (here 0.1) are still used.
    db = experience_table([("real", easy, [(SPEC_A, scores(0.9)), (SPEC_B, scores(0.8)), (SPEC_C, scores(0.85))])])
    perf = predict_pipeline_scores(meta(), db, k=1).set_index(SPEC_KEYS)["performance"]
    assert perf.loc[tuple(SPEC_A[key] for key in SPEC_KEYS)] == pytest.approx(1.0)
    assert perf.loc[tuple(SPEC_B[key] for key in SPEC_KEYS)] == pytest.approx(0.0)


def test_min_gain_off_changes_nothing_and_huge_gain_leaves_only_the_default(db):
    query = meta()
    base = recommend(query, db, k=2, n=5)
    assert recommend(query, db, k=2, n=5, min_gain=0.0) == base
    strict = recommend(query, db, k=2, n=5, min_gain=5.0)
    assert all(spec == DEFAULT_SPEC for spec in strict)