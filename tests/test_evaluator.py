"""Value tests for evaluator.py."""
import numpy as np
import pandas as pd
import pytest

from backend.evaluator import (
    RAW_KEYS,
    SCORE_KEYS,
    SPEC_KEYS,
    build_experience_db,
    efficiency_score,
    evaluate_many,
    evaluate_pipeline,
    performance_score,
    robustness_score,
    simplicity_score,
)
from backend.meta_features import META_FEATURE_KEYS

SIMPLE = dict(imputer="median", encoder="onehot", scaler="none", outlier="none", imbalance="none")
BUSY = dict(imputer="knn", encoder="target", scaler="robust", outlier="isolation_forest", imbalance="smote")


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n)})
    df["label"] = (a + rng.normal(scale=0.3, size=n) > 0).astype(int)  # learnable from "a"
    df.loc[:9, "a"] = np.nan
    return df


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(1)
    n = 150
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": rng.normal(size=n), "y": 3 * a + rng.normal(scale=0.2, size=n)})


# ---------------------------------------------------------- score formulas
def test_performance_is_clipped_to_unit_range():
    assert performance_score(0.83) == pytest.approx(0.83)
    assert performance_score(-0.4) == 0.0
    assert performance_score(1.7) == 1.0


def test_efficiency_formula():
    assert efficiency_score(0.0) == 1.0
    assert efficiency_score(1.0) == pytest.approx(0.5)
    assert efficiency_score(9.0) == pytest.approx(0.1)
    assert efficiency_score(2.0) < efficiency_score(1.0)


def test_simplicity_formula():
    # imputer + encoder are always active -> 2 of 5 -> 0.6
    assert simplicity_score(SIMPLE) == pytest.approx(0.6)
    assert simplicity_score(BUSY) == pytest.approx(0.0)
    assert simplicity_score({**SIMPLE, "scaler": "standard"}) == pytest.approx(0.4)


def test_robustness_formula():
    assert robustness_score(0.0) == 1.0
    assert robustness_score(0.1) == pytest.approx(0.5)
    assert robustness_score(0.3) < robustness_score(0.05)


# ------------------------------------------------------ evaluate_pipeline
def test_scores_are_in_range_and_raw_values_present(clf_df):
    result = evaluate_pipeline(SIMPLE, clf_df, "label", "classification")
    for key in SCORE_KEYS:
        assert 0.0 <= result[key] <= 1.0
    for key in RAW_KEYS:
        assert key in result
    assert result["raw_fit_seconds"] > 0


def test_learnable_classification_scores_well(clf_df):
    result = evaluate_pipeline(SIMPLE, clf_df, "label", "classification")
    assert result["performance"] > 0.8


def test_regression_uses_r2(reg_df):
    result = evaluate_pipeline(SIMPLE, reg_df, "y", "regression")
    assert result["performance"] > 0.9


def test_performance_is_repeatable(clf_df):
    first = evaluate_pipeline(BUSY, clf_df, "label", "classification")
    second = evaluate_pipeline(BUSY, clf_df, "label", "classification")
    assert first["performance"] == pytest.approx(second["performance"])
    assert first["raw_cv_std"] == pytest.approx(second["raw_cv_std"])


def test_busier_pipeline_is_less_simple_and_less_explainable(clf_df):
    simple = evaluate_pipeline(SIMPLE, clf_df, "label", "classification")
    busy = evaluate_pipeline(BUSY, clf_df, "label", "classification")
    assert busy["simplicity"] < simple["simplicity"]
    assert busy["explainability"] < simple["explainability"]


def test_class_weight_option_runs(clf_df):
    result = evaluate_pipeline({**SIMPLE, "imbalance": "class_weight"}, clf_df, "label", "classification")
    assert np.isfinite(result["raw_cv_mean"])


def test_rows_with_missing_target_are_ignored(clf_df):
    df = clf_df.copy()
    df["label"] = df["label"].astype(float)
    df.loc[:4, "label"] = np.nan
    assert np.isfinite(evaluate_pipeline(SIMPLE, df, "label", "classification")["raw_cv_mean"])


def test_single_class_target_gives_failed_result(clf_df):
    df = clf_df.copy()
    df["label"] = 1
    result = evaluate_pipeline(SIMPLE, df, "label", "classification")
    assert result["performance"] == 0.0 and result["efficiency"] == 0.0 and result["robustness"] == 0.0
    assert np.isnan(result["raw_cv_mean"])
    assert result["simplicity"] == pytest.approx(0.6)


def test_tiny_minority_class_does_not_crash(clf_df):
    df = clf_df.copy()
    df["label"] = 0
    df.loc[:2, "label"] = 1  # 3 minority rows -> folds reduced to 3
    result = evaluate_pipeline({**SIMPLE, "imbalance": "smote"}, df, "label", "classification")
    for key in SCORE_KEYS:
        assert 0.0 <= result[key] <= 1.0


def test_bad_target_name_raises(clf_df):
    with pytest.raises(ValueError):
        evaluate_pipeline(SIMPLE, clf_df, "nope", "classification")


# ------------------------------------------------ evaluate_many / experience DB
def test_evaluate_many_one_row_per_spec(clf_df):
    table = evaluate_many([SIMPLE, BUSY], clf_df, "label", "classification")
    assert len(table) == 2
    assert list(table.columns) == SPEC_KEYS + SCORE_KEYS + RAW_KEYS


def test_build_experience_db_writes_contract_columns(clf_df, reg_df, tmp_path, monkeypatch):
    # The full grid is hundreds of pipelines; use a small fixed set to keep the test fast.
    monkeypatch.setattr("backend.evaluator.generate_candidates", lambda meta: [SIMPLE, {**SIMPLE, "scaler": "standard"}])
    out = tmp_path / "results.csv"
    build_experience_db(
        {"clf": (clf_df, "label", "classification"), "reg": (reg_df, "y", "regression")}, str(out)
    )
    table = pd.read_csv(out)
    assert list(table.columns) == ["dataset_id"] + META_FEATURE_KEYS + SPEC_KEYS + SCORE_KEYS + RAW_KEYS
    assert set(table["dataset_id"]) == {"clf", "reg"}
    assert (table.groupby("dataset_id").size() == 2).all()
    assert (table[table["dataset_id"] == "reg"]["task"] == "regression").all()


def test_completely_empty_columns_are_ignored():
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame({"empty_num": np.nan, "c": rng.choice(["x", "y", "z"], n), "y": rng.normal(size=n)})
    df["y"] = df["y"] + (df["c"] == "x") * 3
    for scaler in ["none", "standard", "minmax", "robust"]:
        result = evaluate_pipeline({**SIMPLE, "scaler": scaler}, df, "y", "regression")
        assert np.isfinite(result["raw_cv_mean"]), f"scaler={scaler} failed on an empty column"