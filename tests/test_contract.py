"""Contract tests: every function must return the shape INTEGRATION_CONTRACT.md promises.

These pass against the stubs and must keep passing as you replace stubs with
real code. Add value-level tests (correct numbers) next to these as you build.
"""
import numpy as np
import pandas as pd
import pytest

from backend.evaluator import RAW_KEYS, SCORE_KEYS, SPEC_KEYS, evaluate_many, evaluate_pipeline
from backend.explainer import explain, explainability_score
from backend.meta_features import META_FEATURE_KEYS, extract_meta_features
from backend.meta_recommender import find_similar_datasets, recommend
from backend.pareto_selector import pareto_front, select_best
from backend.pipeline_space import build_pipeline, generate_candidates


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "age": rng.integers(18, 70, 100).astype(float),
            "income": rng.normal(50000, 15000, 100),
            "city": rng.choice(["A", "B", "C"], 100),
            "label": rng.choice([0, 1], 100, p=[0.8, 0.2]),
        }
    )
    data.loc[:9, "age"] = np.nan
    return data


def test_meta_features_has_all_keys(df):
    result = extract_meta_features(df, "label", "classification")
    assert set(result) == set(META_FEATURE_KEYS)
    assert result["task"] == "classification"
    assert result["n_rows"] == 100
    assert result["n_cols"] == 3


def test_meta_features_regression_has_no_imbalance(df):
    result = extract_meta_features(df, "income", "regression")
    assert result["class_imbalance_ratio"] is None


def test_candidates_are_valid_specs(df):
    meta = extract_meta_features(df, "label", "classification")
    candidates = generate_candidates(meta)
    assert len(candidates) >= 1
    for spec in candidates:
        assert set(spec) == set(SPEC_KEYS)


def test_build_pipeline_returns_object(df):
    spec = generate_candidates({"task": "classification"})[0]
    pipeline = build_pipeline(spec, ["age", "income"], ["city"], "classification")
    assert pipeline is not None


def test_evaluate_pipeline_scores_are_normalised(df):
    spec = generate_candidates({"task": "classification"})[0]
    result = evaluate_pipeline(spec, df, "label", "classification")
    for key in SCORE_KEYS:
        assert 0.0 <= result[key] <= 1.0
    for key in RAW_KEYS:
        assert key in result


def test_evaluate_many_one_row_per_spec(df):
    specs = generate_candidates({"task": "classification"})[:3]  # the full grid is slow to evaluate
    table = evaluate_many(specs, df, "label", "classification")
    assert len(table) == len(specs)
    assert set(SPEC_KEYS + SCORE_KEYS).issubset(table.columns)


def test_recommender_returns_specs(df):
    meta = extract_meta_features(df, "label", "classification")
    specs = generate_candidates(meta)[:6]  # the full grid is slow to evaluate
    experience = evaluate_many(specs, df, "label", "classification")
    similar = find_similar_datasets(meta, experience, k=2)
    assert "distance" in similar.columns
    recommended = recommend(meta, experience, k=2)
    assert all(set(spec) == set(SPEC_KEYS) for spec in recommended)


def test_pareto_front_removes_dominated_rows():
    table = pd.DataFrame(
        [
            dict(performance=0.9, efficiency=0.9, simplicity=0.9, robustness=0.9, explainability=0.9),
            dict(performance=0.5, efficiency=0.5, simplicity=0.5, robustness=0.5, explainability=0.5),
            dict(performance=0.95, efficiency=0.2, simplicity=0.9, robustness=0.9, explainability=0.9),
        ]
    )
    front = pareto_front(table)
    assert len(front) == 2
    assert 0.5 not in front["performance"].tolist()


def test_select_best_equal_weights():
    front = pd.DataFrame(
        [
            dict(performance=0.9, efficiency=0.9, simplicity=0.9, robustness=0.9, explainability=0.9),
            dict(performance=0.95, efficiency=0.2, simplicity=0.9, robustness=0.9, explainability=0.9),
        ]
    )
    assert select_best(front)["efficiency"] == 0.9


def test_explainability_score_in_range():
    spec = generate_candidates({"task": "classification"})[0]
    assert 0.0 <= explainability_score(spec) <= 1.0


def test_explain_returns_expected_keys(df):
    meta = extract_meta_features(df, "label", "classification")
    spec = generate_candidates(meta)[0]
    result = explain(meta, spec, {}, pd.DataFrame(), pd.DataFrame())
    assert set(result) == {"summary", "reasons", "tradeoffs"}
    assert isinstance(result["reasons"], list)