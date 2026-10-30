"""Value tests for pipeline_space.py."""
import numpy as np
import pandas as pd
import pytest

from backend.pipeline_space import (
    ENCODERS,
    IMBALANCE_METHODS,
    IMPUTERS,
    OUTLIER_METHODS,
    SCALERS,
    SPEC_KEYS,
    IQRClipper,
    build_pipeline,
    generate_candidates,
    uses_class_weight,
    validate_spec,
    with_model,
)

NUM = ["age", "fare"]
CAT = ["sex", "port"]
BASE = dict(imputer="median", encoder="onehot", scaler="standard", outlier="none", imbalance="none")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "fare": rng.exponential(30, n),
            "sex": rng.choice(["m", "f"], n),
            "port": rng.choice(["a", "b", "c"], n),
        }
    )
    X.loc[:14, "age"] = np.nan
    X.loc[10:20, "port"] = np.nan
    X.loc[0, "fare"] = 5000.0
    y = pd.Series(rng.choice([0, 1], n, p=[0.85, 0.15]))
    return X, y


def _spec(**changes):
    return {**BASE, **changes}


# ------------------------------------------------------------ candidate grid
def test_full_grid_size_when_nothing_is_known():
    specs = generate_candidates({"task": "classification"})
    assert len(specs) == len(IMPUTERS) * len(ENCODERS) * len(SCALERS) * len(OUTLIER_METHODS) * len(IMBALANCE_METHODS)
    assert all(set(spec) == set(SPEC_KEYS) for spec in specs)
    assert len({tuple(spec.items()) for spec in specs}) == len(specs)  # no duplicates


def test_regression_has_no_imbalance_options():
    specs = generate_candidates({"task": "regression"})
    assert {spec["imbalance"] for spec in specs} == {"none"}


def test_balanced_classes_have_no_imbalance_options():
    specs = generate_candidates({"task": "classification", "class_imbalance_ratio": 0.95})
    assert {spec["imbalance"] for spec in specs} == {"none"}


def test_imbalanced_classes_keep_all_imbalance_options():
    specs = generate_candidates({"task": "classification", "class_imbalance_ratio": 0.2})
    assert {spec["imbalance"] for spec in specs} == set(IMBALANCE_METHODS)


def test_nothing_missing_collapses_imputers():
    specs = generate_candidates({"task": "classification", "pct_missing": 0.0})
    assert len({spec["imputer"] for spec in specs}) == 1


def test_no_categorical_columns_collapses_encoders():
    specs = generate_candidates({"task": "classification", "n_categorical": 0})
    assert len({spec["encoder"] for spec in specs}) == 1


def test_no_numeric_columns_removes_scaling_and_outliers():
    specs = generate_candidates({"task": "classification", "n_numeric": 0})
    assert {spec["scaler"] for spec in specs} == {"none"}
    assert {spec["outlier"] for spec in specs} == {"none"}


def test_no_outliers_removes_outlier_handling():
    specs = generate_candidates({"task": "classification", "outlier_fraction": 0.0})
    assert {spec["outlier"] for spec in specs} == {"none"}


def test_none_values_in_meta_features_keep_all_options():
    meta = {"task": "classification", "pct_missing": None, "n_categorical": None, "outlier_fraction": None}
    assert len(generate_candidates(meta)) == len(generate_candidates({"task": "classification"}))


# --------------------------------------------------------------- validation
def test_validate_rejects_bad_specs():
    with pytest.raises(ValueError):
        validate_spec({"imputer": "median"})
    with pytest.raises(ValueError):
        validate_spec(_spec(scaler="banana"))
    with pytest.raises(ValueError):
        validate_spec(_spec(imbalance="smote"), task="regression")
    with pytest.raises(ValueError):
        validate_spec(_spec(), task="clustering")


def test_uses_class_weight():
    assert uses_class_weight(_spec(imbalance="class_weight"))
    assert not uses_class_weight(_spec(imbalance="smote"))


def test_build_needs_columns():
    with pytest.raises(ValueError):
        build_pipeline(_spec(), [], [], "classification")


# ---------------------------------------------------- every option builds & runs
@pytest.mark.parametrize(
    "step,options",
    [
        ("imputer", IMPUTERS),
        ("encoder", ENCODERS),
        ("scaler", SCALERS),
        ("outlier", OUTLIER_METHODS),
        ("imbalance", IMBALANCE_METHODS),
    ],
)
def test_every_option_produces_clean_numeric_output(data, step, options):
    X, y = data
    for option in options:
        pipe = build_pipeline(_spec(**{step: option}), NUM, CAT, "classification")
        Xr, yr = pipe.fit_resample(X, y)
        assert len(Xr) == len(yr)
        assert not np.isnan(np.asarray(Xr, dtype=float)).any(), f"{step}={option} left missing values"


def test_pipeline_cleans_new_data_with_gaps(data):
    X, y = data
    new = pd.DataFrame({"age": [np.nan, 30.0], "fare": [10.0, np.nan], "sex": ["m", None], "port": ["z", "a"]})
    for imputer in IMPUTERS:  # includes drop_rows: unseen data must still be usable
        pipe = build_pipeline(_spec(imputer=imputer), NUM, CAT, "classification")
        pipe.fit_resample(X, y)
        out = np.asarray(pipe.transform(new), dtype=float)
        assert out.shape[0] == 2 and not np.isnan(out).any()


# --------------------------------------------------------- step-specific facts
def test_drop_rows_removes_incomplete_rows(data):
    X, y = data
    complete = int((~X.isna().any(axis=1)).sum())
    Xr, yr = build_pipeline(_spec(imputer="drop_rows"), NUM, CAT, "classification").fit_resample(X, y)
    assert len(Xr) == complete < len(X)


def test_other_imputers_keep_every_row(data):
    X, y = data
    for imputer in ["mean", "median", "most_frequent", "knn"]:
        Xr, _ = build_pipeline(_spec(imputer=imputer), NUM, CAT, "classification").fit_resample(X, y)
        assert len(Xr) == len(X)


def test_median_imputer_fills_with_median():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, np.nan]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipe = build_pipeline(_spec(scaler="none"), ["a"], [], "classification")
    Xr, _ = pipe.fit_resample(X, y)
    assert Xr[4, 0] == pytest.approx(2.5)  # median of 1, 2, 3, 100


def test_onehot_width_and_ordinal_width(data):
    X, y = data
    onehot, _ = build_pipeline(_spec(encoder="onehot"), NUM, CAT, "classification").fit_resample(X, y)
    ordinal, _ = build_pipeline(_spec(encoder="ordinal"), NUM, CAT, "classification").fit_resample(X, y)
    assert onehot.shape[1] == 2 + 2 + 3  # 2 numeric + sex(2) + port(3)
    assert ordinal.shape[1] == 2 + 2


def test_onehot_caps_high_cardinality_columns():
    n = 200
    X = pd.DataFrame({"name": [f"person_{i}" for i in range(n)], "a": np.arange(n, dtype=float)})
    y = pd.Series([0, 1] * (n // 2))
    out, _ = build_pipeline(_spec(), ["a"], ["name"], "classification").fit_resample(X, y)
    assert out.shape[1] <= 1 + 20


def test_scalers_give_expected_ranges(data):
    X, y = data
    mm, _ = build_pipeline(_spec(scaler="minmax", encoder="ordinal"), NUM, CAT, "classification").fit_resample(X, y)
    assert mm[:, :2].min() == pytest.approx(0.0) and mm[:, :2].max() == pytest.approx(1.0)
    st, _ = build_pipeline(_spec(scaler="standard", encoder="ordinal"), NUM, CAT, "classification").fit_resample(X, y)
    assert st[:, :2].mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_scaler_none_leaves_numbers_alone():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    out, _ = build_pipeline(_spec(scaler="none"), ["a"], [], "classification").fit_resample(X, y)
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_iqr_clip_caps_extreme_value(data):
    X, y = data
    base, _ = build_pipeline(_spec(scaler="none", encoder="ordinal"), NUM, CAT, "classification").fit_resample(X, y)
    clipped, _ = build_pipeline(
        _spec(scaler="none", encoder="ordinal", outlier="iqr_clip"), NUM, CAT, "classification"
    ).fit_resample(X, y)
    assert base[:, 1].max() == 5000.0
    assert clipped[:, 1].max() < 500.0


def test_iqr_clipper_learns_limits_from_training_data_only():
    clipper = IQRClipper().fit(np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]))
    # Q1=2, Q3=4, IQR=2 -> limits [-1, 7]. Values outside the training range are capped.
    assert clipper.transform(np.array([[100.0], [-50.0], [3.0]])).ravel().tolist() == [7.0, -1.0, 3.0]


def test_isolation_forest_removes_some_rows(data):
    X, y = data
    Xr, yr = build_pipeline(_spec(outlier="isolation_forest"), NUM, CAT, "classification").fit_resample(X, y)
    assert 0 < len(X) - len(Xr) <= int(0.1 * len(X))
    assert len(Xr) == len(yr)


def test_smote_balances_classes(data):
    X, y = data
    _, yr = build_pipeline(_spec(imbalance="smote"), NUM, CAT, "classification").fit_resample(X, y)
    counts = np.bincount(np.asarray(yr))
    assert counts[0] == counts[1]


def test_smote_survives_tiny_minority_class():
    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    y = pd.Series([0] * 19 + [1])
    Xr, yr = build_pipeline(_spec(imbalance="smote"), ["a"], [], "classification").fit_resample(X, y)
    assert len(Xr) == len(yr) >= 20  # no crash; unchanged when SMOTE is impossible


def test_regression_pipeline_with_target_encoder():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=60), "c": rng.choice(["x", "y", "z"], 60)})
    y = pd.Series(rng.normal(size=60))
    out, _ = build_pipeline(_spec(encoder="target"), ["a"], ["c"], "regression").fit_resample(X, y)
    assert out.shape == (60, 2)


def test_results_are_repeatable(data):
    X, y = data
    spec = _spec(outlier="isolation_forest", imbalance="smote", encoder="target")
    first, _ = build_pipeline(spec, NUM, CAT, "classification").fit_resample(X, y)
    second, _ = build_pipeline(spec, NUM, CAT, "classification").fit_resample(X, y)
    assert np.allclose(first, second)


def test_with_model_cross_validates_without_leakage(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = data
    spec = _spec(imputer="drop_rows", outlier="isolation_forest", imbalance="smote", scaler="robust")
    pipe = with_model(build_pipeline(spec, NUM, CAT, "classification"), LogisticRegression(max_iter=500))
    scores = cross_val_score(pipe, X, y, cv=3)
    assert len(scores) == 3 and np.all(np.isfinite(scores))