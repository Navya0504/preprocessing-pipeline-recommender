"""The exported script must run and behave like the pipeline the evaluator tested."""
import numpy as np
import pandas as pd
import pytest

from backend.code_export import export_code
from backend.pipeline_space import build_pipeline, with_model, uses_class_weight
from sklearn.linear_model import LogisticRegression, Ridge

SPECS = [
    {"imputer": "median", "encoder": "onehot", "scaler": "standard", "outlier": "none", "imbalance": "none"},
    {"imputer": "knn", "encoder": "ordinal", "scaler": "robust", "outlier": "iqr_clip", "imbalance": "class_weight"},
    {"imputer": "drop_rows", "encoder": "target", "scaler": "minmax", "outlier": "isolation_forest", "imbalance": "smote"},
    {"imputer": "mean", "encoder": "onehot", "scaler": "none", "outlier": "none", "imbalance": "none"},
]


def make_data(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n) * 5,
        "c": rng.choice(["x", "y", "z"], size=n),
    })
    df.loc[rng.choice(n, 15, replace=False), "a"] = np.nan
    df["label"] = ((df["a"].fillna(0) + (df["c"] == "x")) > 0.5).astype(int)
    return df


def run(code):
    namespace = {"__name__": "exported"}
    exec(compile(code, "exported.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("spec", SPECS)
def test_exported_script_runs_and_matches_the_tested_pipeline(spec):
    df = make_data()
    X, y = df[["a", "b", "c"]], df["label"]
    code = export_code(spec, ["a", "b"], ["c"], "classification", target="label")
    exported = run(code)["pipeline"].fit(X, y)

    weight = "balanced" if uses_class_weight(spec) else None
    reference = with_model(
        build_pipeline(spec, ["a", "b"], ["c"], "classification"),
        LogisticRegression(max_iter=1000, class_weight=weight, random_state=42),
    ).fit(X, y)
    assert np.array_equal(exported.predict(X), reference.predict(X))


def test_regression_export_uses_ridge_and_runs():
    df = make_data()
    spec = SPECS[0]
    code = export_code(spec, ["a", "b"], ["c"], "regression", target="b")
    assert "Ridge" in code and "LogisticRegression" not in code
    pipe = run(code)["pipeline"].fit(df[["a", "b", "c"]], df["b"])
    assert len(pipe.predict(df[["a", "b", "c"]])) == len(df)


def test_code_only_contains_what_the_spec_uses():
    code = export_code(SPECS[0], ["a", "b"], ["c"], "classification")
    for unused in ("SMOTE", "IsolationForest", "IQRClipper", "KNNImputer"):
        assert unused not in code


def test_bad_inputs_raise_clear_errors():
    with pytest.raises(ValueError):
        export_code({**SPECS[0], "scaler": "weird"}, ["a"], [], "classification")
    with pytest.raises(ValueError):
        export_code(SPECS[0], [], [], "classification")