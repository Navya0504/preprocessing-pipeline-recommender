"""Value tests for meta_features.py, using tiny hand-made tables with known answers."""
import numpy as np
import pandas as pd
import pytest

from backend.meta_features import extract_meta_features


def test_counts_and_types():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [5, 6, 7, 8], "c": ["x", "y", "x", "z"], "y": [0, 1, 0, 1]}
    )
    m = extract_meta_features(df, "y", "classification")
    assert m["n_rows"] == 4
    assert m["n_cols"] == 3
    assert m["n_numeric"] == 2
    assert m["n_categorical"] == 1
    assert m["categorical_ratio"] == pytest.approx(1 / 3)


def test_missing_percentages():
    # 4 rows x 2 feature columns = 8 cells; 2 are missing, in 2 different rows.
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, np.nan, 4.0], "y": [0, 1, 0, 1]})
    m = extract_meta_features(df, "y", "classification")
    assert m["pct_missing"] == pytest.approx(25.0)
    assert m["pct_rows_with_missing"] == pytest.approx(50.0)


def test_no_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})
    m = extract_meta_features(df, "y", "classification")
    assert m["pct_missing"] == 0.0
    assert m["pct_rows_with_missing"] == 0.0


def test_mean_cardinality():
    # column c has 2 distinct values, column d has 4 -> mean 3.
    df = pd.DataFrame(
        {"c": ["x", "y", "x", "y"], "d": ["p", "q", "r", "s"], "y": [0, 1, 0, 1]}
    )
    m = extract_meta_features(df, "y", "classification")
    assert m["mean_cardinality"] == pytest.approx(3.0)


def test_class_imbalance_ratio():
    # 8 of class 0 and 2 of class 1 -> 2 / 8 = 0.25.
    df = pd.DataFrame({"a": range(10), "y": [0] * 8 + [1] * 2})
    m = extract_meta_features(df, "y", "classification")
    assert m["class_imbalance_ratio"] == pytest.approx(0.25)


def test_balanced_classes_give_one():
    df = pd.DataFrame({"a": range(10), "y": [0, 1] * 5})
    assert extract_meta_features(df, "y", "classification")["class_imbalance_ratio"] == pytest.approx(1.0)


def test_regression_has_no_imbalance():
    df = pd.DataFrame({"a": range(10), "y": np.linspace(0, 1, 10)})
    assert extract_meta_features(df, "y", "regression")["class_imbalance_ratio"] is None


def test_outlier_fraction():
    # 1..9 plus 100: Q1=3.25, Q3=7.75, IQR=4.5, upper fence=14.5 -> only 100 is an outlier -> 1 of 10.
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], "y": [0, 1] * 5})
    m = extract_meta_features(df, "y", "classification")
    assert m["outlier_fraction"] == pytest.approx(0.1)


def test_skewness_is_higher_for_skewed_data():
    symmetric = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7], "y": [0, 1] * 3 + [0]})
    skewed = pd.DataFrame({"a": [1, 1, 1, 1, 2, 3, 50], "y": [0, 1] * 3 + [0]})
    s1 = extract_meta_features(symmetric, "y", "classification")["mean_skewness"]
    s2 = extract_meta_features(skewed, "y", "classification")["mean_skewness"]
    assert s1 == pytest.approx(0.0, abs=1e-9)
    assert s2 > 1.0


def test_correlation_of_perfectly_related_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "y": [0, 1, 0, 1]})
    assert extract_meta_features(df, "y", "classification")["mean_abs_correlation"] == pytest.approx(1.0)


def test_numeric_only_features_do_not_apply_to_categorical_datasets():
    df = pd.DataFrame({"c": ["x", "y", "x", "y"], "y": [0, 1, 0, 1]})
    m = extract_meta_features(df, "y", "classification")
    assert m["n_numeric"] == 0
    assert m["mean_skewness"] is None
    assert m["outlier_fraction"] is None
    assert m["mean_abs_correlation"] is None


def test_numeric_only_dataset_has_no_cardinality():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "y": [0, 1, 0]})
    assert extract_meta_features(df, "y", "classification")["mean_cardinality"] is None


def test_bad_inputs_raise_clear_errors():
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})
    with pytest.raises(ValueError):
        extract_meta_features(df, "missing_column", "classification")
    with pytest.raises(ValueError):
        extract_meta_features(df, "y", "clustering")
