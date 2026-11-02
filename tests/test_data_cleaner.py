import numpy as np
import pandas as pd
import pytest

from backend.data_cleaner import clean_dataset

BASE = {"imputer": "median", "encoder": "onehot", "scaler": "standard", "outlier": "none", "imbalance": "none"}


def make_data(n=200, imbalance=False):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n) * 4, "c": rng.choice(["x", "y", "z"], size=n)})
    df.loc[:9, "a"] = np.nan
    df.loc[10:14, "c"] = None
    df["empty"] = np.nan
    cut = 0.9 if imbalance else 0.0
    df["label"] = (rng.normal(size=n) > cut).astype(int)
    return df


def test_no_gaps_text_becomes_numbers_and_target_is_last():
    out = clean_dataset(make_data(), "label", BASE, "classification")
    assert out.columns[-1] == "label"
    assert not out.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)
    assert {"c_x", "c_y", "c_z"} <= set(out.columns)
    assert "empty" not in out.columns and len(out) == 200


def test_drop_rows_removes_incomplete_rows_and_keeps_targets_aligned():
    df = make_data()
    out = clean_dataset(df, "label", {**BASE, "imputer": "drop_rows"}, "classification")
    complete = df.drop(columns=["empty"]).dropna()
    assert len(out) == len(complete)
    assert np.array_equal(out["label"].to_numpy(), complete["label"].to_numpy())


def test_smote_rows_only_added_when_asked():
    df = make_data(imbalance=True)
    spec = {**BASE, "imbalance": "smote"}
    plain = clean_dataset(df, "label", spec, "classification")
    boosted = clean_dataset(df, "label", spec, "classification", add_synthetic_rows=True)
    assert len(plain) == len(df)
    assert len(boosted) > len(plain)


def test_regression_and_other_methods_run():
    df = make_data()
    spec = {"imputer": "knn", "encoder": "ordinal", "scaler": "robust", "outlier": "iqr_clip", "imbalance": "none"}
    out = clean_dataset(df, "b", spec, "regression")
    assert out.columns[-1] == "b" and not out.isna().any().any()
    iso = clean_dataset(df, "b", {**spec, "outlier": "isolation_forest"}, "regression")
    assert len(iso) < len(df)


def test_original_table_is_not_changed_and_bad_input_is_clear():
    df = make_data()
    before = df.copy()
    clean_dataset(df, "label", BASE, "classification")
    pd.testing.assert_frame_equal(df, before)
    with pytest.raises(ValueError):
        clean_dataset(df, "nope", BASE, "classification")
    with pytest.raises(ValueError):
        clean_dataset(df, "label", {**BASE, "scaler": "weird"}, "classification")