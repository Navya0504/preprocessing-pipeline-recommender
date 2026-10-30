"""Candidate preprocessing pipelines.

Contract: see INTEGRATION_CONTRACT.md, section 2.3 and 3.
STATUS: real implementation.

Two public functions:
    generate_candidates(meta_features) -> list of spec dicts (pruned to what applies)
    build_pipeline(spec, numeric_cols, categorical_cols, task) -> fit-ready pipeline

A spec picks one option for each of five steps (see the constants below).
The pipeline returned by build_pipeline contains NO model. The evaluator adds
a model as the last step.

Notes on how each step is built:
    imputer   "drop_rows" removes incomplete rows when fitting (a sampler step), and
              falls back to median / most-frequent so unseen data with gaps still works.
    outlier   "iqr_clip" caps numeric values at 1.5 x IQR (learned on training data only).
              "isolation_forest" removes the most isolated training rows (fit time only).
    imbalance "smote" creates synthetic minority rows (fit time only).
              "class_weight" is a MODEL setting, not a data step, so the pipeline is left
              unchanged; the evaluator checks uses_class_weight(spec) and passes
              class_weight="balanced" to the model.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn import FunctionSampler
from sklearn import __version__ as _sk_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

_SKLEARN_VERSION = tuple(int(part) for part in _sk_version.split(".")[:2])

IMPUTERS = ["mean", "median", "most_frequent", "knn", "drop_rows"]
ENCODERS = ["onehot", "ordinal", "target"]
SCALERS = ["none", "standard", "minmax", "robust"]
OUTLIER_METHODS = ["none", "iqr_clip", "isolation_forest"]
IMBALANCE_METHODS = ["none", "class_weight", "smote"]  # classification only

SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"]
_OPTIONS = {
    "imputer": IMPUTERS,
    "encoder": ENCODERS,
    "scaler": SCALERS,
    "outlier": OUTLIER_METHODS,
    "imbalance": IMBALANCE_METHODS,
}

# A dataset with a minority/majority ratio at or above this counts as balanced.
BALANCED_RATIO = 0.8
# One-hot keeps at most this many categories per column (the rest become "other").
ONEHOT_MAX_CATEGORIES = 20
# Share of training rows that "isolation_forest" removes as outliers.
ISOLATION_CONTAMINATION = 0.05
RANDOM_STATE = 42


# ---------------------------------------------------------------- spec helpers
def validate_spec(spec: dict, task: str | None = None) -> None:
    """Raise ValueError if the spec has wrong keys or unknown options."""
    if set(spec) != set(SPEC_KEYS):
        raise ValueError(f"spec must have exactly these keys: {SPEC_KEYS}, got {sorted(spec)}")
    for step, choice in spec.items():
        if choice not in _OPTIONS[step]:
            raise ValueError(f"unknown {step} option '{choice}'; choose from {_OPTIONS[step]}")
    if task is not None:
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if task == "regression" and spec["imbalance"] != "none":
            raise ValueError("imbalance handling only applies to classification")


def uses_class_weight(spec: dict) -> bool:
    """True if the model should be trained with class_weight='balanced'."""
    return spec.get("imbalance") == "class_weight"


# ------------------------------------------------------------- candidate grid
def _known(meta_features: dict, key: str):
    """Return meta_features[key] or None when it is missing/unknown."""
    return meta_features.get(key)


def generate_candidates(meta_features: dict) -> list[dict]:
    """Return candidate pipeline specs (contract 2.3) that suit this dataset.

    The full grid is every combination of the five steps. Options that cannot
    matter for this dataset are collapsed to a single value so no time is
    wasted evaluating identical pipelines:
        nothing missing            -> one imputer only
        no categorical columns     -> one encoder only
        no numeric columns         -> no scaling and no outlier handling
        no outliers found          -> no outlier handling
        regression or balanced     -> no imbalance handling
    Unknown (missing) meta-features keep every option.
    """
    n_numeric = _known(meta_features, "n_numeric")
    n_categorical = _known(meta_features, "n_categorical")
    pct_missing = _known(meta_features, "pct_missing")
    outlier_fraction = _known(meta_features, "outlier_fraction")
    ratio = _known(meta_features, "class_imbalance_ratio")
    is_classification = meta_features.get("task", "classification") == "classification"

    imputers = IMPUTERS
    if pct_missing is not None and pct_missing == 0:
        imputers = ["median"]

    encoders = ENCODERS
    if n_categorical is not None and n_categorical == 0:
        encoders = ["onehot"]

    scalers = SCALERS
    outliers = OUTLIER_METHODS
    if n_numeric is not None and n_numeric == 0:
        scalers = ["none"]
        outliers = ["none"]
    elif outlier_fraction is not None and outlier_fraction == 0:
        outliers = ["none"]

    imbalances = IMBALANCE_METHODS
    if not is_classification or (ratio is not None and ratio >= BALANCED_RATIO):
        imbalances = ["none"]

    return [
        dict(zip(SPEC_KEYS, combo))
        for combo in product(imputers, encoders, scalers, outliers, imbalances)
    ]


# -------------------------------------------------------- custom pipeline parts
class _RowSampler(FunctionSampler):
    """Identity FunctionSampler that also exposes transform (a no-op). Used only as the
    final step so the pipeline supports both fit_resample and transform. Row-changing
    steps act only in fit_resample; at transform time they are skipped."""

    def transform(self, X):
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Cap values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Limits are learned on fit data only."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.low_ = q1 - 1.5 * iqr
        self.high_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def _take(values, mask):
    """Row-select from a DataFrame/Series/array with a boolean mask."""
    return values.iloc[mask] if hasattr(values, "iloc") else values[mask]


def _drop_incomplete_rows(X, y):
    """Sampler function: keep only rows with no missing feature values."""
    keep = ~np.asarray(X.isna().any(axis=1))
    return _take(X, keep), _take(y, keep)


def _remove_isolated_rows(X, y):
    """Sampler function: drop the rows an isolation forest finds most outlying."""
    if len(X) < 20:
        return X, y
    forest = IsolationForest(contamination=ISOLATION_CONTAMINATION, random_state=RANDOM_STATE)
    keep = forest.fit_predict(X) == 1
    return _take(X, keep), _take(y, keep)


def _safe_smote(X, y):
    """Sampler function: SMOTE that adapts k to tiny minority classes (and skips if impossible)."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) < 2:
        return X, y
    k = min(5, int(counts.min()) - 1)
    if k < 1:
        return X, y
    return SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)


# ------------------------------------------------------------- pipeline builder
def _numeric_imputer(choice: str):
    if choice == "knn":
        return KNNImputer(n_neighbors=5)
    if choice in ("mean", "median", "most_frequent"):
        return SimpleImputer(strategy=choice)
    return SimpleImputer(strategy="median")  # drop_rows fallback for unseen data


def _scaler(choice: str):
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }.get(choice)


def _encoder(choice: str):
    if choice == "onehot":
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=ONEHOT_MAX_CATEGORIES,
            sparse_output=False,
        )
    if choice == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    # scikit-learn 1.9 deprecates random_state here; older versions need it for repeatable results.
    if _SKLEARN_VERSION < (1, 9):
        return TargetEncoder(random_state=RANDOM_STATE)
    return TargetEncoder(cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))


def build_pipeline(
    spec: dict, numeric_cols: list[str], categorical_cols: list[str], task: str
) -> Pipeline:
    """Turn a spec into a fit-ready preprocessing pipeline (no model).

    Use pipeline.fit_resample(X, y) to get the cleaned numeric table and its
    matching target (rows may be removed or added by drop_rows, isolation_forest
    or smote). pipeline.transform(X) cleans new data without touching rows.
    The last step is a harmless identity sampler; the evaluator replaces or
    follows it with a model.
    Categorical missing values are always filled with the most common value,
    because the numeric imputers do not apply to text.
    """
    validate_spec(spec, task)

    numeric_steps = [("impute", _numeric_imputer(spec["imputer"]))]
    if spec["outlier"] == "iqr_clip":
        numeric_steps.append(("clip", IQRClipper()))
    scaler = _scaler(spec["scaler"])
    if scaler is not None:
        numeric_steps.append(("scale", scaler))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", _encoder(spec["encoder"])),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", SkPipeline(numeric_steps), list(numeric_cols)))
    if categorical_cols:
        transformers.append(("categorical", SkPipeline(categorical_steps), list(categorical_cols)))
    if not transformers:
        raise ValueError("need at least one numeric or categorical column")

    steps = []
    if spec["imputer"] == "drop_rows":
        steps.append(("drop_missing", FunctionSampler(func=_drop_incomplete_rows, validate=False)))
    steps.append(("prep", ColumnTransformer(transformers, remainder="drop")))
    if spec["outlier"] == "isolation_forest":
        steps.append(("outlier_removal", FunctionSampler(func=_remove_isolated_rows, validate=False)))
    if spec["imbalance"] == "smote":
        steps.append(("smote", FunctionSampler(func=_safe_smote, validate=False)))
    # Identity final step (the only step allowed to be both sampler and transformer):
    # pipeline.fit_resample(X, y) returns aligned (X, y) and pipeline.transform(X) cleans new data.
    steps.append(("rows", _RowSampler(validate=False)))
    return Pipeline(steps)


def with_model(pipeline: Pipeline, model) -> Pipeline:
    """Return the pipeline with its identity final step replaced by a model.

    Use this in the evaluator for cross-validation: all row-changing steps then
    run on the training folds only, so no test data leaks into preprocessing.
    """
    return Pipeline(list(pipeline.steps[:-1]) + [("model", model)])