"""Turn a recommended pipeline into copy-and-run scikit-learn code.

Contract: see INTEGRATION_CONTRACT.md (export).
STATUS: real implementation.

    export_code(spec, numeric_cols, categorical_cols, task, target="target") -> str

The returned text is a complete Python script. It builds the same pipeline that the
evaluator tested (it reuses the exact source of the custom steps), ends with a model
(logistic regression for classification, ridge for regression) and shows how to fit it.
Row-changing steps (drop_rows, isolation_forest, smote) act only while fitting, so the
fitted pipeline can predict on new data with no rows removed.
"""
from __future__ import annotations

import inspect

from backend import pipeline_space as ps


def _imputer_line(choice: str) -> str:
    if choice == "knn":
        return "KNNImputer(n_neighbors=5)"
    if choice in ("mean", "median", "most_frequent"):
        return f'SimpleImputer(strategy="{choice}")'
    return 'SimpleImputer(strategy="median")'  # drop_rows: rows with gaps are removed when fitting


def _encoder_line(choice: str) -> str:
    if choice == "onehot":
        return f'OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories={ps.ONEHOT_MAX_CATEGORIES}, sparse_output=False)'
    if choice == "ordinal":
        return 'OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)'
    if ps._SKLEARN_VERSION < (1, 9):
        return f"TargetEncoder(random_state={ps.RANDOM_STATE})"
    return f"TargetEncoder(cv=KFold(n_splits=5, shuffle=True, random_state={ps.RANDOM_STATE}))"


_SCALER_LINES = {"standard": "StandardScaler()", "minmax": "MinMaxScaler()", "robust": "RobustScaler()"}


def export_code(
    spec: dict,
    numeric_cols: list[str],
    categorical_cols: list[str],
    task: str,
    target: str = "target",
) -> str:
    """Return a runnable Python script for the pipeline described by `spec`."""
    ps.validate_spec(spec, task)
    if not numeric_cols and not categorical_cols:
        raise ValueError("need at least one numeric or categorical column")

    drop_rows = spec["imputer"] == "drop_rows"
    isolation = spec["outlier"] == "isolation_forest"
    smote = spec["imbalance"] == "smote"
    iqr = spec["outlier"] == "iqr_clip"
    sampler_steps = drop_rows or isolation or smote

    imports = [
        "import numpy as np",
        "import pandas as pd",
        "from imblearn import FunctionSampler",
        "from imblearn.pipeline import Pipeline",
        "from sklearn.base import BaseEstimator, TransformerMixin",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import " + ("KNNImputer, SimpleImputer" if spec["imputer"] == "knn" else "SimpleImputer"),
        "from sklearn.pipeline import Pipeline as SkPipeline",
    ]
    preprocessing = {"onehot": "OneHotEncoder", "ordinal": "OrdinalEncoder", "target": "TargetEncoder"}[spec["encoder"]]
    names = [preprocessing] + [s for s in ("StandardScaler", "MinMaxScaler", "RobustScaler") if _SCALER_LINES.get(spec["scaler"]) == f"{s}()"]
    imports.append("from sklearn.preprocessing import " + ", ".join(sorted(names)))
    if spec["encoder"] == "target" and ps._SKLEARN_VERSION >= (1, 9):
        imports.append("from sklearn.model_selection import KFold")
    if isolation:
        imports.append("from sklearn.ensemble import IsolationForest")
    if smote:
        imports.append("from imblearn.over_sampling import SMOTE")
    if task == "classification":
        imports.append("from sklearn.linear_model import LogisticRegression")
    else:
        imports.append("from sklearn.linear_model import Ridge")

    helpers = []
    constants = [f"RANDOM_STATE = {ps.RANDOM_STATE}"]
    if isolation:
        constants.append(f"ISOLATION_CONTAMINATION = {ps.ISOLATION_CONTAMINATION}")
    if iqr:
        helpers.append(inspect.getsource(ps.IQRClipper))
    if drop_rows or isolation or smote:
        helpers.append(inspect.getsource(ps._take))
    if drop_rows:
        helpers.append(inspect.getsource(ps._drop_incomplete_rows))
    if isolation:
        helpers.append(inspect.getsource(ps._remove_isolated_rows))
    if smote:
        helpers.append(inspect.getsource(ps._safe_smote))

    lines = [
        '"""Preprocessing pipeline recommended for your dataset.',
        "",
        f"Imputer {spec['imputer']}, encoder {spec['encoder']}, scaler {spec['scaler']}, "
        f"outliers {spec['outlier']}, imbalance {spec['imbalance']}.",
        '"""',
        *imports,
        "",
        *constants,
        f"TARGET = {target!r}",
        f"NUMERIC_COLS = {list(numeric_cols)!r}",
        f"CATEGORICAL_COLS = {list(categorical_cols)!r}",
        "",
    ]
    for helper in helpers:
        lines += [helper.rstrip(), "", ""]

    numeric_steps = [f'("impute", {_imputer_line(spec["imputer"])})']
    if iqr:
        numeric_steps.append('("clip", IQRClipper())')
    if spec["scaler"] != "none":
        numeric_steps.append(f'("scale", {_SCALER_LINES[spec["scaler"]]})')
    transformers = []
    if numeric_cols:
        transformers.append(f'    ("numeric", SkPipeline([{", ".join(numeric_steps)}]), NUMERIC_COLS),')
    if categorical_cols:
        transformers.append(
            '    ("categorical", SkPipeline([("impute", SimpleImputer(strategy="most_frequent")), '
            f'("encode", {_encoder_line(spec["encoder"])})]), CATEGORICAL_COLS),'
        )
    lines += ["preprocessing = ColumnTransformer([", *transformers, '], remainder="drop")', ""]

    if task == "classification":
        weight = '"balanced"' if ps.uses_class_weight(spec) else "None"
        model = f"LogisticRegression(max_iter=1000, class_weight={weight}, random_state=RANDOM_STATE)"
    else:
        model = "Ridge(random_state=RANDOM_STATE)"

    steps = []
    if drop_rows:
        steps.append('    ("drop_missing", FunctionSampler(func=_drop_incomplete_rows, validate=False)),')
    steps.append('    ("prep", preprocessing),')
    if isolation:
        steps.append('    ("outlier_removal", FunctionSampler(func=_remove_isolated_rows, validate=False)),')
    if smote:
        steps.append('    ("smote", FunctionSampler(func=_safe_smote, validate=False)),')
    steps.append(f'    ("model", {model}),')
    lines += ["pipeline = Pipeline([", *steps, "])", ""]

    lines += [
        'if __name__ == "__main__":',
        '    data = pd.read_csv("your_data.csv")  # <- change this to your file',
        "    data = data.dropna(subset=[TARGET])",
        "    X = data[NUMERIC_COLS + CATEGORICAL_COLS]",
        "    y = data[TARGET]",
        "    pipeline.fit(X, y)",
        '    print("Training score:", pipeline.score(X, y))',
    ]
    if sampler_steps:
        lines.insert(len(lines) - 5, "    # Steps that remove or add rows run only while fitting; new data is never changed.")
    return "\n".join(lines) + "\n"