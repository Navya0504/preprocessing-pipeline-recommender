"""Value tests for explainer.py."""
import numpy as np
import pandas as pd
import pytest

from backend.explainer import SCORE_KEYS, SPEC_KEYS, explain, explainability_score,step_status

SPEC = dict(imputer="median", encoder="onehot", scaler="standard", outlier="none", imbalance="none")


def meta(**changes):
    base = dict(
        n_rows=1000, n_cols=10, n_numeric=6, n_categorical=4, categorical_ratio=0.4,
        pct_missing=12.0, pct_rows_with_missing=40.0, mean_cardinality=7.0,
        class_imbalance_ratio=0.3, mean_skewness=1.0, outlier_fraction=0.06,
        mean_abs_correlation=0.3, task="classification",
    )
    base.update(changes)
    return base


def scores(**changes):
    base = dict(performance=0.8, efficiency=0.8, simplicity=0.6, robustness=0.8, explainability=0.9)
    base.update(changes)
    return base


def front_table(*rows):
    return pd.DataFrame([{**spec, **sc} for spec, sc in rows])


def text(result):
    return " ".join([result["summary"]] + result["reasons"] + result["tradeoffs"])


# -------------------------------------------------------------- explainability
def test_explainability_score_values():
    assert explainability_score(SPEC) == pytest.approx((1.0 + 1.0 + 0.9 + 1.0 + 1.0) / 5)
    assert explainability_score(dict(imputer="knn", encoder="target", scaler="robust", outlier="isolation_forest", imbalance="smote")) == pytest.approx(
        (0.6 + 0.5 + 0.8 + 0.4 + 0.5) / 5
    )
    assert explainability_score({}) == 0.0


# ---------------------------------------------------------------------- shape
def test_result_has_exactly_the_contract_keys():
    result = explain(meta(), SPEC, scores(), pd.DataFrame(), pd.DataFrame())
    assert set(result) == {"summary", "reasons", "tradeoffs", "steps"}
    assert isinstance(result["summary"], str) and isinstance(result["reasons"], list) and isinstance(result["tradeoffs"], list)


def test_works_with_no_scores_neighbours_or_front():
    result = explain(meta(), SPEC, {}, pd.DataFrame(), pd.DataFrame())
    assert result["reasons"] and result["tradeoffs"]
    assert explain(meta(), SPEC, None, None, None)["tradeoffs"]


def test_unknown_meta_features_do_not_crash():
    sparse = {"task": "classification", "pct_missing": None, "mean_cardinality": float("nan")}
    result = explain(sparse, SPEC, scores(), pd.DataFrame(), pd.DataFrame())
    assert result["summary"]


# -------------------------------------------------------------------- summary
def test_summary_names_every_step_in_plain_words():
    spec = dict(imputer="knn", encoder="ordinal", scaler="robust", outlier="iqr_clip", imbalance="smote")
    summary = explain(meta(), spec, scores(), pd.DataFrame(), pd.DataFrame())["summary"]
    for phrase in ["KNN", "ordinal", "robust scaling", "IQR", "SMOTE"]:
        assert phrase in summary


def test_regression_has_no_imbalance_in_summary_or_reasons():
    m = meta(task="regression", class_imbalance_ratio=None)
    result = explain(m, SPEC, scores(), pd.DataFrame(), pd.DataFrame())
    assert "balancing" not in result["summary"]
    assert not any("class" in reason.lower() and "smallest" in reason.lower() for reason in result["reasons"])


# -------------------------------------------------------------------- reasons
def test_missing_values_reason_quotes_the_numbers():
    reasons = explain(meta(), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    first = reasons[0]
    assert "12%" in first and "40%" in first and "median" in first.lower()


def test_no_missing_values_reason():
    reasons = explain(meta(pct_missing=0.0, pct_rows_with_missing=0.0), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert "No values are missing" in reasons[0]


def test_each_imputer_has_its_own_explanation():
    seen = set()
    for imputer in ["mean", "median", "most_frequent", "knn", "drop_rows"]:
        seen.add(explain(meta(), {**SPEC, "imputer": imputer}, scores(), pd.DataFrame(), pd.DataFrame())["reasons"][0])
    assert len(seen) == 5


def test_no_text_columns_means_no_encoding_needed():
    reasons = explain(meta(n_categorical=0, mean_cardinality=None), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert any("no text columns" in reason.lower() for reason in reasons)


def test_encoder_reason_mentions_category_count():
    reasons = explain(meta(), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert any("4 text column" in reason and "7 categories" in reason for reason in reasons)


def test_no_numeric_columns_skips_scaling_and_outliers():
    reasons = explain(meta(n_numeric=0, outlier_fraction=None), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert sum("no numeric columns" in reason.lower() for reason in reasons) == 2


def test_outlier_reasons_use_the_measured_share():
    iqr = explain(meta(), {**SPEC, "outlier": "iqr_clip"}, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert any("6.0%" in reason for reason in iqr)
    none = explain(meta(outlier_fraction=0.0), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert any("No extreme values" in reason for reason in none)


def test_imbalance_reasons():
    skewed = explain(meta(class_imbalance_ratio=0.2), {**SPEC, "imbalance": "smote"}, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert any("20%" in reason and "SMOTE" in reason for reason in skewed)
    balanced = explain(meta(class_imbalance_ratio=0.95), SPEC, scores(), pd.DataFrame(), pd.DataFrame())["reasons"]
    assert any("fairly balanced" in reason for reason in balanced)


def test_neighbour_names_appear_closest_first():
    neighbours = pd.DataFrame({"dataset_id": ["heart", "bank", "adult", "iris"], "distance": [0.1, 0.2, 0.3, 0.4]})
    reasons = explain(meta(), SPEC, scores(), neighbours, pd.DataFrame())["reasons"]
    last = reasons[-1]
    assert "heart, bank, adult" in last and "iris" not in last


# ------------------------------------------------------------------ trade-offs
def test_tradeoffs_name_what_was_given_up_and_how_the_alternative_differs():
    chosen = scores(performance=0.9, efficiency=0.4)
    fast = dict(SPEC, scaler="none", imputer="mean")
    front = front_table((SPEC, chosen), (fast, scores(performance=0.85, efficiency=0.95)))
    tradeoffs = explain(meta(), SPEC, chosen, pd.DataFrame(), front)["tradeoffs"]
    joined = " ".join(tradeoffs)
    assert "Gave up some speed" in joined and "0.40" in joined and "0.95" in joined
    assert "scaling = none" in joined and "gap filling = mean" in joined
    assert "Among the best of the options on" in joined and "accuracy" in joined
    assert "2 pipeline(s)" in joined


def test_no_tradeoff_when_chosen_is_best_everywhere():
    chosen = scores(performance=0.95, efficiency=0.95, simplicity=0.95, robustness=0.95, explainability=0.95)
    front = front_table((SPEC, chosen))
    tradeoffs = explain(meta(), SPEC, chosen, pd.DataFrame(), front)["tradeoffs"]
    assert not any("Gave up" in line for line in tradeoffs)


def test_missing_scores_message():
    assert "not provided" in explain(meta(), SPEC, {"performance": 0.9}, pd.DataFrame(), front_table((SPEC, scores())))["tradeoffs"][0]


def test_explanation_is_plain_english_not_code_keys():
    spec = dict(imputer="most_frequent", encoder="onehot", scaler="minmax", outlier="isolation_forest", imbalance="class_weight")
    result = explain(meta(), spec, scores(), pd.DataFrame(), pd.DataFrame())
    for raw in ["most_frequent", "class_weight", "isolation_forest", "iqr_clip"]:
        assert raw not in text(result)

# ---------------------------------------------------------------- step status
def test_step_status_when_nothing_to_fix():
    m = meta(pct_missing=0.0, n_categorical=0, class_imbalance_ratio=1.0)
    status = step_status(SPEC, m, True)
    assert status["imputer"]["status"] == "not_needed"
    assert status["encoder"]["status"] == "not_needed"
    assert status["scaler"]["status"] == "applied"  # SPEC uses standard scaling
    assert status["outlier"]["status"] == "skipped"
    assert status["imbalance"]["status"] == "skipped"


def test_step_status_applied_when_the_data_needs_it():
    spec = {**SPEC, "outlier": "iqr_clip", "imbalance": "smote"}
    status = step_status(spec, meta(), True)
    assert all(status[step]["status"] == "applied" for step in SPEC_KEYS)
    assert status["imputer"]["method"] == "median"


def test_step_status_regression_has_no_class_balancing():
    status = step_status(SPEC, meta(task="regression"), False)
    assert status["imbalance"]["status"] == "not_applicable"


def test_step_status_no_numeric_columns():
    status = step_status(SPEC, meta(n_numeric=0), True)
    assert status["scaler"]["status"] == "not_applicable"
    assert status["outlier"]["status"] == "not_applicable"