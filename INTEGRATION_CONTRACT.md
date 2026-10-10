# Integration Contract

Project: Multi-Objective Meta-Learning for Explainable Preprocessing Pipeline Recommendation

This file is the agreement between all 6 modules. Code against these signatures and data formats exactly. If you need to change anything here, tell the whole group first and update this file in the same pull request.

Python 3.11+, type hints on every function, a docstring on every function, and at least one test per function in `tests/`.

---

## 1. Ownership

| Person | Module (file in `backend/`) | Branch name |
|---|---|---|
| 1 | `meta_features.py` | `p1-meta-features` |
| 2 | `pipeline_space.py` | `p2-pipeline-space` |
| 3 | `evaluator.py` | `p3-evaluator` |
| 4 | `meta_recommender.py`, `pareto_selector.py` | `p4-recommender` |
| 5 | `explainer.py` | `p5-explainer` |
| 6 | `api.py`, `frontend/`, deployment | `p6-app` |

---

## 2. Shared data formats

### 2.1 Dataset input
- `df: pandas.DataFrame` containing features and the target column.
- `target: str` name of the target column.
- `task: str` either `"classification"` or `"regression"`.

### 2.2 Meta-features (a flat `dict[str, float | int | str | None]`)
Exactly these keys. Use `None` where a value does not apply.

| Key | Meaning |
|---|---|
| `n_rows` | number of rows |
| `n_cols` | number of feature columns (target excluded) |
| `n_numeric` | number of numeric features |
| `n_categorical` | number of categorical features |
| `categorical_ratio` | `n_categorical / n_cols` |
| `pct_missing` | percent of all cells that are missing (0 to 100) |
| `pct_rows_with_missing` | percent of rows with at least one missing value |
| `mean_cardinality` | mean number of unique values across categorical columns |
| `class_imbalance_ratio` | minority class count / majority class count (0 to 1). `None` for regression |
| `mean_skewness` | mean absolute skewness of numeric columns |
| `outlier_fraction` | fraction of numeric values outside 1.5 x IQR (0 to 1) |
| `mean_abs_correlation` | mean absolute pairwise correlation between numeric columns |
| `task` | `"classification"` or `"regression"` |

### 2.3 Pipeline spec (a plain `dict[str, str]`, JSON-friendly)
Pipelines are described as small dictionaries, not as sklearn objects, so they can be stored, compared, explained and sent through the API.

```python
spec = {
    "imputer":   "median",   # "mean" | "median" | "most_frequent" | "knn" | "drop_rows"
    "encoder":   "onehot",   # "onehot" | "ordinal" | "target"
    "scaler":    "standard", # "none" | "standard" | "minmax" | "robust"
    "outlier":   "none",     # "none" | "iqr_clip" | "isolation_forest"
    "imbalance": "none",     # "none" | "class_weight" | "smote"  (classification only)
}
```

### 2.4 Scores (a `dict[str, float]`)
All five values are normalised to 0 to 1 where **higher is always better**, so Pareto logic needs no special cases.

```python
scores = {
    "performance":    0.87,  # mean CV score (F1 macro for classification, R2 for regression), clipped to 0..1
    "efficiency":     0.64,  # 1 - normalised fit+transform time
    "simplicity":     0.80,  # fewer active (non-"none") steps = higher
    "robustness":     0.91,  # 1 - normalised standard deviation across CV folds
    "explainability": 0.75,  # from the interpretability table in explainer.py
}
```

Raw (unnormalised) values are stored in the experience database as `raw_*` columns (for example `raw_fit_seconds`, `raw_cv_std`).

### 2.5 Experience database
File: `experiments/results.csv` (one row per dataset and pipeline spec).

Columns: `dataset_id`, every meta-feature key from 2.2, the five spec keys from 2.3, the five score keys from 2.4, and the `raw_*` columns.

---

## 3. Function signatures

### Person 1: `backend/meta_features.py`
```python
def extract_meta_features(df: pd.DataFrame, target: str, task: str) -> dict:
    """Return the meta-feature dict defined in 2.2."""
```

### Person 2: `backend/pipeline_space.py`
```python
def generate_candidates(meta_features: dict) -> list[dict]:
    """Return candidate pipeline specs (2.3) that make sense for this dataset.
    Example: do not include imbalance handling for regression."""

def build_pipeline(spec: dict, numeric_cols: list[str], categorical_cols: list[str], task: str):
    """Turn a spec into a fit-ready sklearn / imblearn Pipeline (preprocessing only, no model)."""
```

### Person 3: `backend/evaluator.py`
```python
def evaluate_pipeline(spec: dict, df: pd.DataFrame, target: str, task: str,
                      cv: int = 5, random_state: int = 42) -> dict:
    """Run cross-validation with a fixed reference model and return the scores dict (2.4)
    plus raw_* values. Must not leak test data into preprocessing."""

def evaluate_many(specs: list[dict], df: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    """Evaluate every spec and return one row per spec (spec columns + scores)."""

def build_experience_db(datasets: dict[str, tuple[pd.DataFrame, str, str]], out_path: str) -> None:
    """Run all candidates on all datasets and write the experience database (2.5)."""
```

### Person 4: `backend/meta_recommender.py` and `backend/pareto_selector.py`
```python
# meta_recommender.py
def find_similar_datasets(meta_features: dict, experience_db: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Return the k most similar past datasets with a `distance` column."""

def recommend(meta_features: dict, experience_db: pd.DataFrame, k: int = 5) -> list[dict]:
    """Return ranked candidate specs learned from similar datasets."""

# pareto_selector.py
def pareto_front(results: pd.DataFrame) -> pd.DataFrame:
    """Return only non-dominated rows across the five score columns (higher is better)."""

def select_best(front: pd.DataFrame, weights: dict[str, float] | None = None) -> dict:
    """Pick one pipeline from the front. weights=None means equal weights. Return the row as a dict."""
```

### Person 5: `backend/explainer.py`
```python
INTERPRETABILITY: dict[str, dict[str, float]]  # per-step interpretability table used for the explainability score

def explainability_score(spec: dict) -> float:
    """Return 0..1 (higher = easier for a human to understand)."""

def explain(meta_features: dict, chosen_spec: dict, scores: dict,
            neighbours: pd.DataFrame, front: pd.DataFrame) -> dict:
    """Return {"summary": str, "reasons": list[str], "tradeoffs": list[str]}
    in plain English: why this pipeline, which dataset traits mattered, what was given up."""
```

### Person 6: `backend/api.py` (FastAPI)
| Endpoint | Input | Output |
|---|---|---|
| `POST /profile` | uploaded CSV, `target`, `task` | meta-features dict |
| `POST /recommend` | uploaded CSV, `target`, `task` | `{recommended_spec, scores, pareto_front, neighbours}` |
| `POST /explain` | the output of `/recommend` | explanation dict |
| `GET /export/{id}` | recommendation id | downloadable Python code for the chosen scikit-learn pipeline |
| `GET /health` | none | `{"status": "ok"}` |

Uploaded files are processed in memory or in a temporary folder and deleted after the request.

---

## 4. Working rules
1. Never push directly to `main`. Use your branch and open a pull request. At least one teammate reviews before merging.
2. Commit messages are short and clear, for example `Add outlier_fraction to meta_features`.
3. Fixed random seed (`random_state=42`) everywhere, so results can be reproduced for the paper.
4. No hard-coded file paths. Use relative paths from the project root.
5. Add every new library to `requirements.txt` in the same pull request.
6. Until your real module is ready, keep a working stub that returns correctly shaped dummy output, so Person 6 can build the app in parallel.
7. Contract changes need agreement from the whole group.
