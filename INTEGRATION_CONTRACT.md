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
| `POST /profile` | Uploaded CSV, `target`, `task` | Meta-features dict |
| `POST /recommend` | Uploaded CSV, `target`, `task` | `{recommendation_id, recommended_spec, scores, pareto_front, neighbours}` |
| `POST /explain` | JSON body: `{"recommendation_id": "<id>"}` | Explanation dict |
| `GET /export/{recommendation_id}` | Recommendation ID | Downloadable Python code for the chosen scikit-learn pipeline |
| `GET /health` | None | `{"status": "ok"}` |

**Note:** `pct_missing` returned by `/profile` is a fraction between 0 and 1, not a percentage between 0 and 100. For example, `0.25` represents 25% missing values.

Uploaded files are processed in memory or in a temporary folder and deleted after the request.

---

### `/clean`

- **Method:** POST
- **Path:** `/clean`
- **Input:** Multipart form data:
  - `recommendation_id`: ID returned by `/recommend`.
  - `file`: CSV file to clean.
- **Success response:** Cleaned CSV as a downloadable file named `cleaned_dataset.csv`.
- **Error cases:**
  - `404`: Unknown recommendation ID.
  - `422`: Missing recommendation ID or file.
  - `400`: Empty or unreadable CSV file, or cleaning error.

**Example request:**

```bash
curl -X POST "http://127.0.0.1:8000/clean" \
  -F "recommendation_id=YOUR_ID" \
  -F "file=@data.csv" \
  -o cleaned_dataset.csv
```

**Example response:**

```http
HTTP/1.1 200 OK
Content-Disposition: attachment; filename=cleaned_dataset.csv
Content-Type: text/csv
```

The downloaded file contains the cleaned CSV data.
