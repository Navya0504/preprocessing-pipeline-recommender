
"""FastAPI app. Run from the project root with:

    uvicorn backend.api:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API page.

Contract: see INTEGRATION_CONTRACT.md, section 3 (Person 6 table).
"""

from __future__ import annotations

import io
import json
from fastapi.responses import PlainTextResponse
from uuid import uuid4
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend.meta_features import extract_meta_features
from backend.explainer import explain
from backend.meta_recommender import (
    find_similar_datasets,
    predict_pipeline_scores,
)
from backend.pareto_selector import pareto_front, select_best


# Experience database path
EXPERIENCE_DB = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "results.csv"
)
# Store recommendations temporarily using their unique IDs
RECOMMENDATIONS = {}


def load_experience_db() -> pd.DataFrame:
    """Load previously evaluated datasets and pipeline results."""
    if not EXPERIENCE_DB.exists():
        raise HTTPException(
            status_code=500,
            detail="Experience database not found."
        )

    return pd.read_csv(EXPERIENCE_DB)


app = FastAPI(
    title="Preprocessing Pipeline Recommender",
    description="Multi-objective, explainable preprocessing pipeline recommendation.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Check whether the API is running."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    """Return the API welcome message and documentation location."""
    return {
        "message": "Preprocessing Pipeline Recommender API",
        "docs": "/docs"
    }


@app.post("/profile")
async def profile(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...),
) -> dict:
    """Return the meta-features of an uploaded CSV."""
    if task not in ("classification", "regression"):
        raise HTTPException(
            status_code=422,
            detail="task must be 'classification' or 'regression'"
        )

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or empty CSV file."
        )

    if target not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"target column '{target}' not found"
        )

    return extract_meta_features(df, target, task)



@app.post("/recommend")
async def get_recommendation(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...),
) -> dict:
    """Recommend a preprocessing pipeline and assign it a unique ID."""

    if task not in ("classification", "regression"):
        raise HTTPException(
            status_code=422,
            detail="task must be 'classification' or 'regression'"
        )

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or empty CSV file."
        )

    if target not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"target column '{target}' not found"
        )

    meta_features = extract_meta_features(df, target, task)
    experience_db = load_experience_db()

    predictions = predict_pipeline_scores(
        meta_features,
        experience_db
    )

    if predictions.empty:
        raise HTTPException(
            status_code=503,
            detail="No pipeline predictions are available."
        )

    front = pareto_front(predictions)

    if front.empty:
        raise HTTPException(
            status_code=503,
            detail="No valid Pareto recommendations are available."
        )

    chosen = select_best(front)

    spec_keys = [
        "imputer",
        "encoder",
        "scaler",
        "outlier",
        "imbalance"
    ]

    score_keys = [
        "performance",
        "efficiency",
        "simplicity",
        "robustness",
        "explainability"
    ]

    recommended_spec = {
        key: chosen[key] for key in spec_keys
    }

    scores = {
        key: float(chosen[key]) for key in score_keys
    }

    neighbours = find_similar_datasets(
        meta_features,
        experience_db
    )

    recommendation_id = str(uuid4())

    RECOMMENDATIONS[recommendation_id] = {
        "meta_features": meta_features,
        "recommended_spec": recommended_spec,
        "scores": scores,
        "pareto_front": front.to_dict(orient="records"),
        "neighbours": neighbours.to_dict(orient="records"),
    }

    return {
        "recommendation_id": recommendation_id,
        "recommended_spec": recommended_spec,
        "scores": scores,
        "pareto_front": front.to_dict(orient="records"),
        "neighbours": neighbours.to_dict(orient="records"),
    }


@app.post("/explain")
async def get_explanation(request: dict) -> dict:
    """Explain a previously generated recommendation."""

    recommendation_id = request.get("recommendation_id")

    if not recommendation_id:
        raise HTTPException(
            status_code=422,
            detail="recommendation_id is required."
        )

    if recommendation_id not in RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail="Recommendation not found. Generate a new recommendation."
        )

    recommendation = RECOMMENDATIONS[recommendation_id]

    meta_features = recommendation["meta_features"]
    recommended_spec = recommendation["recommended_spec"]
    scores = recommendation["scores"]

    neighbours = pd.DataFrame(recommendation["neighbours"])
    front = pd.DataFrame(recommendation["pareto_front"])

    explanation = explain(
        meta_features,
        recommended_spec,
        scores,
        neighbours,
        front
    )

    return explanation
@app.get("/export/{recommendation_id}")
async def export_pipeline(recommendation_id: str) -> PlainTextResponse:
    """Export the recommended preprocessing pipeline as Python code."""

    if recommendation_id not in RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail="Recommendation not found. Generate a new recommendation."
        )

    recommendation = RECOMMENDATIONS[recommendation_id]

    spec = recommendation["recommended_spec"]
    task = recommendation["meta_features"]["task"]

    spec_json = json.dumps(spec)

    python_code = f'''"""Recommended preprocessing pipeline."""

import json
import pandas as pd
from backend.pipeline_space import build_pipeline

CSV_PATH = "your_data.csv"
TARGET = "target_column"
TASK = "{task}"

SPEC = json.loads({spec_json!r})

# Load the dataset
df = pd.read_csv(CSV_PATH)

# Separate input features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# Build the recommended preprocessing pipeline
pipeline = build_pipeline(
    SPEC,
    numeric_cols,
    categorical_cols,
    TASK
)

# Fit the preprocessing pipeline
X_processed, y_processed = pipeline.fit_resample(X, y)

print("Pipeline fitted successfully!")
print("Processed data shape:", X_processed.shape)
'''

    return PlainTextResponse(
        content=python_code,
        media_type="text/x-python",
        headers={
            "Content-Disposition": "attachment; filename=recommended_pipeline.py"
        }
    )