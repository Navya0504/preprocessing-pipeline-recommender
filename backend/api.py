
import io
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from backend.meta_features import extract_meta_features
from backend.explainer import explain
from backend.meta_recommender import (
    find_similar_datasets,
    predict_pipeline_scores,
)
from backend.pareto_selector import pareto_front, select_best
from backend.code_export import export_code
from backend.data_cleaner import clean_dataset


EXPERIENCE_DB = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "results.csv"
)

RECOMMENDATIONS = {}


class ExplainRequest(BaseModel):
    recommendation_id: str


app = FastAPI(
    title="Preprocessing Pipeline Recommender",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_experience_db():
    """Load previous experiment results."""

    if not EXPERIENCE_DB.exists():
        raise HTTPException(
            status_code=500,
            detail="Experience database not found.",
        )

    return pd.read_csv(EXPERIENCE_DB)


def read_csv_file(content: bytes) -> pd.DataFrame:
    """Read uploaded CSV content."""

    try:
        return pd.read_csv(io.BytesIO(content))

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded CSV file.",
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty or invalid.",
        )


def validate_task(task: str):
    """Check whether the task is supported."""

    if task not in ["classification", "regression"]:
        raise HTTPException(
            status_code=422,
            detail="Task must be classification or regression.",
        )


def get_feature_columns(df: pd.DataFrame, target: str):
    """Separate numeric and categorical feature columns."""

    if target not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{target}' not found.",
        )

    df = df.dropna(subset=[target])

    feature_df = df.drop(columns=[target])
    feature_df = feature_df.dropna(axis=1, how="all")

    if feature_df.empty or len(feature_df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="No usable feature columns found.",
        )

    numeric_cols = feature_df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_cols = [
        col
        for col in feature_df.columns
        if col not in numeric_cols
    ]

    return df, numeric_cols, categorical_cols


@app.get("/")
def home():
    return {
        "message": "Preprocessing Pipeline Recommender API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/profile")
async def profile_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...),
):
    validate_task(task)

    df = read_csv_file(await file.read())

    df, numeric_cols, categorical_cols = get_feature_columns(
        df, target
    )

    meta_features = extract_meta_features(
        df,
        target,
        task,
    )

    total_features = len(numeric_cols) + len(categorical_cols)

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "categorical_ratio": (
            len(categorical_cols) / total_features
            if total_features else 0
        ),
        "pct_missing": float(df.isnull().mean().mean()),
        "meta_features": meta_features,
    }


@app.post("/recommend")
async def recommend_pipeline(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...),
):
    validate_task(task)

    df = read_csv_file(await file.read())

    df, numeric_cols, categorical_cols = get_feature_columns(
        df, target
    )

    meta_features = extract_meta_features(
        df,
        target,
        task,
    )

    experience_db = load_experience_db()

    predictions = predict_pipeline_scores(
        meta_features,
        experience_db,
        k=3,
    )

    if predictions is None or predictions.empty:
        raise HTTPException(
            status_code=503,
            detail="No pipeline predictions available.",
        )

    front = pareto_front(predictions)

    if front is None or front.empty:
        raise HTTPException(
            status_code=503,
            detail="No Pareto-optimal pipelines found.",
        )

    chosen = select_best(front)

    spec_columns = [
        "imputer",
        "encoder",
        "scaler",
        "outlier",
        "imbalance",
    ]

    score_columns = [
        "performance",
        "efficiency",
        "simplicity",
        "robustness",
        "explainability",
    ]

    recommended_spec = {
        key: chosen[key]
        for key in spec_columns
        if key in chosen
    }

    scores = {
        key: float(chosen[key])
        for key in score_columns
        if key in chosen
    }

    neighbours = find_similar_datasets(
        meta_features,
        experience_db,
        k=3,
    )

    recommendation_id = str(uuid4())

    RECOMMENDATIONS[recommendation_id] = {
        "meta_features": meta_features,
        "recommended_spec": recommended_spec,
        "scores": scores,
        "pareto_front": front.to_dict(orient="records"),
        "neighbours": neighbours.to_dict(orient="records"),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target": target,
        "task": task,
    }

    return {
        "recommendation_id": recommendation_id,
        "recommended_spec": recommended_spec,
        "scores": scores,
        "pareto_front": front.to_dict(orient="records"),
        "neighbours": neighbours.to_dict(orient="records"),
    }


@app.post("/explain")
async def explain_pipeline(request: ExplainRequest):

    recommendation_id = request.recommendation_id

    if recommendation_id not in RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail="Recommendation not found.",
        )

    recommendation = RECOMMENDATIONS[recommendation_id]

    explanation = explain(
        recommendation["meta_features"],
        recommendation["recommended_spec"],
        recommendation["scores"],
        pd.DataFrame(recommendation["neighbours"]),
        pd.DataFrame(recommendation["pareto_front"]),
    )

    return {
        "recommendation_id": recommendation_id,
        **explanation,
    }


@app.get("/export/{recommendation_id}")
async def export_pipeline(recommendation_id: str):

    if recommendation_id not in RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail="Recommendation not found.",
        )

    recommendation = RECOMMENDATIONS[recommendation_id]

    try:
        code = export_code(
            recommendation["recommended_spec"],
            recommendation["numeric_cols"],
            recommendation["categorical_cols"],
            recommendation["task"],
            recommendation["target"],
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return PlainTextResponse(
        content=code,
        media_type="text/plain",
        headers={
            "Content-Disposition":
                "attachment; filename=preprocessing_pipeline.py"
        },
    )


@app.post("/clean")
async def clean_uploaded_dataset(
    recommendation_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Clean a CSV using the previously recommended pipeline."""

    if recommendation_id not in RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail="Recommendation not found. Generate a new recommendation.",
        )

    df = read_csv_file(await file.read())

    recommendation = RECOMMENDATIONS[recommendation_id]

    try:
        cleaned = clean_dataset(
            df,
            recommendation["target"],
            recommendation["recommended_spec"],
            recommendation["task"],
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    csv_output = cleaned.to_csv(index=False)

    return PlainTextResponse(
        content=csv_output,
        media_type="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=cleaned_dataset.csv"
        },
    )
