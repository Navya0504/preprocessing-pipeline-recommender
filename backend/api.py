"""FastAPI app. Run from the project root with:

    uvicorn backend.api:app --reload

Then open http://127.0.0.1:8000/docs for the interactive API page.

Contract: see INTEGRATION_CONTRACT.md, section 3 (Person 6 table).
STATUS: /health and /profile work with the stub modules. /recommend, /explain
and /export are TODO until the engine is real.
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend.meta_features import extract_meta_features

app = FastAPI(
    title="Preprocessing Pipeline Recommender",
    description="Multi-objective, explainable preprocessing pipeline recommendation.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/")
def root() -> dict:
    return {"message": "Preprocessing Pipeline Recommender API", "docs": "/docs"}



@app.post("/profile")
async def profile(
    file: UploadFile = File(...),
    target: str = Form(...),
    task: str = Form(...),
) -> dict:
    """Return the meta-features of an uploaded CSV. The file is read in memory only."""
    if task not in ("classification", "regression"):
        raise HTTPException(status_code=422, detail="task must be 'classification' or 'regression'")
    df = pd.read_csv(io.BytesIO(await file.read()))
    if target not in df.columns:
        raise HTTPException(status_code=422, detail=f"target column '{target}' not found")
    return extract_meta_features(df, target, task)


# TODO: POST /recommend, POST /explain, GET /export/{id}
