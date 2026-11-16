
import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.api as api


client = TestClient(api.app)

DATA = Path(__file__).resolve().parent.parent / "experiments" / "data"

CSV_DATA = b"age,city,label\n22,Delhi,1\n25,Mumbai,0\n30,Delhi,1\n35,Chennai,0\n"

NOT_FOUND_DETAIL = (
    "Recommendation not found or expired. Please upload your file again."
)

SPEC = {
    "imputer": "median",
    "encoder": "onehot",
    "scaler": "standard",
    "outlier": "none",
    "imbalance": "none",
}

SCORES = {
    "performance": 0.85,
    "efficiency": 0.80,
    "simplicity": 0.75,
    "robustness": 0.90,
    "explainability": 0.80,
}


def setup_recommendation_mocks(monkeypatch):
    """Mock recommendation dependencies for API tests."""

    predictions = pd.DataFrame([{**SPEC, **SCORES}])

    monkeypatch.setattr(
        api,
        "extract_meta_features",
        lambda df, target, task: {"task": task, "n_rows": len(df)}
    )

    monkeypatch.setattr(
        api,
        "load_experience_db",
        lambda: pd.DataFrame({"dataset_id": ["test"]})
    )

    monkeypatch.setattr(
        api,
        "predict_pipeline_scores",
        lambda meta, db, k=5: predictions
    )

    monkeypatch.setattr(api, "pareto_front", lambda df: df)

    monkeypatch.setattr(
        api,
        "select_best",
        lambda df: {**SPEC, **SCORES}
    )

    monkeypatch.setattr(
        api,
        "find_similar_datasets",
        lambda meta, db, k=5: pd.DataFrame(
            [{"dataset_id": "test", "distance": 0.1}]
        )
    )


def make_recommendation(monkeypatch):
    """Create a recommendation for testing dependent endpoints."""

    setup_recommendation_mocks(monkeypatch)

    response = client.post(
        "/recommend",
        files={"file": ("data.csv", CSV_DATA, "text/csv")},
        data={"target": "label", "task": "classification"}
    )

    assert response.status_code == 200
    return response.json()


def test_recommend(monkeypatch):
    """Test successful pipeline recommendation."""

    result = make_recommendation(monkeypatch)

    assert "recommendation_id" in result
    assert result["recommended_spec"] == SPEC
    assert "scores" in result
    assert "pareto_front" in result
    assert "neighbours" in result


def test_recommend_samples_large_dataset(monkeypatch):
    """Check that large datasets are sampled to 2000 rows using seed 42."""

    setup_recommendation_mocks(monkeypatch)

    captured = {}

    def fake_extract_meta_features(df, target, task):
        captured["rows"] = len(df)
        captured["ages"] = df["age"].tolist()
        return {"task": task, "n_rows": len(df)}

    monkeypatch.setattr(
        api,
        "extract_meta_features",
        fake_extract_meta_features,
    )

    large_df = pd.DataFrame({
        "age": range(2500),
        "label": [i % 2 for i in range(2500)],
    })

    expected = (
        large_df.sample(n=2000, random_state=42)
        .reset_index(drop=True)
    )

    response = client.post(
        "/recommend",
        files={
            "file": (
                "large.csv",
                large_df.to_csv(index=False).encode("utf-8"),
                "text/csv",
            )
        },
        data={"target": "label", "task": "classification"},
    )

    assert response.status_code == 200
    assert captured["rows"] == 2000
    assert captured["ages"] == expected["age"].tolist()


def test_explain(monkeypatch):
    """Test successful recommendation explanation."""

    result = make_recommendation(monkeypatch)

    monkeypatch.setattr(
        api,
        "explain",
        lambda *args: {
            "summary": "Recommended pipeline",
            "steps": {"imputer": {"method": "median", "status": "applied", "note": "Fills missing values."}},
            "reasons": ["Suitable for the dataset"],
            "tradeoffs": ["Some extra processing"]
        }
    )

    response = client.post(
        "/explain",
        json={"recommendation_id": result["recommendation_id"]}
    )

    assert response.status_code == 200
    assert "summary" in response.json()
    assert "steps" in response.json()


def test_export(monkeypatch):
    """Test downloading generated pipeline code."""

    result = make_recommendation(monkeypatch)

    monkeypatch.setattr(
        api,
        "export_code",
        lambda spec, numeric, categorical, task, target:
            "# Recommended pipeline"
    )

    response = client.get(
        f"/export/{result['recommendation_id']}"
    )

    assert response.status_code == 200
    assert "Recommended pipeline" in response.text


def test_unknown_target():
    """Test recommendation with a missing target column."""

    response = client.post(
        "/recommend",
        files={"file": ("data.csv", CSV_DATA, "text/csv")},
        data={"target": "unknown", "task": "classification"}
    )

    assert response.status_code == 422


def test_invalid_task():
    """Test recommendation with an unsupported task."""

    response = client.post(
        "/recommend",
        files={"file": ("data.csv", CSV_DATA, "text/csv")},
        data={"target": "label", "task": "invalid"}
    )

    assert response.status_code == 422


def test_empty_file():
    """Test recommendation with an empty CSV."""

    response = client.post(
        "/recommend",
        files={"file": ("data.csv", b"", "text/csv")},
        data={"target": "label", "task": "classification"}
    )

    assert response.status_code == 400


def test_binary_file():
    """Test recommendation with a binary file."""

    response = client.post(
        "/recommend",
        files={
            "file": ("data.csv", b"\xff\xfe\xfa\x00", "text/csv")
        },
        data={"target": "label", "task": "classification"}
    )

    assert response.status_code == 400


def test_health():
    """Test the health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_clean(monkeypatch):
    """Test cleaning a dataset using a saved recommendation."""

    result = make_recommendation(monkeypatch)

    expected = pd.DataFrame({
        "age": [22, 25, 30],
        "label": [1, 0, 1]
    })

    monkeypatch.setattr(
        api,
        "clean_dataset",
        lambda df, target, spec, task: expected
    )

    response = client.post(
        "/clean",
        data={"recommendation_id": result["recommendation_id"]},
        files={"file": ("data.csv", CSV_DATA, "text/csv")}
    )

    assert response.status_code == 200
    assert "age,label" in response.text


def test_clean_unknown_recommendation():
    """Test cleaning with an unknown recommendation ID."""

    response = client.post(
        "/clean",
        data={"recommendation_id": "invalid-id"},
        files={"file": ("data.csv", CSV_DATA, "text/csv")}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == NOT_FOUND_DETAIL


def test_recommend_uses_k_three(monkeypatch):
    """Check that recommendation functions receive k=3."""

    calls = {}

    setup_recommendation_mocks(monkeypatch)

    def fake_predict(meta, db, k=5):
        calls["predict"] = k
        return pd.DataFrame([{**SPEC, **SCORES}])

    def fake_neighbours(meta, db, k=5):
        calls["neighbours"] = k
        return pd.DataFrame(
            [{"dataset_id": "test", "distance": 0.1}]
        )

    monkeypatch.setattr(api, "predict_pipeline_scores", fake_predict)
    monkeypatch.setattr(api, "find_similar_datasets", fake_neighbours)

    response = client.post(
        "/recommend",
        files={"file": ("data.csv", CSV_DATA, "text/csv")},
        data={"target": "label", "task": "classification"}
    )

    assert response.status_code == 200
    assert calls["predict"] == 3
    assert calls["neighbours"] == 3


def test_clean_empty_file(monkeypatch):
    """Test cleaning with an empty uploaded file."""

    result = make_recommendation(monkeypatch)

    response = client.post(
        "/clean",
        data={"recommendation_id": result["recommendation_id"]},
        files={"file": ("empty.csv", b"", "text/csv")}
    )

    assert response.status_code == 400


def test_explain_unknown_recommendation():
    """Test explanation with an unknown recommendation ID."""

    response = client.post(
        "/explain",
        json={"recommendation_id": "invalid-id"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == NOT_FOUND_DETAIL


def test_export_unknown_recommendation():
    """Test export with an unknown recommendation ID."""

    response = client.get("/export/invalid-id")

    assert response.status_code == 404
    assert response.json()["detail"] == NOT_FOUND_DETAIL


@pytest.mark.parametrize(
    "file_name,target,task",
    [
        ("iris.csv", "species", "classification"),
        ("insurance.csv", "charges", "regression"),
    ],
)
def test_end_to_end_real_engine(file_name, target, task):
    """Run the whole API flow with the real engine and no mocks."""

    path = DATA / file_name

    if not path.exists():
        pytest.skip("dataset file not available")

    content = path.read_bytes()

    recommendation = client.post(
        "/recommend",
        files={"file": (file_name, content, "text/csv")},
        data={"target": target, "task": task},
    )

    assert recommendation.status_code == 200, recommendation.text

    recommendation_id = recommendation.json()["recommendation_id"]

    explanation = client.post(
        "/explain",
        json={"recommendation_id": recommendation_id},
    )

    assert explanation.status_code == 200, explanation.text
    steps = explanation.json()["steps"]
    assert set(steps) == {"imputer", "encoder", "scaler", "outlier", "imbalance"}
    assert all(
        step["status"] in {"applied", "skipped", "not_needed", "not_applicable"}
        for step in steps.values()
    )

    export = client.get(f"/export/{recommendation_id}")

    assert export.status_code == 200, export.text

    cleaned = client.post(
        "/clean",
        data={"recommendation_id": recommendation_id},
        files={"file": (file_name, content, "text/csv")},
    )

    assert cleaned.status_code == 200, cleaned.text


def test_recommendation_persists_after_memory_clear(monkeypatch, tmp_path):
    """Recommendation should be available after clearing in-memory storage."""

    monkeypatch.setattr(api, "DATA_DIR", tmp_path)

    response = make_recommendation(monkeypatch)

    recommendation_id = response["recommendation_id"]

    api.RECOMMENDATIONS.clear()

    saved_recommendation = api.get_recommendation(recommendation_id)

    assert saved_recommendation is not None
    assert saved_recommendation["task"] == "classification"
    assert (tmp_path / f"{recommendation_id}.json").exists()


def test_explain_rejects_path_traversal():
    """Test that explain rejects path traversal IDs."""

    response = client.post(
        "/explain",
        json={"recommendation_id": "../experiments/results"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == NOT_FOUND_DETAIL


def test_clean_rejects_path_traversal():
    """Test that clean rejects path traversal IDs."""

    response = client.post(
        "/clean",
        data={"recommendation_id": "../experiments/results"},
        files={"file": ("data.csv", CSV_DATA, "text/csv")},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == NOT_FOUND_DETAIL


def test_serve_homepage():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_serve_upload_page():
    response = client.get("/upload.html")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_api_routes_still_work():
    health_response = client.get("/health")
    docs_response = client.get("/docs")

    assert health_response.status_code == 200
    assert docs_response.status_code == 200


def test_tidy_messy_dataset():
    """Test that tidy cleans spaces, missing markers, and numeric text."""

    df = pd.DataFrame({
        "age": [" 25 ", "30", "?"],
        "city": [" Delhi ", "Mumbai ", " Chennai "],
    })

    result = api.tidy(df)

    assert result["age"].dtype == "float64"
    assert result["age"].iloc[0] == 25.0
    assert result["age"].iloc[1] == 30.0
    assert pd.isna(result["age"].iloc[2])

    assert result["city"].tolist() == [
        "Delhi",
        "Mumbai",
        "Chennai",
    ]
