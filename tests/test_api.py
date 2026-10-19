from __future__ import annotations

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_and_model_info():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "verification_code": "AI-A2-7953"}
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["features"] == [
        "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
        "field_size_ha", "season", "cooperative_support",
    ]


def test_api_prediction(valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert response.json()["verification_code"] == "AI-A2-7953"


def test_api_missing_wrong_type_and_range_return_422(valid_payload):
    missing = dict(valid_payload)
    missing.pop("season")
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**valid_payload, "rainfall_mm": "950"}).status_code == 422
    assert client.post("/predict", json={**valid_payload, "rainfall_mm": 2000.0}).status_code == 422


def test_cli_and_api_share_identical_result(valid_payload, tmp_path):
    from conftest import ROOT
    from src.inference import ModelService
    from src.validation import FarmInput

    direct = ModelService(ROOT).predict(FarmInput.model_validate(valid_payload))
    api_result = client.post("/predict", json=valid_payload).json()
    assert api_result == direct

