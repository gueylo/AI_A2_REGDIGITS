"""FastAPI interface over locally persisted model pipelines."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.inference import ModelService
from src.validation import FarmInput

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0")


def get_service() -> ModelService:
    try:
        return ModelService(ROOT)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict:
    service = get_service()
    return {"status": "healthy", "verification_code": service.metadata["verification_code"]}


@app.get("/model-info")
async def model_info() -> dict:
    metadata = get_service().metadata
    return {
        **metadata,
        "limitations": [
            "Trained only on synthetic data.",
            "Probabilities are model estimates, not agronomic truths.",
            "Must not determine finance, insurance, or public-service access.",
            "Seek human agronomist review before operational use.",
        ],
    }


@app.post("/predict")
async def predict(farm: FarmInput) -> dict:
    return get_service().predict(farm)

