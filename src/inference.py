"""Shared saved-model inference for the CLI and HTTP API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.validation import FEATURES, FarmInput


class ModelService:
    def __init__(self, project_root: Path):
        model_dir = project_root / "models"
        required = [
            model_dir / "regression.joblib",
            model_dir / "classification.joblib",
            model_dir / "clustering.joblib",
            model_dir / "model_metadata.json",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "saved model artifacts are unavailable: " + ", ".join(missing) + "; run run_all.py first"
            )
        self.regression = joblib.load(required[0])
        self.classification = joblib.load(required[1])
        self.clustering = joblib.load(required[2])
        self.metadata = json.loads(required[3].read_text(encoding="utf-8"))

    def predict(self, farm: FarmInput) -> dict:
        row = pd.DataFrame([farm.model_dump()], columns=FEATURES)
        estimated = float(self.regression.predict(row)[0])
        risk = int(self.classification.predict(row)[0])
        classes = self.classification.named_steps["model"].classes_
        index = int(np.where(classes == 1)[0][0])
        probability = float(self.classification.predict_proba(row)[0, index])
        cluster = int(self.clustering.predict(row)[0])
        return {
            "verification_code": self.metadata["verification_code"],
            "estimated_yield_tons": estimated,
            "high_risk": risk,
            "high_risk_probability": probability,
            "cluster": cluster,
            "synthetic_prototype": True,
        }

