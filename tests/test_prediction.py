from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.inference import ModelService
from src.validation import FarmInput


def test_valid_saved_model_inference(valid_payload):
    from conftest import ROOT

    result = ModelService(ROOT).predict(FarmInput.model_validate(valid_payload))
    assert result["verification_code"] == "AI-A2-7953"
    assert isinstance(result["estimated_yield_tons"], float)
    assert result["high_risk"] in (0, 1)
    assert 0 <= result["high_risk_probability"] <= 1
    assert isinstance(result["cluster"], int)
    assert result["synthetic_prototype"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"season": None},
        {"rainfall_mm": "950"},
        {"rainfall_mm": 2000.0},
        {"cooperative_support": 2},
        {"extra_field": 10},
    ],
)
def test_invalid_prediction_input_is_rejected(valid_payload, change):
    payload = {**valid_payload, **change}
    if change == {"season": None}:
        payload.pop("season")
    with pytest.raises(ValidationError):
        FarmInput.model_validate(payload)

