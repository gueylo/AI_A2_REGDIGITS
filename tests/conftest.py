from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def trained_fixture_models():
    """Build the student's models when run_all.py was not run first."""
    metadata = ROOT / "models" / "model_metadata.json"
    valid = False
    if metadata.is_file():
        valid = json.loads(metadata.read_text(encoding="utf-8")).get("verification_code") == "AI-A2-7953"
    if not valid:
        from run_all import run

        return run("ug2527953@ines.ac.rw", 7953, ROOT)
    return {"models": "already built"}


@pytest.fixture()
def valid_payload():
    return {
        "rainfall_mm": 950.0,
        "avg_temp_c": 18.0,
        "soil_moisture_pct": 55.0,
        "fertilizer_kg": 90.0,
        "field_size_ha": 1.5,
        "season": "A",
        "cooperative_support": 1,
    }

