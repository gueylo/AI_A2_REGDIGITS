"""Dataset and request validation rules."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]
TARGETS = ["yield_tons", "high_risk"]
COLUMNS = FEATURES + TARGETS
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}


class FarmInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rainfall_mm: StrictFloat = Field(ge=600, le=1400)
    avg_temp_c: StrictFloat = Field(ge=12, le=26)
    soil_moisture_pct: StrictFloat = Field(ge=20, le=80)
    fertilizer_kg: StrictFloat = Field(ge=0, le=250)
    field_size_ha: StrictFloat = Field(ge=0.2, le=4.0)
    season: Literal["A", "B"]
    cooperative_support: StrictInt = Field(ge=0, le=1)

    @field_validator(
        "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha"
    )
    @classmethod
    def finite_numbers(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


def validate_raw_data(frame: pd.DataFrame, base_rows: int) -> dict:
    """Fail on invalid raw schema/content and return an audit record."""
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"expected columns {COLUMNS}, received {list(frame.columns)}")
    if len(frame) != base_rows + 3:
        raise ValueError(f"expected {base_rows + 3} rows, received {len(frame)}")
    duplicates = int(frame.duplicated().sum())
    if duplicates != 3:
        raise ValueError(f"expected exactly 3 duplicate rows, received {duplicates}")
    for column, (low, high) in RANGES.items():
        non_null = frame[column].dropna()
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValueError(f"{column} contains non-finite or non-numeric values")
        if not numeric.between(low, high, inclusive="both").all():
            raise ValueError(f"{column} contains values outside [{low}, {high}]")
    allowed_missing = {"soil_moisture_pct", "fertilizer_kg"}
    unexpected = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if count and column not in allowed_missing
    }
    if unexpected:
        raise ValueError(f"unexpected missing values: {unexpected}")
    if not set(frame["season"].dropna().unique()).issubset({"A", "B"}):
        raise ValueError("season must contain only A or B")
    for column in ("cooperative_support", "high_risk"):
        if not set(frame[column].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{column} must be binary")
    if (frame["yield_tons"] < 0.5).any() or not np.isfinite(frame["yield_tons"]).all():
        raise ValueError("yield_tons must be finite and at least 0.5")
    return {
        "base_rows": base_rows,
        "raw_rows": len(frame),
        "duplicate_rows": duplicates,
        "clean_rows": len(frame.drop_duplicates()),
        "missing_counts": {key: int(value) for key, value in frame.isna().sum().items()},
        "schema_valid": True,
    }


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates only; keep missing inputs for train-only imputation."""
    return frame.drop_duplicates().copy()

