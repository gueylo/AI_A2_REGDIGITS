"""Deterministic synthetic farm dataset generator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.identity import Identity
from src.validation import COLUMNS


def generate_dataset(identity: Identity, csv_path: Path) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(identity.seed)
    base_rows = 240 + (identity.seed % 41)
    missing_each = round(0.05 * base_rows)

    rainfall = rng.uniform(600, 1400, base_rows)
    temperature = rng.uniform(12, 26, base_rows)
    moisture = rng.uniform(20, 80, base_rows)
    fertilizer = rng.uniform(0, 250, base_rows)
    field_size = rng.uniform(0.2, 4.0, base_rows)
    season = rng.choice(np.array(["A", "B"]), base_rows)
    support = rng.integers(0, 2, base_rows)
    noise = rng.normal(0, 0.80, base_rows)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * moisture
        + 0.02 * fertilizer
        + 1.30 * field_size
        - 0.25 * np.abs(temperature - 18)
        + 0.70 * support
        + np.where(season == "A", 0.50, -0.20)
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)
    yield_per_hectare = yield_tons / field_size
    risk_median = float(np.median(yield_per_hectare))
    high_risk = (yield_per_hectare < risk_median).astype(int)

    frame = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temperature,
            "soil_moisture_pct": moisture,
            "fertilizer_kg": fertilizer,
            "field_size_ha": field_size,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        },
        columns=COLUMNS,
    )
    moisture_rows = rng.choice(base_rows, size=missing_each, replace=False)
    fertilizer_rows = rng.choice(base_rows, size=missing_each, replace=False)
    frame.loc[moisture_rows, "soil_moisture_pct"] = np.nan
    frame.loc[fertilizer_rows, "fertilizer_kg"] = np.nan
    frame = pd.concat([frame, frame.iloc[:3].copy()], ignore_index=True)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    saved = pd.read_csv(csv_path)
    details = {
        "verification_code": identity.verification_code,
        "base_rows": base_rows,
        "raw_rows": len(saved),
        "missing_positions_per_column": missing_each,
        "risk_median_before_missing_and_duplicates": risk_median,
        "sha256": digest,
    }
    return saved, details

