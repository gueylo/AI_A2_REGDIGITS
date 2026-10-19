from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from data.generate_data import generate_dataset
from src.identity import identity_from_student_id
from src.validation import FEATURES, clean_data, validate_raw_data


def test_identity_and_wrong_seed_rejection():
    identity = identity_from_student_id("ug2527953@ines.ac.rw", 7953)
    assert identity.last_four == "7953"
    assert identity.verification_code == "AI-A2-7953"
    with pytest.raises(ValueError, match="seed mismatch"):
        identity_from_student_id("ug2527953@ines.ac.rw", 9999)


def test_generation_is_byte_reproducible(tmp_path):
    identity = identity_from_student_id("ug2527953@ines.ac.rw", 7953)
    path_one, path_two = tmp_path / "one.csv", tmp_path / "two.csv"
    first, details = generate_dataset(identity, path_one)
    second, _ = generate_dataset(identity, path_two)
    assert path_one.read_bytes() == path_two.read_bytes()
    assert hashlib.sha256(path_one.read_bytes()).hexdigest() == details["sha256"]
    assert first.equals(second)


def test_raw_duplicates_missingness_and_cleaning(tmp_path):
    identity = identity_from_student_id("ug2527953@ines.ac.rw", 7953)
    frame, details = generate_dataset(identity, tmp_path / "farms.csv")
    audit = validate_raw_data(frame, details["base_rows"])
    assert audit["duplicate_rows"] == 3
    assert len(clean_data(frame)) == details["base_rows"]
    assert frame["soil_moisture_pct"].isna().sum() >= details["missing_positions_per_column"]
    assert frame["fertilizer_kg"].isna().sum() >= details["missing_positions_per_column"]


def test_feature_allowlist_has_no_target_leakage():
    assert FEATURES == [
        "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg",
        "field_size_ha", "season", "cooperative_support",
    ]
    assert not {"yield_tons", "yield_per_hectare", "high_risk", "row_id"}.intersection(FEATURES)

