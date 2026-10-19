from __future__ import annotations

import json

import joblib


def test_all_saved_pipelines_load_and_metadata_is_consistent():
    from conftest import ROOT

    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    for name in ("regression", "classification", "clustering"):
        pipeline = joblib.load(ROOT / "models" / f"{name}.joblib")
        assert list(pipeline.named_steps) == ["preprocess", "model"]
    assert metadata["verification_code"] == "AI-A2-7953"
    assert metadata["synthetic_prototype"] is True


def test_every_required_k_was_evaluated():
    from conftest import ROOT

    report = json.loads((ROOT / "artifacts" / "clustering_report.json").read_text())
    assert set(report["scores_k_2_to_6"]) == {"2", "3", "4", "5", "6"}
    assert report["selected_k"] in range(2, 7)

