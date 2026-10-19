#!/usr/bin/env python3
"""Regenerate every machine-produced project artifact."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.generate_data import generate_dataset
from src.evaluation import plot_clusters, plot_confusion, plot_residuals, write_json
from src.identity import identity_from_student_id
from src.reporting import generate_declaration_template, generate_technical_report
from src.training import train_classification, train_clustering, train_regression
from src.validation import FEATURES, clean_data, validate_raw_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True, help="institutional username/email")
    parser.add_argument("--seed", required=True, type=int, help="integer value of last four registration digits")
    return parser.parse_args()


def run(student_id: str, seed: int, root: Path = ROOT) -> dict:
    identity = identity_from_student_id(student_id, seed)
    data_path = root / "data" / f"AI_A2_{identity.registration_digits}.csv"
    frame, generation = generate_dataset(identity, data_path)
    validation = validate_raw_data(frame, generation["base_rows"])
    validation.update(identity.as_dict())
    validation["sha256"] = generation["sha256"]
    write_json(root / "artifacts" / "data_validation.json", validation)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "data_fingerprint.txt").write_text(
        f"verification_code={identity.verification_code}\nsha256={generation['sha256']}\n",
        encoding="utf-8",
    )

    cleaned = clean_data(frame)
    cleaned["row_id"] = cleaned.index.astype(int)
    train_ids, test_ids = train_test_split(
        cleaned.index,
        test_size=0.20,
        random_state=seed,
        stratify=cleaned["high_risk"],
    )
    train = cleaned.loc[train_ids].copy()
    test = cleaned.loc[test_ids].copy()
    X_train, X_test = train[FEATURES], test[FEATURES]

    regression, yield_prediction, regression_report = train_regression(
        X_train, train["yield_tons"], X_test, test["yield_tons"], seed
    )
    classification, risk_prediction, risk_probability, classification_report = train_classification(
        X_train, train["high_risk"], X_test, test["high_risk"], seed
    )
    clustering, cluster_points, train_clusters, test_clusters, cluster_report = train_clustering(
        X_train, X_test, seed
    )

    regression_report.update(identity.as_dict())
    regression_report["largest_test_error"]["row_id"] = int(test.iloc[regression_report["largest_test_error"]["position"]]["row_id"])
    classification_report.update(identity.as_dict())
    if classification_report["example_classification_error"] is not None:
        error = classification_report["example_classification_error"]
        error["row_id"] = int(test.iloc[error["position"]]["row_id"])
    cluster_report.update(identity.as_dict())

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regression, model_dir / "regression.joblib")
    joblib.dump(classification, model_dir / "classification.joblib")
    joblib.dump(clustering, model_dir / "clustering.joblib")

    metadata = {
        **identity.as_dict(),
        "features": FEATURES,
        "data_sha256": generation["sha256"],
        "selected_regression_model": regression_report["selected_model"],
        "selected_classification_model": classification_report["selected_model"],
        "selected_k": cluster_report["selected_k"],
        "targets": {
            "yield_tons": "synthetic yield in tons from the prescribed formula",
            "high_risk": "1 when base-row yield per hectare is strictly below its median",
        },
        "synthetic_prototype": True,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    write_json(model_dir / "model_metadata.json", metadata)
    write_json(root / "artifacts" / "regression_metrics.json", regression_report)
    write_json(root / "artifacts" / "classification_metrics.json", classification_report)
    write_json(root / "artifacts" / "clustering_report.json", cluster_report)

    actual_yield = test["yield_tons"].to_numpy()
    predictions = pd.DataFrame(
        {
            "row_id": test["row_id"].to_numpy(),
            "actual_yield_tons": actual_yield,
            "predicted_yield_tons": yield_prediction,
            "residual": actual_yield - yield_prediction,
            "absolute_error": abs(actual_yield - yield_prediction),
            "actual_high_risk": test["high_risk"].to_numpy(dtype=int),
            "predicted_high_risk": risk_prediction,
            "high_risk_probability": risk_probability,
            "cluster": test_clusters,
        }
    )
    predictions.to_csv(root / "artifacts" / "test_predictions.csv", index=False, float_format="%.6f")
    plot_dir = root / "artifacts" / "plots"
    plot_residuals(actual_yield, yield_prediction, plot_dir / "residuals.png")
    plot_confusion(
        confusion_matrix(test["high_risk"], risk_prediction, labels=[0, 1]),
        plot_dir / "confusion_matrix.png",
    )
    plot_clusters(cluster_points, train_clusters, plot_dir / "clusters.png", seed)
    generate_technical_report(root)
    declaration_path = root / "evidence" / "AI_A2_2527953_DECLARATION.pdf"
    if not declaration_path.exists():
        generate_declaration_template(root)

    summary = {
        "verification_code": identity.verification_code,
        "student_id": identity.student_id,
        "base_rows": generation["base_rows"],
        "raw_rows": len(frame),
        "clean_rows": len(cleaned),
        "train_rows": len(train),
        "test_rows": len(test),
        "sha256": generation["sha256"],
        "selected_regression": regression_report["selected_model"],
        "selected_classification": classification_report["selected_model"],
        "selected_k": cluster_report["selected_k"],
    }
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args.student_id, args.seed)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("Musanze Harvest build complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("Report and declaration template refreshed; student review/signature and personal video remain required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

