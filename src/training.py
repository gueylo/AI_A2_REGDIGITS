"""Deterministic model comparison, selection and profiling."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])


def train_regression(X_train, y_train, X_test, y_test, seed: int):
    candidates = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    selection: dict[str, dict] = {}
    held_out: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipe = _pipeline(estimator)
        mae_scores = -cross_val_score(
            pipe, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1
        )
        rmse_scores = -cross_val_score(
            pipe, X_train, y_train, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1
        )
        selection[name] = {
            "cv_mae_mean": float(mae_scores.mean()),
            "cv_mae_std": float(mae_scores.std()),
            "cv_rmse_mean": float(rmse_scores.mean()),
        }
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        prediction = pipe.predict(X_test)
        held_out[name] = {
            "mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "r2": float(r2_score(y_test, prediction)),
        }
    chosen = min(candidates, key=lambda name: (selection[name]["cv_mae_mean"], selection[name]["cv_rmse_mean"]))
    final = fitted[chosen]
    prediction = final.predict(X_test)
    worst = int(np.argmax(np.abs(np.asarray(y_test) - prediction)))
    report = {
        "selection_method": "lowest mean 5-fold training-only CV MAE; CV RMSE tie-break",
        "cv_results": selection,
        "held_out_metrics": held_out,
        "selected_model": chosen,
        "selection_reason": f"{chosen} had the lowest training-only mean CV MAE",
        "largest_test_error": {
            "position": worst,
            "actual_yield_tons": float(np.asarray(y_test)[worst]),
            "predicted_yield_tons": float(prediction[worst]),
            "absolute_error_tons": float(abs(np.asarray(y_test)[worst] - prediction[worst])),
        },
    }
    return final, prediction, report


def train_classification(X_train, y_train, X_test, y_test, seed: int):
    candidates = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    selection: dict[str, dict] = {}
    held_out: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipe = _pipeline(estimator)
        f1_scores = cross_val_score(pipe, X_train, y_train, scoring="f1", cv=cv, n_jobs=1)
        recall_scores = cross_val_score(pipe, X_train, y_train, scoring="recall", cv=cv, n_jobs=1)
        selection[name] = {
            "cv_f1_mean": float(f1_scores.mean()),
            "cv_f1_std": float(f1_scores.std()),
            "cv_recall_mean": float(recall_scores.mean()),
        }
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        prediction = pipe.predict(X_test).astype(int)
        classes = pipe.named_steps["model"].classes_
        class_one_index = int(np.where(classes == 1)[0][0])
        probability = pipe.predict_proba(X_test)[:, class_one_index]
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        auc = float(roc_auc_score(y_test, probability)) if len(np.unique(y_test)) == 2 else None
        held_out[name] = {
            "precision_high_risk": float(precision_score(y_test, prediction, zero_division=0)),
            "recall_high_risk": float(recall_score(y_test, prediction, zero_division=0)),
            "f1_high_risk": float(f1_score(y_test, prediction, zero_division=0)),
            "confusion_matrix_labels_0_1": matrix.tolist(),
            "roc_auc": auc,
            "zero_division": 0,
        }
    chosen = max(candidates, key=lambda name: (selection[name]["cv_f1_mean"], selection[name]["cv_recall_mean"]))
    final = fitted[chosen]
    prediction = final.predict(X_test).astype(int)
    classes = final.named_steps["model"].classes_
    probability = final.predict_proba(X_test)[:, int(np.where(classes == 1)[0][0])]
    errors = np.flatnonzero(prediction != np.asarray(y_test))
    example = None
    if len(errors):
        pos = int(errors[0])
        example = {
            "position": pos,
            "actual_high_risk": int(np.asarray(y_test)[pos]),
            "predicted_high_risk": int(prediction[pos]),
            "high_risk_probability": float(probability[pos]),
            "error_type": "false_negative" if y_test.iloc[pos] == 1 else "false_positive",
        }
    report = {
        "selection_method": "highest mean 5-fold stratified training-only CV F1; recall tie-break",
        "cv_results": selection,
        "held_out_metrics": held_out,
        "selected_model": chosen,
        "selection_reason": f"{chosen} had the highest training-only mean CV F1",
        "example_classification_error": example,
    }
    return final, prediction, probability, report


def train_clustering(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int):
    preprocess = make_preprocessor()
    transformed = preprocess.fit_transform(X_train)
    scores: dict[str, dict] = {}
    models: dict[int, KMeans] = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = model.fit_predict(transformed)
        scores[str(k)] = {
            "silhouette": float(silhouette_score(transformed, labels)),
            "cluster_counts": {str(key): int(value) for key, value in zip(*np.unique(labels, return_counts=True))},
        }
        models[k] = model
    selected_k = max(range(2, 7), key=lambda k: (scores[str(k)]["silhouette"], -k))
    cluster_model = models[selected_k]
    pipeline = Pipeline([("preprocess", preprocess), ("model", cluster_model)])
    train_labels = cluster_model.labels_
    test_labels = pipeline.predict(X_test).astype(int)
    profiles = []
    profiled = X_train.reset_index(drop=True).copy()
    profiled["cluster"] = train_labels
    for cluster_id in range(selected_k):
        group = profiled[profiled["cluster"] == cluster_id]
        profiles.append(
            {
                "cluster": cluster_id,
                "count": len(group),
                "numeric_means": {
                    column: float(group[column].mean())
                    for column in [
                        "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha"
                    ]
                },
                "cooperative_support_proportion": float(group["cooperative_support"].mean()),
                "season_distribution": {
                    str(key): float(value) for key, value in group["season"].value_counts(normalize=True).sort_index().items()
                },
                "interpretation_note": "Cluster IDs group similar input conditions and are not quality rankings.",
            }
        )
    report = {
        "selection_method": "maximum training silhouette; smaller K breaks exact ties",
        "scores_k_2_to_6": scores,
        "selected_k": selected_k,
        "profiles_original_units": profiles,
        "limitations": "Unsupervised similarity groups are descriptive, not agronomic quality grades.",
    }
    return pipeline, transformed, train_labels, test_labels, report

