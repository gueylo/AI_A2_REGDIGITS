"""Metrics, JSON and plot helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_residuals(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    residual = actual - predicted
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(predicted, residual, alpha=0.75, edgecolor="none")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Predicted yield (tons)")
    axis.set_ylabel("Residual: actual - predicted (tons)")
    axis.set_title("Held-out regression residuals")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5))
    display = ConfusionMatrixDisplay(matrix, display_labels=["not high risk", "high risk"])
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Held-out classification results")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_clusters(points: np.ndarray, labels: np.ndarray, path: Path, seed: int) -> None:
    from sklearn.decomposition import PCA

    projection = PCA(n_components=2, random_state=seed).fit_transform(points)
    fig, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(projection[:, 0], projection[:, 1], c=labels, cmap="tab10", alpha=0.8)
    axis.set_xlabel("PCA component 1 (visualization only)")
    axis.set_ylabel("PCA component 2 (visualization only)")
    axis.set_title("Training farms by selected K-Means cluster")
    fig.colorbar(scatter, ax=axis, label="Cluster ID (not a ranking)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

