"""Plotting utilities for age estimation benchmarking and model comparison."""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Union_Path = Union[str, Path]


def plot_ground_truth_vs_prediction(
    y_true: List[float],
    y_pred: List[float],
    output_path: Union_Path,
    model_name: str = "MiVOLO Pretrained (Official)",
) -> Path:
    """Generates and saves a scatter plot comparing Ground Truth vs. Predicted Age."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    # Scatter points
    ax.scatter(yt, yp, alpha=0.7, edgecolors="none", s=50, c="#1f77b4", label="Predictions")

    # Perfect prediction line (y = x)
    min_val = min(float(np.min(yt)), float(np.min(yp)))
    max_val = max(float(np.max(yt)), float(np.max(yp)))
    margin = (max_val - min_val) * 0.05
    line_min = max(0.0, min_val - margin)
    line_max = max_val + margin

    ax.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=1.5, label="Ideal (y = x)")

    # Linear trendline
    if len(yt) > 1 and np.std(yt) > 1e-6:
        poly = np.polyfit(yt, yp, 1)
        poly_fn = np.poly1d(poly)
        ax.plot(
            np.linspace(line_min, line_max, 50),
            poly_fn(np.linspace(line_min, line_max, 50)),
            "g-",
            linewidth=1.2,
            label=f"Fit (y = {poly[0]:.2f}x + {poly[1]:.2f})",
        )

    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))

    ax.set_title(f"{model_name}\nGround Truth vs. Predicted Age (MAE: {mae:.2f}, RMSE: {rmse:.2f})", fontsize=12)
    ax.set_xlabel("Ground Truth Age (Years)", fontsize=11)
    ax.set_ylabel("Predicted Age (Years)", fontsize=11)
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.legend(loc="upper left")
    fig.tight_layout()

    fig.savefig(str(output_path))
    plt.close(fig)
    return output_path


def plot_error_distribution(
    y_true: List[float],
    y_pred: List[float],
    output_path: Union_Path,
    model_name: str = "MiVOLO Pretrained (Official)",
) -> Path:
    """Generates and saves a residual error distribution plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors = np.asarray(y_pred) - np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    sns.histplot(errors, kde=True, color="#2ca02c", ax=ax, bins=15, stat="density")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvline(float(np.mean(errors)), color="blue", linestyle=":", linewidth=1.5, label=f"Mean Error: {np.mean(errors):.2f}")

    ax.set_title(f"{model_name}\nResidual Error Distribution (Predicted - True)", fontsize=12)
    ax.set_xlabel("Prediction Error in Years", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig.savefig(str(output_path))
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    output_path: Union_Path,
    title: str = "Child Safety Classification Confusion Matrix",
) -> Path:
    """Generates and saves a confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.set_theme(style="white")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Ground Truth Label", fontsize=11)
    fig.tight_layout()

    fig.savefig(str(output_path))
    plt.close(fig)
    return output_path
