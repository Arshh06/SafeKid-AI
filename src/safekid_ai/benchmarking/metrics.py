"""Benchmarking metrics computation for age estimation and child safety evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class RegressionMetrics:
    """Quantitative regression metrics for continuous age estimation."""
    count: int
    mae: float
    rmse: float
    median_absolute_error: float
    max_error: float
    pearson_r: float
    r_squared: float
    cumulative_score_3yr: float  # Percentage of predictions within +-3 years
    cumulative_score_5yr: float  # Percentage of predictions within +-5 years

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "median_absolute_error": round(self.median_absolute_error, 3),
            "max_error": round(self.max_error, 3),
            "pearson_r": round(self.pearson_r, 4),
            "r_squared": round(self.r_squared, 4),
            "cumulative_score_3yr_pct": round(self.cumulative_score_3yr, 2),
            "cumulative_score_5yr_pct": round(self.cumulative_score_5yr, 2),
        }


@dataclass
class ClassificationMetrics:
    """Binary or multi-class demographic classification metrics (e.g. child vs. adult)."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]
    labels: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


@dataclass
class LatencyMetrics:
    """Execution timing and throughput metrics."""
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    fps: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "throughput_fps": round(self.fps, 2),
        }


def compute_regression_metrics(
    y_true: List[float], y_pred: List[float]
) -> Optional[RegressionMetrics]:
    """Computes MAE, RMSE, Median Absolute Error, Pearson R, R2, and CS metrics.

    Args:
        y_true: Ground-truth target ages.
        y_pred: Predicted target ages.

    Returns:
        RegressionMetrics dataclass or None if insufficient samples.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    # Filter out NaNs
    valid_mask = (~np.isnan(yt)) & (~np.isnan(yp))
    yt = yt[valid_mask]
    yp = yp[valid_mask]

    n = len(yt)
    if n == 0:
        return None

    errors = yp - yt
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    med_ae = float(np.median(abs_errors))
    max_err = float(np.max(abs_errors))

    # Pearson correlation and R2
    if n > 1 and np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
        r, _ = stats.pearsonr(yt, yp)
        r_val = float(r)
        # Coefficient of determination R^2
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    else:
        r_val = 1.0 if np.allclose(yt, yp) else 0.0
        r2 = 1.0 if np.allclose(yt, yp) else 0.0

    cs3 = float(np.mean(abs_errors <= 3.0) * 100.0)
    cs5 = float(np.mean(abs_errors <= 5.0) * 100.0)

    return RegressionMetrics(
        count=n,
        mae=mae,
        rmse=rmse,
        median_absolute_error=med_ae,
        max_error=max_err,
        pearson_r=r_val,
        r_squared=r2,
        cumulative_score_3yr=cs3,
        cumulative_score_5yr=cs5,
    )


def compute_child_classification_metrics(
    y_true: List[float], y_pred: List[float], child_age_cutoff: float = 13.0
) -> Optional[ClassificationMetrics]:
    """Computes binary classification metrics for identifying children (< cutoff).

    Args:
        y_true: Ground truth numeric ages.
        y_pred: Predicted numeric ages.
        child_age_cutoff: Threshold below which a subject is categorized as CHILD.

    Returns:
        ClassificationMetrics for CHILD vs NON_CHILD.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    valid_mask = (~np.isnan(yt)) & (~np.isnan(yp))
    yt = yt[valid_mask]
    yp = yp[valid_mask]

    if len(yt) == 0:
        return None

    # Binary labels: 1 = CHILD, 0 = NON_CHILD (ADULT / TEEN)
    true_labels = (yt < child_age_cutoff).astype(int)
    pred_labels = (yp < child_age_cutoff).astype(int)

    acc = float(accuracy_score(true_labels, pred_labels))
    prec = float(precision_score(true_labels, pred_labels, zero_division=0))
    rec = float(recall_score(true_labels, pred_labels, zero_division=0))
    f1 = float(f1_score(true_labels, pred_labels, zero_division=0))
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 0]).tolist()

    return ClassificationMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm,
        labels=["CHILD", "NON_CHILD"],
    )


def compute_latency_metrics(times_ms: List[float]) -> Optional[LatencyMetrics]:
    """Computes mean, median, min, max latency and throughput FPS."""
    valid_times = [t for t in times_ms if t is not None and t > 0.0]
    if not valid_times:
        return None

    arr = np.array(valid_times, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    med_ms = float(np.median(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return LatencyMetrics(
        mean_ms=mean_ms,
        median_ms=med_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        fps=fps,
    )

