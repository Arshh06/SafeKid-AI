"""Benchmarking and evaluation package for SafeKid AI."""

from .benchmark import BenchmarkEvaluator, BenchmarkRecord, BenchmarkSummary
from .metrics import (
    ClassificationMetrics,
    LatencyMetrics,
    RegressionMetrics,
    compute_child_classification_metrics,
    compute_latency_metrics,
    compute_regression_metrics,
)
from .plots import (
    plot_confusion_matrix,
    plot_error_distribution,
    plot_ground_truth_vs_prediction,
)

__all__ = [
    "BenchmarkEvaluator",
    "BenchmarkRecord",
    "BenchmarkSummary",
    "RegressionMetrics",
    "ClassificationMetrics",
    "LatencyMetrics",
    "compute_regression_metrics",
    "compute_child_classification_metrics",
    "compute_latency_metrics",
    "plot_ground_truth_vs_prediction",
    "plot_error_distribution",
    "plot_confusion_matrix",
]

