"""Unit tests for benchmarking, metrics computation, and artifact generation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safekid_ai.age_estimator import MiVOLOAgeEstimator
from safekid_ai.benchmarking.benchmark import BenchmarkEvaluator, BenchmarkRecord
from safekid_ai.benchmarking.metrics import (
    compute_child_classification_metrics,
    compute_latency_metrics,
    compute_regression_metrics,
)
from safekid_ai.config import AgeEstimatorConfig


def test_compute_regression_metrics():
    """Tests exact mathematical computation of MAE, RMSE, and MedAE."""
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 19.0, 34.0, 40.0]
    # Errors: [2.0, -1.0, 4.0, 0.0]
    # Abs errors: [2.0, 1.0, 4.0, 0.0]
    # MAE = (2 + 1 + 4 + 0) / 4 = 7 / 4 = 1.75
    # MSE = (4 + 1 + 16 + 0) / 4 = 21 / 4 = 5.25
    # RMSE = sqrt(5.25) ~= 2.2912878
    # MedAE = median([0, 1, 2, 4]) = 1.5

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics is not None
    assert metrics.count == 4
    assert pytest.approx(metrics.mae, 0.001) == 1.75
    assert pytest.approx(metrics.rmse, 0.001) == np.sqrt(5.25)
    assert pytest.approx(metrics.median_absolute_error, 0.001) == 1.5
    assert pytest.approx(metrics.max_error, 0.001) == 4.0
    assert metrics.cumulative_score_3yr == 75.0  # 3 of 4 are <= 3.0


def test_compute_child_classification_metrics():
    """Tests accuracy, precision, recall, and confusion matrix for child detection."""
    # Ages: children (< 13): 8, 11; adults: 25, 40
    y_true = [8.0, 11.0, 25.0, 40.0]
    y_pred = [9.0, 14.0, 26.0, 38.0]
    # True binary (< 13): [1, 1, 0, 0]
    # Pred binary (< 13): [1, 0, 0, 0]
    # TP: 1, FN: 1, FP: 0, TN: 2
    # Accuracy: 3/4 = 0.75
    # Precision: 1 / (1 + 0) = 1.0
    # Recall: 1 / (1 + 1) = 0.5
    # F1 = 2 * (1 * 0.5) / (1 + 0.5) = 2/3 ~= 0.6667

    cls_m = compute_child_classification_metrics(y_true, y_pred, child_age_cutoff=13.0)
    assert cls_m is not None
    assert pytest.approx(cls_m.accuracy, 0.01) == 0.75
    assert pytest.approx(cls_m.precision, 0.01) == 1.0
    assert pytest.approx(cls_m.recall, 0.01) == 0.5
    assert pytest.approx(cls_m.f1, 0.01) == 2.0 / 3.0


def test_compute_latency_metrics():
    """Tests latency mean, median, min, max, and FPS."""
    times = [10.0, 20.0, 30.0]
    lat = compute_latency_metrics(times)
    assert lat is not None
    assert lat.mean_ms == 20.0
    assert lat.median_ms == 20.0
    assert lat.min_ms == 10.0
    assert lat.max_ms == 30.0
    assert pytest.approx(lat.fps, 0.1) == 50.0  # 1000 / 20 = 50


def test_benchmark_evaluator_end_to_end():
    """Tests running the evaluator on mock crops, saving CSV, JSON summary and plots."""
    config = AgeEstimatorConfig(
        device="cpu",
        half=False,
        allow_research_inference=True,
        consent_verified=True,
    )
    estimator = MiVOLOAgeEstimator(config)
    evaluator = BenchmarkEvaluator(estimator, dataset_name="UnitTest_Benchmark")

    # Evaluate 3 synthetic samples
    for i, age in enumerate([7.0, 15.0, 32.0]):
        crop = np.full((120, 120, 3), 100 + i * 20, dtype=np.uint8)
        evaluator.evaluate_sample(
            sample_id=f"test_{i}",
            image_or_path=crop,
            ground_truth_age=age,
            is_face_crop=True,
        )

    summary = evaluator.compute_summary()
    assert summary.total_samples == 3
    assert len(evaluator.records) == 3

    with tempfile.TemporaryDirectory() as tmp_dir:
        files = evaluator.save_results(tmp_dir)
        assert Path(files["csv"]).exists()
        assert Path(files["json"]).exists()

        # Check CSV content
        df = pd.read_csv(files["csv"])
        assert len(df) == 3
        assert "ground_truth_age" in df.columns
        assert "predicted_age" in df.columns
        assert "error" in df.columns

        # Verify plots were generated
        assert Path(files["parity_plot"]).exists()
        assert Path(files["error_plot"]).exists()
        assert Path(files["confusion_matrix"]).exists()

