"""Benchmarking engine for evaluating age estimation models on labelled datasets."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd

from safekid_ai.age_estimator import (
    AgeEstimationResult,
    AgeEstimationStatus,
    MiVOLOAgeEstimator,
    classify_age_group,
)
from safekid_ai.benchmarking.metrics import (
    ClassificationMetrics,
    LatencyMetrics,
    RegressionMetrics,
    compute_child_classification_metrics,
    compute_latency_metrics,
    compute_regression_metrics,
)
from safekid_ai.benchmarking.plots import (
    plot_confusion_matrix,
    plot_error_distribution,
    plot_ground_truth_vs_prediction,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.benchmark")


@dataclass
class BenchmarkRecord:
    """Individual benchmark evaluation record."""
    sample_id: str
    image_path: str
    ground_truth_age: float
    ground_truth_age_group: str
    predicted_age: Optional[float]
    predicted_age_group: str
    error: Optional[float]
    abs_error: Optional[float]
    inference_time_ms: float
    status: str
    predicted_gender: Optional[str] = None
    reason: str = ""


@dataclass
class BenchmarkSummary:
    """Consolidated performance evaluation summary."""
    dataset_name: str
    total_samples: int
    successful_predictions: int
    failed_predictions: int
    regression_metrics: Optional[RegressionMetrics]
    classification_metrics: Optional[ClassificationMetrics]
    latency_metrics: Optional[LatencyMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "regression": self.regression_metrics.to_dict() if self.regression_metrics else None,
            "child_classification": self.classification_metrics.to_dict() if self.classification_metrics else None,
            "latency": self.latency_metrics.to_dict() if self.latency_metrics else None,
        }


class BenchmarkEvaluator:
    """Evaluates age estimation models against consented benchmark datasets."""

    def __init__(self, estimator: MiVOLOAgeEstimator, dataset_name: str = "Research Benchmark"):
        self.estimator = estimator
        self.dataset_name = dataset_name
        self.records: List[BenchmarkRecord] = []

    def evaluate_sample(
        self,
        sample_id: str,
        image_or_path: Union[str, Path, np.ndarray],
        ground_truth_age: float,
        is_face_crop: bool = True,
        is_person_crop: bool = False,
    ) -> BenchmarkRecord:
        """Evaluates a single research sample.

        Args:
            sample_id: Unique identifier for the sample.
            image_or_path: Filepath or numpy image array (BGR).
            ground_truth_age: Known ground truth age in years.
            is_face_crop: Whether the input image is a cropped face.
            is_person_crop: Whether the input image is a full person crop.

        Returns:
            BenchmarkRecord for this sample.
        """
        if isinstance(image_or_path, (str, Path)):
            path_str = str(image_or_path)
            img = cv2.imread(path_str)
            if img is None:
                record = BenchmarkRecord(
                    sample_id=sample_id,
                    image_path=path_str,
                    ground_truth_age=float(ground_truth_age),
                    ground_truth_age_group=classify_age_group(ground_truth_age),
                    predicted_age=None,
                    predicted_age_group="UNKNOWN",
                    error=None,
                    abs_error=None,
                    inference_time_ms=0.0,
                    status="IMAGE_READ_FAILED",
                    reason="Could not read or decode image file.",
                )
                self.records.append(record)
                return record
        else:
            path_str = f"in_memory_{sample_id}"
            img = image_or_path

        # Explicitly invoke with consented research flags
        face_input = img if is_face_crop else None
        person_input = img if is_person_crop else None

        res: AgeEstimationResult = self.estimator.estimate(
            face_crop=face_input,
            person_crop=person_input,
            consent_override=True,
            research_mode_override=True,
        )

        gt_age = float(ground_truth_age)
        pred_age = res.estimated_age
        err = float(pred_age - gt_age) if pred_age is not None else None
        abs_err = abs(err) if err is not None else None

        record = BenchmarkRecord(
            sample_id=sample_id,
            image_path=path_str,
            ground_truth_age=gt_age,
            ground_truth_age_group=classify_age_group(gt_age),
            predicted_age=pred_age,
            predicted_age_group=res.age_group,
            error=err,
            abs_error=abs_err,
            inference_time_ms=res.inference_time_ms,
            status=res.status.value,
            predicted_gender=res.gender,
            reason=res.reason,
        )
        self.records.append(record)
        return record

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        image_col: str = "image_path",
        age_col: str = "age",
        id_col: Optional[str] = "id",
        is_face_crop: bool = True,
    ) -> BenchmarkSummary:
        """Evaluates an annotated DataFrame of research samples."""
        logger.info(f"Starting benchmark evaluation across {len(df)} samples...")
        for idx, row in df.iterrows():
            sid = str(row[id_col]) if id_col and id_col in row else f"sample_{idx:05d}"
            img_path = str(row[image_col])
            gt_age = float(row[age_col])
            self.evaluate_sample(
                sample_id=sid,
                image_or_path=img_path,
                ground_truth_age=gt_age,
                is_face_crop=is_face_crop,
            )

        return self.compute_summary()

    def compute_summary(self) -> BenchmarkSummary:
        """Aggregates all recorded samples into a BenchmarkSummary."""
        valid_records = [r for r in self.records if r.predicted_age is not None]
        total = len(self.records)
        succ = len(valid_records)
        failed = total - succ

        y_true = [r.ground_truth_age for r in valid_records]
        y_pred = [r.predicted_age for r in valid_records]
        times = [r.inference_time_ms for r in valid_records]

        reg_metrics = compute_regression_metrics(y_true, y_pred) if valid_records else None
        cls_metrics = compute_child_classification_metrics(y_true, y_pred) if valid_records else None
        lat_metrics = compute_latency_metrics(times) if valid_records else None

        summary = BenchmarkSummary(
            dataset_name=self.dataset_name,
            total_samples=total,
            successful_predictions=succ,
            failed_predictions=failed,
            regression_metrics=reg_metrics,
            classification_metrics=cls_metrics,
            latency_metrics=lat_metrics,
        )
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Converts benchmark records to a pandas DataFrame."""
        return pd.DataFrame([asdict(r) for r in self.records])

    def save_results(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Saves results as CSV, summary JSON, and generates model comparison plots.

        Args:
            output_dir: Directory path for saving benchmark artifacts.

        Returns:
            Dictionary of generated file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save detailed CSV
        csv_path = out_dir / "benchmark_predictions.csv"
        df = self.to_dataframe()
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved benchmark records to CSV: {csv_path}")

        # 2. Save summary JSON
        summary = self.compute_summary()
        json_path = out_dir / "benchmark_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark metrics to JSON: {json_path}")

        saved_files = {
            "csv": csv_path,
            "json": json_path,
        }

        # 3. Generate visualization plots if predictions exist
        valid_records = [r for r in self.records if r.predicted_age is not None]
        if valid_records:
            yt = [r.ground_truth_age for r in valid_records]
            yp = [r.predicted_age for r in valid_records]

            # Parity plot
            p1 = out_dir / "parity_scatter_plot.png"
            plot_ground_truth_vs_prediction(yt, yp, p1, model_name=f"MiVOLO ({self.dataset_name})")
            saved_files["parity_plot"] = p1

            # Residual error distribution
            p2 = out_dir / "error_distribution.png"
            plot_error_distribution(yt, yp, p2, model_name=f"MiVOLO ({self.dataset_name})")
            saved_files["error_plot"] = p2

            # Confusion matrix
            if summary.classification_metrics:
                p3 = out_dir / "child_classification_confusion_matrix.png"
                plot_confusion_matrix(
                    summary.classification_metrics.confusion_matrix,
                    summary.classification_metrics.labels,
                    p3,
                )
                saved_files["confusion_matrix"] = p3

        return saved_files

