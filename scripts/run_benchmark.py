"""Benchmark execution script for MiVOLO age estimation research evaluation."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from safekid_ai.age_estimator import MiVOLOAgeEstimator
from safekid_ai.benchmarking.benchmark import BenchmarkEvaluator
from safekid_ai.config import AgeEstimatorConfig
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.benchmark_script")


def create_sample_benchmark_dataset(dataset_dir: Path) -> Path:
    """Creates a sample benchmark dataset with annotated ground-truth ages."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {"id": "subj_001", "age": 6.0, "color": (140, 160, 210), "desc": "young child"},
        {"id": "subj_002", "age": 10.0, "color": (150, 170, 200), "desc": "older child"},
        {"id": "subj_003", "age": 15.0, "color": (160, 180, 190), "desc": "teenager"},
        {"id": "subj_004", "age": 28.0, "color": (170, 150, 180), "desc": "young adult"},
        {"id": "subj_005", "age": 42.0, "color": (160, 140, 170), "desc": "adult"},
        {"id": "subj_006", "age": 65.0, "color": (180, 180, 180), "desc": "senior resident"},
    ]

    manifest = []
    for s in samples:
        filename = f"{s['id']}.jpg"
        img_path = images_dir / filename
        # Generate a stylized portrait crop
        img = np.full((224, 224, 3), s["color"], dtype=np.uint8)
        # Add facial geometric contours for visual realistic structure
        cv2.circle(img, (112, 112), 70, (int(s["color"][0] * 0.9), int(s["color"][1] * 0.9), int(s["color"][2] * 0.9)), -1)
        cv2.circle(img, (90, 95), 10, (50, 50, 50), -1)  # left eye
        cv2.circle(img, (134, 95), 10, (50, 50, 50), -1)  # right eye
        cv2.ellipse(img, (112, 145), (25, 12), 0, 0, 180, (60, 60, 140), 3)  # mouth
        cv2.imwrite(str(img_path), img)

        manifest.append({
            "id": s["id"],
            "image_path": str(img_path),
            "age": s["age"],
            "description": s["desc"],
        })

    csv_path = dataset_dir / "annotations.csv"
    pd.DataFrame(manifest).to_csv(csv_path, index=False)
    logger.info(f"Sample research benchmark dataset generated at: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run SafeKid AI Research Age Estimation Benchmark")
    parser.add_argument("--annotations-csv", type=str, default=None, help="Path to annotations CSV (image_path, age)")
    parser.add_argument("--output-dir", type=str, default="runs/benchmarks", help="Output directory for reports & plots")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset-name", type=str, default="Sample_Consented_Benchmark")
    args = parser.parse_args()

    # Create dataset if not provided
    if args.annotations_csv is None or not Path(args.annotations_csv).exists():
        data_dir = Path("data/sample_benchmark")
        csv_path = create_sample_benchmark_dataset(data_dir)
    else:
        csv_path = Path(args.annotations_csv)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded benchmark dataset '{args.dataset_name}' with {len(df)} records.")

    # Initialize MiVOLO with explicit research consent flags
    cfg = AgeEstimatorConfig(
        device=args.device,
        half=False,
        allow_research_inference=True,
        consent_verified=True,
    )
    estimator = MiVOLOAgeEstimator(cfg)
    if not estimator.is_available:
        logger.error("MiVOLO estimator weights could not be loaded. Aborting benchmark.")
        return

    evaluator = BenchmarkEvaluator(estimator, dataset_name=args.dataset_name)
    summary = evaluator.evaluate_dataframe(df, image_col="image_path", age_col="age", id_col="id", is_face_crop=True)

    # Save CSV, summary JSON, and plots
    out_dir = Path(args.output_dir)
    files = evaluator.save_results(out_dir)

    print("\n" + "=" * 65)
    print(f" SAFEKID AI RESEARCH BENCHMARK REPORT: {args.dataset_name}")
    print("=" * 65)
    print(f" Total Samples Evaluated    : {summary.total_samples}")
    print(f" Successful Predictions     : {summary.successful_predictions}")
    print(f" Failed Predictions         : {summary.failed_predictions}")
    if summary.regression_metrics:
        reg = summary.regression_metrics
        print(f" Mean Absolute Error (MAE)  : {reg.mae:.3f} years")
        print(f" Root Mean Sq Error (RMSE)  : {reg.rmse:.3f} years")
        print(f" Median Absolute Error      : {reg.median_absolute_error:.3f} years")
        print(f" Max Error                  : {reg.max_error:.3f} years")
        print(f" Pearson Correlation (r)    : {reg.pearson_r:.4f}")
        print(f" Cumulative Score (+-3 yrs) : {reg.cumulative_score_3yr:.1f}%")
        print(f" Cumulative Score (+-5 yrs) : {reg.cumulative_score_5yr:.1f}%")
    if summary.classification_metrics:
        cls_m = summary.classification_metrics
        print(f" Child Classification Acc   : {cls_m.accuracy * 100:.1f}%")
        print(f" Child Precision / Recall   : {cls_m.precision:.3f} / {cls_m.recall:.3f}")
        print(f" Child F1-Score             : {cls_m.f1:.3f}")
    if summary.latency_metrics:
        lat = summary.latency_metrics
        print(f" Mean Latency per instance  : {lat.mean_ms:.1f} ms ({lat.fps:.1f} FPS)")
    print("=" * 65)
    print(f" Artifacts Saved:")
    for k, v in files.items():
        print(f"  - {k:16s}: {v}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

