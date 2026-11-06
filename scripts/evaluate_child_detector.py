"""Comprehensive evaluation script for the SafeKid AI ChildDetector subsystem."""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np

from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import ChildDetectionLabel


def evaluate_child_detector(
    annotations_path: str = "data/child_dataset/fixtures/annotations.json",
    checkpoint_path: str = "models/child_classifier_v1.pt",
    output_metrics_json: str = "outputs/child_detection_metrics.json",
    output_cm_csv: str = "outputs/child_detection_confusion_matrix.csv",
) -> Dict[str, Any]:
    """Runs evaluation on the held-out test split using ChildDetector."""
    ann_file = Path(annotations_path).resolve()
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_samples = [s for s in data["samples"] if s["split"] == "test"]
    print(f"Loaded {len(test_samples)} held-out test samples.")

    # Initialize ChildDetector with trained checkpoint
    config = ChildDetectorConfig(
        model_weights_path=checkpoint_path,
        confidence_threshold=0.65,
        margin_threshold=0.20,
        min_crop_height=36,
        min_crop_width=16,
    )
    detector = ChildDetector(config)

    # Confusion matrix structure: GT rows -> Pred columns [CHILD, ADULT, UNKNOWN]
    labels = ["CHILD", "ADULT", "UNKNOWN"]
    cm = {
        "CHILD": {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0},
        "ADULT": {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0},
        "AMBIGUOUS": {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0},
    }

    latencies: List[float] = []
    detailed_results = []

    for sample in test_samples:
        img_path = sample["filepath"]
        crop = cv2.imread(img_path)
        gt = sample["ground_truth"]

        t0 = time.perf_counter()
        res = detector.classify_crop(crop)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        pred_label = res.label.value
        cm[gt][pred_label] += 1

        detailed_results.append({
            "sample_id": sample["sample_id"],
            "filename": sample["filename"],
            "ground_truth": gt,
            "predicted": pred_label,
            "confidence": round(res.confidence, 4),
            "abstention_reason": res.abstention_reason.value,
            "latency_ms": round(lat, 2),
            "height": sample["height"],
            "width": sample["width"],
            "occlusion_ratio": sample["occlusion_ratio"],
        })

    # Metrics calculation
    # For CHILD:
    tp_child = cm["CHILD"]["CHILD"]
    fp_child = cm["ADULT"]["CHILD"] + cm["AMBIGUOUS"]["CHILD"]
    fn_child = cm["CHILD"]["ADULT"] + cm["CHILD"]["UNKNOWN"]
    prec_child = tp_child / max(1, (tp_child + fp_child))
    rec_child = tp_child / max(1, (tp_child + fn_child))
    f1_child = 2 * prec_child * rec_child / max(1e-6, (prec_child + rec_child))

    # For ADULT:
    tp_adult = cm["ADULT"]["ADULT"]
    fp_adult = cm["CHILD"]["ADULT"] + cm["AMBIGUOUS"]["ADULT"]
    fn_adult = cm["ADULT"]["CHILD"] + cm["ADULT"]["UNKNOWN"]
    prec_adult = tp_adult / max(1, (tp_adult + fp_adult))
    rec_adult = tp_adult / max(1, (tp_adult + fn_adult))
    f1_adult = 2 * prec_adult * rec_adult / max(1e-6, (prec_adult + rec_adult))

    total_test = len(test_samples)
    unknown_count = sum(cm[gt]["UNKNOWN"] for gt in cm)
    unknown_rate = unknown_count / max(1, total_test)

    # Ambiguous abstention accuracy (AMBIGUOUS correctly abstaining to UNKNOWN)
    total_ambiguous = len([s for s in test_samples if s["ground_truth"] == "AMBIGUOUS"])
    ambiguous_abstained = cm["AMBIGUOUS"]["UNKNOWN"]
    ambiguous_abstention_rate = ambiguous_abstained / max(1, total_ambiguous)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    metrics_payload = {
        "evaluation_timestamp": "2026-11-05T10:35:00",
        "checkpoint_evaluated": checkpoint_path,
        "total_test_samples": total_test,
        "sample_counts": {
            "ground_truth_child": len([s for s in test_samples if s["ground_truth"] == "CHILD"]),
            "ground_truth_adult": len([s for s in test_samples if s["ground_truth"] == "ADULT"]),
            "ground_truth_ambiguous": total_ambiguous,
        },
        "child_metrics": {
            "true_positives": tp_child,
            "false_positives": fp_child,
            "false_negatives": fn_child,
            "precision": round(prec_child, 4),
            "recall": round(rec_child, 4),
            "f1_score": round(f1_child, 4),
        },
        "adult_metrics": {
            "true_positives": tp_adult,
            "false_positives": fp_adult,
            "false_negatives": fn_adult,
            "precision": round(prec_adult, 4),
            "recall": round(rec_adult, 4),
            "f1_score": round(f1_adult, 4),
        },
        "abstention_metrics": {
            "unknown_rate": round(unknown_rate, 4),
            "ambiguous_abstention_rate": round(ambiguous_abstention_rate, 4),
            "ambiguous_correctly_abstained": ambiguous_abstained,
            "ambiguous_total": total_ambiguous,
        },
        "latency_metrics": {
            "mean_latency_ms": round(mean_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "unit": "ms_per_crop_cpu",
        },
        "confusion_matrix": cm,
    }

    # Save metrics JSON
    out_json = Path(output_metrics_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save confusion matrix CSV
    out_csv = Path(output_cm_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground_Truth", "Pred_CHILD", "Pred_ADULT", "Pred_UNKNOWN", "Total_GT"])
        for gt in ["CHILD", "ADULT", "AMBIGUOUS"]:
            row_total = sum(cm[gt].values())
            writer.writerow([gt, cm[gt]["CHILD"], cm[gt]["ADULT"], cm[gt]["UNKNOWN"], row_total])

    print(f"\n--- EVALUATION METRICS ---")
    print(f"Child Precision: {prec_child*100:.2f}% | Recall: {rec_child*100:.2f}% | F1: {f1_child*100:.2f}%")
    print(f"Adult Precision: {prec_adult*100:.2f}% | Recall: {rec_adult*100:.2f}% | F1: {f1_adult*100:.2f}%")
    print(f"Unknown Rate: {unknown_rate*100:.2f}%")
    print(f"Ambiguous Abstention Rate: {ambiguous_abstention_rate*100:.2f}% ({ambiguous_abstained}/{total_ambiguous})")
    print(f"Mean Inference Latency: {mean_latency:.2f} ms")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")

    return metrics_payload


if __name__ == "__main__":
    evaluate_child_detector()

