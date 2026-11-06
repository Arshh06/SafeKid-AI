"""Comprehensive, multi-tier evaluation protocol with statistical confidence intervals."""

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import ChildDetectionLabel


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence standard normal quantile
    p_hat = successes / total
    denom = 1.0 + (z ** 2) / total
    centre = (p_hat + (z ** 2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z ** 2) / (4.0 * (total ** 2))))) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (round(lower, 4), round(upper, 4))


def run_comprehensive_evaluation() -> Dict[str, Any]:
    """Executes multi-tier evaluation with separate reporting for fixtures and real-world tiers."""
    fixture_ann_path = Path("data/child_dataset/fixtures/annotations.json")
    with open(fixture_ann_path, "r", encoding="utf-8") as f:
        fix_data = json.load(f)

    test_samples = [s for s in fix_data["samples"] if s["split"] == "test"]
    n_test = len(test_samples)

    detector = ChildDetector(ChildDetectorConfig(
        model_weights_path="models/child_classifier_v1.pt",
        confidence_threshold=0.65,
        margin_threshold=0.20,
    ))

    # Confusion matrix
    cm = {
        "CHILD": {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0},
        "ADULT": {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0},
        "AMBIGUOUS": {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0},
    }

    subgroup_performance = {
        "distant_figures": {"total": 0, "abstained_unknown": 0, "predicted_child": 0, "predicted_adult": 0},
        "occluded_figures": {"total": 0, "abstained_unknown": 0, "predicted_child": 0, "predicted_adult": 0},
        "borderline_figures": {"total": 0, "abstained_unknown": 0, "predicted_child": 0, "predicted_adult": 0},
        "standard_figures": {"total": 0, "correct": 0, "abstained": 0, "misclassified": 0},
    }

    latencies: List[float] = []

    for s in test_samples:
        img = cv2.imread(s["filepath"])
        gt = s["ground_truth"]

        t0 = time.perf_counter()
        res = detector.classify_crop(img)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        pred = res.label.value
        cm[gt][pred] += 1

        # Subgroup tracking
        h = s["height"]
        occ = s.get("occlusion_ratio", 0.0)
        cat = s.get("ambiguous_category", "")

        if h < 36 or cat == "DISTANT_PEDESTRIAN":
            subgroup_performance["distant_figures"]["total"] += 1
            if pred == "UNKNOWN":
                subgroup_performance["distant_figures"]["abstained_unknown"] += 1
            elif pred == "CHILD":
                subgroup_performance["distant_figures"]["predicted_child"] += 1
            else:
                subgroup_performance["distant_figures"]["predicted_adult"] += 1

        elif occ > 0.40 or cat == "SEVERE_OCCLUSION":
            subgroup_performance["occluded_figures"]["total"] += 1
            if pred == "UNKNOWN":
                subgroup_performance["occluded_figures"]["abstained_unknown"] += 1
            elif pred == "CHILD":
                subgroup_performance["occluded_figures"]["predicted_child"] += 1
            else:
                subgroup_performance["occluded_figures"]["predicted_adult"] += 1

        elif cat == "BORDERLINE_TEEN":
            subgroup_performance["borderline_figures"]["total"] += 1
            if pred == "UNKNOWN":
                subgroup_performance["borderline_figures"]["abstained_unknown"] += 1
            elif pred == "CHILD":
                subgroup_performance["borderline_figures"]["predicted_child"] += 1
            else:
                subgroup_performance["borderline_figures"]["predicted_adult"] += 1

        else:
            subgroup_performance["standard_figures"]["total"] += 1
            if pred == gt:
                subgroup_performance["standard_figures"]["correct"] += 1
            elif pred == "UNKNOWN":
                subgroup_performance["standard_figures"]["abstained"] += 1
            else:
                subgroup_performance["standard_figures"]["misclassified"] += 1

    # Metrics
    tp_child = cm["CHILD"]["CHILD"]
    fp_child = cm["ADULT"]["CHILD"] + cm["AMBIGUOUS"]["CHILD"]
    fn_child = cm["CHILD"]["ADULT"] + cm["CHILD"]["UNKNOWN"]
    pred_child_total = tp_child + fp_child
    gt_child_total = tp_child + fn_child

    prec_child = tp_child / max(1, pred_child_total)
    rec_child = tp_child / max(1, gt_child_total)
    f1_child = 2 * prec_child * rec_child / max(1e-6, prec_child + rec_child)
    prec_child_ci = wilson_score_interval(tp_child, pred_child_total)
    rec_child_ci = wilson_score_interval(tp_child, gt_child_total)

    tp_adult = cm["ADULT"]["ADULT"]
    fp_adult = cm["CHILD"]["ADULT"] + cm["AMBIGUOUS"]["ADULT"]
    fn_adult = cm["ADULT"]["CHILD"] + cm["ADULT"]["UNKNOWN"]
    pred_adult_total = tp_adult + fp_adult
    gt_adult_total = tp_adult + fn_adult

    prec_adult = tp_adult / max(1, pred_adult_total)
    rec_adult = tp_adult / max(1, gt_adult_total)
    f1_adult = 2 * prec_adult * rec_adult / max(1e-6, prec_adult + rec_adult)
    prec_adult_ci = wilson_score_interval(tp_adult, pred_adult_total)
    rec_adult_ci = wilson_score_interval(tp_adult, gt_adult_total)

    total_unknown = sum(cm[gt]["UNKNOWN"] for gt in cm)
    unknown_rate = total_unknown / n_test
    unknown_ci = wilson_score_interval(total_unknown, n_test)

    # False child rate among adult samples: adults falsely classified as child / total adult samples
    false_child_per_adult = cm["ADULT"]["CHILD"] / max(1, gt_adult_total)

    report_payload = {
        "evaluation_protocol_version": "2.0_rigorous",
        "evaluation_timestamp": "2026-11-05T11:35:00",
        "tiers": {
            "tier_1_synthetic_fixtures": {
                "status": "EVALUATED_PRELIMINARY_FIXTURE_SET",
                "sample_count": n_test,
                "notes": "Preliminary reference fixture test set. Reflects in-distribution template accuracy, not real-world CCTV generalization.",
                "metrics": {
                    "child_precision": round(prec_child, 4),
                    "child_precision_95ci": prec_child_ci,
                    "child_recall": round(rec_child, 4),
                    "child_recall_95ci": rec_child_ci,
                    "child_f1": round(f1_child, 4),
                    "adult_precision": round(prec_adult, 4),
                    "adult_precision_95ci": prec_adult_ci,
                    "adult_recall": round(rec_adult, 4),
                    "adult_recall_95ci": rec_adult_ci,
                    "adult_f1": round(f1_adult, 4),
                    "unknown_rate": round(unknown_rate, 4),
                    "unknown_rate_95ci": unknown_ci,
                    "false_child_predictions_per_adult": round(false_child_per_adult, 4),
                    "latency_mean_ms": round(float(np.mean(latencies)), 2),
                    "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                },
                "confusion_matrix": cm,
                "subgroup_stress_analysis": subgroup_performance,
            },
            "tier_2_real_world_academic_cctv": {
                "status": "BLOCKED_AWAITING_ACADEMIC_EULA",
                "sample_count": 0,
                "target_datasets": ["RAP v2 (CASIA/CUHK)", "PETA (CUHK)"],
                "blocking_rationale": "Sandbox network isolation and required signed institutional academic EULA agreement.",
                "metrics": None,
            },
            "tier_3_consented_residential_cctv": {
                "status": "AWAITING_ON_SITE_PILOT_ENROLLMENT",
                "sample_count": 0,
                "blocking_rationale": "Requires parental opt-in enrollment (COPPA/GDPR Art. 8) and camera signage in a live gated community.",
                "metrics": None,
            }
        }
    }

    # Save metrics JSON
    out_json = Path("outputs/prompt_22_evaluation_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Save CSV comparison
    out_csv = Path("outputs/prompt_22_comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset_Tier", "Metric", "Sample_Count", "Point_Estimate", "Confidence_Interval_95", "Unit"])
        writer.writerow(["TIER_1_FIXTURE", "Child_Precision", pred_child_total, f"{prec_child*100:.2f}%", f"[{prec_child_ci[0]*100:.1f}%, {prec_child_ci[1]*100:.1f}%]", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "Child_Recall", gt_child_total, f"{rec_child*100:.2f}%", f"[{rec_child_ci[0]*100:.1f}%, {rec_child_ci[1]*100:.1f}%]", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "Child_F1", gt_child_total, f"{f1_child*100:.2f}%", "N/A", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "Adult_Precision", pred_adult_total, f"{prec_adult*100:.2f}%", f"[{prec_adult_ci[0]*100:.1f}%, {prec_adult_ci[1]*100:.1f}%]", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "Adult_Recall", gt_adult_total, f"{rec_adult*100:.2f}%", f"[{rec_adult_ci[0]*100:.1f}%, {rec_adult_ci[1]*100:.1f}%]", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "Adult_F1", gt_adult_total, f"{f1_adult*100:.2f}%", "N/A", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "Unknown_Abstention_Rate", n_test, f"{unknown_rate*100:.2f}%", f"[{unknown_ci[0]*100:.1f}%, {unknown_ci[1]*100:.1f}%]", "percent"])
        writer.writerow(["TIER_1_FIXTURE", "False_Child_Per_Adult", gt_adult_total, f"{false_child_per_adult*100:.2f}%", "N/A", "percent"])
        writer.writerow(["TIER_2_REAL_WORLD", "Status", 0, "BLOCKED", "Awaiting Academic EULA", "status"])
        writer.writerow(["TIER_3_RESIDENTIAL", "Status", 0, "AWAITING_PILOT", "Awaiting Parental Opt-In", "status"])

    print(f"Comprehensive Evaluation Completed:")
    print(f"Child Precision: {prec_child*100:.2f}% (95% CI: [{prec_child_ci[0]*100:.1f}%, {prec_child_ci[1]*100:.1f}%]) [N={pred_child_total}]")
    print(f"Child Recall:    {rec_child*100:.2f}% (95% CI: [{rec_child_ci[0]*100:.1f}%, {rec_child_ci[1]*100:.1f}%]) [N={gt_child_total}]")
    print(f"Child F1:        {f1_child*100:.2f}%")
    print(f"Adult Precision: {prec_adult*100:.2f}% (95% CI: [{prec_adult_ci[0]*100:.1f}%, {prec_adult_ci[1]*100:.1f}%]) [N={pred_adult_total}]")
    print(f"Adult Recall:    {rec_adult*100:.2f}% (95% CI: [{rec_adult_ci[0]*100:.1f}%, {rec_adult_ci[1]*100:.1f}%]) [N={gt_adult_total}]")
    print(f"Adult F1:        {f1_adult*100:.2f}%")
    print(f"Unknown Rate:    {unknown_rate*100:.2f}% (95% CI: [{unknown_ci[0]*100:.1f}%, {unknown_ci[1]*100:.1f}%]) [N={n_test}]")
    print(f"False Child Rate on Adults: {false_child_per_adult*100:.2f}%")
    print(f"Metrics saved to: {out_json}")
    print(f"CSV saved to:     {out_csv}")

    return report_payload


if __name__ == "__main__":
    run_comprehensive_evaluation()
