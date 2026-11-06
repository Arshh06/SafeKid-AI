"""SafeKid AI — Prompt 23 Comparative Object Detection Study: YOLOv8 vs. RT-DETRv2.

Executes a fair, reproducible comparative benchmark between:
1. Baseline: YOLOv8n (Pretrained CNN architecture with C2f feature pyramid)
2. Candidate: RT-DETRv2-L (Pretrained Real-Time Detection Transformer with hybrid encoder & cross-attention)

Both detectors receive identical input frames, identical 640x640 resolution,
identical confidence threshold (0.35) and IoU threshold (0.45), evaluated against
identical ground truth annotations on CPU hardware.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.datasets.virat_annotations import VIRATObjectDetection
from safekid_ai.datasets.virat_evaluator import (
    VIRATDetectionEvaluator,
    VIRATDetectionMetrics,
    VIRATTrackingEvaluator,
    VIRATTrackingMetrics,
    calculate_iou,
)
from safekid_ai.datasets.virat_loader import VIRATDatasetLoader, VIRATSequence
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.detectors import RTDETRv2DetectorAdapter, YOLOv8DetectorAdapter
from safekid_ai.utils.logger import get_logger
from safekid_ai.video_input import VideoInput

logger = get_logger("safekid_ai.comparative_study")


def compute_scale_breakdown(
    records: Dict[int, Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Computes recall broken down by object scale (small, medium, large).

    Scale definitions (area in pixels^2):
    - Small: area < 1024 (e.g. < 32x32, distant pedestrians)
    - Medium: 1024 <= area < 9216 (e.g. 32x32 to 96x96, intermediate distance)
    - Large: area >= 9216 (e.g. >= 96x96, nearby pedestrians)
    """
    scale_counts = {
        "small": {"gt": 0, "tp": 0, "fn": 0},
        "medium": {"gt": 0, "tp": 0, "fn": 0},
        "large": {"gt": 0, "tp": 0, "fn": 0},
    }

    for r in records.values():
        gt_boxes = list(r["gt"])
        pred_items = sorted(r["pred"], key=lambda x: x[1], reverse=True)
        matched_gt = set()

        # Classify GT boxes into scales
        gt_scales = []
        for g_box in gt_boxes:
            w = max(0.0, g_box[2] - g_box[0])
            h = max(0.0, g_box[3] - g_box[1])
            area = w * h
            if area < 1024:
                scale = "small"
            elif area < 9216:
                scale = "medium"
            else:
                scale = "large"
            gt_scales.append(scale)
            scale_counts[scale]["gt"] += 1

        for p_box, conf in pred_items:
            best_iou = 0.0
            best_idx = -1
            for idx, g_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx != -1:
                matched_gt.add(best_idx)
                sc = gt_scales[best_idx]
                scale_counts[sc]["tp"] += 1

    result = {}
    for sc, counts in scale_counts.items():
        gt = counts["gt"]
        tp = counts["tp"]
        fn = gt - tp
        rec = (tp / gt) if gt > 0 else 0.0
        result[sc] = {
            "ground_truth": gt,
            "true_positives": tp,
            "false_negatives": fn,
            "recall": round(rec, 4),
        }

    return result


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[DetectionBox],
    color: Tuple[int, int, int],
    model_name: str,
) -> np.ndarray:
    """Draws styled detection bounding boxes on an image copy."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det.confidence:.2f}"
        if det.track_id is not None:
            label = f"ID:{det.track_id} {label}"
        
        # Label banner
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def render_side_by_side(
    frame: np.ndarray,
    frame_id: int,
    yolo_dets: List[DetectionBox],
    yolo_lat_ms: float,
    rtdetr_dets: List[DetectionBox],
    rtdetr_lat_ms: float,
    target_width: int = 1280,
) -> np.ndarray:
    """Creates a side-by-side comparison frame: YOLOv8 (Left) vs RT-DETRv2 (Right)."""
    h, w = frame.shape[:2]
    
    # Left: YOLOv8 in Cyan (255, 200, 0)
    left_img = draw_bounding_boxes(frame, yolo_dets, (255, 200, 0), "YOLOv8n")
    # Right: RT-DETRv2 in Magenta (200, 0, 255)
    right_img = draw_bounding_boxes(frame, rtdetr_dets, (200, 0, 255), "RT-DETRv2-L")

    # Resize panels if needed for composite
    half_w = target_width // 2
    aspect = h / w
    panel_h = int(half_w * aspect)

    left_resized = cv2.resize(left_img, (half_w, panel_h))
    right_resized = cv2.resize(right_img, (half_w, panel_h))

    # Add header bars to each panel
    bar_h = 36
    canvas = np.zeros((panel_h + bar_h, target_width, 3), dtype=np.uint8)

    # Top headers
    cv2.rectangle(canvas, (0, 0), (half_w, bar_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (half_w, 0), (target_width, bar_h), (20, 20, 45), -1)

    yolo_text = f"YOLOv8n (CNN Baseline) | Frame {frame_id} | Det: {len(yolo_dets)} | {yolo_lat_ms:.1f}ms"
    rtdetr_text = f"RT-DETRv2-L (Transformer) | Frame {frame_id} | Det: {len(rtdetr_dets)} | {rtdetr_lat_ms:.1f}ms"

    cv2.putText(canvas, yolo_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, rtdetr_text, (half_w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 100, 255), 1, cv2.LINE_AA)

    # Place resized images
    canvas[bar_h:, :half_w] = left_resized
    canvas[bar_h:, half_w:] = right_resized

    # Draw divider line
    cv2.line(canvas, (half_w, 0), (half_w, panel_h + bar_h), (80, 80, 80), 2)

    return canvas


def run_comparative_study(
    sequence_id: str = "VIRAT_S_000001_sample",
    confidence_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    max_frames: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs the comparative benchmark on identical frames between YOLOv8n and RT-DETRv2-L."""
    out_dir = output_dir or (ROOT_DIR / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" SAFEKID AI — PROMPT 23 COMPARATIVE STUDY: YOLOv8n vs. RT-DETRv2-L")
    print("=" * 80)

    # 1. Load Sequence
    loader = VIRATDatasetLoader(root_dir=ROOT_DIR / "data" / "virat")
    sequence = loader.load_sequence(sequence_id)
    meta = sequence.metadata
    print(f"\n[1/5] Loaded Benchmark Sequence: {meta.sequence_id}")
    print(f"      Video: {meta.video_path.name} ({meta.total_frames} frames @ {meta.fps} FPS, {meta.width}x{meta.height})")
    print(f"      Annotations: {meta.person_instance_count} person instances across {meta.unique_person_tracks} unique tracks.")
    print(f"      Dataset Provenance: is_genuine={meta.is_genuine}, tier='{meta.tier}'")
    if not meta.is_genuine:
        print("      * STRICT AUDIT NOTE: Using verified benchmark fixture sequence.")

    # 2. Instantiate Detectors
    print("\n[2/5] Initializing Detectors with Fair Configuration (Device=CPU, conf=0.35, iou=0.45, imgsz=640)...")
    
    # YOLOv8n Baseline
    yolo_cfg = DetectorConfig(
        model_name_or_path="yolov8n.pt",
        confidence_threshold=confidence_thresh,
        iou_threshold=iou_thresh,
        imgsz=640,
        device="cpu",
        num_threads=4,
        fuse_model=True,
    )
    tracker_cfg = TrackerConfig(
        tracker_type="bytetrack",
        tracker_config_path="bytetrack.yaml",
        confidence_threshold=confidence_thresh,
        match_thresh=0.75,
        track_buffer=60,
    )
    yolo_adapter = YOLOv8DetectorAdapter(config=yolo_cfg, tracker_config=tracker_cfg)
    print(f"      [Baseline]  {yolo_adapter.model_name.upper()} ({yolo_adapter.backend_name}): {yolo_adapter.parameter_count:,} params ({yolo_adapter.parameter_count/1e6:.2f}M), {yolo_adapter.model_size_mb} MB")

    # RT-DETRv2 Candidate
    rtdetr_cfg = DetectorConfig(
        backend="rtdetrv2",
        model_name_or_path="rtdetr-l.pt",
        confidence_threshold=confidence_thresh,
        iou_threshold=iou_thresh,
        imgsz=640,
        device="cpu",
        num_threads=4,
    )
    rtdetr_adapter = RTDETRv2DetectorAdapter(
        config=rtdetr_cfg,
        tracker_config=tracker_cfg,
        checkpoint_path="rtdetr-l.pt",
    )
    print(f"      [Candidate] {rtdetr_adapter.model_name.upper()} ({rtdetr_adapter.backend_name}): {rtdetr_adapter.parameter_count:,} params ({rtdetr_adapter.parameter_count/1e6:.2f}M), {rtdetr_adapter.model_size_mb} MB")

    # 3. Setup Evaluators
    yolo_det_eval = VIRATDetectionEvaluator(iou_threshold=0.50)
    yolo_track_eval = VIRATTrackingEvaluator(iou_threshold=0.50)

    rtdetr_det_eval = VIRATDetectionEvaluator(iou_threshold=0.50)
    rtdetr_track_eval = VIRATTrackingEvaluator(iou_threshold=0.50)

    yolo_latencies: List[float] = []
    rtdetr_latencies: List[float] = []

    # Frames for visual contact sheet
    contact_frame_indices = [10, 22, 35, 48, 60, 70]
    contact_renders: List[Tuple[int, np.ndarray]] = []

    # 4. Run Sequential Synchronized Inference Frame-by-Frame
    print(f"\n[3/5] Executing Synchronized Comparative Inference across video frames...")
    v_input = VideoInput(source=str(meta.video_path))
    frame_count = 0

    t_total_start = time.perf_counter()

    for frame_id, timestamp_ms, frame in v_input.frames():
        if max_frames is not None and frame_count >= max_frames:
            break
        frame_count += 1

        gt_ann = sequence.get_frame_annotation(frame_id)
        gt_persons = gt_ann.person_detections if gt_ann else []

        # --- A. YOLOv8n Inference ---
        t0 = time.perf_counter()
        yolo_result = yolo_adapter.track(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh,
        )
        t_yolo_ms = (time.perf_counter() - t0) * 1000.0
        yolo_latencies.append(t_yolo_ms)

        yolo_det_eval.record_frame(
            frame_id=frame_id,
            ground_truth=gt_persons,
            predictions=yolo_result.detections,
            inference_time_ms=t_yolo_ms,
        )
        yolo_track_eval.record_frame(
            frame_id=frame_id,
            ground_truth=gt_persons,
            tracked_predictions=yolo_result.detections,
        )

        # --- B. RT-DETRv2-L Inference ---
        t1 = time.perf_counter()
        rtdetr_result = rtdetr_adapter.track(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=confidence_thresh,
            iou_thresh=iou_thresh,
        )
        t_rtdetr_ms = (time.perf_counter() - t1) * 1000.0
        rtdetr_latencies.append(t_rtdetr_ms)

        rtdetr_det_eval.record_frame(
            frame_id=frame_id,
            ground_truth=gt_persons,
            predictions=rtdetr_result.detections,
            inference_time_ms=t_rtdetr_ms,
        )
        rtdetr_track_eval.record_frame(
            frame_id=frame_id,
            ground_truth=gt_persons,
            tracked_predictions=rtdetr_result.detections,
        )

        # Progress log every 10 frames
        if frame_count % 10 == 0 or frame_count == meta.total_frames:
            print(
                f"      Frame {frame_count:02d}/{meta.total_frames} | "
                f"YOLOv8n: {t_yolo_ms:.1f}ms ({len(yolo_result.detections)} det) | "
                f"RT-DETR-L: {t_rtdetr_ms:.1f}ms ({len(rtdetr_result.detections)} det)"
            )

        # Render contact frame if target index
        if frame_id in contact_frame_indices:
            composite = render_side_by_side(
                frame=frame,
                frame_id=frame_id,
                yolo_dets=yolo_result.detections,
                yolo_lat_ms=t_yolo_ms,
                rtdetr_dets=rtdetr_result.detections,
                rtdetr_lat_ms=t_rtdetr_ms,
                target_width=1280,
            )
            contact_renders.append((frame_id, composite))

    total_duration = time.perf_counter() - t_total_start
    print(f"\n[4/5] Benchmark completed in {total_duration:.1f}s ({frame_count} frames processed).")

    # 5. Compute Quantitative Metrics
    print("\n[5/5] Computing Quantitative Metrics and Statistical Breakdown...")
    yolo_det_metrics = yolo_det_eval.compute_metrics()
    yolo_track_metrics = yolo_track_eval.compute_metrics()
    yolo_scale = compute_scale_breakdown(yolo_det_eval.records, iou_thresh=0.50)

    rtdetr_det_metrics = rtdetr_det_eval.compute_metrics()
    rtdetr_track_metrics = rtdetr_track_eval.compute_metrics()
    rtdetr_scale = compute_scale_breakdown(rtdetr_det_eval.records, iou_thresh=0.50)

    # Compute Latency Percentiles
    yolo_lat_arr = np.array(yolo_latencies)
    rtdetr_lat_arr = np.array(rtdetr_latencies)

    yolo_lat_stats = {
        "mean_ms": round(float(np.mean(yolo_lat_arr)), 2),
        "p50_ms": round(float(np.median(yolo_lat_arr)), 2),
        "p95_ms": round(float(np.percentile(yolo_lat_arr, 95)), 2),
        "min_ms": round(float(np.min(yolo_lat_arr)), 2),
        "max_ms": round(float(np.max(yolo_lat_arr)), 2),
        "fps": round(1000.0 / float(np.mean(yolo_lat_arr)), 2),
    }

    rtdetr_lat_stats = {
        "mean_ms": round(float(np.mean(rtdetr_lat_arr)), 2),
        "p50_ms": round(float(np.median(rtdetr_lat_arr)), 2),
        "p95_ms": round(float(np.percentile(rtdetr_lat_arr, 95)), 2),
        "min_ms": round(float(np.min(rtdetr_lat_arr)), 2),
        "max_ms": round(float(np.max(rtdetr_lat_arr)), 2),
        "fps": round(1000.0 / float(np.mean(rtdetr_lat_arr)), 2),
    }

    # Consolidated Results Dictionary
    study_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sequence_id": meta.sequence_id,
            "evaluated_frames": frame_count,
            "resolution": [meta.width, meta.height],
            "input_resolution": [640, 640],
            "confidence_threshold": confidence_thresh,
            "iou_threshold": iou_thresh,
            "device": "cpu",
            "cpu_threads": 4,
            "dataset_tier": meta.tier,
            "is_genuine_virat": meta.is_genuine,
        },
        "yolov8n_baseline": {
            "model_name": yolo_adapter.model_name,
            "architecture": "CNN (CSPDarknet + C2f)",
            "parameters": yolo_adapter.parameter_count,
            "model_size_mb": yolo_adapter.model_size_mb,
            "latency": yolo_lat_stats,
            "detection": yolo_det_metrics.to_dict(),
            "tracking": yolo_track_metrics.to_dict(),
            "scale_breakdown": yolo_scale,
        },
        "rtdetrv2_candidate": {
            "model_name": rtdetr_adapter.model_name,
            "architecture": "Transformer (HGNetv2 + Hybrid Encoder + Decoder)",
            "parameters": rtdetr_adapter.parameter_count,
            "model_size_mb": rtdetr_adapter.model_size_mb,
            "latency": rtdetr_lat_stats,
            "detection": rtdetr_det_metrics.to_dict(),
            "tracking": rtdetr_track_metrics.to_dict(),
            "scale_breakdown": rtdetr_scale,
        },
    }

    # Export JSON
    json_path = out_dir / "prompt_23_detection_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(study_results, fp, indent=2)
    print(f"\nSaved metrics JSON: {json_path}")

    # Export Detection CSV
    det_csv_path = out_dir / "prompt_23_detection_comparison.csv"
    with open(det_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "Model", "Architecture", "Params_M", "Size_MB",
            "Precision", "Recall", "F1_Score", "mAP_50", "mAP_50_95", "Mean_IoU",
            "True_Positives", "False_Positives", "False_Negatives",
            "Small_Obj_Recall", "Med_Obj_Recall", "Large_Obj_Recall",
            "Mean_Latency_ms", "P50_Latency_ms", "P95_Latency_ms", "Throughput_FPS"
        ])
        
        # YOLOv8 row
        yd = yolo_det_metrics
        writer.writerow([
            "YOLOv8n",
            "CNN",
            f"{yolo_adapter.parameter_count/1e6:.2f}",
            yolo_adapter.model_size_mb,
            yd.precision,
            yd.recall,
            yd.f1_score,
            yd.ap_50,
            yd.ap_50_95,
            yd.mean_iou,
            yd.true_positives,
            yd.false_positives,
            yd.false_negatives,
            yolo_scale["small"]["recall"],
            yolo_scale["medium"]["recall"],
            yolo_scale["large"]["recall"],
            yolo_lat_stats["mean_ms"],
            yolo_lat_stats["p50_ms"],
            yolo_lat_stats["p95_ms"],
            yolo_lat_stats["fps"],
        ])

        # RT-DETRv2 row
        rd = rtdetr_det_metrics
        writer.writerow([
            "RT-DETRv2-L",
            "Transformer",
            f"{rtdetr_adapter.parameter_count/1e6:.2f}",
            rtdetr_adapter.model_size_mb,
            rd.precision,
            rd.recall,
            rd.f1_score,
            rd.ap_50,
            rd.ap_50_95,
            rd.mean_iou,
            rd.true_positives,
            rd.false_positives,
            rd.false_negatives,
            rtdetr_scale["small"]["recall"],
            rtdetr_scale["medium"]["recall"],
            rtdetr_scale["large"]["recall"],
            rtdetr_lat_stats["mean_ms"],
            rtdetr_lat_stats["p50_ms"],
            rtdetr_lat_stats["p95_ms"],
            rtdetr_lat_stats["fps"],
        ])
    print(f"Saved detection CSV: {det_csv_path}")

    # Export Tracking CSV
    track_csv_path = out_dir / "prompt_23_tracking_comparison.csv"
    with open(track_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "Model", "Architecture", "Evaluated_Frames", "Total_GT_Tracks", "Predicted_Tracks",
            "MOTA", "IDF1", "ID_Precision", "ID_Recall", "ID_Switches", "Fragmentations", "Track_Continuity"
        ])
        yt = yolo_track_metrics
        writer.writerow([
            "YOLOv8n",
            "CNN",
            yt.evaluated_frames,
            yt.total_gt_tracks,
            yt.total_predicted_tracks,
            yt.mota,
            yt.idf1,
            yt.id_precision,
            yt.id_recall,
            yt.id_switches,
            yt.fragmentations,
            yt.mean_track_continuity,
        ])
        rt = rtdetr_track_metrics
        writer.writerow([
            "RT-DETRv2-L",
            "Transformer",
            rt.evaluated_frames,
            rt.total_gt_tracks,
            rt.total_predicted_tracks,
            rt.mota,
            rt.idf1,
            rt.id_precision,
            rt.id_recall,
            rt.id_switches,
            rt.fragmentations,
            rt.mean_track_continuity,
        ])
    print(f"Saved tracking CSV: {track_csv_path}")

    # Render Multi-Frame Contact Sheet
    if contact_renders:
        contact_sheet_path = out_dir / "prompt_23_detection_comparison_contact_sheet.jpg"
        rows = []
        for fid, composite in contact_renders:
            rows.append(composite)
        
        # Stack vertically
        full_sheet = np.vstack(rows)
        cv2.imwrite(str(contact_sheet_path), full_sheet)
        print(f"Saved contact sheet: {contact_sheet_path} ({len(contact_renders)} side-by-side frames stacked)")

    # Print Formatted Comparison Summary Table
    print("\n" + "=" * 80)
    print(" EMPIRICAL COMPARISON SUMMARY: YOLOv8n vs. RT-DETRv2-L (CPU)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'YOLOv8n (Baseline)':<22} | {'RT-DETRv2-L (Candidate)':<22}")
    print("-" * 80)
    print(f"{'Architecture Family':<28} | {'CNN (CSPDarknet/C2f)':<22} | {'Transformer (Hybrid)':<22}")
    print(f"{'Model Parameters':<28} | {yolo_adapter.parameter_count/1e6:.2f}M {'':<16} | {rtdetr_adapter.parameter_count/1e6:.2f}M {'':<16}")
    print(f"{'Model File Size':<28} | {yolo_adapter.model_size_mb:.2f} MB {'':<14} | {rtdetr_adapter.model_size_mb:.2f} MB {'':<14}")
    print("-" * 80)
    print(f"{'Mean CPU Latency':<28} | {yolo_lat_stats['mean_ms']:.1f} ms {'':<14} | {rtdetr_lat_stats['mean_ms']:.1f} ms {'':<14}")
    print(f"{'P95 CPU Latency':<28} | {yolo_lat_stats['p95_ms']:.1f} ms {'':<14} | {rtdetr_lat_stats['p95_ms']:.1f} ms {'':<14}")
    print(f"{'Throughput (FPS)':<28} | {yolo_lat_stats['fps']:.2f} FPS {'':<13} | {rtdetr_lat_stats['fps']:.2f} FPS {'':<13}")
    print("-" * 80)
    print(f"{'Precision':<28} | {yolo_det_metrics.precision:.4f} {'':<15} | {rtdetr_det_metrics.precision:.4f} {'':<15}")
    print(f"{'Recall':<28} | {yolo_det_metrics.recall:.4f} {'':<15} | {rtdetr_det_metrics.recall:.4f} {'':<15}")
    print(f"{'F1-Score':<28} | {yolo_det_metrics.f1_score:.4f} {'':<15} | {rtdetr_det_metrics.f1_score:.4f} {'':<15}")
    print(f"{'mAP @ 0.50':<28} | {yolo_det_metrics.ap_50:.4f} {'':<15} | {rtdetr_det_metrics.ap_50:.4f} {'':<15}")
    print(f"{'mAP @ [0.50:0.95]':<28} | {yolo_det_metrics.ap_50_95:.4f} {'':<15} | {rtdetr_det_metrics.ap_50_95:.4f} {'':<15}")
    print(f"{'Mean IoU (Matched)':<28} | {yolo_det_metrics.mean_iou:.4f} {'':<15} | {rtdetr_det_metrics.mean_iou:.4f} {'':<15}")
    print(f"{'False Positives':<28} | {yolo_det_metrics.false_positives:<22} | {rtdetr_det_metrics.false_positives:<22}")
    print(f"{'False Negatives':<28} | {yolo_det_metrics.false_negatives:<22} | {rtdetr_det_metrics.false_negatives:<22}")
    print("-" * 80)
    print(f"{'Small Object Recall (<1k px)':<28} | {yolo_scale['small']['recall']:.4f} {'':<15} | {rtdetr_scale['small']['recall']:.4f} {'':<15}")
    print(f"{'Medium Object Recall':<28} | {yolo_scale['medium']['recall']:.4f} {'':<15} | {rtdetr_scale['medium']['recall']:.4f} {'':<15}")
    print(f"{'Large Object Recall (>9k px)':<28} | {yolo_scale['large']['recall']:.4f} {'':<15} | {rtdetr_scale['large']['recall']:.4f} {'':<15}")
    print("-" * 80)
    print(f"{'MOTA (ByteTrack)':<28} | {yolo_track_metrics.mota:.4f} {'':<15} | {rtdetr_track_metrics.mota:.4f} {'':<15}")
    print(f"{'IDF1 (ByteTrack)':<28} | {yolo_track_metrics.idf1:.4f} {'':<15} | {rtdetr_track_metrics.idf1:.4f} {'':<15}")
    print(f"{'ID Switches':<28} | {yolo_track_metrics.id_switches:<22} | {rtdetr_track_metrics.id_switches:<22}")
    print(f"{'Fragmentations':<28} | {yolo_track_metrics.fragmentations:<22} | {rtdetr_track_metrics.fragmentations:<22}")
    print("=" * 80)

    return study_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt 23 Comparative Object Detection Study")
    parser.add_argument("--sequence", default="VIRAT_S_000001_sample", help="Sequence ID")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    run_comparative_study(
        sequence_id=args.sequence,
        confidence_thresh=args.conf,
        iou_thresh=args.iou,
        max_frames=args.max_frames,
    )
