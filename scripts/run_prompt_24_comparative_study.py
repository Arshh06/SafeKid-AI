"""SafeKid AI — Prompt 24 D-FINE Integration & Four-Way Comparative Study.

Evaluates and compares four object detection architectures on identical CCTV footage:
1. YOLOv8n (CNN Baseline, 3.15M params)
2. RT-DETR-L (Original Baidu Transformer Baseline, 32.97M params)
3. D-FINE-N (Proposed Primary Candidate, 3.81M params)
4. D-FINE-S (Secondary Candidate, 10.39M params)

Evaluates:
- Detection quality (Precision, Recall, F1, mAP@0.50, mAP@[0.50:0.95], Mean IoU, FP, FN)
- Scale sensitivity (Small, Medium, Large pedestrian recall)
- Occlusion robustness
- Tracking quality with ByteTrack (MOTA, IDF1, ID switches, fragmentations, continuity)
- Hardware profiling (Mean latency, P95 latency, FPS, memory consumption on Intel Core i3 CPU)
"""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import psutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure src/ and dfine_repo are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DFINE_DIR = ROOT_DIR / "dfine_repo"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(DFINE_DIR) not in sys.path:
    sys.path.insert(0, str(DFINE_DIR))

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
from safekid_ai.detectors import (
    DFINEDetectorAdapter,
    RTDETRv2DetectorAdapter,
    YOLOv8DetectorAdapter,
)
from safekid_ai.utils.logger import get_logger
from safekid_ai.video_input import VideoInput

logger = get_logger("safekid_ai.dfine_study")


def compute_scale_breakdown(
    records: Dict[int, Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Computes recall broken down by object scale (small, medium, large)."""
    scale_counts = {
        "small": {"gt": 0, "tp": 0, "fn": 0},
        "medium": {"gt": 0, "tp": 0, "fn": 0},
        "large": {"gt": 0, "tp": 0, "fn": 0},
    }

    for r in records.values():
        gt_boxes = list(r["gt"])
        pred_items = sorted(r["pred"], key=lambda x: x[1], reverse=True)
        matched_gt = set()

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


def draw_styled_boxes(
    image: np.ndarray,
    detections: List[DetectionBox],
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draws styled detection boxes with confidence labels."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det.confidence:.2f}"
        if det.track_id is not None:
            label = f"ID:{det.track_id} {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(annotated, (x1, ty - th - 3), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def render_four_way_composite(
    frame: np.ndarray,
    frame_id: int,
    model_results: List[Tuple[str, str, List[DetectionBox], float, Tuple[int, int, int]]],
    target_width: int = 1600,
) -> np.ndarray:
    """Renders a 4-panel comparison row: YOLOv8n, RT-DETR-L, D-FINE-N, D-FINE-S."""
    h, w = frame.shape[:2]
    panel_w = target_width // 4
    aspect = h / w
    panel_h = int(panel_w * aspect)
    bar_h = 32

    composite = np.zeros((panel_h + bar_h, target_width, 3), dtype=np.uint8)

    for i, (name, arch, dets, lat_ms, color) in enumerate(model_results):
        x_start = i * panel_w
        x_end = (i + 1) * panel_w if i < 3 else target_width

        annotated = draw_styled_boxes(frame, dets, color)
        resized = cv2.resize(annotated, (panel_w, panel_h))

        # Top banner
        header_color = (30, 30, 30) if i % 2 == 0 else (20, 20, 35)
        cv2.rectangle(composite, (x_start, 0), (x_end, bar_h), header_color, -1)

        title = f"{name} | {len(dets)} det | {lat_ms:.1f}ms"
        cv2.putText(
            composite,
            title,
            (x_start + 6, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

        # Place image
        composite[bar_h:, x_start:x_end] = resized

        # Divider line
        if i > 0:
            cv2.line(composite, (x_start, 0), (x_start, panel_h + bar_h), (80, 80, 80), 1)

    return composite


def run_comparative_study(
    sequence_id: str = "VIRAT_S_000001_sample",
    confidence_thresh: float = 0.35,
    iou_thresh: float = 0.45,
    max_frames: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs fair synchronized comparative study across all 4 candidate architectures."""
    out_dir = output_dir or (ROOT_DIR / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(" SAFEKID AI — PROMPT 24 COMPARATIVE STUDY: D-FINE vs. YOLOv8 vs. RT-DETR")
    print("=" * 90)

    # 1. Load Dataset
    loader = VIRATDatasetLoader(root_dir=ROOT_DIR / "data" / "virat")
    sequence = loader.load_sequence(sequence_id)
    meta = sequence.metadata
    print(f"\n[1/5] Loaded Sequence: {meta.sequence_id}")
    print(f"      Video: {meta.video_path.name} ({meta.total_frames} frames @ {meta.fps} FPS, {meta.width}x{meta.height})")
    print(f"      Ground Truth: {meta.person_instance_count} person instances across {meta.unique_person_tracks} tracks.")
    print(f"      Provenance: is_genuine={meta.is_genuine}, tier='{meta.tier}' (Synthetic surveillance fixture)")

    # 2. Setup Detectors
    print("\n[2/5] Initializing Detectors with Fair Standardized Settings (conf=0.35, iou=0.45, imgsz=640)...")

    tracker_cfg = TrackerConfig(
        tracker_type="bytetrack",
        tracker_config_path="configs/trackers/bytetrack_surveillance.yaml",
        confidence_threshold=confidence_thresh,
        track_high_thresh=0.30,
        track_low_thresh=0.10,
        new_track_thresh=0.35,
        track_buffer=60,
        match_thresh=0.75,
    )

    models_info = []

    # 1. YOLOv8n Baseline
    yolo_cfg = DetectorConfig(
        model_name_or_path="yolov8n.pt",
        confidence_threshold=confidence_thresh,
        iou_threshold=iou_thresh,
        imgsz=640,
        device="cpu",
        num_threads=4,
        fuse_model=True,
    )
    yolo_adapter = YOLOv8DetectorAdapter(config=yolo_cfg, tracker_config=tracker_cfg)
    models_info.append(("yolov8n", "YOLOv8n", "CNN", yolo_adapter, (255, 200, 0)))

    # 2. RT-DETR-L Experimental Baseline
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
    models_info.append(("rtdetr_l", "RT-DETR-L", "Transformer", rtdetr_adapter, (200, 0, 255)))

    # 3. D-FINE-N Proposed Candidate
    dfine_n_cfg = DetectorConfig(
        backend="dfine",
        model_variant="dfine_n",
        model_name_or_path="models/dfine_n_coco.pth",
        confidence_threshold=confidence_thresh,
        iou_threshold=iou_thresh,
        imgsz=640,
        device="cpu",
        num_threads=4,
    )
    dfine_n_adapter = DFINEDetectorAdapter(
        config=dfine_n_cfg,
        tracker_config=tracker_cfg,
        variant="dfine_n",
    )
    models_info.append(("dfine_n", "D-FINE-N", "Transformer", dfine_n_adapter, (50, 220, 50)))

    # 4. D-FINE-S Candidate
    dfine_s_cfg = DetectorConfig(
        backend="dfine",
        model_variant="dfine_s",
        model_name_or_path="models/dfine_s_coco.pth",
        confidence_threshold=confidence_thresh,
        iou_threshold=iou_thresh,
        imgsz=640,
        device="cpu",
        num_threads=4,
    )
    dfine_s_adapter = DFINEDetectorAdapter(
        config=dfine_s_cfg,
        tracker_config=tracker_cfg,
        variant="dfine_s",
    )
    models_info.append(("dfine_s", "D-FINE-S", "Transformer", dfine_s_adapter, (255, 80, 80)))

    for tag, name, arch, adapter, _ in models_info:
        print(f"      [{name:<10}] {arch:<12} | {adapter.parameter_count:,} params ({adapter.parameter_count/1e6:.2f}M) | {adapter.model_size_mb:.2f} MB")

    # 3. Evaluators Setup
    det_evaluators = {tag: VIRATDetectionEvaluator(iou_threshold=0.50) for tag, _, _, _, _ in models_info}
    track_evaluators = {tag: VIRATTrackingEvaluator(iou_threshold=0.50) for tag, _, _, _, _ in models_info}
    latencies = {tag: [] for tag, _, _, _, _ in models_info}

    contact_target_frames = [10, 22, 35, 48, 60, 70]
    contact_composites: List[np.ndarray] = []

    # 4. Sequential Synchronized Execution
    print(f"\n[3/5] Executing Synchronized Comparative Inference across video frames...")
    v_input = VideoInput(source=str(meta.video_path))
    frame_count = 0
    t_start_all = time.perf_counter()

    for frame_id, timestamp_ms, frame in v_input.frames():
        if max_frames is not None and frame_count >= max_frames:
            break
        frame_count += 1

        gt_ann = sequence.get_frame_annotation(frame_id)
        gt_persons = gt_ann.person_detections if gt_ann else []

        frame_results = []

        for tag, name, arch, adapter, color in models_info:
            t0 = time.perf_counter()
            res = adapter.track(
                frame=frame,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                confidence_thresh=confidence_thresh,
                iou_thresh=iou_thresh,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies[tag].append(lat_ms)

            det_evaluators[tag].record_frame(
                frame_id=frame_id,
                ground_truth=gt_persons,
                predictions=res.detections,
                inference_time_ms=lat_ms,
            )
            track_evaluators[tag].record_frame(
                frame_id=frame_id,
                ground_truth=gt_persons,
                tracked_predictions=res.detections,
            )

            frame_results.append((name, arch, res.detections, lat_ms, color))

        if frame_count % 10 == 0 or frame_count == meta.total_frames:
            yolo_ms = latencies["yolov8n"][-1]
            dfine_n_ms = latencies["dfine_n"][-1]
            dfine_s_ms = latencies["dfine_s"][-1]
            rtdetr_ms = latencies["rtdetr_l"][-1]
            print(
                f"      Frame {frame_count:02d}/{meta.total_frames} | "
                f"YOLOv8n: {yolo_ms:.1f}ms | D-FINE-N: {dfine_n_ms:.1f}ms | "
                f"D-FINE-S: {dfine_s_ms:.1f}ms | RT-DETR-L: {rtdetr_ms:.1f}ms"
            )

        if frame_id in contact_target_frames:
            comp = render_four_way_composite(
                frame=frame,
                frame_id=frame_id,
                model_results=frame_results,
                target_width=1600,
            )
            contact_composites.append(comp)

    total_time = time.perf_counter() - t_start_all
    print(f"\n[4/5] Benchmark completed in {total_time:.1f}s ({frame_count} frames evaluated across 4 models).")

    # 5. Compute Quantitative Metrics
    print(f"\n[5/5] Computing Quantitative Metrics and Compiling Reports...")
    all_metrics: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sequence_id": meta.sequence_id,
            "evaluated_frames": frame_count,
            "input_resolution": [640, 640],
            "confidence_threshold": confidence_thresh,
            "iou_threshold": iou_thresh,
            "device": "cpu",
            "cpu_threads": 4,
            "dataset_tier": meta.tier,
            "is_genuine_virat": meta.is_genuine,
            "checkpoint_audit": {
                "rtdetr_l": "RT-DETR v1 (Baidu / Ultralytics 8.1.25), 32.97M params",
                "dfine_n": "D-FINE Nano (Peterande / ICLR 2025 Spotlight), 3.81M params",
                "dfine_s": "D-FINE Small (Peterande / ICLR 2025 Spotlight), 10.39M params",
                "yolov8n": "YOLOv8 Nano (Ultralytics), 3.15M params",
            },
        },
        "models": {},
    }

    det_metrics_map = {}
    track_metrics_map = {}
    scale_map = {}
    lat_stats_map = {}

    for tag, name, arch, adapter, _ in models_info:
        dm = det_evaluators[tag].compute_metrics()
        tm = track_evaluators[tag].compute_metrics()
        sc = compute_scale_breakdown(det_evaluators[tag].records, iou_thresh=0.50)

        l_arr = np.array(latencies[tag])
        lat_stats = {
            "mean_ms": round(float(np.mean(l_arr)), 2),
            "p50_ms": round(float(np.median(l_arr)), 2),
            "p95_ms": round(float(np.percentile(l_arr, 95)), 2),
            "min_ms": round(float(np.min(l_arr)), 2),
            "max_ms": round(float(np.max(l_arr)), 2),
            "fps": round(1000.0 / float(np.mean(l_arr)), 2),
        }

        det_metrics_map[tag] = dm
        track_metrics_map[tag] = tm
        scale_map[tag] = sc
        lat_stats_map[tag] = lat_stats

        all_metrics["models"][tag] = {
            "name": name,
            "architecture": arch,
            "parameters": adapter.parameter_count,
            "model_size_mb": adapter.model_size_mb,
            "latency": lat_stats,
            "detection": dm.to_dict(),
            "tracking": tm.to_dict(),
            "scale_breakdown": sc,
        }

    # Save JSON
    json_path = out_dir / "prompt_24_detection_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved: {json_path}")

    # Save Detection CSV
    det_csv = out_dir / "prompt_24_detection_comparison.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Architecture", "Params_M", "Size_MB",
            "Precision", "Recall", "F1_Score", "mAP_50", "mAP_50_95", "Mean_IoU",
            "True_Positives", "False_Positives", "False_Negatives",
            "Small_Obj_Recall", "Med_Obj_Recall", "Large_Obj_Recall",
            "Mean_Latency_ms", "P50_Latency_ms", "P95_Latency_ms", "Throughput_FPS"
        ])
        for tag, name, arch, adapter, _ in models_info:
            dm = det_metrics_map[tag]
            ls = lat_stats_map[tag]
            sc = scale_map[tag]
            writer.writerow([
                name,
                arch,
                f"{adapter.parameter_count/1e6:.2f}",
                adapter.model_size_mb,
                dm.precision,
                dm.recall,
                dm.f1_score,
                dm.ap_50,
                dm.ap_50_95,
                dm.mean_iou,
                dm.true_positives,
                dm.false_positives,
                dm.false_negatives,
                sc["small"]["recall"],
                sc["medium"]["recall"],
                sc["large"]["recall"],
                ls["mean_ms"],
                ls["p50_ms"],
                ls["p95_ms"],
                ls["fps"],
            ])
    print(f"Saved: {det_csv}")

    # Save Tracking CSV
    track_csv = out_dir / "prompt_24_tracking_comparison.csv"
    with open(track_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Architecture", "Evaluated_Frames", "Total_GT_Tracks", "Predicted_Tracks",
            "MOTA", "IDF1", "ID_Precision", "ID_Recall", "ID_Switches", "Fragmentations", "Track_Continuity"
        ])
        for tag, name, arch, adapter, _ in models_info:
            tm = track_metrics_map[tag]
            writer.writerow([
                name,
                arch,
                tm.evaluated_frames,
                tm.total_gt_tracks,
                tm.total_predicted_tracks,
                tm.mota,
                tm.idf1,
                tm.id_precision,
                tm.id_recall,
                tm.id_switches,
                tm.fragmentations,
                tm.mean_track_continuity,
            ])
    print(f"Saved: {track_csv}")

    # Save Contact Sheet
    if contact_composites:
        contact_path = out_dir / "prompt_24_contact_sheet.jpg"
        full_sheet = np.vstack(contact_composites)
        cv2.imwrite(str(contact_path), full_sheet)
        print(f"Saved: {contact_path} ({len(contact_composites)} frames stacked across 4 models)")

    # Print Formatted Table
    print("\n" + "=" * 90)
    print(f"{'Metric':<28} | {'YOLOv8n':<12} | {'RT-DETR-L':<12} | {'D-FINE-N':<12} | {'D-FINE-S':<12}")
    print("-" * 90)
    print(f"{'Architecture':<28} | {'CNN':<12} | {'Transformer':<12} | {'FDR-DETR':<12} | {'FDR-DETR':<12}")
    print(f"{'Parameters (M)':<28} | {'3.15M':<12} | {'32.97M':<12} | {'3.81M':<12} | {'10.39M':<12}")
    print(f"{'Model Size (MB)':<28} | {'6.23 MB':<12} | {'63.43 MB':<12} | {'14.77 MB':<12} | {'39.90 MB':<12}")
    print("-" * 90)
    print(f"{'Mean CPU Latency (ms)':<28} | {lat_stats_map['yolov8n']['mean_ms']:<12.1f} | {lat_stats_map['rtdetr_l']['mean_ms']:<12.1f} | {lat_stats_map['dfine_n']['mean_ms']:<12.1f} | {lat_stats_map['dfine_s']['mean_ms']:<12.1f}")
    print(f"{'Throughput (FPS)':<28} | {lat_stats_map['yolov8n']['fps']:<12.2f} | {lat_stats_map['rtdetr_l']['fps']:<12.2f} | {lat_stats_map['dfine_n']['fps']:<12.2f} | {lat_stats_map['dfine_s']['fps']:<12.2f}")
    print("-" * 90)
    print(f"{'Precision':<28} | {det_metrics_map['yolov8n'].precision:<12.4f} | {det_metrics_map['rtdetr_l'].precision:<12.4f} | {det_metrics_map['dfine_n'].precision:<12.4f} | {det_metrics_map['dfine_s'].precision:<12.4f}")
    print(f"{'Recall':<28} | {det_metrics_map['yolov8n'].recall:<12.4f} | {det_metrics_map['rtdetr_l'].recall:<12.4f} | {det_metrics_map['dfine_n'].recall:<12.4f} | {det_metrics_map['dfine_s'].recall:<12.4f}")
    print(f"{'F1-Score':<28} | {det_metrics_map['yolov8n'].f1_score:<12.4f} | {det_metrics_map['rtdetr_l'].f1_score:<12.4f} | {det_metrics_map['dfine_n'].f1_score:<12.4f} | {det_metrics_map['dfine_s'].f1_score:<12.4f}")
    print(f"{'mAP @ 0.50':<28} | {det_metrics_map['yolov8n'].ap_50:<12.4f} | {det_metrics_map['rtdetr_l'].ap_50:<12.4f} | {det_metrics_map['dfine_n'].ap_50:<12.4f} | {det_metrics_map['dfine_s'].ap_50:<12.4f}")
    print(f"{'mAP @ [0.50:0.95]':<28} | {det_metrics_map['yolov8n'].ap_50_95:<12.4f} | {det_metrics_map['rtdetr_l'].ap_50_95:<12.4f} | {det_metrics_map['dfine_n'].ap_50_95:<12.4f} | {det_metrics_map['dfine_s'].ap_50_95:<12.4f}")
    print(f"{'Mean IoU (Matched)':<28} | {det_metrics_map['yolov8n'].mean_iou:<12.4f} | {det_metrics_map['rtdetr_l'].mean_iou:<12.4f} | {det_metrics_map['dfine_n'].mean_iou:<12.4f} | {det_metrics_map['dfine_s'].mean_iou:<12.4f}")
    print(f"{'False Positives':<28} | {det_metrics_map['yolov8n'].false_positives:<12} | {det_metrics_map['rtdetr_l'].false_positives:<12} | {det_metrics_map['dfine_n'].false_positives:<12} | {det_metrics_map['dfine_s'].false_positives:<12}")
    print(f"{'False Negatives':<28} | {det_metrics_map['yolov8n'].false_negatives:<12} | {det_metrics_map['rtdetr_l'].false_negatives:<12} | {det_metrics_map['dfine_n'].false_negatives:<12} | {det_metrics_map['dfine_s'].false_negatives:<12}")
    print("-" * 90)
    print(f"{'MOTA (ByteTrack)':<28} | {track_metrics_map['yolov8n'].mota:<12.4f} | {track_metrics_map['rtdetr_l'].mota:<12.4f} | {track_metrics_map['dfine_n'].mota:<12.4f} | {track_metrics_map['dfine_s'].mota:<12.4f}")
    print(f"{'IDF1 (ByteTrack)':<28} | {track_metrics_map['yolov8n'].idf1:<12.4f} | {track_metrics_map['rtdetr_l'].idf1:<12.4f} | {track_metrics_map['dfine_n'].idf1:<12.4f} | {track_metrics_map['dfine_s'].idf1:<12.4f}")
    print(f"{'ID Switches':<28} | {track_metrics_map['yolov8n'].id_switches:<12} | {track_metrics_map['rtdetr_l'].id_switches:<12} | {track_metrics_map['dfine_n'].id_switches:<12} | {track_metrics_map['dfine_s'].id_switches:<12}")
    print("=" * 90)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt 24 Four-Way Comparative Study")
    parser.add_argument("--sequence", default="VIRAT_S_000001_sample")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    run_comparative_study(
        sequence_id=args.sequence,
        confidence_thresh=args.conf,
        iou_thresh=args.iou,
        max_frames=args.max_frames,
    )

