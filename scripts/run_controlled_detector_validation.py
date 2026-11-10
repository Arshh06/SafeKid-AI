"""SafeKid AI — Controlled Ground-Truth-Based YOLOv8n vs D-FINE-N Evaluation Engine.

Executes controlled validation across JAAD (2,100 frames) and VIRAT (72 frames).
Produces raw predictions, per-frame matching records, subgroup metrics, and CPU latency statistics.
"""

import gc
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch

from safekid_ai.config import DetectorConfig
from safekid_ai.datasets.jaad_parser import JAADAnnotationParser, JAADBoundingBox
from safekid_ai.datasets.virat_loader import VIRATDatasetLoader
from safekid_ai.detector import DetectionBox, FrameDetections, PersonDetector
from safekid_ai.detectors.dfine_detector import DFINEDetectorAdapter


def calculate_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_coco_ap(all_detections: List[Dict[str, Any]], total_gt: int, iou_thresh: float = 0.50) -> float:
    """Computes standard 101-point interpolated AP at a given IoU threshold."""
    if total_gt == 0:
        return 0.0
    if not all_detections:
        return 0.0

    # Sort detections by confidence descending
    sorted_dets = sorted(all_detections, key=lambda x: x["confidence"], reverse=True)

    tp_flags = []
    fp_flags = []
    for d in sorted_dets:
        if d["matched_gt_iou"] >= iou_thresh and d["is_tp_at_iou"].get(f"{iou_thresh:.2f}", False):
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # 101-point interpolation
    ap = 0.0
    for r_thresh in np.linspace(0.0, 1.0, 101):
        prec_at_r = precisions[recalls >= r_thresh]
        p_val = np.max(prec_at_r) if len(prec_at_r) > 0 else 0.0
        ap += p_val / 101.0

    return float(ap)


def evaluate_sequence(
    video_path: Path,
    frame_gt_map: Dict[int, List[Dict[str, Any]]],
    yolo_model: PersonDetector,
    dfine_model: DFINEDetectorAdapter,
    warmup_frames: int = 10,
    operating_conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    """Evaluates both models on a single video sequence."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = video_path.stem

    # Storage for predictions and timing
    results = {
        "video_id": video_id,
        "total_frames": total_frames,
        "evaluated_frames": 0,
        "warmup_frames": warmup_frames,
        "total_gt_boxes": 0,
        "yolo": {
            "predictions": [],
            "latencies_ms": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
            "scale_stats": {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0}, "large": {"gt": 0, "tp": 0}},
            "occl_stats": {"none": {"gt": 0, "tp": 0}, "part": {"gt": 0, "tp": 0}},
            "demo_stats": {"child": {"gt": 0, "tp": 0}, "adult": {"gt": 0, "tp": 0}},
        },
        "dfine": {
            "predictions": [],
            "latencies_ms": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
            "scale_stats": {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0}, "large": {"gt": 0, "tp": 0}},
            "occl_stats": {"none": {"gt": 0, "tp": 0}, "part": {"gt": 0, "tp": 0}},
            "demo_stats": {"child": {"gt": 0, "tp": 0}, "adult": {"gt": 0, "tp": 0}},
        },
    }

    iou_test_thresholds = [round(0.50 + 0.05 * k, 2) for k in range(10)]

    for fid in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        is_warmup = fid < warmup_frames
        gt_boxes = frame_gt_map.get(fid, [])

        if not is_warmup:
            results["evaluated_frames"] += 1
            results["total_gt_boxes"] += len(gt_boxes)

            # Accumulate GT subgroup totals
            for g in gt_boxes:
                # Scale
                area = g["area"]
                if area < 32 * 32:
                    scale = "small"
                elif area < 96 * 96:
                    scale = "medium"
                else:
                    scale = "large"
                results["yolo"]["scale_stats"][scale]["gt"] += 1
                results["dfine"]["scale_stats"][scale]["gt"] += 1

                # Occlusion
                occl = g.get("occlusion", "none")
                if occl in results["yolo"]["occl_stats"]:
                    results["yolo"]["occl_stats"][occl]["gt"] += 1
                    results["dfine"]["occl_stats"][occl]["gt"] += 1

                # Demographic
                lbl = g.get("label", "").lower()
                if lbl in results["yolo"]["demo_stats"]:
                    results["yolo"]["demo_stats"][lbl]["gt"] += 1
                    results["dfine"]["demo_stats"][lbl]["gt"] += 1

        # Run YOLOv8n
        t0 = time.perf_counter()
        yolo_res = yolo_model.detect(frame, frame_id=fid, confidence_thresh=0.05)
        t_yolo = (time.perf_counter() - t0) * 1000.0

        # Run D-FINE-N
        t0 = time.perf_counter()
        dfine_res = dfine_model.detect(frame, frame_id=fid, confidence_thresh=0.05)
        t_dfine = (time.perf_counter() - t0) * 1000.0

        if not is_warmup:
            results["yolo"]["latencies_ms"].append(t_yolo)
            results["dfine"]["latencies_ms"].append(t_dfine)

            # Match YOLO predictions at operating threshold and for AP
            _match_frame(
                gt_boxes=gt_boxes,
                predictions=yolo_res.detections,
                model_key="yolo",
                results=results,
                fid=fid,
                operating_conf=operating_conf_thresh,
                iou_thresholds=iou_test_thresholds,
            )

            # Match D-FINE predictions at operating threshold and for AP
            _match_frame(
                gt_boxes=gt_boxes,
                predictions=dfine_res.detections,
                model_key="dfine",
                results=results,
                fid=fid,
                operating_conf=operating_conf_thresh,
                iou_thresholds=iou_test_thresholds,
            )

    cap.release()

    # Compute sequence summary metrics
    for m_key in ["yolo", "dfine"]:
        m_dict = results[m_key]
        tp = m_dict["tp"]
        fp = m_dict["fp"]
        total_gt = results["total_gt_boxes"]
        fn = max(0, total_gt - tp)
        m_dict["fn"] = fn

        m_dict["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        m_dict["recall"] = float(tp / total_gt) if total_gt > 0 else 0.0
        m_dict["f1"] = (
            float(2 * m_dict["precision"] * m_dict["recall"] / (m_dict["precision"] + m_dict["recall"]))
            if (m_dict["precision"] + m_dict["recall"]) > 0
            else 0.0
        )
        m_dict["mean_iou"] = float(np.mean(m_dict["matched_ious"])) if m_dict["matched_ious"] else 0.0

        # AP@0.50, AP@0.75, and mAP@0.50:0.95
        ap_50 = compute_coco_ap(m_dict["predictions"], total_gt, iou_thresh=0.50)
        ap_75 = compute_coco_ap(m_dict["predictions"], total_gt, iou_thresh=0.75)

        ap_steps = [compute_coco_ap(m_dict["predictions"], total_gt, iou_thresh=th) for th in iou_test_thresholds]
        map_50_95 = float(np.mean(ap_steps)) if ap_steps else 0.0

        m_dict["ap_50"] = ap_50
        m_dict["ap_75"] = ap_75
        m_dict["map_50_95"] = map_50_95

        # Latencies
        lats = m_dict["latencies_ms"]
        m_dict["mean_latency_ms"] = float(np.mean(lats)) if lats else 0.0
        m_dict["median_latency_ms"] = float(np.median(lats)) if lats else 0.0
        m_dict["p90_latency_ms"] = float(np.percentile(lats, 90)) if lats else 0.0
        m_dict["p99_latency_ms"] = float(np.percentile(lats, 99)) if lats else 0.0
        m_dict["throughput_fps"] = (1000.0 / m_dict["mean_latency_ms"]) if m_dict["mean_latency_ms"] > 0 else 0.0

    return results


def _match_frame(
    gt_boxes: List[Dict[str, Any]],
    predictions: List[DetectionBox],
    model_key: str,
    results: Dict[str, Any],
    fid: int,
    operating_conf: float,
    iou_thresholds: List[float],
) -> None:
    """Matches predictions to ground truth on a single frame."""
    m_dict = results[model_key]

    # Sort predictions by confidence descending
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    # 1. Multi-IoU flags for AP calculation across all predictions (conf >= 0.05)
    # For each IoU threshold, determine which predictions are True Positives
    for iou_th in iou_thresholds:
        matched_gt_indices = set()
        iou_key = f"{iou_th:.2f}"
        for p in sorted_preds:
            p_box = p.bbox_xyxy
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                iou = calculate_iou(p_box, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_th and best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                # Store match flag in prediction record
                if not hasattr(p, "_tp_flags"):
                    p._tp_flags = {}
                p._tp_flags[iou_key] = True
            else:
                if not hasattr(p, "_tp_flags"):
                    p._tp_flags = {}
                p._tp_flags[iou_key] = False

    # 2. Operating threshold evaluation (conf >= operating_conf, IoU >= 0.50)
    matched_gt_at_op = set()
    for p in sorted_preds:
        p_box = p.bbox_xyxy
        conf = float(p.confidence)

        # Best IoU against any GT
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g in enumerate(gt_boxes):
            iou = calculate_iou(p_box, g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        # Check operating threshold matching
        is_op_tp = False
        matched_gt_obj = None
        if conf >= operating_conf:
            # Check greedy unclaimed match
            best_unclaimed_iou = 0.0
            best_unclaimed_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt_at_op:
                    continue
                iou = calculate_iou(p_box, g["bbox"])
                if iou > best_unclaimed_iou:
                    best_unclaimed_iou = iou
                    best_unclaimed_idx = g_idx

            if best_unclaimed_iou >= 0.50 and best_unclaimed_idx != -1:
                is_op_tp = True
                matched_gt_at_op.add(best_unclaimed_idx)
                matched_gt_obj = gt_boxes[best_unclaimed_idx]
                m_dict["tp"] += 1
                m_dict["matched_ious"].append(best_unclaimed_iou)

                # Accumulate subgroup TP
                area = matched_gt_obj["area"]
                scale = "small" if area < 32 * 32 else ("medium" if area < 96 * 96 else "large")
                m_dict["scale_stats"][scale]["tp"] += 1

                occl = matched_gt_obj.get("occlusion", "none")
                if occl in m_dict["occl_stats"]:
                    m_dict["occl_stats"][occl]["tp"] += 1

                lbl = matched_gt_obj.get("label", "").lower()
                if lbl in m_dict["demo_stats"]:
                    m_dict["demo_stats"][lbl]["tp"] += 1
            else:
                m_dict["fp"] += 1

        # Save raw prediction record for archival
        m_dict["predictions"].append({
            "frame_id": fid,
            "bbox": [round(c, 1) for c in p_box],
            "confidence": round(conf, 4),
            "matched_gt_iou": round(best_iou, 4),
            "is_op_tp": is_op_tp,
            "is_tp_at_iou": getattr(p, "_tp_flags", {}),
            "matched_gt_track": matched_gt_obj.get("track_id") if matched_gt_obj else None,
        })


def run_full_benchmark():
    print("================================================================================")
    print("SafeKid AI — Controlled Ground-Truth-Based YOLOv8n vs D-FINE-N Benchmark")
    print("================================================================================")

    # Clamping CPU threads
    torch.set_num_threads(4)
    print(f"PyTorch threads clamped to: {torch.get_num_threads()}")
    print(f"Host CPU affinity: {psutil.Process().cpu_affinity()} (Unpinned)")

    # 1. Instantiate Detectors
    print("\n--- Initializing Detectors ---")
    yolo_cfg = DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu", confidence_threshold=0.05, imgsz=640)
    yolo_det = PersonDetector(yolo_cfg)
    print("YOLOv8n initialized.")

    dfine_cfg = DetectorConfig(model_name_or_path="models/dfine_n_coco.pth", device="cpu", confidence_threshold=0.05, imgsz=640)
    dfine_det = DFINEDetectorAdapter(dfine_cfg, variant="dfine_n")
    print("D-FINE-N initialized.")

    # 2. Parse JAAD Datasets
    print("\n--- Loading JAAD Annotations ---")
    jaad_dir = Path("data/external/jaad")
    parser = JAADAnnotationParser()
    jaad_videos = sorted(list((jaad_dir / "videos").glob("*.mp4")))

    jaad_gt_by_video = {}
    for v in jaad_videos:
        ann_path = jaad_dir / "repo" / "annotations" / f"{v.stem}.xml"
        attr_path = jaad_dir / "repo" / "annotations_attributes" / f"{v.stem}_attributes.xml"
        boxes, _ = parser.parse_video_annotations(ann_path, attr_path, 1920, 1080)

        # Build frame -> list of gt boxes dict
        f_map = {}
        for b in boxes:
            fid = b.frame_id
            if fid not in f_map:
                f_map[fid] = []
            f_map[fid].append({
                "bbox": (b.xtl, b.ytl, b.xbr, b.ybr),
                "area": b.width * b.height,
                "track_id": b.track_id,
                "label": b.harmonized_label.value,
                "occlusion": b.occlusion_level,
                "action": b.action,
            })
        jaad_gt_by_video[v.stem] = f_map

    # 3. Evaluate on JAAD Sequences
    print("\n--- Evaluating on 9 JAAD Sequences ---")
    all_jaad_results = []
    t_start_all = time.perf_counter()
    raw_dir = Path("outputs/prompt_36_raw_predictions")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for idx, v in enumerate(jaad_videos, 1):
        cache_file = raw_dir / f"{v.stem}_eval_cache.json"
        if cache_file.exists():
            print(f"\n[{idx}/9] Loading cached results for {v.name} ({v.stem})...", flush=True)
            with open(cache_file, "r", encoding="utf-8") as f:
                seq_res = json.load(f)
        else:
            print(f"\n[{idx}/9] Evaluating {v.name} ({v.stem})...", flush=True)
            t_seq = time.perf_counter()
            seq_res = evaluate_sequence(
                video_path=v,
                frame_gt_map=jaad_gt_by_video[v.stem],
                yolo_model=yolo_det,
                dfine_model=dfine_det,
                warmup_frames=10,
                operating_conf_thresh=0.25,
            )
            dur = time.perf_counter() - t_seq
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(seq_res, f)
            print(f"   Completed and cached {v.name} in {dur:.1f}s | GT: {seq_res['total_gt_boxes']} boxes", flush=True)

        y_res = seq_res["yolo"]
        d_res = seq_res["dfine"]

        print(
            f"   YOLOv8n:  P={y_res['precision']*100:.1f}%, R={y_res['recall']*100:.1f}%, F1={y_res['f1']*100:.1f}%, AP50={y_res['ap_50']*100:.1f}%, mAP={y_res['map_50_95']*100:.1f}%, Lat={y_res['mean_latency_ms']:.1f}ms\n"
            f"   D-FINE-N: P={d_res['precision']*100:.1f}%, R={d_res['recall']*100:.1f}%, F1={d_res['f1']*100:.1f}%, AP50={d_res['ap_50']*100:.1f}%, mAP={d_res['map_50_95']*100:.1f}%, Lat={d_res['mean_latency_ms']:.1f}ms",
            flush=True
        )
        all_jaad_results.append(seq_res)

    total_jaad_wall_time = time.perf_counter() - t_start_all
    print(f"\nAll 9 JAAD sequences processed in {total_jaad_wall_time:.1f} seconds.", flush=True)

    # 4. Evaluate on VIRAT Supplementary Sample
    print("\n--- Evaluating on VIRAT Surveillance Sample ---", flush=True)
    virat_loader = VIRATDatasetLoader(root_dir="data/virat")
    virat_seq = virat_loader.load_sequence("VIRAT_S_000001_sample")

    virat_gt_map = {}
    for fid, fa in virat_seq.frame_annotations.items():
        virat_gt_map[fid] = []
        for det in fa.person_detections:
            virat_gt_map[fid].append({
                "bbox": det.bbox_xyxy,
                "area": det.width * det.height,
                "track_id": str(det.track_id),
                "label": "ADULT",  # VIRAT has no demographic labels
                "occlusion": "none",
            })

    virat_video_path = Path("data/virat/videos/VIRAT_S_000001_sample.mp4")
    virat_results = evaluate_sequence(
        video_path=virat_video_path,
        frame_gt_map=virat_gt_map,
        yolo_model=yolo_det,
        dfine_model=dfine_det,
        warmup_frames=5,
        operating_conf_thresh=0.25,
    )
    y_virat = virat_results["yolo"]
    d_virat = virat_results["dfine"]
    print(
        f"VIRAT Sample Results (72 frames, {virat_results['total_gt_boxes']} GT boxes):\n"
        f"   YOLOv8n:  TP={y_virat['tp']}, FP={y_virat['fp']}, FN={y_virat['fn']}, P={y_virat['precision']*100:.1f}%, R={y_virat['recall']*100:.1f}%, F1={y_virat['f1']*100:.1f}%, AP50={y_virat['ap_50']*100:.1f}%\n"
        f"   D-FINE-N: TP={d_virat['tp']}, FP={d_virat['fp']}, FN={d_virat['fn']}, P={d_virat['precision']*100:.1f}%, R={d_virat['recall']*100:.1f}%, F1={d_virat['f1']*100:.1f}%, AP50={d_virat['ap_50']*100:.1f}%"
    )

    # 5. Measure Peak Memory RSS & CPU Utilization
    mem_info = psutil.Process().memory_info()
    peak_rss_mb = mem_info.rss / (1024 * 1024)

    # 6. Aggregate JAAD Metrics Across All Sequences
    # Pool all predictions for global dataset-level AP
    yolo_all_preds = []
    dfine_all_preds = []
    total_eval_frames = sum(s["evaluated_frames"] for s in all_jaad_results)
    total_eval_gt = sum(s["total_gt_boxes"] for s in all_jaad_results)

    yolo_total_tp = sum(s["yolo"]["tp"] for s in all_jaad_results)
    yolo_total_fp = sum(s["yolo"]["fp"] for s in all_jaad_results)
    yolo_total_fn = total_eval_gt - yolo_total_tp
    yolo_all_lats = []

    dfine_total_tp = sum(s["dfine"]["tp"] for s in all_jaad_results)
    dfine_total_fp = sum(s["dfine"]["fp"] for s in all_jaad_results)
    dfine_total_fn = total_eval_gt - dfine_total_tp
    dfine_all_lats = []

    # Subgroups aggregation
    subgroups = {
        "yolo": {
            "scale": {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0}, "large": {"gt": 0, "tp": 0}},
            "occl": {"none": {"gt": 0, "tp": 0}, "part": {"gt": 0, "tp": 0}},
            "demo": {"child": {"gt": 0, "tp": 0}, "adult": {"gt": 0, "tp": 0}},
        },
        "dfine": {
            "scale": {"small": {"gt": 0, "tp": 0}, "medium": {"gt": 0, "tp": 0}, "large": {"gt": 0, "tp": 0}},
            "occl": {"none": {"gt": 0, "tp": 0}, "part": {"gt": 0, "tp": 0}},
            "demo": {"child": {"gt": 0, "tp": 0}, "adult": {"gt": 0, "tp": 0}},
        },
    }

    for s in all_jaad_results:
        yolo_all_preds.extend(s["yolo"]["predictions"])
        dfine_all_preds.extend(s["dfine"]["predictions"])
        yolo_all_lats.extend(s["yolo"]["latencies_ms"])
        dfine_all_lats.extend(s["dfine"]["latencies_ms"])

        for k_sub in ["scale", "occl", "demo"]:
            sub_key = f"{k_sub}_stats"
            for sub_cat, vals in s["yolo"][sub_key].items():
                subgroups["yolo"][k_sub][sub_cat]["gt"] += vals["gt"]
                subgroups["yolo"][k_sub][sub_cat]["tp"] += vals["tp"]
            for sub_cat, vals in s["dfine"][sub_key].items():
                subgroups["dfine"][k_sub][sub_cat]["gt"] += vals["gt"]
                subgroups["dfine"][k_sub][sub_cat]["tp"] += vals["tp"]

    # Global APs
    iou_steps = [round(0.50 + 0.05 * k, 2) for k in range(10)]
    yolo_global_ap50 = compute_coco_ap(yolo_all_preds, total_eval_gt, 0.50)
    yolo_global_ap75 = compute_coco_ap(yolo_all_preds, total_eval_gt, 0.75)
    yolo_global_map = float(np.mean([compute_coco_ap(yolo_all_preds, total_eval_gt, th) for th in iou_steps]))

    dfine_global_ap50 = compute_coco_ap(dfine_all_preds, total_eval_gt, 0.50)
    dfine_global_ap75 = compute_coco_ap(dfine_all_preds, total_eval_gt, 0.75)
    dfine_global_map = float(np.mean([compute_coco_ap(dfine_all_preds, total_eval_gt, th) for th in iou_steps]))

    yolo_p = yolo_total_tp / (yolo_total_tp + yolo_total_fp) if (yolo_total_tp + yolo_total_fp) > 0 else 0.0
    yolo_r = yolo_total_tp / total_eval_gt if total_eval_gt > 0 else 0.0
    yolo_f1 = 2 * yolo_p * yolo_r / (yolo_p + yolo_r) if (yolo_p + yolo_r) > 0 else 0.0

    dfine_p = dfine_total_tp / (dfine_total_tp + dfine_total_fp) if (dfine_total_tp + dfine_total_fp) > 0 else 0.0
    dfine_r = dfine_total_tp / total_eval_gt if total_eval_gt > 0 else 0.0
    dfine_f1 = 2 * dfine_p * dfine_r / (dfine_p + dfine_r) if (dfine_p + dfine_r) > 0 else 0.0

    print("\n================================================================================")
    print("GLOBAL JAAD BENCHMARK SUMMARY (2,100 frames, 13,682 Ground-Truth Boxes)")
    print("================================================================================")
    print(f"YOLOv8n:  P={yolo_p*100:.2f}%, R={yolo_r*100:.2f}%, F1={yolo_f1*100:.2f}%, AP50={yolo_global_ap50*100:.2f}%, AP75={yolo_global_ap75*100:.2f}%, mAP={yolo_global_map*100:.2f}% | Latency: {np.mean(yolo_all_lats):.1f}ms ({1000.0/np.mean(yolo_all_lats):.2f} FPS)")
    print(f"D-FINE-N: P={dfine_p*100:.2f}%, R={dfine_r*100:.2f}%, F1={dfine_f1*100:.2f}%, AP50={dfine_global_ap50*100:.2f}%, AP75={dfine_global_ap75*100:.2f}%, mAP={dfine_global_map*100:.2f}% | Latency: {np.mean(dfine_all_lats):.1f}ms ({1000.0/np.mean(dfine_all_lats):.2f} FPS)")

    # 7. Export Outputs
    outputs_dir = Path("outputs")
    raw_dir = outputs_dir / "prompt_36_raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save raw predictions per video
    for s in all_jaad_results:
        vid = s["video_id"]
        with open(raw_dir / f"{vid}_yolo_predictions.json", "w", encoding="utf-8") as f:
            json.dump({"video_id": vid, "predictions": s["yolo"]["predictions"]}, f)
        with open(raw_dir / f"{vid}_dfine_predictions.json", "w", encoding="utf-8") as f:
            json.dump({"video_id": vid, "predictions": s["dfine"]["predictions"]}, f)

    with open(raw_dir / "virat_yolo_predictions.json", "w", encoding="utf-8") as f:
        json.dump(virat_results["yolo"]["predictions"], f)
    with open(raw_dir / "virat_dfine_predictions.json", "w", encoding="utf-8") as f:
        json.dump(virat_results["dfine"]["predictions"], f)

    # Export prompt_36_jaad_metrics.csv
    jaad_csv_path = outputs_dir / "prompt_36_jaad_metrics.csv"
    with open(jaad_csv_path, "w", encoding="utf-8") as f:
        f.write("Model,Evaluated Frames,Total GT,TP,FP,FN,Precision,Recall,F1_Score,AP_50,AP_75,mAP_50_95,Mean_IoU,Mean_Latency_ms,Median_Latency_ms,P90_Latency_ms,P99_Latency_ms,Throughput_FPS\n")
        f.write(f"YOLOv8n,{total_eval_frames},{total_eval_gt},{yolo_total_tp},{yolo_total_fp},{yolo_total_fn},{yolo_p:.4f},{yolo_r:.4f},{yolo_f1:.4f},{yolo_global_ap50:.4f},{yolo_global_ap75:.4f},{yolo_global_map:.4f},{np.mean([s['yolo']['mean_iou'] for s in all_jaad_results]):.4f},{np.mean(yolo_all_lats):.2f},{np.median(yolo_all_lats):.2f},{np.percentile(yolo_all_lats, 90):.2f},{np.percentile(yolo_all_lats, 99):.2f},{1000.0/np.mean(yolo_all_lats):.2f}\n")
        f.write(f"D-FINE-N,{total_eval_frames},{total_eval_gt},{dfine_total_tp},{dfine_total_fp},{dfine_total_fn},{dfine_p:.4f},{dfine_r:.4f},{dfine_f1:.4f},{dfine_global_ap50:.4f},{dfine_global_ap75:.4f},{dfine_global_map:.4f},{np.mean([s['dfine']['mean_iou'] for s in all_jaad_results]):.4f},{np.mean(dfine_all_lats):.2f},{np.median(dfine_all_lats):.2f},{np.percentile(dfine_all_lats, 90):.2f},{np.percentile(dfine_all_lats, 99):.2f},{1000.0/np.mean(dfine_all_lats):.2f}\n")

    # Export prompt_36_virat_metrics.csv
    virat_csv_path = outputs_dir / "prompt_36_virat_metrics.csv"
    with open(virat_csv_path, "w", encoding="utf-8") as f:
        f.write("Model,Evaluated Frames,Total GT,TP,FP,FN,Precision,Recall,F1_Score,AP_50,Mean_IoU,Mean_Latency_ms,Throughput_FPS\n")
        f.write(f"YOLOv8n,{virat_results['evaluated_frames']},{virat_results['total_gt_boxes']},{y_virat['tp']},{y_virat['fp']},{y_virat['fn']},{y_virat['precision']:.4f},{y_virat['recall']:.4f},{y_virat['f1']:.4f},{y_virat['ap_50']:.4f},{y_virat['mean_iou']:.4f},{y_virat['mean_latency_ms']:.2f},{y_virat['throughput_fps']:.2f}\n")
        f.write(f"D-FINE-N,{virat_results['evaluated_frames']},{virat_results['total_gt_boxes']},{d_virat['tp']},{d_virat['fp']},{d_virat['fn']},{d_virat['precision']:.4f},{d_virat['recall']:.4f},{d_virat['f1']:.4f},{d_virat['ap_50']:.4f},{d_virat['mean_iou']:.4f},{d_virat['mean_latency_ms']:.2f},{d_virat['throughput_fps']:.2f}\n")

    # Export prompt_36_per_sequence_metrics.csv
    seq_csv_path = outputs_dir / "prompt_36_per_sequence_metrics.csv"
    with open(seq_csv_path, "w", encoding="utf-8") as f:
        f.write("Sequence,Total Frames,Total GT,YOLO_Precision,YOLO_Recall,YOLO_F1,YOLO_AP50,YOLO_mAP,YOLO_Latency_ms,DFINE_Precision,DFINE_Recall,DFINE_F1,DFINE_AP50,DFINE_mAP,DFINE_Latency_ms,F1_Diff,AP50_Diff\n")
        for s in all_jaad_results:
            y = s["yolo"]
            d = s["dfine"]
            f.write(f"{s['video_id']},{s['evaluated_frames']},{s['total_gt_boxes']},{y['precision']:.4f},{y['recall']:.4f},{y['f1']:.4f},{y['ap_50']:.4f},{y['map_50_95']:.4f},{y['mean_latency_ms']:.2f},{d['precision']:.4f},{d['recall']:.4f},{d['f1']:.4f},{d['ap_50']:.4f},{d['map_50_95']:.4f},{d['mean_latency_ms']:.2f},{d['f1'] - y['f1']:.4f},{d['ap_50'] - y['ap_50']:.4f}\n")

    # Export prompt_36_subgroup_analysis.csv
    subgroup_csv_path = outputs_dir / "prompt_36_subgroup_analysis.csv"
    with open(subgroup_csv_path, "w", encoding="utf-8") as f:
        f.write("Dimension,Subgroup,Total GT Instances,YOLO_TP,YOLO_Recall,DFINE_TP,DFINE_Recall,Recall_Difference\n")
        # Scale
        for sc in ["small", "medium", "large"]:
            y_sc = subgroups["yolo"]["scale"][sc]
            d_sc = subgroups["dfine"]["scale"][sc]
            gt_sc = y_sc["gt"]
            y_rec = y_sc["tp"] / gt_sc if gt_sc > 0 else 0.0
            d_rec = d_sc["tp"] / gt_sc if gt_sc > 0 else 0.0
            f.write(f"Scale,{sc.capitalize()},{gt_sc},{y_sc['tp']},{y_rec:.4f},{d_sc['tp']},{d_rec:.4f},{d_rec - y_rec:.4f}\n")
        # Occlusion
        for oc in ["none", "part"]:
            y_oc = subgroups["yolo"]["occl"][oc]
            d_oc = subgroups["dfine"]["occl"][oc]
            gt_oc = y_oc["gt"]
            y_rec = y_oc["tp"] / gt_oc if gt_oc > 0 else 0.0
            d_rec = d_oc["tp"] / gt_oc if gt_oc > 0 else 0.0
            f.write(f"Occlusion,{oc.capitalize()},{gt_oc},{y_oc['tp']},{y_rec:.4f},{d_oc['tp']},{d_rec:.4f},{d_rec - y_rec:.4f}\n")
        # Demographic
        for dm in ["child", "adult"]:
            y_dm = subgroups["yolo"]["demo"][dm]
            d_dm = subgroups["dfine"]["demo"][dm]
            gt_dm = y_dm["gt"]
            y_rec = y_dm["tp"] / gt_dm if gt_dm > 0 else 0.0
            d_rec = d_dm["tp"] / gt_dm if gt_dm > 0 else 0.0
            f.write(f"Demographics,{dm.capitalize()},{gt_dm},{y_dm['tp']},{y_rec:.4f},{d_dm['tp']},{d_rec:.4f},{d_rec - y_rec:.4f}\n")

    # Export prompt_36_cpu_benchmark.csv
    cpu_csv_path = outputs_dir / "prompt_36_cpu_benchmark.csv"
    with open(cpu_csv_path, "w", encoding="utf-8") as f:
        f.write("Model,Hardware,PyTorch Threads,Affinity,Evaluated Frames,Mean Latency (ms),Median Latency (ms),P90 Latency (ms),P99 Latency (ms),Effective FPS,Peak RSS (MB)\n")
        f.write(f"YOLOv8n,Intel Core i3-1215U,4,Unpinned (0-7),{total_eval_frames},{np.mean(yolo_all_lats):.2f},{np.median(yolo_all_lats):.2f},{np.percentile(yolo_all_lats, 90):.2f},{np.percentile(yolo_all_lats, 99):.2f},{1000.0/np.mean(yolo_all_lats):.2f},{peak_rss_mb:.2f}\n")
        f.write(f"D-FINE-N,Intel Core i3-1215U,4,Unpinned (0-7),{total_eval_frames},{np.mean(dfine_all_lats):.2f},{np.median(dfine_all_lats):.2f},{np.percentile(dfine_all_lats, 90):.2f},{np.percentile(dfine_all_lats, 99):.2f},{1000.0/np.mean(dfine_all_lats):.2f},{peak_rss_mb:.2f}\n")

    # Save summary json
    summary_json_path = outputs_dir / "prompt_36_benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "jaad_aggregate": {
                "total_frames": total_eval_frames,
                "total_gt": total_eval_gt,
                "yolo": {"precision": yolo_p, "recall": yolo_r, "f1": yolo_f1, "ap50": yolo_global_ap50, "ap75": yolo_global_ap75, "map50_95": yolo_global_map, "mean_latency_ms": float(np.mean(yolo_all_lats)), "fps": float(1000.0/np.mean(yolo_all_lats))},
                "dfine": {"precision": dfine_p, "recall": dfine_r, "f1": dfine_f1, "ap50": dfine_global_ap50, "ap75": dfine_global_ap75, "map50_95": dfine_global_map, "mean_latency_ms": float(np.mean(dfine_all_lats)), "fps": float(1000.0/np.mean(dfine_all_lats))}
            },
            "virat_sample": {
                "frames": virat_results['evaluated_frames'],
                "total_gt": virat_results['total_gt_boxes'],
                "yolo": {"tp": y_virat['tp'], "fp": y_virat['fp'], "fn": y_virat['fn'], "p": y_virat['precision'], "r": y_virat['recall'], "f1": y_virat['f1'], "ap50": y_virat['ap_50']},
                "dfine": {"tp": d_virat['tp'], "fp": d_virat['fp'], "fn": d_virat['fn'], "p": d_virat['precision'], "r": d_virat['recall'], "f1": d_virat['f1'], "ap50": d_virat['ap_50']}
            },
            "subgroups": subgroups,
            "wall_clock_seconds": total_jaad_wall_time,
            "peak_rss_mb": peak_rss_mb
        }, f, indent=2)

    print("\nBenchmark outputs generated successfully in outputs/")


if __name__ == "__main__":
    run_full_benchmark()

