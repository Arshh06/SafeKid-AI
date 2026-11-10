"""Evaluates Keyframe Sampling Intervals (1, 2, 3) on JAAD sequence video_0317."""

import json
from pathlib import Path
import time
import cv2
import numpy as np

from safekid_ai.config import DetectorConfig, TrackerConfig
from safekid_ai.detector import PersonDetector
from safekid_ai.detectors.dfine_detector import DFINEDetectorAdapter
from safekid_ai.tracker import PersonTracker
from safekid_ai.datasets.jaad_parser import JAADAnnotationParser


def calculate_iou(box_a, box_b):
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
    return float(inter_area / union_area) if union_area > 0 else 0.0


def run_sampling_eval():
    print("Running frame sampling analysis on video_0317 (270 frames)...")
    v_path = Path("data/external/jaad/videos/video_0317.mp4")
    ann_path = Path("data/external/jaad/repo/annotations/video_0317.xml")
    attr_path = Path("data/external/jaad/repo/annotations_attributes/video_0317_attributes.xml")

    parser = JAADAnnotationParser()
    boxes, _ = parser.parse_video_annotations(ann_path, attr_path, 1920, 1080)
    frame_gt_map = {}
    for b in boxes:
        if b.frame_id not in frame_gt_map:
            frame_gt_map[b.frame_id] = []
        frame_gt_map[b.frame_id].append((b.xtl, b.ytl, b.xbr, b.ybr))

    sampling_results = []

    for model_name in ["yolov8n", "dfine_n"]:
        for interval in [1, 2, 3]:
            cap = cv2.VideoCapture(str(v_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if model_name == "yolov8n":
                det_cfg = DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu", confidence_threshold=0.25)
                detector = PersonDetector(det_cfg)
                trk_cfg = TrackerConfig(frame_interval=interval)
                tracker = PersonTracker(detector_model=detector.model, config=trk_cfg)
            else:
                det_cfg = DetectorConfig(model_name_or_path="models/dfine_n_coco.pth", device="cpu", confidence_threshold=0.25, frame_interval=interval)
                trk_cfg = TrackerConfig(frame_interval=interval)
                tracker = DFINEDetectorAdapter(det_cfg, tracker_config=trk_cfg, variant="dfine_n")

            fresh_detections_count = 0
            extrapolated_boxes_count = 0
            fresh_tp = 0
            fresh_fp = 0
            total_eval_gt = 0
            frame_times = []

            for fid in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                gt_boxes = frame_gt_map.get(fid, [])

                t0 = time.perf_counter()
                tracked_res = tracker.track(frame, frame_id=fid, confidence_thresh=0.25)
                t_frame = (time.perf_counter() - t0) * 1000.0

                if fid >= 10:  # discard warmup
                    frame_times.append(t_frame)
                    total_eval_gt += len(gt_boxes)

                    for trk in tracked_res.detections:
                        if getattr(trk, "is_predicted", False):
                            extrapolated_boxes_count += 1
                        else:
                            fresh_detections_count += 1
                            # Match only fresh detections to GT
                            matched = False
                            for gb in gt_boxes:
                                if calculate_iou(trk.bbox_xyxy, gb) >= 0.50:
                                    matched = True
                                    break
                            if matched:
                                fresh_tp += 1
                            else:
                                fresh_fp += 1

            cap.release()
            total_boxes = fresh_detections_count + extrapolated_boxes_count
            extrap_pct = (extrapolated_boxes_count / total_boxes * 100.0) if total_boxes > 0 else 0.0
            mean_lat = np.mean(frame_times)
            effective_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

            res = {
                "model": model_name,
                "interval": interval,
                "evaluated_frames": len(frame_times),
                "total_gt": total_eval_gt,
                "fresh_detections": fresh_detections_count,
                "extrapolated_boxes": extrapolated_boxes_count,
                "total_rendered_boxes": total_boxes,
                "extrapolation_percentage": round(extrap_pct, 2),
                "fresh_tp": fresh_tp,
                "fresh_fp": fresh_fp,
                "fresh_precision": round(fresh_tp / (fresh_tp + fresh_fp), 4) if (fresh_tp + fresh_fp) > 0 else 0.0,
                "fresh_recall_relative_to_interval_gt": round(fresh_tp / (total_eval_gt / interval), 4) if total_eval_gt > 0 else 0.0,
                "mean_frame_time_ms": round(float(mean_lat), 2),
                "effective_fps": round(float(effective_fps), 2)
            }
            sampling_results.append(res)
            print(f"[{model_name.upper()} Int={interval}] Extrapolated: {extrap_pct:.1f}%, Mean Time: {mean_lat:.1f}ms ({effective_fps:.1f} FPS)")

    with open("scratch/sampling_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(sampling_results, f, indent=2)
    print("Sampling results saved to scratch/sampling_analysis_results.json")


if __name__ == "__main__":
    run_sampling_eval()
