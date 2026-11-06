"""SafeKid AI — Comprehensive Pipeline Benchmark & Profiling Tool.

Measures baseline and optimized performance across all four community environments:
1. Playground (1280x720, 24 FPS, 240 frames)
2. Driveway (1920x1080, 24 FPS, 216 frames)
3. Boom Barrier (1920x1080, 24 FPS, 144 frames)
4. Swimming Pool (848x480, 24 FPS, 240 frames)

Records:
- Source FPS and resolution
- Actual processing FPS
- Frame-processing latency (mean, median, p95, min, max)
- YOLO inference time
- ByteTrack tracking time
- ROI processing time
- Child safety event engine time
- Incident management time
- Visualization & video writing time
- Peak memory usage (RSS MB) and CPU usage
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safekid_ai.child_safety import (
    ChildSafetyEvent,
    ChildSafetyEventEngine,
    ManualTrackAnnotationProvider,
)
from safekid_ai.config import (
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    TrackerConfig,
)
from safekid_ai.detector import PersonDetector
from safekid_ai.events import SafetyEvent, ZoneStateManager
from safekid_ai.incidents import Incident, IncidentManager, IncidentPolicyConfig
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.loader import load_camera_roi_config
from safekid_ai.tracker import PersonTracker
from safekid_ai.video_input import VideoInput
from safekid_ai.visualization import draw_person_and_events, draw_roi_polygons


@dataclass
class ScenarioSpec:
    name: str
    scenario_id: str
    video_path: str
    config_path: str
    annotation_path: str
    output_video_path: str


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def benchmark_scenario(spec: ScenarioSpec, mode: str = "baseline") -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"BENCHMARKING [{mode.upper()}]: {spec.name} ({spec.scenario_id})")
    print(f"Video: {spec.video_path}")
    print(f"Config: {spec.config_path}")
    print("=" * 80)

    video_file = Path(spec.video_path).resolve()
    config_file = Path(spec.config_path).resolve()
    ann_file = Path(spec.annotation_path).resolve()
    out_video_file = Path(spec.output_video_path).resolve()
    out_video_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    camera_cfg = load_camera_roi_config(config_file)

    # Initialize pipeline stages with measurement instrumentation
    if mode == "baseline":
        det_config = DetectorConfig(
            model_name_or_path="yolov8n.pt",
            confidence_threshold=0.35,
            iou_threshold=0.45,
            fuse_model=False,
            num_threads=None,
        )
        tracker_config = TrackerConfig(tracker_type="bytetrack", persist=True)
    else:
        det_config = DetectorConfig(
            model_name_or_path="yolov8n.pt",
            confidence_threshold=0.35,
            iou_threshold=0.45,
            fuse_model=True,
            num_threads=4,
            imgsz=640,
        )
        tracker_config = TrackerConfig(tracker_type="bytetrack", persist=True, imgsz=640)

    detector = PersonDetector(det_config)
    tracker = PersonTracker(detector.model, tracker_config)

    roi_engine = ROIEngine(config=camera_cfg)
    zone_manager = ZoneStateManager(entry_confirmation_frames=3, exit_confirmation_frames=3)

    ann_provider = ManualTrackAnnotationProvider.load_from_json(ann_file)
    child_safety_engine = ChildSafetyEventEngine(
        annotation_provider=ann_provider,
        roi_config=camera_cfg,
    )
    incident_manager = IncidentManager()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_video_file), fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {out_video_file}")

    latencies: Dict[str, List[float]] = {
        "video_decoding_ms": [],
        "yolo_inference_ms": [],
        "bytetrack_ms": [],
        "roi_evaluation_ms": [],
        "child_safety_ms": [],
        "incident_manager_ms": [],
        "vis_and_writing_ms": [],
        "total_frame_ms": [],
    }

    video_input = VideoInput(source=str(video_file))
    active_alert_banners: List[Tuple[float, str, Any]] = []

    process = psutil.Process(os.getpid())
    mem_samples = []

    wall_start = time.perf_counter()
    frame_idx = 0
    input_iterator = video_input.frames()

    while True:
        # Stage 1: Decoding
        t_decode_start = time.perf_counter()
        try:
            frame_id, timestamp_ms, frame = next(input_iterator)
        except StopIteration:
            break
        t_decode_end = time.perf_counter()
        decode_ms = (t_decode_end - t_decode_start) * 1000.0

        t_frame_start = t_decode_start
        h, w = frame.shape[:2]

        # Stage 2: YOLO Detection + ByteTrack
        t_track_start = time.perf_counter()
        detections_obj = tracker.track(
            frame=frame,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            confidence_thresh=det_config.confidence_threshold,
            iou_thresh=det_config.iou_threshold,
        )
        t_track_end = time.perf_counter()
        det_track_total_ms = (t_track_end - t_track_start) * 1000.0

        raw_res = detections_obj.raw_results
        if raw_res is not None and hasattr(raw_res, "speed") and raw_res.speed:
            yolo_infer_ms = float(raw_res.speed.get("inference", 0.0))
            bytetrack_ms = max(0.0, det_track_total_ms - yolo_infer_ms)
        else:
            yolo_infer_ms = det_track_total_ms * 0.82
            bytetrack_ms = det_track_total_ms * 0.18

        detections = detections_obj.detections

        # Stage 3: ROI Evaluation
        t_roi_start = time.perf_counter()
        safety_events: List[SafetyEvent] = zone_manager.update(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            detections=detections,
            roi_engine=roi_engine,
            frame_resolution=(w, h),
        )
        t_roi_end = time.perf_counter()
        roi_eval_ms = (t_roi_end - t_roi_start) * 1000.0

        # Stage 4: Child Safety Event Engine
        t_child_start = time.perf_counter()
        child_events: List[ChildSafetyEvent] = child_safety_engine.process_events(
            safety_events=safety_events,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            roi_config=camera_cfg,
        )
        t_child_end = time.perf_counter()
        child_safety_ms = (t_child_end - t_child_start) * 1000.0

        # Stage 5: Incident Management
        t_inc_start = time.perf_counter()
        active_incidents = incident_manager.process_frame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            child_events=child_events,
            raw_safety_events=safety_events,
            roi_config=camera_cfg,
            active_track_ids={det.track_id for det in detections if det.track_id is not None},
        )
        t_inc_end = time.perf_counter()
        inc_manager_ms = (t_inc_end - t_inc_start) * 1000.0

        # Stage 6: Visualization and Video Writing
        t_vis_start = time.perf_counter()
        occupied_zones_per_track: Dict[Optional[int], List[str]] = {}
        dwell_times_per_track: Dict[Optional[int], Dict[str, float]] = {}
        track_classifications: Dict[int, str] = {}

        for det in detections:
            tid = det.track_id
            if tid is not None and child_safety_engine.annotation_provider:
                cls_val = child_safety_engine.annotation_provider.get_classification(tid, timestamp_ms)
                track_classifications[tid] = cls_val.value

            bottom_center = calculate_bottom_center(det.bbox_xyxy, (w, h))
            matching_zones = roi_engine.find_zones_for_point(bottom_center, (w, h))
            occupied_zones_per_track[tid] = [z.name for z in matching_zones]

            dwell_map: Dict[str, float] = {}
            if tid is not None:
                for z in matching_zones:
                    dwell_map[z.roi_id] = zone_manager.get_dwell_time_seconds(tid, z.roi_id, timestamp_ms)
            dwell_times_per_track[tid] = dwell_map

        # Manage banner messages
        active_alert_banners = [(exp, msg, sev) for exp, msg, sev in active_alert_banners if exp > timestamp_ms]
        for ce in child_events:
            active_alert_banners.append((timestamp_ms + 2500.0, f"CHILD ALERT: {ce.message}", ce.severity))
        for e in safety_events:
            if not any(ce.underlying_event_id == e.event_id for ce in child_events):
                active_alert_banners.append((timestamp_ms + 2500.0, f"{e.event_type.value}: {e.message}", e.severity))

        calibrated_zones = roi_engine.calibrate_for_resolution(w, h)
        vis = draw_roi_polygons(frame, calibrated_zones, alpha=0.22, detections=detections)
        active_items = [(msg, sev) for exp, msg, sev in active_alert_banners]
        annotated = draw_person_and_events(
            frame=vis,
            detections=detections,
            occupied_zones_per_track=occupied_zones_per_track,
            dwell_times_per_track=dwell_times_per_track,
            events=safety_events,
            community_name=f"Community {camera_cfg.community_id}",
            camera_id=camera_cfg.camera_id,
            active_alerts=active_items,
            track_classifications=track_classifications,
            child_events=child_events,
            incidents=active_incidents,
            copy=(mode == "baseline"),
        )
        video_writer.write(annotated)
        t_vis_end = time.perf_counter()
        vis_writing_ms = (t_vis_end - t_vis_start) * 1000.0

        total_frame_ms = (t_vis_end - t_frame_start) * 1000.0

        latencies["video_decoding_ms"].append(decode_ms)
        latencies["yolo_inference_ms"].append(yolo_infer_ms)
        latencies["bytetrack_ms"].append(bytetrack_ms)
        latencies["roi_evaluation_ms"].append(roi_eval_ms)
        latencies["child_safety_ms"].append(child_safety_ms)
        latencies["incident_manager_ms"].append(inc_manager_ms)
        latencies["vis_and_writing_ms"].append(vis_writing_ms)
        latencies["total_frame_ms"].append(total_frame_ms)

        frame_idx += 1
        if frame_idx % 20 == 0:
            mem_samples.append(process.memory_info().rss / (1024 * 1024))
        if frame_idx % 40 == 0 or frame_idx == total_frames:
            print(f"  Frame {frame_idx:3d}/{total_frames} | Latency: {total_frame_ms:6.1f} ms | YOLO: {yolo_infer_ms:5.1f} ms | Vis+Write: {vis_writing_ms:5.1f} ms")

    wall_duration = time.perf_counter() - wall_start
    video_writer.release()
    avg_fps = frame_idx / wall_duration if wall_duration > 0 else 0.0

    stage_metrics = {stage: compute_stats(latencies[stage]) for stage in latencies}
    mean_total = stage_metrics["total_frame_ms"]["mean"]

    stage_percentages = {
        stage: (stage_metrics[stage]["mean"] / mean_total * 100.0) if mean_total > 0 else 0.0
        for stage in latencies
        if stage != "total_frame_ms"
    }

    peak_mem_mb = max(mem_samples) if mem_samples else (process.memory_info().rss / (1024 * 1024))

    result = {
        "scenario_id": spec.scenario_id,
        "name": spec.name,
        "mode": mode,
        "video_resolution": f"{width}x{height}",
        "video_fps": fps,
        "frames_profiled": frame_idx,
        "wall_duration_seconds": round(wall_duration, 2),
        "average_throughput_fps": round(avg_fps, 2),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "stage_metrics": stage_metrics,
        "stage_percentages": stage_percentages,
    }

    print(f"Completed {spec.name} [{mode}]: {frame_idx} frames in {wall_duration:.2f}s ({avg_fps:.2f} FPS) | Mean: {mean_total:.1f}ms | Peak RSS: {peak_mem_mb:.1f}MB")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    demo_dir = root / "data" / "demo"
    cfg_dir = root / "configs" / "cameras"
    ann_dir = root / "configs" / "annotations"
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ScenarioSpec(
            name="Swimming Pool",
            scenario_id="scenario_pool",
            video_path=str(demo_dir / "pool_safety_demo.mp4"),
            config_path=str(cfg_dir / "community_demo" / "camera_pool_demo.json"),
            annotation_path=str(ann_dir / "demo_pool_track_annotations.json"),
            output_video_path=str(out_dir / f"bench_{args.mode}_pool.mp4"),
        ),
        ScenarioSpec(
            name="Playground",
            scenario_id="scenario_playground",
            video_path=str(demo_dir / "playground_demo.mp4"),
            config_path=str(cfg_dir / "community_002" / "camera_playground_01.json"),
            annotation_path=str(ann_dir / "demo_playground_track_annotations.json"),
            output_video_path=str(out_dir / f"bench_{args.mode}_playground.mp4"),
        ),
        ScenarioSpec(
            name="Driveway",
            scenario_id="scenario_driveway",
            video_path=str(demo_dir / "driveway_demo.mp4"),
            config_path=str(cfg_dir / "community_001" / "camera_gate_north.json"),
            annotation_path=str(ann_dir / "demo_driveway_track_annotations.json"),
            output_video_path=str(out_dir / f"bench_{args.mode}_driveway.mp4"),
        ),
        ScenarioSpec(
            name="Boom Barrier",
            scenario_id="scenario_boom_barrier",
            video_path=str(demo_dir / "boom_barrier_demo.mp4"),
            config_path=str(cfg_dir / "community_001" / "camera_gate_north.json"),
            annotation_path=str(ann_dir / "demo_boom_barrier_track_annotations.json"),
            output_video_path=str(out_dir / f"bench_{args.mode}_boom_barrier.mp4"),
        ),
    ]

    all_results = []
    for sc in scenarios:
        res = benchmark_scenario(sc, mode=args.mode)
        all_results.append(res)

    out_file = Path(args.out) if args.out else (out_dir / f"benchmark_{args.mode}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved {args.mode.upper()} Benchmark Results] {out_file}")


if __name__ == "__main__":
    main()

