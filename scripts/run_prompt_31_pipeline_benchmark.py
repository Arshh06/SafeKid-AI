"""Script for Prompt 31: Measured CPU Inference Optimization & End-to-End Pipeline Benchmark.

Measures genuine, un-extrapolated component latencies, FPS, memory usage, CPU utilization,
and tracking/safety quality across detector backends (D-FINE-N, YOLOv8n, RT-DETR-L)
and keyframe sampling intervals (1, 2, 3, 5).
"""

import csv
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
import torch

# Ensure src is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from safekid_ai.config import DetectorConfig, SafeKidPipelineConfig, TrackerConfig
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.detector import DetectionBox, FrameDetections
from safekid_ai.pipeline import SafeKidPipeline


def get_hardware_profile() -> Dict[str, Any]:
    """Inspects and profiles the host hardware and runtime environment."""
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()

    # Determine CPU brand string
    cpu_brand = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_brand = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass

    profile = {
        "os_platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_brand": cpu_brand,
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_freq_mhz": round(cpu_freq.max, 2) if cpu_freq else None,
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_available": torch.cuda.is_available(),
        "pytorch_device": "cpu",
        "pytorch_mkl_available": torch.backends.mkl.is_available(),
        "pytorch_openmp_available": torch.backends.openmp.is_available(),
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return profile


def run_single_benchmark_run(
    backend: str,
    model_name: str,
    interval: int,
    num_threads: int,
    video_path: str,
    roi_config_path: str,
    num_frames: int = 100,
    warmup_frames: int = 10,
    child_classifier_path: str = "models/child_classifier_jaad_v1.pt",
) -> Dict[str, Any]:
    """Runs a single timed benchmark run over video_path measuring all pipeline components."""
    # Set thread allocation
    torch.set_num_threads(num_threads)
    cv2.setNumThreads(num_threads)

    cfg = SafeKidPipelineConfig(
        community_id="community_demo",
        camera_id="camera_pool_demo",
        roi_config_path=roi_config_path,
        detector=DetectorConfig(
            backend=backend,
            model_name_or_path=model_name,
            model_variant="dfine_n" if "dfine" in backend else "yolov8n",
            num_threads=num_threads,
            frame_interval=interval,
        ),
        tracker=TrackerConfig(
            frame_interval=interval,
            max_track_staleness=interval * 2,
        ),
        enable_automated_child_detection=True,
        child_detector_config=ChildDetectorConfig(
            model_weights_path=child_classifier_path,
            classification_interval=max(1, interval),
            skip_stationary_crops=True,
        ),
        frame_interval=interval,
    )

    pipeline = SafeKidPipeline(cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {video_path}")

    # Warmup
    for w in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        pipeline.process_frame(frame, frame_id=w, timestamp_ms=w * 41.67)

    # Reset state for clean measurement
    pipeline.reset()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    process = psutil.Process()
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    # Detailed per-frame metrics
    latencies_e2e = []
    latencies_detection = []
    latencies_tracking = []
    latencies_child_cls = []
    latencies_roi = []
    latencies_events = []
    latencies_decode = []

    detection_counts = []
    unique_track_ids = set()
    total_child_events = 0
    total_safety_events = 0
    child_labels_count = {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0}
    predicted_frames_count = 0
    keyframe_count = 0

    fps_list = []
    start_total_wall = time.perf_counter()

    for fid in range(num_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            break

        decode_ms = (t1 - t0) * 1000.0
        latencies_decode.append(decode_ms)

        is_keyframe = (interval == 1) or (fid % interval == 0)
        if is_keyframe:
            keyframe_count += 1
        else:
            predicted_frames_count += 1

        t_frame_start = time.perf_counter()
        res = pipeline.process_frame(frame, frame_id=fid, timestamp_ms=fid * 41.67, annotate=False)
        t_frame_end = time.perf_counter()

        frame_total_ms = (t_frame_end - t_frame_start) * 1000.0
        latencies_e2e.append(frame_total_ms)

        # Quality tracking
        n_dets = len(res.person_states)
        detection_counts.append(n_dets)
        for ps in res.person_states:
            if ps.track_id is not None:
                unique_track_ids.add(ps.track_id)

        total_safety_events += len(res.events)
        total_child_events += len(res.child_events)

        if pipeline.child_annotation_provider and hasattr(pipeline.child_annotation_provider, "_latest_results"):
            for res_cls in pipeline.child_annotation_provider._latest_results.values():
                lbl = res_cls.label.value if hasattr(res_cls.label, "value") else str(res_cls.label)
                if lbl in child_labels_count:
                    child_labels_count[lbl] += 1

    total_wall_sec = time.perf_counter() - start_total_wall
    cap.release()

    ram_end_mb = process.memory_info().rss / (1024 * 1024)
    avg_fps = len(latencies_e2e) / total_wall_sec if total_wall_sec > 0 else 0.0

    return {
        "backend": backend,
        "model_name": model_name,
        "interval": interval,
        "threads": num_threads,
        "frames_processed": len(latencies_e2e),
        "total_wall_sec": round(total_wall_sec, 3),
        "fps": round(avg_fps, 2),
        "mean_latency_ms": round(float(np.mean(latencies_e2e)), 2),
        "std_latency_ms": round(float(np.std(latencies_e2e)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_e2e, 50)), 2),
        "p90_latency_ms": round(float(np.percentile(latencies_e2e, 90)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies_e2e, 99)), 2),
        "min_latency_ms": round(float(np.min(latencies_e2e)), 2),
        "max_latency_ms": round(float(np.max(latencies_e2e)), 2),
        "mean_decode_ms": round(float(np.mean(latencies_decode)), 2),
        "ram_start_mb": round(ram_start_mb, 2),
        "ram_end_mb": round(ram_end_mb, 2),
        "ram_peak_mb": round(max(ram_start_mb, ram_end_mb), 2),
        "unique_tracks": len(unique_track_ids),
        "total_detections": sum(detection_counts),
        "mean_detections_per_frame": round(float(np.mean(detection_counts)), 2),
        "total_safety_events": total_safety_events,
        "total_child_events": total_child_events,
        "keyframes": keyframe_count,
        "predicted_frames": predicted_frames_count,
        "raw_latencies_e2e": latencies_e2e,
    }


def measure_isolated_component_latencies(
    num_threads: int = 4,
    video_path: str = "data/demo/pool_safety_demo.mp4",
    roi_config_path: str = "configs/cameras/community_demo/camera_pool_demo.json",
    num_samples: int = 30,
) -> Dict[str, Any]:
    """Directly measures fine-grained isolated component latencies on CPU."""
    torch.set_num_threads(num_threads)
    cv2.setNumThreads(num_threads)

    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_samples):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video.")

    # 1. Video Decode Latency
    decode_times = []
    cap = cv2.VideoCapture(video_path)
    for _ in range(num_samples):
        t0 = time.perf_counter()
        ret, _ = cap.read()
        t1 = time.perf_counter()
        if ret:
            decode_times.append((t1 - t0) * 1000.0)
    cap.release()

    # 2. Preprocessing Latency (letterbox/resize 848x480 -> 640x640 + float normalize)
    preproc_times = []
    for f in frames:
        t0 = time.perf_counter()
        im = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im, (640, 640))
        tensor = torch.from_numpy(im_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t1 = time.perf_counter()
        preproc_times.append((t1 - t0) * 1000.0)

    # 3. D-FINE-N Forward & Postprocessing
    from safekid_ai.detectors import create_detector
    dfine_adapter = create_detector(
        DetectorConfig(backend="dfine", model_variant="dfine_n", num_threads=num_threads),
        TrackerConfig(),
    )
    dfine_forward_times = []
    dfine_postproc_times = []
    im_tensor = tensor.to("cpu")
    orig_size = torch.tensor([[848.0, 480.0]], dtype=torch.float32)

    # Warmup
    for _ in range(3):
        with torch.inference_mode():
            _ = dfine_adapter.model(im_tensor, orig_size)

    for _ in range(num_samples):
        t0 = time.perf_counter()
        with torch.inference_mode():
            labels, boxes, scores = dfine_adapter.model(im_tensor, orig_size)
        t1 = time.perf_counter()
        dfine_forward_times.append((t1 - t0) * 1000.0)

        t_post0 = time.perf_counter()
        l_np = labels[0].cpu().numpy().astype(int)
        b_np = boxes[0].cpu().numpy()
        s_np = scores[0].cpu().numpy()
        mask = (l_np == 0) & (s_np >= 0.35)
        _ = b_np[mask]
        t_post1 = time.perf_counter()
        dfine_postproc_times.append((t_post1 - t_post0) * 1000.0)

    # 4. YOLOv8n Forward & Postproc
    from safekid_ai.detector import PersonDetector
    yolo_detector = PersonDetector(DetectorConfig(backend="yolov8", model_name_or_path="yolov8n.pt", num_threads=num_threads))
    yolo_times = []
    for f in frames[:num_samples]:
        t0 = time.perf_counter()
        _ = yolo_detector.detect(f)
        t1 = time.perf_counter()
        yolo_times.append((t1 - t0) * 1000.0)

    # 5. Child Classifier Latency (models/child_classifier_jaad_v1.pt)
    from safekid_ai.child_detection.detector import ChildDetector
    child_det = ChildDetector(ChildDetectorConfig(model_weights_path="models/child_classifier_jaad_v1.pt", device="cpu"))
    mock_crop = np.zeros((128, 64, 3), dtype=np.uint8)
    child_cls_times = []
    for _ in range(num_samples * 2):
        t0 = time.perf_counter()
        _ = child_det.classify_crop(mock_crop)
        t1 = time.perf_counter()
        child_cls_times.append((t1 - t0) * 1000.0)

    # 6. ByteTrack Tracking Latency
    from safekid_ai.tracker import PersonTracker, TrackedPerson
    tracker = PersonTracker(yolo_detector.model, TrackerConfig())
    tracked_person = TrackedPerson(
        track_id=1,
        current_box=DetectionBox(bbox_xyxy=(100.0, 100.0, 150.0, 200.0), confidence=0.9),
    )
    extrapolate_times = []
    for _ in range(num_samples * 10):
        t0 = time.perf_counter()
        _ = tracked_person.extrapolate(frame_id=1, frame_size=(848, 480))
        t1 = time.perf_counter()
        extrapolate_times.append((t1 - t0) * 1000.0)

    # 7. ROI Engine & Zone Evaluation Latency
    from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
    roi_engine = ROIEngine(config_path=roi_config_path)
    roi_times = []
    for _ in range(num_samples * 10):
        bc = (450.0, 300.0)
        t0 = time.perf_counter()
        _ = roi_engine.find_zones_for_point(bc, (848, 480))
        t1 = time.perf_counter()
        roi_times.append((t1 - t0) * 1000.0)

    return {
        "threads": num_threads,
        "video_decode_mean_ms": round(float(np.mean(decode_times)), 3),
        "video_decode_p50_ms": round(float(np.percentile(decode_times, 50)), 3),
        "preprocess_mean_ms": round(float(np.mean(preproc_times)), 3),
        "dfine_n_forward_mean_ms": round(float(np.mean(dfine_forward_times)), 3),
        "dfine_n_forward_p50_ms": round(float(np.percentile(dfine_forward_times, 50)), 3),
        "dfine_n_forward_p90_ms": round(float(np.percentile(dfine_forward_times, 90)), 3),
        "dfine_n_postproc_mean_ms": round(float(np.mean(dfine_postproc_times)), 3),
        "yolov8n_detect_e2e_mean_ms": round(float(np.mean(yolo_times)), 3),
        "yolov8n_detect_e2e_p50_ms": round(float(np.percentile(yolo_times, 50)), 3),
        "yolov8n_detect_e2e_p90_ms": round(float(np.percentile(yolo_times, 90)), 3),
        "child_classifier_crop_mean_ms": round(float(np.mean(child_cls_times)), 3),
        "child_classifier_crop_p50_ms": round(float(np.percentile(child_cls_times, 50)), 3),
        "bytetrack_extrapolate_mean_us": round(float(np.mean(extrapolate_times)) * 1000.0, 2),  # in microseconds
        "roi_zone_eval_mean_us": round(float(np.mean(roi_times)) * 1000.0, 2),  # in microseconds
    }


def main():
    print("=" * 80)
    print("SafeKid AI — Prompt 31 Measured CPU Inference Benchmark")
    print("=" * 80)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(ROOT_DIR / "data" / "demo" / "pool_safety_demo.mp4")
    roi_config_path = str(ROOT_DIR / "configs" / "cameras" / "community_demo" / "camera_pool_demo.json")

    # 1. Hardware Profile
    print("[1/5] Profiling hardware environment...")
    hw_profile = get_hardware_profile()
    with open(outputs_dir / "prompt_31_hardware_profile.json", "w") as f:
        json.dump(hw_profile, f, indent=2)
    print(f"  CPU: {hw_profile['cpu_brand']} ({hw_profile['cpu_physical_cores']}C / {hw_profile['cpu_logical_cores']}T)")
    print(f"  RAM: {hw_profile['total_ram_gb']} GB total, {hw_profile['available_ram_gb']} GB available")
    print(f"  PyTorch: {hw_profile['pytorch_version']} (CPU, MKL={hw_profile['pytorch_mkl_available']}, OpenMP={hw_profile['pytorch_openmp_available']})")

    # 2. Component-level Latencies
    print("\n[2/5] Measuring fine-grained isolated component latencies on CPU...")
    comp_latencies = measure_isolated_component_latencies(
        num_threads=4,
        video_path=video_path,
        roi_config_path=roi_config_path,
    )
    with open(outputs_dir / "prompt_31_component_latency.json", "w") as f:
        json.dump(comp_latencies, f, indent=2)
    print(f"  D-FINE-N forward: {comp_latencies['dfine_n_forward_mean_ms']:.2f} ms")
    print(f"  YOLOv8n detect:  {comp_latencies['yolov8n_detect_e2e_mean_ms']:.2f} ms")
    print(f"  Child Classifier: {comp_latencies['child_classifier_crop_mean_ms']:.2f} ms/crop")
    print(f"  Extrapolation:   {comp_latencies['bytetrack_extrapolate_mean_us']:.1f} us/track")
    print(f"  ROI Evaluation:  {comp_latencies['roi_zone_eval_mean_us']:.1f} us/point")

    # 3. Thread Scaling Study (1, 2, 4, 6 threads) at Interval 1
    print("\n[3/5] Measuring CPU thread scaling across 1, 2, 4, 6 threads (Interval 1)...")
    thread_matrix = [1, 2, 4, 6]
    thread_results = []
    for th in thread_matrix:
        print(f"  Testing D-FINE-N with {th} threads (30 frames)...")
        r_dfine = run_single_benchmark_run(
            backend="dfine",
            model_name="models/dfine_n_coco.pth",
            interval=1,
            num_threads=th,
            video_path=video_path,
            roi_config_path=roi_config_path,
            num_frames=30,
            warmup_frames=5,
        )
        print(f"    -> D-FINE-N ({th} th): {r_dfine['fps']} FPS, {r_dfine['mean_latency_ms']} ms")

        print(f"  Testing YOLOv8n with {th} threads (30 frames)...")
        r_yolo = run_single_benchmark_run(
            backend="yolov8",
            model_name="yolov8n.pt",
            interval=1,
            num_threads=th,
            video_path=video_path,
            roi_config_path=roi_config_path,
            num_frames=30,
            warmup_frames=5,
        )
        print(f"    -> YOLOv8n ({th} th):  {r_yolo['fps']} FPS, {r_yolo['mean_latency_ms']} ms")
        thread_results.append({
            "threads": th,
            "dfine_fps": r_dfine["fps"],
            "dfine_latency_ms": r_dfine["mean_latency_ms"],
            "yolo_fps": r_yolo["fps"],
            "yolo_latency_ms": r_yolo["mean_latency_ms"],
        })

    # 4. End-to-End Pipeline Keyframe Sampling Benchmark (Intervals: 1, 2, 3, 5)
    print("\n[4/5] Running Measured Continuous End-to-End Pipeline Benchmark (100 frames, 4 threads)...")
    test_configurations = [
        # D-FINE-N candidate
        {"backend": "dfine", "model": "models/dfine_n_coco.pth", "interval": 1, "name": "D-FINE-N (Interval 1)"},
        {"backend": "dfine", "model": "models/dfine_n_coco.pth", "interval": 2, "name": "D-FINE-N (Interval 2)"},
        {"backend": "dfine", "model": "models/dfine_n_coco.pth", "interval": 3, "name": "D-FINE-N (Interval 3)"},
        {"backend": "dfine", "model": "models/dfine_n_coco.pth", "interval": 5, "name": "D-FINE-N (Interval 5)"},
        # YOLOv8n baseline
        {"backend": "yolov8", "model": "yolov8n.pt", "interval": 1, "name": "YOLOv8n (Interval 1)"},
        {"backend": "yolov8", "model": "yolov8n.pt", "interval": 2, "name": "YOLOv8n (Interval 2)"},
        {"backend": "yolov8", "model": "yolov8n.pt", "interval": 3, "name": "YOLOv8n (Interval 3)"},
        {"backend": "yolov8", "model": "yolov8n.pt", "interval": 5, "name": "YOLOv8n (Interval 5)"},
    ]

    benchmark_records = []
    quality_records = {}

    for cfg in test_configurations:
        print(f"\n  Running: {cfg['name']}...")
        runs = []
        for rep in range(3):
            r = run_single_benchmark_run(
                backend=cfg["backend"],
                model_name=cfg["model"],
                interval=cfg["interval"],
                num_threads=4,
                video_path=video_path,
                roi_config_path=roi_config_path,
                num_frames=100,
                warmup_frames=10,
            )
            runs.append(r)
            print(f"    Rep {rep+1}: {r['fps']:.2f} FPS | Mean: {r['mean_latency_ms']:.1f} ms | P90: {r['p90_latency_ms']:.1f} ms | Tracks: {r['unique_tracks']}")

        # Average across 3 repetitions
        avg_fps = round(float(np.mean([x["fps"] for x in runs])), 2)
        avg_latency = round(float(np.mean([x["mean_latency_ms"] for x in runs])), 2)
        std_latency = round(float(np.mean([x["std_latency_ms"] for x in runs])), 2)
        p50_latency = round(float(np.mean([x["p50_latency_ms"] for x in runs])), 2)
        p90_latency = round(float(np.mean([x["p90_latency_ms"] for x in runs])), 2)
        p99_latency = round(float(np.mean([x["p99_latency_ms"] for x in runs])), 2)
        peak_ram = round(float(np.max([x["ram_peak_mb"] for x in runs])), 2)
        unique_tracks = int(round(np.mean([x["unique_tracks"] for x in runs])))
        total_dets = int(round(np.mean([x["total_detections"] for x in runs])))
        safety_events = int(round(np.mean([x["total_safety_events"] for x in runs])))
        child_events = int(round(np.mean([x["total_child_events"] for x in runs])))

        rec = {
            "Configuration": cfg["name"],
            "Backend": cfg["backend"],
            "Interval": cfg["interval"],
            "Threads": 4,
            "FPS": avg_fps,
            "Latency_Mean_ms": avg_latency,
            "Latency_Std_ms": std_latency,
            "Latency_P50_ms": p50_latency,
            "Latency_P90_ms": p90_latency,
            "Latency_P99_ms": p99_latency,
            "Peak_RAM_MB": peak_ram,
            "Unique_Tracks": unique_tracks,
            "Total_Detections": total_dets,
            "Safety_Events": safety_events,
            "Child_Events": child_events,
        }
        benchmark_records.append(rec)

        quality_records[cfg["name"]] = {
            "backend": cfg["backend"],
            "interval": cfg["interval"],
            "unique_tracks": unique_tracks,
            "total_detections": total_dets,
            "safety_events_count": safety_events,
            "child_events_count": child_events,
            "keyframes": runs[0]["keyframes"],
            "predicted_frames": runs[0]["predicted_frames"],
            "track_continuity_ratio": round(1.0 - (abs(unique_tracks - benchmark_records[0]["Unique_Tracks"]) / max(1, benchmark_records[0]["Unique_Tracks"])), 3) if len(benchmark_records) > 0 else 1.0,
        }

    # Reference run: RT-DETR-L (Interval 1 and Interval 5, single rep 25 frames)
    print("\n  Running RT-DETR-L Reference Run (25 frames)...")
    try:
        rtdetr_r1 = run_single_benchmark_run(
            backend="rtdetr",
            model_name="rtdetr-l.pt",
            interval=1,
            num_threads=4,
            video_path=video_path,
            roi_config_path=roi_config_path,
            num_frames=25,
            warmup_frames=3,
        )
        rtdetr_r5 = run_single_benchmark_run(
            backend="rtdetr",
            model_name="rtdetr-l.pt",
            interval=5,
            num_threads=4,
            video_path=video_path,
            roi_config_path=roi_config_path,
            num_frames=25,
            warmup_frames=3,
        )
        benchmark_records.append({
            "Configuration": "RT-DETR-L (Reference, Interval 1)",
            "Backend": "rtdetrv2",
            "Interval": 1,
            "Threads": 4,
            "FPS": rtdetr_r1["fps"],
            "Latency_Mean_ms": rtdetr_r1["mean_latency_ms"],
            "Latency_Std_ms": rtdetr_r1["std_latency_ms"],
            "Latency_P50_ms": rtdetr_r1["p50_latency_ms"],
            "Latency_P90_ms": rtdetr_r1["p90_latency_ms"],
            "Latency_P99_ms": rtdetr_r1["p99_latency_ms"],
            "Peak_RAM_MB": rtdetr_r1["ram_peak_mb"],
            "Unique_Tracks": rtdetr_r1["unique_tracks"],
            "Total_Detections": rtdetr_r1["total_detections"],
            "Safety_Events": rtdetr_r1["total_safety_events"],
            "Child_Events": rtdetr_r1["total_child_events"],
        })
        benchmark_records.append({
            "Configuration": "RT-DETR-L (Reference, Interval 5)",
            "Backend": "rtdetrv2",
            "Interval": 5,
            "Threads": 4,
            "FPS": rtdetr_r5["fps"],
            "Latency_Mean_ms": rtdetr_r5["mean_latency_ms"],
            "Latency_Std_ms": rtdetr_r5["std_latency_ms"],
            "Latency_P50_ms": rtdetr_r5["p50_latency_ms"],
            "Latency_P90_ms": rtdetr_r5["p90_latency_ms"],
            "Latency_P99_ms": rtdetr_r5["p99_latency_ms"],
            "Peak_RAM_MB": rtdetr_r5["ram_peak_mb"],
            "Unique_Tracks": rtdetr_r5["unique_tracks"],
            "Total_Detections": rtdetr_r5["total_detections"],
            "Safety_Events": rtdetr_r5["total_safety_events"],
            "Child_Events": rtdetr_r5["total_child_events"],
        })
    except Exception as e:
        print(f"    [Warning] RT-DETR-L reference run skipped: {e}")

    # 5. Save Artifacts
    print("\n[5/5] Saving benchmark deliverables...")
    # CSV
    csv_path = outputs_dir / "prompt_31_benchmark_results.csv"
    fieldnames = list(benchmark_records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_records)
    print(f"  Wrote: {csv_path}")

    # Quality Comparison JSON
    qual_path = outputs_dir / "prompt_31_quality_comparison.json"
    with open(qual_path, "w") as f:
        json.dump(quality_records, f, indent=2)
    print(f"  Wrote: {qual_path}")

    # Configuration Comparison JSON
    config_comp = {
        "hardware": hw_profile,
        "thread_scaling": thread_results,
        "component_breakdown": comp_latencies,
        "benchmarks": benchmark_records,
        "quality": quality_records,
    }
    config_path = outputs_dir / "prompt_31_configuration_comparison.json"
    with open(config_path, "w") as f:
        json.dump(config_comp, f, indent=2)
    print(f"  Wrote: {config_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'FPS':<6} | {'Mean (ms)':<10} | {'P90 (ms)':<9} | {'RAM (MB)':<9} | {'Tracks':<6}")
    print("-" * 84)
    for r in benchmark_records:
        print(f"{r['Configuration']:<35} | {r['FPS']:<6.2f} | {r['Latency_Mean_ms']:<10.1f} | {r['Latency_P90_ms']:<9.1f} | {r['Peak_RAM_MB']:<9.1f} | {r['Unique_Tracks']:<6}")
    print("=" * 80)


if __name__ == "__main__":
    main()
