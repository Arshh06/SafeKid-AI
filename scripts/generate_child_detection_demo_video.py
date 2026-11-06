"""Generates an annotated demonstration video and visual contact sheet for Prompt 21 Child Detection."""

from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.child_detection.provider import AutomatedChildClassificationProvider
from safekid_ai.child_safety import ChildSafetyEventEngine
from safekid_ai.config import DetectorConfig, SafeKidPipelineConfig, SafetyZoneType
from safekid_ai.detector import PersonDetector
from safekid_ai.events import ZoneStateManager
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.roi.engine import ROIEngine, calculate_bottom_center
from safekid_ai.roi.models import (
    AlertSeverity,
    CameraROIConfig,
    Polygon,
    SafetyZone,
    ZoneRole,
    ZoneSafetyPolicy,
)
from safekid_ai.tracker import PersonTracker
from safekid_ai.video_input import VideoInput
from safekid_ai.visualization import draw_person_and_events, draw_roi_polygons


def generate_demo() -> None:
    """Processes video through SafeKid AI pipeline with dedicated ChildDetector."""
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / "child_detection_demo_annotated.mp4"
    out_sheet_path = output_dir / "child_detection_demo_contact_sheet.jpg"

    input_video = "data/demo/playground_demo.mp4"
    if not Path(input_video).exists():
        input_video = "data/demo/pool_safety_demo.mp4"

    print(f"Loading input video fixture: {input_video}")
    v_in = VideoInput(source=input_video)

    # Configure dedicated automated child detection pipeline
    pipeline_cfg = SafeKidPipelineConfig(
        community_id="community_greenfield",
        community_name="Greenfield Residential Enclave",
        camera_id="cam_playground_north",
        zone_type=SafetyZoneType.PLAYGROUND,
        enable_automated_child_detection=True,
        child_detector_config=ChildDetectorConfig(
            model_weights_path="models/child_classifier_v1.pt",
            confidence_threshold=0.65,
            margin_threshold=0.20,
        ),
        roi_config_path="configs/cameras/community_002/camera_playground_01.json",
    )
    pipeline = SafeKidPipeline(pipeline_cfg)

    writer = None
    frames_for_sheet = []
    total_frames = 0
    child_alerts_fired = 0
    classifications_seen = {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0}

    print("Running end-to-end pipeline with dedicated ChildDetector...")
    for frame_res in pipeline.process_video(v_in, annotate=True):
        total_frames += 1
        vis = frame_res.annotated_frame

        if writer is None:
            fh, fw = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, 20.0, (fw, fh))

        writer.write(vis)

        if frame_res.child_events:
            child_alerts_fired += len(frame_res.child_events)

        for p in frame_res.person_states:
            # Check track classification
            tid = p.track_id
            if tid is not None and pipeline.child_safety_engine and pipeline.child_safety_engine.annotation_provider:
                cls_v = pipeline.child_safety_engine.annotation_provider.get_classification(tid).value
                if cls_v in classifications_seen:
                    classifications_seen[cls_v] += 1

        # Capture key frames for contact sheet (e.g. frame 10, 25, 40, 55, 70, 85)
        if total_frames in (12, 28, 44, 60, 76, 92) and len(frames_for_sheet) < 6:
            frames_for_sheet.append(vis.copy())

    if writer:
        writer.release()
    print(f"Saved annotated video: {out_video_path} ({total_frames} frames)")

    # Build 2x3 contact sheet
    if frames_for_sheet:
        th, tw = 360, 640
        resized = [cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA) for f in frames_for_sheet]
        while len(resized) < 6:
            resized.append(np.zeros((th, tw, 3), dtype=np.uint8))

        row1 = np.hstack([resized[0], resized[1], resized[2]])
        row2 = np.hstack([resized[3], resized[4], resized[5]])
        contact_sheet = np.vstack([row1, row2])
        cv2.imwrite(str(out_sheet_path), contact_sheet)
        print(f"Saved contact sheet: {out_sheet_path}")

    print(f"Summary: Processed {total_frames} frames, {child_alerts_fired} child safety alerts fired.")


if __name__ == "__main__":
    generate_demo()

