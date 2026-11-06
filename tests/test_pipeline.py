"""Integration tests for the unified SafeKid AI pipeline."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from safekid_ai.age_estimator import AgeEstimationStatus
from safekid_ai.config import (
    AgeEstimatorConfig,
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    SafetyZoneType,
    TrackerConfig,
    VideoInputConfig,
)
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.video_input import VideoInput


@pytest.fixture(scope="module")
def pipeline():
    """Initializes SafeKidPipeline in operational community monitoring mode."""
    cfg = SafeKidPipelineConfig(
        mode=OperationalMode.COMMUNITY_MONITORING,
        community_name="Palm Grove Gated Community",
        zone_type=SafetyZoneType.SWIMMING_POOL_GATE,
        detector=DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu"),
        tracker=TrackerConfig(tracker_config_path="bytetrack.yaml"),
        age_estimator=AgeEstimatorConfig(device="cpu", half=False),
    )
    return SafeKidPipeline(cfg)


def test_pipeline_operational_mode_processes_frame(pipeline):
    """Verifies that operational mode processes frames and preserves UNKNOWN for age."""
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    res = pipeline.process_frame(frame, frame_id=1, timestamp_ms=33.3, annotate=True)

    assert res.frame_id == 1
    assert res.community_name == "Palm Grove Gated Community"
    assert res.zone_type == SafetyZoneType.SWIMMING_POOL_GATE
    assert res.operational_mode == OperationalMode.COMMUNITY_MONITORING
    assert res.annotated_frame is not None
    assert res.annotated_frame.shape == (360, 480, 3)


def test_pipeline_video_processing(pipeline):
    """Tests processing a sequence of frames from VideoInput."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        vid_path = f.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(vid_path, fourcc, 10.0, (160, 120))
    for i in range(4):
        f = np.full((120, 160, 3), i * 30, dtype=np.uint8)
        out.write(f)
    out.release()

    try:
        with VideoInput(source=vid_path) as vin:
            frame_results = list(pipeline.process_video(vin, max_frames=3))
            assert len(frame_results) == 3
            for idx, fr in enumerate(frame_results):
                assert fr.frame_id == idx
    finally:
        if Path(vid_path).exists():
            Path(vid_path).unlink()
