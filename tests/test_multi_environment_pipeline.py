"""End-to-End Multi-Environment Video Simulation Integration Tests.

Validates the full SafeKid AI pipeline on deterministic synthetic environment clips:
1. Scenario A — Playground (camera_playground_01.json, 1280x720)
2. Scenario B — Residential Driveway (camera_gate_north.json, 1920x1080)
3. Scenario C — Boom Barrier (camera_gate_north.json, 1920x1080)

Verifies:
- Complete pipeline execution: VideoInput -> PersonDetector -> ByteTrack -> ROIEngine -> ZoneStateManager -> ChildSafetyEventEngine
- Zone policy compliance (SAFE vs DANGER roles, entry alert suppression, dwell timing)
- Strict adult exclusion (Adult tracks never generate child alerts)
- Correct severity levels (LOW, HIGH, CRITICAL)
- Exit alert suppression when disabled
"""

from pathlib import Path
import pytest

from safekid_ai.config import (
    DetectorConfig,
    OperationalMode,
    SafeKidPipelineConfig,
    TrackerConfig,
)
from safekid_ai.child_safety import ChildSafetyEventType
from safekid_ai.events import AlertSeverity
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.video_input import VideoInput

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def project_root():
    return PROJECT_ROOT


def test_playground_simulation_pipeline(project_root):
    """Scenario A: Playground — Verifies safe zone entry suppression, danger zone entry suppression, and dwell alert emission."""
    video_path = project_root / "data" / "demo" / "playground_demo.mp4"
    config_path = project_root / "configs" / "cameras" / "community_002" / "camera_playground_01.json"
    annotation_path = project_root / "configs" / "annotations" / "demo_playground_track_annotations.json"

    assert video_path.exists(), f"Playground demo video missing: {video_path}"
    assert config_path.exists(), f"Camera config missing: {config_path}"
    assert annotation_path.exists(), f"Annotation missing: {annotation_path}"

    pipeline_cfg = SafeKidPipelineConfig(
        community_name="Palm Grove Villa Enclave",
        community_id="community_002",
        camera_id="camera_playground_01",
        mode=OperationalMode.COMMUNITY_MONITORING,
        detector=DetectorConfig(model_name_or_path="yolov8n.pt", confidence_threshold=0.35),
        tracker=TrackerConfig(tracker_type="bytetrack", persist=True),
        roi_config_path=str(config_path),
        entry_confirmation_frames=3,
        exit_confirmation_frames=3,
        child_annotation_path=str(annotation_path),
        enable_child_safety_layer=True,
    )

    pipeline = SafeKidPipeline(pipeline_cfg)
    video_input = VideoInput(source=str(video_path))

    frame_count = 0
    all_child_events = []
    all_raw_events = []

    for result in pipeline.process_video(video_input, annotate=False):
        frame_count += 1
        all_child_events.extend(result.child_events)
        all_raw_events.extend(result.events)

    assert frame_count == 240, f"Expected 240 frames, got {frame_count}"

    # Child safety engine history
    emitted = list(pipeline.child_safety_engine.child_events_history)

    # 1. Adult (Track 2) must produce zero child alerts
    adult_alerts = [e for e in emitted if e.track_id == 2]
    assert len(adult_alerts) == 0, f"Adult generated child alerts: {adult_alerts}"

    # 2. Child (Track 1) must generate exactly 1 alert: dwell on perimeter_hedge_boundary
    child_alerts = [e for e in emitted if e.track_id == 1]
    assert len(child_alerts) == 1, f"Expected 1 child alert, got {len(child_alerts)}: {child_alerts}"

    dwell_alert = child_alerts[0]
    assert dwell_alert.event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED
    assert dwell_alert.roi_id == "perimeter_hedge_boundary"
    assert dwell_alert.severity == AlertSeverity.HIGH
    assert dwell_alert.dwell_time_seconds >= 4.0
    assert "remained in danger zone" in dwell_alert.message.lower()


def test_driveway_simulation_pipeline(project_root):
    """Scenario B: Driveway — Verifies immediate HIGH entry alert and 2.0s dwell alert on vehicle thoroughfare."""
    video_path = project_root / "data" / "demo" / "driveway_demo.mp4"
    config_path = project_root / "configs" / "cameras" / "community_001" / "camera_gate_north.json"
    annotation_path = project_root / "configs" / "annotations" / "demo_driveway_track_annotations.json"

    assert video_path.exists(), f"Driveway demo video missing: {video_path}"
    assert config_path.exists(), f"Camera config missing: {config_path}"
    assert annotation_path.exists(), f"Annotation missing: {annotation_path}"

    pipeline_cfg = SafeKidPipelineConfig(
        community_name="Greenfield Residential Enclave",
        community_id="community_001",
        camera_id="camera_gate_north",
        mode=OperationalMode.COMMUNITY_MONITORING,
        detector=DetectorConfig(model_name_or_path="yolov8n.pt", confidence_threshold=0.35),
        tracker=TrackerConfig(tracker_type="bytetrack", persist=True),
        roi_config_path=str(config_path),
        entry_confirmation_frames=3,
        exit_confirmation_frames=3,
        child_annotation_path=str(annotation_path),
        enable_child_safety_layer=True,
    )

    pipeline = SafeKidPipeline(pipeline_cfg)
    video_input = VideoInput(source=str(video_path))

    frame_count = 0
    for result in pipeline.process_video(video_input, annotate=False):
        frame_count += 1

    assert frame_count == 192, f"Expected 192 frames, got {frame_count}"

    emitted = list(pipeline.child_safety_engine.child_events_history)

    # 1. Adult (Track 1) must produce zero child alerts
    adult_alerts = [e for e in emitted if e.track_id == 1]
    assert len(adult_alerts) == 0, f"Adult generated child alerts: {adult_alerts}"

    # 2. Child (Track 2) must generate entry and dwell alerts on vehicle_driveway_lane
    child_alerts = [e for e in emitted if e.track_id == 2]
    assert len(child_alerts) == 2, f"Expected 2 child alerts, got {len(child_alerts)}: {child_alerts}"

    entry_alert = [e for e in child_alerts if e.event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY][0]
    assert entry_alert.roi_id == "vehicle_driveway_lane"
    assert entry_alert.severity == AlertSeverity.HIGH
    assert "entered configured danger zone" in entry_alert.message.lower()

    dwell_alert = [e for e in child_alerts if e.event_type == ChildSafetyEventType.CHILD_DWELL_TIME_EXCEEDED][0]
    assert dwell_alert.roi_id == "vehicle_driveway_lane"
    assert dwell_alert.severity == AlertSeverity.HIGH
    assert dwell_alert.dwell_time_seconds >= 2.0


def test_boom_barrier_simulation_pipeline(project_root):
    """Scenario C: Boom Barrier — Verifies immediate CRITICAL alert and suppression of exit alert."""
    video_path = project_root / "data" / "demo" / "boom_barrier_demo.mp4"
    config_path = project_root / "configs" / "cameras" / "community_001" / "camera_gate_north.json"
    annotation_path = project_root / "configs" / "annotations" / "demo_boom_barrier_track_annotations.json"

    assert video_path.exists(), f"Boom barrier demo video missing: {video_path}"
    assert config_path.exists(), f"Camera config missing: {config_path}"
    assert annotation_path.exists(), f"Annotation missing: {annotation_path}"

    pipeline_cfg = SafeKidPipelineConfig(
        community_name="Greenfield Residential Enclave",
        community_id="community_001",
        camera_id="camera_gate_north",
        mode=OperationalMode.COMMUNITY_MONITORING,
        detector=DetectorConfig(model_name_or_path="yolov8n.pt", confidence_threshold=0.35),
        tracker=TrackerConfig(tracker_type="bytetrack", persist=True),
        roi_config_path=str(config_path),
        entry_confirmation_frames=3,
        exit_confirmation_frames=3,
        child_annotation_path=str(annotation_path),
        enable_child_safety_layer=True,
    )

    pipeline = SafeKidPipeline(pipeline_cfg)
    video_input = VideoInput(source=str(video_path))

    frame_count = 0
    for result in pipeline.process_video(video_input, annotate=False):
        frame_count += 1

    assert frame_count == 144, f"Expected 144 frames, got {frame_count}"

    emitted = list(pipeline.child_safety_engine.child_events_history)

    # 1. Adult (Track 1) must produce zero child alerts
    adult_alerts = [e for e in emitted if e.track_id == 1]
    assert len(adult_alerts) == 0, f"Adult generated child alerts: {adult_alerts}"

    # 2. Child (Track 2) enters boom_barrier_danger_zone (CRITICAL)
    barrier_alerts = [e for e in emitted if e.roi_id == "boom_barrier_danger_zone"]
    assert len(barrier_alerts) == 1, f"Expected 1 barrier alert, got {len(barrier_alerts)}: {barrier_alerts}"

    barrier_entry = barrier_alerts[0]
    assert barrier_entry.event_type == ChildSafetyEventType.CHILD_ZONE_ENTRY
    assert barrier_entry.severity == AlertSeverity.CRITICAL

    # 3. Verify no exit alert emitted for boom barrier
    exit_alerts = [e for e in emitted if e.event_type == ChildSafetyEventType.CHILD_ZONE_EXIT]
    assert len(exit_alerts) == 0, f"Expected 0 exit alerts, got {exit_alerts}"


def test_unclassified_tracks_produce_no_child_alerts(project_root):
    """Verifies that in the absence of explicit child annotations, no child alerts are generated."""
    video_path = project_root / "data" / "demo" / "playground_demo.mp4"
    config_path = project_root / "configs" / "cameras" / "community_002" / "camera_playground_01.json"

    pipeline_cfg = SafeKidPipelineConfig(
        community_name="Palm Grove Villa Enclave",
        community_id="community_002",
        camera_id="camera_playground_01",
        mode=OperationalMode.COMMUNITY_MONITORING,
        detector=DetectorConfig(model_name_or_path="yolov8n.pt", confidence_threshold=0.35),
        tracker=TrackerConfig(tracker_type="bytetrack", persist=True),
        roi_config_path=str(config_path),
        entry_confirmation_frames=3,
        exit_confirmation_frames=3,
        child_annotation_path=None,  # No annotations -> all UNCLASSIFIED
        enable_child_safety_layer=True,
    )

    pipeline = SafeKidPipeline(pipeline_cfg)
    video_input = VideoInput(source=str(video_path))

    for result in pipeline.process_video(video_input, max_frames=60, annotate=False):
        pass

    emitted = list(pipeline.child_safety_engine.child_events_history)
    assert len(emitted) == 0, f"Unclassified tracks produced child alerts: {emitted}"

