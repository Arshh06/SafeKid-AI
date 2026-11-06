"""Demo execution script for SafeKid AI pipeline in a gated residential community."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from safekid_ai import (
    DetectorConfig,
    OperationalMode,
    SafeKidPipeline,
    SafeKidPipelineConfig,
    SafetyZoneType,
    TrackerConfig,
    VideoInput,
    VideoInputConfig,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.demo")


def generate_synthetic_community_video(output_path: Path, num_frames: int = 30) -> Path:
    """Generates a synthetic test video simulating a community swimming pool gate."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 15.0, (640, 480))

    logger.info(f"Generating synthetic community test video: {output_path} ({num_frames} frames)...")
    for i in range(num_frames):
        # Create background
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        # Draw pool zone marker (red caution area)
        cv2.rectangle(frame, (400, 50), (620, 430), (0, 0, 120), 2)
        cv2.putText(
            frame,
            "SWIMMING POOL ZONE",
            (410, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1,
        )

        # Simulate a person walking toward the pool gate
        px = int(150 + i * 10)
        py = 220
        # Draw torso and head
        cv2.rectangle(frame, (px - 25, py - 40), (px + 25, py + 80), (180, 180, 180), -1)
        cv2.circle(frame, (px, py - 60), 22, (200, 200, 200), -1)

        writer.write(frame)

    writer.release()
    logger.info("Synthetic video generated successfully.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="SafeKid AI Gated Community Pipeline Demo")
    parser.add_argument("--source", type=str, default=None, help="Path to video/image file or camera index")
    parser.add_argument("--roi-config", type=str, default="configs/cameras/community_001/camera_pool_01.json", help="Path to camera-specific ROI JSON configuration")
    parser.add_argument("--zone", type=str, default="swimming_pool_gate", choices=[z.value for z in SafetyZoneType])
    parser.add_argument("--community", type=str, default="Greenfield Enclave")
    parser.add_argument("--camera-id", type=str, default="camera_pool_01", help="Camera identifier")
    parser.add_argument("--max-frames", type=int, default=20, help="Max frames to process")
    parser.add_argument("--output-video", type=str, default="runs/safekid/demo_output.mp4")
    args = parser.parse_args()

    # If no source provided, generate synthetic community test video
    if args.source is None:
        source_path = Path("data/sample_community_test.mp4")
        generate_synthetic_community_video(source_path, num_frames=args.max_frames)
        source_str = str(source_path)
    else:
        source_str = args.source

    config = SafeKidPipelineConfig(
        mode=OperationalMode.COMMUNITY_MONITORING,
        community_name=args.community,
        camera_id=args.camera_id,
        roi_config_path=args.roi_config if Path(args.roi_config).exists() else None,
        zone_type=SafetyZoneType(args.zone),
        video=VideoInputConfig(source=source_str),
        detector=DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu", confidence_threshold=0.25),
        tracker=TrackerConfig(tracker_config_path="bytetrack.yaml"),
    )

    pipeline = SafeKidPipeline(config)
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running SafeKid AI pipeline on source: {source_str}")
    writer = None

    with VideoInput(config=config.video) as vin:
        meta = vin.get_metadata()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, meta.fps, (meta.width, meta.height))

        count = 0
        for frame_result in pipeline.process_video(vin, max_frames=args.max_frames, annotate=True):
            if frame_result.annotated_frame is not None and writer is not None:
                writer.write(frame_result.annotated_frame)
            count += 1
            logger.info(
                f"Frame {frame_result.frame_id:03d} processed: {len(frame_result.person_states)} person tracks. "
                f"Safety events: {len(frame_result.events)} "
                f"([{[e.event_type.value for e in frame_result.events]}])"
            )

        if writer:
            writer.release()

    logger.info(f"Demo complete! Processed {count} frames. Output saved to: {output_path}")


if __name__ == "__main__":
    main()

