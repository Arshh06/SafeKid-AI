"""Creates a verified, standard-compliant VIRAT reference sample sequence and annotations.

Used for offline testing, CI/CD regression, and baseline validation when full multi-gigabyte
VIRAT archives have not yet been downloaded by the user.
Adheres strictly to the official VIRAT Release 2.0 schema:
- 1920x1080 @ 24 FPS MP4 video
- Exact 8-column .viratdata.objects.txt
- Exact .viratdata.events.txt and .viratdata.mapping.txt
"""

import os
from pathlib import Path
import cv2
import numpy as np

def create_sample_fixture():
    root = Path(__file__).resolve().parent.parent
    videos_dir = root / "data" / "virat" / "videos"
    ann_dir = root / "data" / "virat" / "annotations"
    cfg_dir = root / "configs" / "cameras" / "community_virat"

    videos_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    seq_id = "VIRAT_S_000001_sample"
    video_path = videos_dir / f"{seq_id}.mp4"
    objects_path = ann_dir / f"{seq_id}.viratdata.objects.txt"
    events_path = ann_dir / f"{seq_id}.viratdata.events.txt"
    mapping_path = ann_dir / f"{seq_id}.viratdata.mapping.txt"
    roi_config_path = cfg_dir / "camera_virat_s01.json"

    width, height = 1920, 1080
    fps = 24.0
    num_frames = 72

    print(f"Creating VIRAT reference sample sequence: {seq_id} ({width}x{height} @ {fps} FPS, {num_frames} frames)...")

    # 1. Generate Video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")

    # Track 1: Person 1 (walking horizontally across sidewalk/driveway)
    # Track 2: Car (stationary vehicle parked on left)
    # Track 3: Person 2 (walking diagonally towards facility door)
    obj_records = []

    for fid in range(num_frames):
        # Background: outdoor facility / parking environment
        frame = np.full((height, width, 3), (120, 125, 125), dtype=np.uint8)
        # Asphalt roadway
        cv2.rectangle(frame, (0, 420), (width, height), (55, 60, 65), -1)
        # Concrete walkway / facility frontage
        cv2.rectangle(frame, (0, 200), (width, 420), (145, 150, 152), -1)
        # Building wall
        cv2.rectangle(frame, (0, 0), (width, 200), (90, 95, 105), -1)
        # Road markings
        for mx in range(100, width, 250):
            cv2.rectangle(frame, (mx, 700), (mx + 120, 715), (220, 220, 225), -1)

        # Object 2: Stationary Car (parked at x=200, y=500, w=340, h=180)
        car_x, car_y, car_w, car_h = 200, 520, 340, 180
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (40, 40, 160), -1)
        cv2.rectangle(frame, (car_x + 30, car_y + 20), (car_x + car_w - 30, car_y + 90), (160, 190, 200), -1)
        # Record car in objects.txt (object_type = 2)
        obj_records.append(f"2 {num_frames} {fid} {car_x} {car_y} {car_w} {car_h} 2")

        # Object 1: Person 1 (walking across from x=450 to x=850, y=360, w=60, h=170)
        p1_x = int(450 + (850 - 450) * (fid / num_frames))
        p1_y = 360
        p1_w, p1_h = 65, 175
        # Draw realistic human figure for YOLO detection
        # Head
        cv2.circle(frame, (p1_x + p1_w // 2, p1_y + 25), 18, (190, 170, 150), -1)
        # Torso
        cv2.rectangle(frame, (p1_x + 10, p1_y + 45), (p1_x + p1_w - 10, p1_y + 115), (140, 50, 40), -1)
        # Legs
        leg_offset = 6 if (fid // 4) % 2 == 0 else -6
        cv2.line(frame, (p1_x + 20, p1_y + 115), (p1_x + 16 + leg_offset, p1_y + p1_h), (30, 30, 40), 8)
        cv2.line(frame, (p1_x + 45, p1_y + 115), (p1_x + 49 - leg_offset, p1_y + p1_h), (30, 30, 40), 8)
        # Record Person 1 (object_type = 1)
        obj_records.append(f"1 {num_frames} {fid} {p1_x} {p1_y} {p1_w} {p1_h} 1")

        # Object 3: Person 2 (walking from x=1250, y=480 towards entrance x=1350, y=310)
        p2_x = int(1250 + (1350 - 1250) * (fid / num_frames))
        p2_y = int(480 + (310 - 480) * (fid / num_frames))
        p2_w, p2_h = 60, 165
        # Head
        cv2.circle(frame, (p2_x + p2_w // 2, p2_y + 22), 16, (180, 160, 140), -1)
        # Torso
        cv2.rectangle(frame, (p2_x + 10, p2_y + 40), (p2_x + p2_w - 10, p2_y + 105), (35, 110, 60), -1)
        # Legs
        leg_offset2 = 5 if (fid // 3) % 2 == 0 else -5
        cv2.line(frame, (p2_x + 18, p2_y + 105), (p2_x + 15 + leg_offset2, p2_y + p2_h), (25, 25, 30), 7)
        cv2.line(frame, (p2_x + 42, p2_y + 105), (p2_x + 45 - leg_offset2, p2_y + p2_h), (25, 25, 30), 7)
        # Record Person 2 (object_type = 1)
        obj_records.append(f"3 {num_frames} {fid} {p2_x} {p2_y} {p2_w} {p2_h} 1")

        writer.write(frame)

    writer.release()
    print(f"  [Created Video] {video_path}")

    # 2. Write .viratdata.objects.txt
    with open(objects_path, "w", encoding="utf-8") as f:
        f.write("# VIRAT Release 2.0 Objects Ground Truth\n")
        f.write("# object_id duration current_frame x y width height object_type\n")
        f.write("\n".join(obj_records) + "\n")
    print(f"  [Created Objects Annotation] {objects_path} ({len(obj_records)} entries)")

    # 3. Write .viratdata.events.txt
    with open(events_path, "w", encoding="utf-8") as f:
        f.write("# VIRAT Release 2.0 Events Ground Truth\n")
        f.write("# event_id event_type duration start_frame end_frame current_frame x y width height\n")
        f.write(f"1 1 {num_frames} 0 {num_frames - 1} 0 450 360 465 175\n")
        f.write(f"2 1 {num_frames} 0 {num_frames - 1} 0 1250 310 160 335\n")
    print(f"  [Created Events Annotation] {events_path}")

    # 4. Write .viratdata.mapping.txt
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("# VIRAT Release 2.0 Event-Object Mapping\n")
        f.write("1 1 1\n")
        f.write("2 1 3\n")
    print(f"  [Created Mapping Annotation] {mapping_path}")

    # 5. Write experimental camera ROI configuration
    import json
    camera_roi = {
        "camera_id": "camera_virat_s01",
        "community_id": "community_virat",
        "frame_width": 1920,
        "frame_height": 1080,
        "zones": [
            {
                "roi_id": "virat_monitored_walkway",
                "name": "Surveillance Walkway & Loading Bay",
                "zone_type": "general_common_area",
                "severity": "LOW",
                "enabled": True,
                "is_normalized": False,
                "policy": {
                    "zone_role": "MONITORED",
                    "entry_alert": False,
                    "exit_alert": False,
                    "dwell_alert": True,
                    "dwell_threshold_seconds": 3.0,
                    "severity": "LOW"
                },
                "polygon": [
                    [400.0, 250.0],
                    [1000.0, 250.0],
                    [1000.0, 600.0],
                    [400.0, 600.0]
                ]
            },
            {
                "roi_id": "virat_restricted_driveway",
                "name": "Vehicular Transit Lane",
                "zone_type": "VEHICULAR_GATEWAY",
                "severity": "HIGH",
                "enabled": True,
                "policy": {
                    "zone_role": "DANGER",
                    "entry_alert": True,
                    "exit_alert": False,
                    "dwell_alert": True,
                    "dwell_threshold_seconds": 2.0,
                    "severity": "HIGH"
                },
                "polygon": [
                    [1150.0, 300.0],
                    [1600.0, 300.0],
                    [1600.0, 750.0],
                    [1150.0, 750.0]
                ]
            }
        ]
    }
    with open(roi_config_path, "w", encoding="utf-8") as f:
        json.dump(camera_roi, f, indent=2)
    print(f"  [Created Experimental Camera ROI] {roi_config_path}")
    print("VIRAT sample fixture setup complete.\n")

if __name__ == "__main__":
    create_sample_fixture()
