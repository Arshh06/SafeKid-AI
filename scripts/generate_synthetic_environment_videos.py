"""SafeKid AI — Synthetic Environment Video Generator.

Generates deterministic synthetic demonstration videos for:
1. Scenario A — Playground (camera_playground_01.json, 1280x720, 24 FPS, 10s = 240 frames)
2. Scenario B — Residential Driveway (camera_gate_north.json, 1920x1080, 24 FPS, 8s = 192 frames)
3. Scenario C — Boom Barrier (camera_gate_north.json, 1920x1080, 24 FPS, 6s = 144 frames)

Reuses real person detection crops to ensure bona fide YOLO person detection and
deterministic ByteTrack tracking across all frames.
"""

from pathlib import Path
import cv2
import numpy as np


def paste_person_crop(
    canvas: np.ndarray,
    crop: np.ndarray,
    bottom_center_x: float,
    bottom_center_y: float,
) -> None:
    """Pastes person crop onto canvas centered horizontally at bottom_center_x and bottom_center_y."""
    ch, cw = crop.shape[:2]
    h, w = canvas.shape[:2]

    x1 = int(round(bottom_center_x - cw / 2.0))
    y1 = int(round(bottom_center_y - ch))
    x2 = x1 + cw
    y2 = y1 + ch

    # Ensure bounds
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w, x2)
        cy2 = min(h, y2)
        crop_part = crop[cy1 - y1 : cy1 - y1 + (cy2 - cy1), cx1 - x1 : cx1 - x1 + (cx2 - cx1)]
        canvas[cy1:cy2, cx1:cx2] = crop_part
    else:
        canvas[y1:y2, x1:x2] = crop


def draw_playground_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """Renders realistic playground scene matching camera_playground_01 geometry."""
    bg = np.full((height, width, 3), (75, 155, 75), dtype=np.uint8)  # Grass lawn

    # 1. Toddler Sandbox polygon [[150, 180], [680, 190], [650, 580], [120, 560]]
    sandbox_poly = np.array([[150, 180], [680, 190], [650, 580], [120, 560]], dtype=np.int32)
    cv2.fillPoly(bg, [sandbox_poly], (165, 215, 235))
    cv2.polylines(bg, [sandbox_poly], True, (60, 100, 160), 6)

    # 2. Perimeter Hedge & Fence polygon [[750, 60], [1240, 60], [1250, 260], [740, 240]]
    hedge_poly = np.array([[750, 60], [1240, 60], [1250, 260], [740, 240]], dtype=np.int32)
    cv2.fillPoly(bg, [hedge_poly], (30, 85, 30))
    cv2.polylines(bg, [hedge_poly], True, (20, 60, 20), 4)

    # Wire fence along north boundary
    cv2.line(bg, (750, 60), (1240, 60), (180, 180, 180), 3)
    for fx in range(750, 1240, 40):
        cv2.line(bg, (fx, 50), (fx, 75), (120, 120, 120), 2)

    # Decorative bench near sandbox for adult supervisor
    cv2.rectangle(bg, (210, 360), (280, 375), (50, 90, 140), -1)
    cv2.rectangle(bg, (215, 375), (225, 395), (40, 40, 40), -1)
    cv2.rectangle(bg, (265, 375), (275, 395), (40, 40, 40), -1)

    # Swing frame inside sandbox
    cv2.line(bg, (480, 210), (510, 320), (50, 50, 200), 5)
    cv2.line(bg, (540, 210), (510, 320), (50, 50, 200), 5)
    cv2.line(bg, (600, 210), (630, 320), (50, 50, 200), 5)
    cv2.line(bg, (660, 210), (630, 320), (50, 50, 200), 5)
    cv2.line(bg, (510, 320), (630, 320), (50, 50, 200), 6)

    # Walkway paving stones
    for py in range(250, 550, 60):
        cv2.circle(bg, (710, py), 18, (140, 150, 160), -1)

    return bg


def draw_gate_background(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Renders realistic vehicular gate & driveway scene matching camera_gate_north geometry."""
    bg = np.full((height, width, 3), (75, 155, 75), dtype=np.uint8)  # Grass lawn

    # 1. Main Vehicular Driveway [[500, 400], [1350, 410], [1500, 1050], [380, 1050]]
    driveway_poly = np.array([[500, 400], [1350, 410], [1500, 1050], [380, 1050]], dtype=np.int32)
    cv2.fillPoly(bg, [driveway_poly], (115, 120, 125))  # Weathered paving roadway
    cv2.polylines(bg, [driveway_poly], True, (90, 95, 100), 2)

    # Road center line dashed
    cv2.line(bg, (925, 405), (940, 1050), (220, 220, 220), 4)

    # 2. Pedestrian Sidewalk [[80, 450], [420, 450], [360, 1050], [50, 1050]]
    sidewalk_poly = np.array([[80, 450], [420, 450], [360, 1050], [50, 1050]], dtype=np.int32)
    cv2.fillPoly(bg, [sidewalk_poly], (175, 180, 185))  # Light concrete pavement
    cv2.polylines(bg, [sidewalk_poly], True, (140, 145, 150), 2)

    # Pedestrian crossing connecting sidewalk to roadway
    crossing_poly = np.array([[300, 780], [650, 780], [650, 920], [300, 920]], dtype=np.int32)
    cv2.fillPoly(bg, [crossing_poly], (175, 180, 185))

    # Bollards separating upper sidewalk from driveway
    for by in range(500, 750, 70):
        bx = int(420 + (360 - 420) * (by - 450) / 600.0)
        cv2.circle(bg, (bx, by), 8, (40, 40, 210), -1)

    # 3. Boom Barrier Mechanical Zone [[720, 480], [1080, 480], [1100, 620], [700, 620]]
    barrier_poly = np.array([[720, 480], [1080, 480], [1100, 620], [700, 620]], dtype=np.int32)
    cv2.fillPoly(bg, [barrier_poly], (140, 170, 190))
    cv2.polylines(bg, [barrier_poly], True, (0, 200, 240), 3)

    # Barrier arm
    cv2.line(bg, (730, 540), (1070, 540), (240, 240, 240), 8)
    for sx in range(750, 1070, 40):
        cv2.line(bg, (sx, 536), (sx + 20, 544), (30, 30, 220), 8)

    # Guard booth (compact, placed on sidewalk without obstructing adult at (220, 600) or (220, 750))
    cv2.rectangle(bg, (100, 460), (200, 530), (190, 190, 200), -1)
    cv2.rectangle(bg, (100, 460), (200, 530), (80, 80, 90), 2)
    cv2.rectangle(bg, (120, 480), (180, 510), (220, 220, 160), -1)

    return bg


def generate_playground_video(
    output_path: Path,
    crop: np.ndarray,
    fps: float = 24.0,
    total_frames: int = 240,
) -> None:
    """Generates Scenario A — Playground video (1280x720, 24 FPS, 10 seconds)."""
    width, height = 1280, 720
    bg_template = draw_playground_background(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Track 2 (Adult): Stationary inside sandbox area near supervisor bench at (250, 350)
    adult_x = 250.0
    adult_y = 350.0

    # Track 1 (Child):
    # - Frames 0..35: In sandbox at (450, 380) [0.0s to 1.5s]
    # - Frames 36..71: Walks from (450, 380) to (950, 160) in perimeter hedge [1.5s to 3.0s]
    # - Frames 72..191: Stationary inside perimeter hedge at (950, 160) [3.0s to 8.0s] (dwell = 5.0s > 4.0s)
    # - Frames 192..239: Walks away toward (600, 350) [8.0s to 10.0s]
    start_pos = (450.0, 380.0)
    hedge_pos = (950.0, 160.0)
    exit_pos = (600.0, 350.0)

    for f in range(total_frames):
        frame = bg_template.copy()

        # Adult position
        paste_person_crop(frame, crop, adult_x, adult_y)

        # Child position
        if f < 36:
            cx, cy = start_pos
        elif f < 72:
            alpha = (f - 36) / 36.0
            cx = start_pos[0] + alpha * (hedge_pos[0] - start_pos[0])
            cy = start_pos[1] + alpha * (hedge_pos[1] - start_pos[1])
        elif f < 192:
            cx, cy = hedge_pos
        else:
            alpha = (f - 192) / 48.0
            cx = hedge_pos[0] + alpha * (exit_pos[0] - hedge_pos[0])
            cy = hedge_pos[1] + alpha * (exit_pos[1] - hedge_pos[1])

        paste_person_crop(frame, crop, cx, cy)
        writer.write(frame)

    writer.release()
    print(f"[Generated] {output_path} ({total_frames} frames, {width}x{height} @ {fps} FPS)")


def generate_driveway_video(
    output_path: Path,
    crop: np.ndarray,
    fps: float = 24.0,
    total_frames: int = 192,
) -> None:
    """Generates Scenario B — Residential Driveway video (1920x1080, 24 FPS, 8 seconds)."""
    width, height = 1920, 1080
    bg_template = draw_gate_background(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Track 1 (Adult): Standing on sidewalk at (220, 600)
    adult_x = 220.0
    adult_y = 600.0

    # Track 2 (Child):
    # - Frames 0..23: On sidewalk at (320, 850) [0.0s to 1.0s]
    # - Frames 24..47: Steps into driveway at (600, 850) [1.0s to 2.0s]
    # - Frames 48..119: Dwells in driveway at (600, 850) [2.0s to 5.0s] (dwell = 3.0s > 2.0s)
    # - Frames 120..167: Steps smoothly back out to sidewalk at (320, 850) [5.0s to 7.0s]
    # - Frames 168..191: Remains on sidewalk at (320, 850) [7.0s to 8.0s]
    sidewalk_pos = (320.0, 850.0)
    driveway_pos = (600.0, 850.0)

    for f in range(total_frames):
        frame = bg_template.copy()

        # Adult
        paste_person_crop(frame, crop, adult_x, adult_y)

        # Child
        if f < 24:
            cx = sidewalk_pos[0]
        elif f < 48:
            alpha = (f - 24) / 24.0
            cx = sidewalk_pos[0] + alpha * (driveway_pos[0] - sidewalk_pos[0])
        elif f < 120:
            cx = driveway_pos[0]
        elif f < 168:
            alpha = (f - 120) / 48.0
            cx = driveway_pos[0] + alpha * (sidewalk_pos[0] - driveway_pos[0])
        else:
            cx = sidewalk_pos[0]

        paste_person_crop(frame, crop, cx, sidewalk_pos[1])
        writer.write(frame)

    writer.release()
    print(f"[Generated] {output_path} ({total_frames} frames, {width}x{height} @ {fps} FPS)")


def generate_boom_barrier_video(
    output_path: Path,
    crop: np.ndarray,
    fps: float = 24.0,
    total_frames: int = 144,
) -> None:
    """Generates Scenario C — Boom Barrier video (1920x1080, 24 FPS, 6 seconds)."""
    width, height = 1920, 1080
    bg_template = draw_gate_background(width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Track 1 (Adult): Standing on sidewalk at (220, 750)
    adult_x = 220.0
    adult_y = 750.0

    # Track 2 (Child):
    # - Frames 0..11: Outside gate at north (900, 350) [0.0s to 0.5s] (outside all zones: y < 400)
    # - Frames 12..35: Walks into boom barrier mechanical zone at (900, 550) [0.5s to 1.5s]
    # - Frames 36..83: Stationary in boom barrier zone at (900, 550) [1.5s to 3.5s] (dwell = 2.0s, dwell_alert=False)
    # - Frames 84..131: Walks out of boom barrier zone to (650, 550) [3.5s to 5.5s] (exit_alert=False)
    # - Frames 132..143: Stationary at (650, 550) [5.5s to 6.0s]
    north_pos = (900.0, 350.0)
    barrier_pos = (900.0, 550.0)
    exit_pos = (650.0, 550.0)

    for f in range(total_frames):
        frame = bg_template.copy()

        # Adult
        paste_person_crop(frame, crop, adult_x, adult_y)

        # Child
        if f < 12:
            cx, cy = north_pos
        elif f < 36:
            alpha = (f - 12) / 24.0
            cx = north_pos[0] + alpha * (barrier_pos[0] - north_pos[0])
            cy = north_pos[1] + alpha * (barrier_pos[1] - north_pos[1])
        elif f < 84:
            cx, cy = barrier_pos
        elif f < 132:
            alpha = (f - 84) / 48.0
            cx = barrier_pos[0] + alpha * (exit_pos[0] - barrier_pos[0])
            cy = barrier_pos[1] + alpha * (exit_pos[1] - barrier_pos[1])
        else:
            cx, cy = exit_pos

        paste_person_crop(frame, crop, cx, cy)
        writer.write(frame)

    writer.release()
    print(f"[Generated] {output_path} ({total_frames} frames, {width}x{height} @ {fps} FPS)")


def main():
    root = Path(__file__).resolve().parent.parent
    demo_dir = root / "data" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    crop_path = root / "outputs" / "track1_crop.png"
    if not crop_path.exists():
        raise FileNotFoundError(f"Person crop not found at {crop_path}")

    crop = cv2.imread(str(crop_path))
    if crop is None:
        raise ValueError(f"Failed to read crop image from {crop_path}")

    print("Generating synthetic environment demo videos...")
    generate_playground_video(demo_dir / "playground_demo.mp4", crop)
    generate_driveway_video(demo_dir / "driveway_demo.mp4", crop)
    generate_boom_barrier_video(demo_dir / "boom_barrier_demo.mp4", crop)
    print("All synthetic videos generated successfully.")


if __name__ == "__main__":
    main()
