"""SafeKid AI — Visual Validation & Representative Frame Extraction Script.

Extracts milestone frames from annotated scenario videos:
1. Playground (outputs/playground_annotated.mp4)
2. Driveway (outputs/driveway_annotated.mp4)
3. Boom Barrier (outputs/boom_barrier_annotated.mp4)
4. Swimming Pool (outputs/pool_safety_annotated.mp4)

Generates:
- Individual representative frames in outputs/visual_contact_sheets/<scenario>/
- Composite 4-panel visual contact sheets for each scenario
- outputs/visual_validation_report.md
"""

from pathlib import Path
import cv2
import numpy as np


def extract_frames_for_scenario(
    video_path: Path,
    frame_indices: list,
    output_dir: Path,
    scenario_title: str,
) -> list:
    """Extracts specified frame indices from video and saves individual images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    extracted_images = []
    for idx, label in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Warning: could not read frame {idx} from {video_path.name}")
            continue

        out_name = f"frame_{idx:03d}_{label}.png"
        out_file = output_dir / out_name
        cv2.imwrite(str(out_file), frame)
        extracted_images.append((idx, label, frame, out_file))
        print(f"  Extracted: {out_file.name} ({frame.shape[1]}x{frame.shape[0]})")

    cap.release()
    return extracted_images


def create_contact_sheet(
    extracted_images: list,
    output_path: Path,
    scenario_title: str,
    target_width: int = 1600,
) -> None:
    """Arranges 4 extracted frames into a 2x2 composite contact sheet with header labels."""
    if len(extracted_images) < 4:
        print(f"Warning: Not enough frames ({len(extracted_images)}) to create 2x2 contact sheet for {scenario_title}")
        return

    # Resize panels to half of target width (preserving aspect ratio)
    panel_w = target_width // 2
    panels = []

    for idx, label, img, _ in extracted_images[:4]:
        ih, iw = img.shape[:2]
        panel_h = int(round(ih * (panel_w / float(iw))))
        resized = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        # Add caption banner at bottom of each panel
        banner_h = 36
        banner = np.full((banner_h, panel_w, 3), (25, 25, 30), dtype=np.uint8)
        caption = f"Frame {idx:03d} | {label.replace('_', ' ').title()}"
        cv2.putText(
            banner,
            caption,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        combined_panel = np.vstack([resized, banner])
        panels.append(combined_panel)

    # Combine 2x2
    top_row = np.hstack([panels[0], panels[1]])
    bottom_row = np.hstack([panels[2], panels[3]])
    grid = np.vstack([top_row, bottom_row])

    # Add header banner across the top
    header_h = 50
    header = np.full((header_h, grid.shape[1], 3), (15, 20, 25), dtype=np.uint8)
    title_text = f"SafeKid AI — Visual Validation Contact Sheet: {scenario_title}"
    cv2.putText(
        header,
        title_text,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    full_sheet = np.vstack([header, grid])

    cv2.imwrite(str(output_path), full_sheet)
    print(f"[Contact Sheet Created] {output_path} ({full_sheet.shape[1]}x{full_sheet.shape[0]})")


def main():
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "outputs" / "visual_contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {
            "name": "Playground",
            "scenario_id": "playground",
            "video_path": root / "outputs" / "playground_annotated.mp4",
            "frames": [
                (15, "sandbox_safe_play"),
                (70, "perimeter_approach"),
                (140, "perimeter_dwell_alert"),
                (220, "return_to_sandbox"),
            ],
            "title": "Scenario A — Playground (camera_playground_01)",
        },
        {
            "name": "Driveway",
            "scenario_id": "driveway",
            "video_path": root / "outputs" / "driveway_annotated.mp4",
            "frames": [
                (15, "sidewalk_safe_presence"),
                (40, "driveway_entry_alert"),
                (80, "driveway_dwell_alert"),
                (180, "sidewalk_return_exit"),
            ],
            "title": "Scenario B — Residential Driveway (camera_gate_north)",
        },
        {
            "name": "Boom Barrier",
            "scenario_id": "boom_barrier",
            "video_path": root / "outputs" / "boom_barrier_annotated.mp4",
            "frames": [
                (8, "north_approach"),
                (25, "driveway_entry_alert"),
                (45, "boom_barrier_critical_alert"),
                (120, "barrier_exit_to_driveway"),
            ],
            "title": "Scenario C — Boom Barrier (camera_gate_north)",
        },
        {
            "name": "Swimming Pool",
            "scenario_id": "pool",
            "video_path": root / "outputs" / "pool_safety_annotated.mp4",
            "frames": [
                (20, "poolside_monitoring"),
                (60, "water_basin_critical_entry"),
                (140, "deck_dwell_alert"),
                (210, "climb_out_recovery"),
            ],
            "title": "Scenario D — Swimming Pool (camera_pool_demo)",
        },
    ]

    for sc in scenarios:
        print(f"\nProcessing visual validation for: {sc['name']}")
        sc_dir = out_dir / sc["scenario_id"]
        extracted = extract_frames_for_scenario(
            sc["video_path"],
            sc["frames"],
            sc_dir,
            sc["title"],
        )
        sheet_path = out_dir / f"{sc['scenario_id']}_contact_sheet.jpg"
        create_contact_sheet(extracted, sheet_path, sc["title"])

    print("\nAll visual contact sheets generated successfully.")


if __name__ == "__main__":
    main()

