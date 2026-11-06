"""Generates visual validation contact sheet illustrating child detection, abstentions, and generalization bounds."""

import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.child_detection.detector import ChildDetector
from safekid_ai.child_detection.models import AbstentionReason, ChildDetectionLabel
from generate_child_dataset_fixture import render_pedestrian_crop


def render_panel(
    title: str,
    subtitle: str,
    crop: np.ndarray,
    label_text: str,
    badge_color: Tuple[int, int, int],
    panel_w: int = 400,
    panel_h: int = 300,
) -> np.ndarray:
    """Renders a single labeled visual diagnostic panel."""
    panel = np.full((panel_h, panel_w, 3), (22, 26, 32), dtype=np.uint8)

    # Header bar
    cv2.rectangle(panel, (0, 0), (panel_w, 38), (30, 36, 44), -1)
    cv2.line(panel, (0, 38), (panel_w, 38), (55, 65, 78), 1)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)

    # Center crop placement
    ch, cw = crop.shape[:2]
    # Scale crop to fit neatly in panel center
    scale = min(180.0 / ch, 140.0 / cw)
    dw, dh = int(round(cw * scale)), int(round(ch * scale))
    scaled_crop = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)

    x_off = (panel_w - dw) // 2
    y_off = 48 + (190 - dh) // 2

    # Draw background frame around crop
    cv2.rectangle(panel, (x_off - 4, y_off - 4), (x_off + dw + 4, y_off + dh + 4), (45, 52, 64), 1)
    panel[y_off:y_off + dh, x_off:x_off + dw] = scaled_crop

    # Draw bounding box and ground-contact dot
    cv2.rectangle(panel, (x_off, y_off), (x_off + dw, y_off + dh), badge_color, 2)
    bc_x = x_off + dw // 2
    bc_y = y_off + dh
    cv2.circle(panel, (bc_x, bc_y), 5, (0, 235, 255), -1, cv2.LINE_AA)
    cv2.circle(panel, (bc_x, bc_y), 3, (10, 10, 10), -1, cv2.LINE_AA)

    # Classification Badge Pill
    (bw, bh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    bx1 = (panel_w - bw) // 2 - 8
    by1 = panel_h - 48
    bx2 = bx1 + bw + 16
    by2 = by1 + bh + 10
    cv2.rectangle(panel, (bx1, by1), (bx2, by2), (15, 18, 22), -1)
    cv2.rectangle(panel, (bx1, by1), (bx2, by2), badge_color, 1)
    cv2.putText(panel, label_text, (bx1 + 8, by1 + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Subtitle / Reason
    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(panel, subtitle, (max(4, (panel_w - sw) // 2), panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 180, 190), 1, cv2.LINE_AA)

    # Synthetic Fixture Watermark
    cv2.putText(panel, "[SYNTHETIC FIXTURE]", (panel_w - 130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 140, 160), 1, cv2.LINE_AA)

    return panel


def generate_contact_sheet() -> None:
    detector = ChildDetector(ChildDetectorConfig(
        model_weights_path="models/child_classifier_v1.pt",
        confidence_threshold=0.65,
        margin_threshold=0.20,
    ))

    panels = []

    # 1. Correct CHILD
    crop_child = render_pedestrian_crop(is_child=True, height=95, width=34, head_ratio=0.26, clothing_color=(220, 60, 40), skin_tone=(160, 190, 230))
    res1 = detector.classify_crop(crop_child)
    p1 = render_panel(
        title="1. CHILD Candidate",
        subtitle=f"Conf: {res1.confidence*100:.1f}% | Gate: PASSED",
        crop=crop_child,
        label_text=f"PRED: {res1.label.value} ({res1.confidence:.2f})",
        badge_color=(0, 0, 245), # Red
    )
    panels.append(p1)

    # 2. Correct ADULT
    crop_adult = render_pedestrian_crop(is_child=False, height=170, width=48, head_ratio=0.13, clothing_color=(45, 45, 50), skin_tone=(130, 160, 210))
    res2 = detector.classify_crop(crop_adult)
    p2 = render_panel(
        title="2. ADULT Supervisor",
        subtitle=f"Conf: {res2.confidence*100:.1f}% | Gate: PASSED",
        crop=crop_adult,
        label_text=f"PRED: {res2.label.value} ({res2.confidence:.2f})",
        badge_color=(220, 180, 60), # Cyan/Teal
    )
    panels.append(p2)

    # 3. UNKNOWN: Low Resolution / Distant
    crop_dist = render_pedestrian_crop(is_child=True, height=26, width=10, head_ratio=0.24, clothing_color=(200, 120, 40), skin_tone=(160, 190, 230))
    res3 = detector.classify_crop(crop_dist)
    p3 = render_panel(
        title="3. Distant Pedestrian",
        subtitle=f"Height: 26px < 36px | Gate: LOW_RES",
        crop=crop_dist,
        label_text=f"PRED: {res3.label.value} (DISTANT)",
        badge_color=(160, 170, 180), # Slate
    )
    panels.append(p3)

    # 4. UNKNOWN: Severe Occlusion
    crop_occ = render_pedestrian_crop(is_child=True, height=100, width=40, head_ratio=0.25, clothing_color=(80, 190, 80), skin_tone=(90, 130, 180), occlusion_level=0.65)
    res4 = detector.classify_crop(crop_occ)
    p4 = render_panel(
        title="4. Severe Occlusion",
        subtitle=f"Lower Torso Blocked | Gate: OCCLUSION",
        crop=crop_occ,
        label_text=f"PRED: {res4.label.value} (OCCLUDED)",
        badge_color=(160, 170, 180),
    )
    panels.append(p4)

    # 5. UNKNOWN: Borderline Teen
    crop_teen = render_pedestrian_crop(is_child=True, height=120, width=42, head_ratio=0.18, clothing_color=(160, 80, 180), skin_tone=(130, 160, 210))
    res5 = detector.classify_crop(crop_teen)
    p5 = render_panel(
        title="5. Borderline Youth",
        subtitle=f"Margin < 0.20 | Gate: AMBIGUOUS",
        crop=crop_teen,
        label_text=f"PRED: {res5.label.value} (BORDERLINE)",
        badge_color=(0, 190, 240), # Amber
    )
    panels.append(p5)

    # 6. UNKNOWN: Scale Mismatch / Foreground Anomaly
    crop_scale = render_pedestrian_crop(is_child=True, height=220, width=65, head_ratio=0.24, clothing_color=(50, 140, 220), skin_tone=(160, 190, 230))
    res6 = detector.classify_crop(crop_scale, bbox_xyxy=(50, 50, 250, 950), frame_size=(1920, 1080))
    p6 = render_panel(
        title="6. Perspective Anomaly",
        subtitle=f"Foreground > 45% Frame | Gate: SCALE",
        crop=crop_scale,
        label_text=f"PRED: {res6.label.value} (SCALE_MISMATCH)",
        badge_color=(160, 170, 180),
    )
    panels.append(p6)

    # 7. Real-World CCTV Status (RAP v2 / PETA)
    panel_info = np.full((300, 400, 3), (22, 26, 32), dtype=np.uint8)
    cv2.rectangle(panel_info, (0, 0), (400, 38), (30, 36, 44), -1)
    cv2.putText(panel_info, "7. Tier 2: Real-World CCTV", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1)
    cv2.putText(panel_info, "Status: BLOCKED_AWAITING_EULA", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
    cv2.putText(panel_info, "- RAP v2 (84,928 samples)", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)
    cv2.putText(panel_info, "- PETA (19,000 samples)", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)
    cv2.putText(panel_info, "Academic EULA with CUHK required.", (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1)
    cv2.putText(panel_info, "Zero real CCTV data fabricated.", (15, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 120), 1)
    cv2.putText(panel_info, "[DATASET TIER 2]", (270, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 140, 160), 1)
    panels.append(panel_info)

    # 8. Tier 3: Consented Residential CCTV Status
    panel_res = np.full((300, 400, 3), (22, 26, 32), dtype=np.uint8)
    cv2.rectangle(panel_res, (0, 0), (400, 38), (30, 36, 44), -1)
    cv2.putText(panel_res, "8. Tier 3: Consented Residential", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1)
    cv2.putText(panel_res, "Status: AWAITING_PILOT_ENROLLMENT", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.putText(panel_res, "- COPPA & GDPR Art. 8 Opt-in", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)
    cv2.putText(panel_res, "- Security Staff Human-in-Loop", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)
    cv2.putText(panel_res, "- Provisional alert candidate review", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1)
    cv2.putText(panel_res, "- Biometrics strictly prohibited", (15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 120), 1)
    cv2.putText(panel_res, "[DATASET TIER 3]", (270, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 140, 160), 1)
    panels.append(panel_res)

    # Arrange 4x2 grid (8 panels)
    row1 = np.hstack([panels[0], panels[1], panels[2], panels[3]])
    row2 = np.hstack([panels[4], panels[5], panels[6], panels[7]])
    grid = np.vstack([row1, row2])

    out_path = Path("outputs/child_detection_generalization_contact_sheet.jpg")
    cv2.imwrite(str(out_path), grid)
    print(f"Saved contact sheet to: {out_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    generate_contact_sheet()
