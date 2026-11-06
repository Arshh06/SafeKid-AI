"""Generates a reproducible, structured child/adult pedestrian reference fixture dataset for SafeKid AI."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def render_pedestrian_crop(
    is_child: bool,
    height: int,
    width: int,
    head_ratio: float,
    clothing_color: Tuple[int, int, int],
    skin_tone: Tuple[int, int, int],
    occlusion_level: float = 0.0,
    background_tone: int = 120,
) -> np.ndarray:
    """Renders a synthetic pedestrian figure with specific anatomical proportions."""
    canvas = np.full((height, width, 3), background_tone, dtype=np.uint8)
    
    # Add subtle asphalt/ground background gradient
    gradient = np.linspace(-15, 15, height).reshape(height, 1, 1).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + gradient, 0, 255).astype(np.uint8)

    cx = width // 2
    head_radius = max(3, int(round(height * head_ratio * 0.45)))
    head_cy = max(head_radius + 2, int(round(height * head_ratio * 0.7)))

    # 1. Torso
    torso_top = head_cy + head_radius
    torso_bot = int(round(height * (0.60 if is_child else 0.55)))
    torso_w = max(4, int(round(width * (0.50 if is_child else 0.42))))
    tx1, tx2 = max(0, cx - torso_w // 2), min(width - 1, cx + torso_w // 2)
    cv2.rectangle(canvas, (tx1, torso_top), (tx2, torso_bot), clothing_color, -1)

    # 2. Legs / Lower Body
    leg_w = max(2, torso_w // 3)
    leg_bot = height - 2
    # Left leg
    cv2.rectangle(canvas, (tx1, torso_bot), (tx1 + leg_w, leg_bot), (40, 45, 55), -1)
    # Right leg
    cv2.rectangle(canvas, (tx2 - leg_w, torso_bot), (tx2, leg_bot), (40, 45, 55), -1)

    # 3. Head
    cv2.circle(canvas, (cx, head_cy), head_radius, skin_tone, -1)

    # 4. Hair / Cap
    hair_color = (25, 20, 15)
    cv2.ellipse(canvas, (cx, head_cy - 1), (head_radius, head_radius // 2), 0, 180, 360, hair_color, -1)

    # 5. Apply occlusion if specified (e.g. car bumper, bench, foliage)
    if occlusion_level > 0.0:
        occ_h = int(round(height * occlusion_level))
        occ_y1 = height - occ_h
        cv2.rectangle(canvas, (0, occ_y1), (width, height), (35, 35, 40), -1)
        cv2.line(canvas, (0, occ_y1), (width, occ_y1), (70, 75, 80), 1)

    return canvas


def build_reference_dataset() -> None:
    """Builds 200 diverse pedestrian crops with metadata and train/val/test splits."""
    root_dir = Path("data/child_dataset/fixtures")
    img_dir = root_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    np.random.seed(42)

    # Palette of clothing and skin tones
    clothing_palettes = [
        (220, 60, 40),   # Red
        (40, 180, 220),  # Yellow/Gold
        (50, 140, 220),  # Orange
        (200, 120, 40),  # Sky blue
        (80, 190, 80),   # Green
        (160, 80, 180),  # Purple
        (45, 45, 50),    # Dark charcoal
        (180, 180, 180), # Grey
        (50, 80, 140),   # Navy/Brown
    ]
    skin_tones = [
        (160, 190, 230), # Light
        (130, 160, 210), # Medium light
        (90, 130, 180),  # Tan
        (60, 90, 130),   # Brown
        (40, 60, 90),    # Deep
    ]

    sample_id = 0

    # Group 1: 80 CHILD samples (diverse heights 60-120px, large head ratios 0.22-0.28)
    for i in range(80):
        sample_id += 1
        h = int(np.random.randint(65, 125))
        w = int(np.random.randint(24, 48))
        head_r = float(np.random.uniform(0.22, 0.28))
        c_color = clothing_palettes[i % len(clothing_palettes)]
        s_tone = skin_tones[i % len(skin_tones)]
        occ = 0.0 if i < 65 else float(np.random.uniform(0.10, 0.25)) # Slight occlusion on some

        img = render_pedestrian_crop(
            is_child=True,
            height=h,
            width=w,
            head_ratio=head_r,
            clothing_color=c_color,
            skin_tone=s_tone,
            occlusion_level=occ,
            background_tone=int(np.random.randint(100, 140)),
        )

        filename = f"crop_child_{sample_id:04d}.png"
        filepath = img_dir / filename
        cv2.imwrite(str(filepath), img)

        split = "train" if i < 56 else ("val" if i < 68 else "test")
        samples.append({
            "sample_id": sample_id,
            "filename": filename,
            "filepath": str(filepath).replace("\\", "/"),
            "ground_truth": "CHILD",
            "split": split,
            "height": h,
            "width": w,
            "head_to_body_ratio": round(head_r, 3),
            "occlusion_ratio": round(occ, 2),
            "distance_tier": "CLOSE" if h > 95 else "MEDIUM",
        })

    # Group 2: 80 ADULT samples (diverse heights 130-220px, small head ratios 0.12-0.16)
    for i in range(80):
        sample_id += 1
        h = int(np.random.randint(130, 220))
        w = int(np.random.randint(35, 65))
        head_r = float(np.random.uniform(0.12, 0.16))
        c_color = clothing_palettes[(i + 3) % len(clothing_palettes)]
        s_tone = skin_tones[(i + 2) % len(skin_tones)]
        occ = 0.0 if i < 65 else float(np.random.uniform(0.10, 0.25))

        img = render_pedestrian_crop(
            is_child=False,
            height=h,
            width=w,
            head_ratio=head_r,
            clothing_color=c_color,
            skin_tone=s_tone,
            occlusion_level=occ,
            background_tone=int(np.random.randint(100, 140)),
        )

        filename = f"crop_adult_{sample_id:04d}.png"
        filepath = img_dir / filename
        cv2.imwrite(str(filepath), img)

        split = "train" if i < 56 else ("val" if i < 68 else "test")
        samples.append({
            "sample_id": sample_id,
            "filename": filename,
            "filepath": str(filepath).replace("\\", "/"),
            "ground_truth": "ADULT",
            "split": split,
            "height": h,
            "width": w,
            "head_to_body_ratio": round(head_r, 3),
            "occlusion_ratio": round(occ, 2),
            "distance_tier": "CLOSE" if h > 165 else "MEDIUM",
        })

    # Group 3: 40 AMBIGUOUS / UNCERTAIN samples designed to trigger UNKNOWN abstention
    # - 15 Low resolution / distant (height 20-32px < 36px threshold)
    # - 15 Heavy occlusion (occlusion ratio > 0.50)
    # - 10 Borderline teen / intermediate stature (head ratio ~ 0.18)
    for i in range(40):
        sample_id += 1
        if i < 15:
            # Low resolution distant
            h = int(np.random.randint(22, 33))
            w = max(8, int(h * 0.40))
            head_r = float(np.random.uniform(0.15, 0.22))
            occ = 0.0
            category = "DISTANT_PEDESTRIAN"
        elif i < 30:
            # Heavy occlusion
            h = int(np.random.randint(90, 150))
            w = int(h * 0.45)
            head_r = float(np.random.uniform(0.15, 0.22))
            occ = float(np.random.uniform(0.55, 0.75))
            category = "SEVERE_OCCLUSION"
        else:
            # Borderline teen
            h = int(np.random.randint(110, 135))
            w = int(h * 0.38)
            head_r = 0.18
            occ = 0.0
            category = "BORDERLINE_TEEN"

        is_ch = (i % 2 == 0)
        c_color = clothing_palettes[i % len(clothing_palettes)]
        s_tone = skin_tones[i % len(skin_tones)]

        img = render_pedestrian_crop(
            is_child=is_ch,
            height=h,
            width=w,
            head_ratio=head_r,
            clothing_color=c_color,
            skin_tone=s_tone,
            occlusion_level=occ,
            background_tone=120,
        )

        filename = f"crop_ambiguous_{sample_id:04d}.png"
        filepath = img_dir / filename
        cv2.imwrite(str(filepath), img)

        # Stratified split for ambiguous sub-categories
        if category == "DISTANT_PEDESTRIAN":
            sub_idx = i
            split = "train" if sub_idx < 10 else ("val" if sub_idx < 12 else "test")
        elif category == "SEVERE_OCCLUSION":
            sub_idx = i - 15
            split = "train" if sub_idx < 10 else ("val" if sub_idx < 12 else "test")
        else:
            sub_idx = i - 30
            split = "train" if sub_idx < 6 else ("val" if sub_idx < 8 else "test")
        samples.append({
            "sample_id": sample_id,
            "filename": filename,
            "filepath": str(filepath).replace("\\", "/"),
            "ground_truth": "AMBIGUOUS",
            "ambiguous_category": category,
            "split": split,
            "height": h,
            "width": w,
            "head_to_body_ratio": round(head_r, 3),
            "occlusion_ratio": round(occ, 2),
            "distance_tier": "DISTANT" if h < 36 else "MEDIUM",
        })

    # Save annotations
    ann_path = root_dir / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": "SafeKid AI Child/Adult Reference Fixture",
            "version": "1.0",
            "total_samples": len(samples),
            "distribution": {
                "CHILD": sum(1 for s in samples if s["ground_truth"] == "CHILD"),
                "ADULT": sum(1 for s in samples if s["ground_truth"] == "ADULT"),
                "AMBIGUOUS": sum(1 for s in samples if s["ground_truth"] == "AMBIGUOUS"),
            },
            "splits": {
                "train": sum(1 for s in samples if s["split"] == "train"),
                "val": sum(1 for s in samples if s["split"] == "val"),
                "test": sum(1 for s in samples if s["split"] == "test"),
            },
            "samples": samples,
        }, f, indent=2)

    print(f"Generated {len(samples)} fixture crops in: {img_dir}")
    print(f"Saved annotations to: {ann_path}")

    # Generate outputs/child_detection_dataset_inventory.json
    out_inv_path = Path("outputs/child_detection_dataset_inventory.json")
    out_inv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_inv_path, "w", encoding="utf-8") as f:
        json.dump({
            "inventory_timestamp": "2026-11-05T10:30:00",
            "genuine_academic_datasets": [
                {
                    "dataset_name": "RAP v2 (Richly Annotated Pedestrian)",
                    "acquisition_status": "IDENTIFIED_ACQUISITION_GUIDE_PROVIDED",
                    "license": "Academic Non-Commercial EULA",
                    "relevant_attribute": "AgeLess16 (Binary attribute 1)",
                    "samples_total": 84928,
                    "location": "data/child_dataset/genuine/rap_v2/"
                },
                {
                    "dataset_name": "PETA (PEdesTrian Attribute)",
                    "acquisition_status": "IDENTIFIED_ACQUISITION_GUIDE_PROVIDED",
                    "license": "Academic Non-Commercial (CUHK)",
                    "relevant_attribute": "personalLess15 (Binary attribute)",
                    "samples_total": 19000,
                    "location": "data/child_dataset/genuine/peta/"
                }
            ],
            "authorized_residential_datasets": [
                {
                    "deployment_type": "Gated Residential Community (COPPA/GDPR Opt-in)",
                    "consent_policy": "Explicit Parental Consent Required",
                    "status": "SPECIFICATION_DEFINED",
                    "location": "data/child_dataset/authorized_residential/"
                }
            ],
            "reference_benchmark_fixtures": [
                {
                    "fixture_name": "SafeKid AI Child/Adult Reference Fixture",
                    "status": "LOCAL_FIXTURE_GENERATED",
                    "total_samples": len(samples),
                    "classes": {"CHILD": 80, "ADULT": 80, "AMBIGUOUS": 40},
                    "splits": {"train": 140, "val": 30, "test": 30},
                    "annotations_path": str(ann_path).replace("\\", "/")
                }
            ]
        }, f, indent=2)
    print(f"Generated dataset inventory in: {out_inv_path}")


if __name__ == "__main__":
    build_reference_dataset()

