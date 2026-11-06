"""Modular dataset ingestion, annotation mapping, deduplication, and leakage-free partitioning."""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.dataset_ingestion")


class DatasetSource(str, Enum):
    """Source provenance of child detection datasets."""
    RAP_V2 = "RAP_V2"
    PETA = "PETA"
    AUTHORIZED_RESIDENTIAL = "AUTHORIZED_RESIDENTIAL"
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"


@dataclass
class IngestionValidationRecord:
    """Audit summary of an ingestion run."""
    source: DatasetSource
    total_raw_candidates: int = 0
    valid_samples: int = 0
    corrupted_images: int = 0
    invalid_bboxes: int = 0
    near_duplicates_removed: int = 0
    splits_distribution: Dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    class_distribution: Dict[str, int] = field(default_factory=lambda: {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0})
    manifest_path: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for perceptual similarity comparison."""
    if image is None or image.size == 0:
        return 0
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(h1: int, h2: int) -> int:
    """Calculates bitwise Hamming distance between two hashes."""
    return bin(h1 ^ h2).count("1")


class ChildDatasetIngestionEngine:
    """Modular ingestion engine for real-world and benchmark child detection data.
    
    Invariants:
    1. Preserves original dataset annotations and provenance metadata.
    2. Maps raw attributes to operational classes (CHILD, ADULT, UNKNOWN) only when semantically justified.
    3. Eliminates near-duplicate crops using perceptual hashing.
    4. Partitions train/val/test splits strictly by camera/subject groups to prevent split leakage.
    5. Isolates real-world samples under data/child_dataset/real_world/, never overwriting fixtures.
    """

    def __init__(self, target_root: Optional[Path] = None):
        self.target_root = target_root or Path("data/child_dataset/real_world")
        self.target_root.mkdir(parents=True, exist_ok=True)

    def map_attribute_to_operational_class(
        self,
        raw_attributes: Dict[str, Any],
        source: DatasetSource,
    ) -> Tuple[ChildDetectionLabel, float, str]:
        """Maps dataset-specific attributes to SafeKid AI operational classes with uncertainty tracking.
        
        Returns:
            Tuple of (operational_label, confidence, semantic_rationale).
        """
        if source == DatasetSource.RAP_V2:
            # RAP v2 attribute #1: AgeLess16
            is_less_16 = raw_attributes.get("AgeLess16", 0) == 1
            is_adult_31_45 = raw_attributes.get("Age31_45", 0) == 1
            is_adult_46_60 = raw_attributes.get("Age46_60", 0) == 1
            is_teen = raw_attributes.get("Age17_30", 0) == 1

            if is_less_16:
                # Semantic Gap: AgeLess16 includes 13-15yo teenagers.
                # Must abstain to UNKNOWN to prevent false child safety alarms.
                return (
                    ChildDetectionLabel.UNKNOWN,
                    0.50,
                    "RAP v2 AgeLess16 bundles 13-15yo teens; mapped conservatively to UNKNOWN.",
                )
            elif is_adult_31_45 or is_adult_46_60:
                return (ChildDetectionLabel.ADULT, 0.95, "Verified mature adult supervisor in RAP v2.")
            else:
                return (ChildDetectionLabel.UNKNOWN, 0.40, "Ambiguous adult/teen age bracket in RAP v2.")

        elif source == DatasetSource.PETA:
            is_less_15 = raw_attributes.get("personalLess15", 0) == 1
            is_adult_30_45 = raw_attributes.get("Age30_45", 0) == 1
            is_adult_45_60 = raw_attributes.get("Age45_60", 0) == 1

            if is_less_15:
                return (
                    ChildDetectionLabel.UNKNOWN,
                    0.55,
                    "PETA personalLess15 bundles 13-14yo teenagers; mapped to UNKNOWN.",
                )
            elif is_adult_30_45 or is_adult_45_60:
                return (ChildDetectionLabel.ADULT, 0.95, "Verified mature adult in PETA.")
            else:
                return (ChildDetectionLabel.UNKNOWN, 0.40, "Ambiguous age bracket in PETA.")

        elif source == DatasetSource.AUTHORIZED_RESIDENTIAL:
            # Exact verified age provided with parental consent
            exact_age = raw_attributes.get("verified_age")
            consent_ok = raw_attributes.get("parental_consent_verified", False)
            if not consent_ok or exact_age is None:
                return (ChildDetectionLabel.UNKNOWN, 0.0, "Missing parental consent or verified age.")
            
            if exact_age < 13.0:
                return (ChildDetectionLabel.CHILD, 1.0, f"Verified enrolled child (Age {exact_age}).")
            elif exact_age >= 18.0:
                return (ChildDetectionLabel.ADULT, 1.0, f"Verified adult resident (Age {exact_age}).")
            else:
                return (ChildDetectionLabel.UNKNOWN, 0.60, f"Independent teenager (Age {exact_age}).")

        else:
            return (ChildDetectionLabel.UNKNOWN, 0.0, "Unknown source provenance.")

    def ingest_candidate_records(
        self,
        raw_candidates: List[Dict[str, Any]],
        source: DatasetSource,
        subfolder_name: str = "ingested",
    ) -> IngestionValidationRecord:
        """Ingests, validates, deduplicates, and partitions candidate pedestrian records.
        
        Args:
            raw_candidates: List of candidate metadata records.
            source: DatasetSource enum.
            subfolder_name: Subfolder under real_world/ to store outputs.
            
        Returns:
            IngestionValidationRecord summarizing ingestion results.
        """
        record = IngestionValidationRecord(source=source, total_raw_candidates=len(raw_candidates))
        out_dir = self.target_root / subfolder_name
        out_crops_dir = out_dir / "crops"
        out_crops_dir.mkdir(parents=True, exist_ok=True)

        accepted_samples: List[Dict[str, Any]] = []
        seen_hashes: Dict[int, str] = {}

        for cand in raw_candidates:
            img_path = Path(cand.get("image_path", ""))
            if not img_path.exists():
                record.corrupted_images += 1
                record.warnings.append(f"Missing image file: {img_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                record.corrupted_images += 1
                record.warnings.append(f"Corrupted unreadable image: {img_path}")
                continue

            fh, fw = img.shape[:2]
            bbox = cand.get("bbox_xyxy", [0, 0, fw, fh])
            x1, y1, x2, y2 = bbox

            # Validate bounding box geometry
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
                record.invalid_bboxes += 1
                record.warnings.append(f"Invalid bbox coordinates {bbox} for image {img_path.name}")
                continue

            # Extract crop
            crop = img[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]
            if crop.size == 0:
                record.invalid_bboxes += 1
                continue

            # Check near-duplicate collision
            dh = compute_dhash(crop)
            is_dup = False
            for prev_h, prev_name in seen_hashes.items():
                if hamming_distance(dh, prev_h) <= 2:
                    is_dup = True
                    record.near_duplicates_removed += 1
                    break
            if is_dup:
                continue

            seen_hashes[dh] = img_path.name

            # Map attributes
            raw_attrs = cand.get("raw_attributes", {})
            op_label, conf, reason = self.map_attribute_to_operational_class(raw_attrs, source)

            # Save clean crop
            crop_filename = f"{source.value.lower()}_{cand.get('sample_id', len(accepted_samples)):06d}.png"
            crop_path = out_crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            accepted_samples.append({
                "sample_id": cand.get("sample_id", len(accepted_samples)),
                "source": source.value,
                "crop_filename": crop_filename,
                "crop_path": str(crop_path).replace("\\", "/"),
                "original_image_path": str(img_path).replace("\\", "/"),
                "bbox_xyxy": [round(float(c), 1) for c in bbox],
                "operational_label": op_label.value,
                "confidence": conf,
                "semantic_rationale": reason,
                "group_id": cand.get("group_id", cand.get("camera_id", cand.get("subject_id", "default_group"))),
                "camera_id": cand.get("camera_id", "unknown_cam"),
                "height": int(crop.shape[0]),
                "width": int(crop.shape[1]),
            })

        # Leakage-Free Group Partitioning (Train: 70%, Val: 15%, Test: 15%)
        # All samples from the same group_id are assigned to the same split!
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for s in accepted_samples:
            gid = str(s["group_id"])
            groups.setdefault(gid, []).append(s)

        sorted_groups = sorted(groups.keys())
        total_accepted = len(accepted_samples)
        train_target = int(total_accepted * 0.70)
        val_target = int(total_accepted * 0.15)

        train_count, val_count, test_count = 0, 0, 0
        for gid in sorted_groups:
            group_samples = groups[gid]
            g_size = len(group_samples)
            if train_count + g_size <= train_target or (train_count == 0 and g_size > 0):
                split_name = "train"
                train_count += g_size
            elif val_count + g_size <= val_target or (val_count == 0 and g_size > 0):
                split_name = "val"
                val_count += g_size
            else:
                split_name = "test"
                test_count += g_size

            for s in group_samples:
                s["split"] = split_name
                record.splits_distribution[split_name] += 1
                record.class_distribution[s["operational_label"]] += 1

        record.valid_samples = total_accepted

        # Save manifest
        manifest_path = out_dir / "manifest.json"
        manifest_payload = {
            "source": source.value,
            "ingestion_engine_version": "1.0",
            "total_accepted_samples": total_accepted,
            "splits": record.splits_distribution,
            "classes": record.class_distribution,
            "near_duplicates_removed": record.near_duplicates_removed,
            "samples": accepted_samples,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        record.manifest_path = str(manifest_path).replace("\\", "/")
        logger.info(
            f"Ingested {total_accepted} samples from {source.value} into {out_dir}. "
            f"Splits: {record.splits_distribution}, Manifest: {manifest_path.name}"
        )
        return record
