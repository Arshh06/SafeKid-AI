"""Dataset validator and audit suite for genuine and fixture VIRAT surveillance sequences."""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from safekid_ai.datasets.virat_annotations import VIRATAnnotationParser, VIRATObjectType
from safekid_ai.datasets.virat_loader import VIRATDatasetLoader, VIRATSequenceMetadata
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.datasets.virat_validator")


@dataclass
class SequenceValidationResult:
    """Rigorous audit report for an individual video-annotation sequence pair."""
    sequence_id: str
    is_valid: bool
    is_genuine: bool
    tier: str  # "genuine" or "fixture"
    video_path: Optional[str] = None
    objects_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    video_properties: Dict[str, Any] = field(default_factory=dict)
    annotation_summary: Dict[str, Any] = field(default_factory=dict)
    synchronization_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "is_valid": self.is_valid,
            "is_genuine": self.is_genuine,
            "tier": self.tier,
            "video_path": self.video_path,
            "objects_path": self.objects_path,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "video_properties": self.video_properties,
            "annotation_summary": self.annotation_summary,
            "synchronization_summary": self.synchronization_summary,
        }


class VIRATDatasetValidator:
    """Validates real and fixture surveillance sequences against VIRAT Release 2.0 standards."""

    def __init__(self, loader: Optional[VIRATDatasetLoader] = None):
        self.loader = loader or VIRATDatasetLoader()

    def validate_sequence(self, meta: VIRATSequenceMetadata) -> SequenceValidationResult:
        """Runs a complete integrity audit on a single sequence."""
        errors: List[str] = []
        warnings: List[str] = []
        video_props: Dict[str, Any] = {}
        ann_props: Dict[str, Any] = {}
        sync_props: Dict[str, Any] = {}

        is_genuine = getattr(meta, "is_genuine", False)
        tier = getattr(meta, "tier", "fixture")

        # 1. Video Integrity Checks
        if not meta.has_video or meta.video_path is None or not meta.video_path.is_file():
            errors.append(f"Video file missing or unreadable: {meta.video_path}")
        else:
            try:
                cap = cv2.VideoCapture(str(meta.video_path))
                if not cap.isOpened():
                    errors.append(f"Failed to open video container: {meta.video_path.name}")
                else:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

                    if w <= 0 or h <= 0:
                        errors.append(f"Invalid frame dimensions: {w}x{h}")
                    if fps <= 0.0 or fps > 120.0:
                        warnings.append(f"Unusual FPS value: {fps}")
                    if frame_count <= 0:
                        errors.append(f"Video contains 0 frames: {frame_count}")

                    # Test decodability at candidate sample positions (start, 25%, 50%, 75%, end)
                    corrupted_frames: List[int] = []
                    sample_positions = [
                        0,
                        frame_count // 4,
                        frame_count // 2,
                        (3 * frame_count) // 4,
                        max(0, frame_count - 1),
                    ]
                    for pos in sample_positions:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                        ret, frame = cap.read()
                        if not ret or frame is None or frame.size == 0:
                            corrupted_frames.append(pos)
                        elif frame.shape[0] != h or frame.shape[1] != w:
                            errors.append(f"Decoded frame shape mismatch at frame {pos}: {frame.shape} vs expected ({h}, {w})")

                    if corrupted_frames:
                        errors.append(f"Frame decoding failure at indices: {corrupted_frames}")

                    video_props = {
                        "width": w,
                        "height": h,
                        "fps": round(fps, 2),
                        "total_frames": frame_count,
                        "codec": codec.strip(),
                        "file_size_bytes": meta.video_path.stat().st_size,
                        "corrupted_sample_frames": corrupted_frames,
                    }
                cap.release()
            except Exception as e:
                errors.append(f"Video inspection encountered exception: {e}")

        # 2. Annotation Integrity & Coordinate Checks
        if not meta.has_annotations or meta.objects_path is None or not meta.objects_path.is_file():
            errors.append(f"Objects annotation file missing or unreadable: {meta.objects_path}")
        else:
            try:
                seen_pairs: Set[Tuple[int, int]] = set()
                duplicate_records = 0
                invalid_bboxes: List[Dict[str, Any]] = []
                out_of_bounds_bboxes: List[Dict[str, Any]] = []
                otype_counts: Dict[str, int] = defaultdict(int)
                unique_tracks: Set[int] = set()
                all_frames: Set[int] = set()
                total_lines = 0
                comment_lines = 0
                malformed_lines = 0

                ref_w = video_props.get("width", meta.width)
                ref_h = video_props.get("height", meta.height)

                with open(meta.objects_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_idx, line in enumerate(f, start=1):
                        raw = line.strip()
                        if not raw:
                            continue
                        if raw.startswith("#"):
                            comment_lines += 1
                            continue
                        total_lines += 1
                        tokens = raw.split()
                        if len(tokens) < 6:
                            malformed_lines += 1
                            continue

                        # Release 2.0: <obj_id> <duration> <curr_frame> <x> <y> <w> <h> <obj_type>
                        if len(tokens) >= 8:
                            oid = int(tokens[0])
                            fid = int(tokens[2])
                            x, y, w, h = float(tokens[3]), float(tokens[4]), float(tokens[5]), float(tokens[6])
                            tid = int(tokens[7])
                        else:
                            oid = int(tokens[0])
                            fid = int(tokens[1])
                            x, y, w, h = float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5])
                            tid = int(tokens[6]) if len(tokens) >= 7 else 1

                        unique_tracks.add(oid)
                        all_frames.add(fid)

                        # Category distribution
                        try:
                            type_name = VIRATObjectType(tid).name
                        except ValueError:
                            type_name = f"UNKNOWN_{tid}"
                        otype_counts[type_name] += 1

                        # Duplicate check
                        pair = (oid, fid)
                        if pair in seen_pairs:
                            duplicate_records += 1
                        seen_pairs.add(pair)

                        # Bounding box geometry checks
                        if w <= 0 or h <= 0 or x < 0 or y < 0:
                            invalid_bboxes.append({"line": line_idx, "track_id": oid, "frame": fid, "bbox": [x, y, w, h]})
                        elif ref_w > 0 and ref_h > 0 and (x + w > ref_w + 5 or y + h > ref_h + 5):
                            out_of_bounds_bboxes.append({"line": line_idx, "track_id": oid, "frame": fid, "bbox": [x, y, w, h]})

                if malformed_lines > 0:
                    errors.append(f"Encountered {malformed_lines} malformed annotation lines in {meta.objects_path.name}")
                if duplicate_records > 0:
                    warnings.append(f"Encountered {duplicate_records} duplicate (track_id, frame_id) entries.")
                if invalid_bboxes:
                    errors.append(f"Encountered {len(invalid_bboxes)} bounding boxes with non-positive dimensions or negative coordinates.")
                if out_of_bounds_bboxes:
                    warnings.append(f"Encountered {len(out_of_bounds_bboxes)} bounding boxes extending beyond reference frame resolution.")

                ann_props = {
                    "total_data_lines": total_lines,
                    "comment_lines": comment_lines,
                    "malformed_lines": malformed_lines,
                    "unique_tracks": len(unique_tracks),
                    "annotated_frames": len(all_frames),
                    "min_frame": min(all_frames) if all_frames else 0,
                    "max_frame": max(all_frames) if all_frames else 0,
                    "object_distribution": dict(otype_counts),
                    "duplicate_records": duplicate_records,
                    "invalid_bbox_count": len(invalid_bboxes),
                    "out_of_bounds_count": len(out_of_bounds_bboxes),
                }

                # 3. Video-Annotation Synchronization Check
                total_vid_frames = video_props.get("total_frames", meta.total_frames)
                if total_vid_frames > 0 and ann_props.get("max_frame", 0) >= total_vid_frames:
                    warnings.append(
                        f"Annotation max frame index ({ann_props['max_frame']}) exceeds video frame count ({total_vid_frames})."
                    )

                sync_props = {
                    "video_frames": total_vid_frames,
                    "annotation_max_frame": ann_props.get("max_frame", 0),
                    "is_synchronized": (ann_props.get("max_frame", 0) < total_vid_frames or total_vid_frames == 0),
                    "frame_coverage_ratio": round(len(all_frames) / max(1, total_vid_frames), 4) if total_vid_frames > 0 else 0.0,
                }

            except Exception as e:
                errors.append(f"Annotation inspection encountered exception: {e}")

        is_valid = (len(errors) == 0)
        return SequenceValidationResult(
            sequence_id=meta.sequence_id,
            is_valid=is_valid,
            is_genuine=is_genuine,
            tier=tier,
            video_path=str(meta.video_path) if meta.video_path else None,
            objects_path=str(meta.objects_path) if meta.objects_path else None,
            errors=errors,
            warnings=warnings,
            video_properties=video_props,
            annotation_summary=ann_props,
            synchronization_summary=sync_props,
        )

    def validate_all(self) -> Dict[str, SequenceValidationResult]:
        """Runs the validation suite on all discovered sequences."""
        results: Dict[str, SequenceValidationResult] = {}
        for seq_id, meta in self.loader.scan_dataset().items():
            results[seq_id] = self.validate_sequence(meta)
        return results

    def generate_inventory_and_validation_reports(self, output_dir: Path) -> Tuple[Path, Path]:
        """Generates virat_dataset_inventory.json and virat_dataset_validation.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        validation_results = self.validate_all()

        # 1. Dataset Inventory JSON
        inventory_data = {
            "dataset_name": "VIRAT Video Dataset Release 2.0",
            "root_directory": str(self.loader.root_dir),
            "total_sequences": len(validation_results),
            "genuine_sequences_count": sum(1 for r in validation_results.values() if r.is_genuine),
            "fixture_sequences_count": sum(1 for r in validation_results.values() if not r.is_genuine),
            "valid_sequences_count": sum(1 for r in validation_results.values() if r.is_valid),
            "sequences": [
                {
                    "sequence_id": r.sequence_id,
                    "tier": r.tier,
                    "is_genuine": r.is_genuine,
                    "is_valid": r.is_valid,
                    "resolution": f"{r.video_properties.get('width', 0)}x{r.video_properties.get('height', 0)}",
                    "total_frames": r.video_properties.get("total_frames", 0),
                    "fps": r.video_properties.get("fps", 0.0),
                    "unique_tracks": r.annotation_summary.get("unique_tracks", 0),
                    "objects_distribution": r.annotation_summary.get("object_distribution", {}),
                    "video_path": r.video_path,
                    "objects_path": r.objects_path,
                }
                for r in validation_results.values()
            ],
        }
        inventory_path = output_dir / "virat_dataset_inventory.json"
        with open(inventory_path, "w", encoding="utf-8") as f:
            json.dump(inventory_data, f, indent=2)
        logger.info(f"Saved dataset inventory to: {inventory_path}")

        # 2. Dataset Validation JSON
        validation_data = {
            "validation_timestamp": np.datetime64("now").astype(str),
            "total_sequences_evaluated": len(validation_results),
            "all_sequences_valid": all(r.is_valid for r in validation_results.values()),
            "results": {seq_id: r.to_dict() for seq_id, r in validation_results.items()},
        }
        validation_path = output_dir / "virat_dataset_validation.json"
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation_data, f, indent=2)
        logger.info(f"Saved dataset validation to: {validation_path}")

        return inventory_path, validation_path

