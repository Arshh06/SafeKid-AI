"""JAAD dataset annotation parser and validator.

Parses official JAAD (Joint Attention in Autonomous Driving) XML annotations:
- video_XXXX.xml: Spatial bounding boxes, frame indices, track IDs, occlusion, and behavioral attributes.
- video_XXXX_attributes.xml: Track-level demographic attributes (age, gender, crossing, direction, etc.).
- video_XXXX_appearance.xml: Appearance attributes (clothing, pose, carried objects, accessories).

Preserves original dataset labels alongside SafeKid AI harmonized labels:
- CHILD (< 13 years)
- ADULT (>= 18 years)
- UNKNOWN (adolescents, ambiguous, low-resolution, or missing demographics)
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.jaad_parser")


@dataclass
class JAADBoundingBox:
    """Represents a validated per-frame bounding box for a pedestrian instance."""
    video_id: str
    frame_id: int
    track_id: str
    old_id: Optional[str]
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    width: float
    height: float
    occluded: bool
    occlusion_level: str  # 'none', 'part', 'full'
    outside: bool
    keyframe: bool
    action: str  # 'standing', 'walking', etc.
    cross: str   # 'crossing', 'not-crossing', etc.
    look: str    # 'looking', 'not-looking', etc.
    raw_age: str
    raw_gender: str
    raw_designated: str
    harmonized_label: ChildDetectionLabel
    harmonization_confidence: float
    harmonization_reason: str
    is_valid_geometry: bool = True
    is_low_res: bool = False
    validation_flags: List[str] = field(default_factory=list)

    @property
    def aspect_ratio(self) -> float:
        """Height / Width ratio (typical upright human is ~2.0 - 3.5)."""
        return self.height / max(1.0, self.width)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes bounding box record to dictionary."""
        return {
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "track_id": self.track_id,
            "old_id": self.old_id,
            "bbox_xyxy": [round(self.xtl, 1), round(self.ytl, 1), round(self.xbr, 1), round(self.ybr, 1)],
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "occluded": self.occluded,
            "occlusion_level": self.occlusion_level,
            "outside": self.outside,
            "action": self.action,
            "cross": self.cross,
            "look": self.look,
            "raw_age": self.raw_age,
            "raw_gender": self.raw_gender,
            "harmonized_label": self.harmonized_label.value,
            "harmonization_confidence": round(self.harmonization_confidence, 2),
            "harmonization_reason": self.harmonization_reason,
            "is_valid_geometry": self.is_valid_geometry,
            "is_low_res": self.is_low_res,
            "validation_flags": self.validation_flags,
        }


@dataclass
class JAADValidationReport:
    """Structured report summarizing the validation of a set of JAAD annotations."""
    total_xml_files: int = 0
    total_tracks: int = 0
    total_boxes: int = 0
    valid_boxes: int = 0
    outside_boxes_filtered: int = 0
    invalid_geometry_boxes: int = 0
    duplicate_annotations_detected: int = 0
    low_resolution_boxes: int = 0
    class_counts: Dict[str, int] = field(default_factory=lambda: {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0})
    raw_age_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_xml_files": self.total_xml_files,
            "total_tracks": self.total_tracks,
            "total_boxes": self.total_boxes,
            "valid_boxes": self.valid_boxes,
            "outside_boxes_filtered": self.outside_boxes_filtered,
            "invalid_geometry_boxes": self.invalid_geometry_boxes,
            "duplicate_annotations_detected": self.duplicate_annotations_detected,
            "low_resolution_boxes": self.low_resolution_boxes,
            "class_counts": self.class_counts,
            "raw_age_counts": self.raw_age_counts,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class JAADAnnotationParser:
    """Parser for JAAD dataset annotations with demographic label harmonization and geometry validation."""

    def __init__(
        self,
        min_crop_height: int = 40,
        min_crop_width: int = 16,
        low_res_height_threshold: int = 80,
        default_frame_width: int = 1920,
        default_frame_height: int = 1080,
    ):
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width
        self.low_res_height_threshold = low_res_height_threshold
        self.default_frame_width = default_frame_width
        self.default_frame_height = default_frame_height

    def harmonize_age(self, raw_age: Optional[str]) -> Tuple[ChildDetectionLabel, float, str]:
        """Harmonizes raw JAAD age string into SafeKid operational label.
        
        Adheres to docs/child_label_harmonization.md:
        - 'child' -> CHILD (conf 0.95)
        - 'adult', 'senior' -> ADULT (conf 0.95)
        - 'young' -> UNKNOWN (conf 0.40, adolescent semantic gap)
        - None/unspecified -> UNKNOWN (conf 0.0)
        """
        if not raw_age:
            return (ChildDetectionLabel.UNKNOWN, 0.0, "Missing or unspecified age attribute.")
        
        clean_age = raw_age.strip().lower()
        if clean_age == "child":
            return (ChildDetectionLabel.CHILD, 0.95, "Verified child pedestrian in JAAD.")
        elif clean_age in ("adult", "senior"):
            return (ChildDetectionLabel.ADULT, 0.95, f"Verified mature adult/senior ({clean_age}) in JAAD.")
        elif clean_age == "young":
            return (
                ChildDetectionLabel.UNKNOWN,
                0.40,
                "JAAD 'young' bundles adolescents (13-17) and young adults; mapped conservatively to UNKNOWN.",
            )
        else:
            return (ChildDetectionLabel.UNKNOWN, 0.20, f"Unrecognized age tag '{raw_age}' in JAAD.")

    def parse_track_attributes(self, attr_xml_path: Path) -> Dict[str, Dict[str, str]]:
        """Parses pedestrian attributes from video_XXXX_attributes.xml.
        
        Returns:
            Dictionary mapping pedestrian ID (e.g. '0_5_13b') to dict of attribute key-value pairs.
        """
        if not attr_xml_path.exists():
            return {}
        
        try:
            tree = ET.parse(attr_xml_path)
            root = tree.getroot()
        except Exception as e:
            logger.warning(f"Failed to parse attribute XML {attr_xml_path}: {e}")
            return {}

        track_attrs: Dict[str, Dict[str, str]] = {}
        for ped in root.findall("pedestrian"):
            ped_id = ped.get("id")
            if ped_id:
                track_attrs[ped_id] = {k: v for k, v in ped.attrib.items()}
        return track_attrs

    def parse_video_annotations(
        self,
        ann_xml_path: Path,
        attr_xml_path: Optional[Path] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[List[JAADBoundingBox], JAADValidationReport]:
        """Parses and validates all bounding boxes for a single video sequence.
        
        Args:
            ann_xml_path: Path to video_XXXX.xml annotation file.
            attr_xml_path: Optional path to corresponding video_XXXX_attributes.xml file.
            frame_width: Frame pixel width for boundary clipping (defaults to 1920).
            frame_height: Frame pixel height for boundary clipping (defaults to 1080).
            
        Returns:
            Tuple of (list of validated JAADBoundingBox, JAADValidationReport).
        """
        fw = frame_width or self.default_frame_width
        fh = frame_height or self.default_frame_height
        report = JAADValidationReport(total_xml_files=1)

        if not ann_xml_path.exists():
            report.errors.append(f"Annotation file not found: {ann_xml_path}")
            return [], report

        # Extract video identifier from filename (e.g. video_0005)
        video_id = ann_xml_path.stem

        # Parse demographic attributes if available
        track_attributes: Dict[str, Dict[str, str]] = {}
        if attr_xml_path and attr_xml_path.exists():
            track_attributes = self.parse_track_attributes(attr_xml_path)

        try:
            tree = ET.parse(ann_xml_path)
            root = tree.getroot()
        except Exception as e:
            report.errors.append(f"Corrupted or invalid XML syntax in {ann_xml_path}: {e}")
            return [], report

        # Check meta task frame size if available
        meta_size = root.find(".//task/size")
        if meta_size is not None and meta_size.text:
            try:
                pass  # Frames count
            except Exception:
                pass

        parsed_boxes: List[JAADBoundingBox] = []
        seen_frame_tracks: Set[Tuple[int, str]] = set()

        tracks = root.findall("track")
        report.total_tracks = len(tracks)

        for track in tracks:
            track_label = track.get("label", "pedestrian")
            if track_label not in ("pedestrian", "ped"):
                continue

            for box in track.findall("box"):
                report.total_boxes += 1
                outside = box.get("outside", "0") == "1"
                if outside:
                    report.outside_boxes_filtered += 1
                    continue

                frame_id_str = box.get("frame")
                if frame_id_str is None:
                    report.invalid_geometry_boxes += 1
                    report.warnings.append(f"Box missing frame index in {video_id}")
                    continue
                try:
                    frame_id = int(frame_id_str)
                except ValueError:
                    report.invalid_geometry_boxes += 1
                    continue

                # Parse box attributes
                box_attrs = {a.get("name"): (a.text or "") for a in box.findall("attribute")}
                track_id = box_attrs.get("id", "")
                old_id = box_attrs.get("old_id")
                action = box_attrs.get("action", "unspecified")
                cross = box_attrs.get("cross", "unspecified")
                look = box_attrs.get("look", "unspecified")
                occlusion_level = box_attrs.get("occlusion", "none")
                occluded = box.get("occluded", "0") == "1" or occlusion_level in ("part", "full")
                keyframe = box.get("keyframe", "0") == "1"

                # Check duplicate (frame_id, track_id)
                frame_track_key = (frame_id, track_id)
                if frame_track_key in seen_frame_tracks:
                    report.duplicate_annotations_detected += 1
                    report.warnings.append(
                        f"Duplicate annotation for track {track_id} at frame {frame_id} in {video_id}"
                    )
                    continue
                seen_frame_tracks.add(frame_track_key)

                # Parse coordinates
                try:
                    xtl = float(box.get("xtl", 0))
                    ytl = float(box.get("ytl", 0))
                    xbr = float(box.get("xbr", 0))
                    ybr = float(box.get("ybr", 0))
                except (ValueError, TypeError):
                    report.invalid_geometry_boxes += 1
                    report.warnings.append(f"Malformed coordinate values in {video_id} frame {frame_id}")
                    continue

                val_flags: List[str] = []

                # Coordinate boundary clipping
                clipped_xtl = max(0.0, min(float(fw), xtl))
                clipped_ytl = max(0.0, min(float(fh), ytl))
                clipped_xbr = max(0.0, min(float(fw), xbr))
                clipped_ybr = max(0.0, min(float(fh), ybr))

                if (clipped_xtl != xtl or clipped_ytl != ytl or 
                    clipped_xbr != xbr or clipped_ybr != ybr):
                    val_flags.append("CLIPPED_TO_FRAME")

                w = clipped_xbr - clipped_xtl
                h = clipped_ybr - clipped_ytl

                # Geometry validation
                if w < self.min_crop_width or h < self.min_crop_height:
                    report.invalid_geometry_boxes += 1
                    report.warnings.append(
                        f"Rejected sub-minimum bbox {w:.1f}x{h:.1f} (min {self.min_crop_width}x{self.min_crop_height}) in {video_id}"
                    )
                    continue

                # Low-resolution check
                is_low_res = h < self.low_res_height_threshold
                if is_low_res:
                    report.low_resolution_boxes += 1
                    val_flags.append("LOW_RESOLUTION")

                # Demographic attribute lookup
                ped_meta = track_attributes.get(track_id, {})
                raw_age = ped_meta.get("age", "unspecified")
                raw_gender = ped_meta.get("gender", "unspecified")
                raw_designated = ped_meta.get("designated", "unspecified")

                # Label harmonization
                if is_low_res and raw_age == "child":
                    # Low-res child requires UNKNOWN abstention per harmonization standard
                    harmonized_label = ChildDetectionLabel.UNKNOWN
                    conf = 0.50
                    reason = f"Child pedestrian crop height ({h:.1f}px) below {self.low_res_height_threshold}px threshold; abstained to UNKNOWN."
                    val_flags.append("LOW_RES_ABSTENTION")
                else:
                    harmonized_label, conf, reason = self.harmonize_age(raw_age)

                # Update statistics
                report.class_counts[harmonized_label.value] += 1
                report.raw_age_counts[raw_age] = report.raw_age_counts.get(raw_age, 0) + 1
                report.valid_boxes += 1

                bbox_record = JAADBoundingBox(
                    video_id=video_id,
                    frame_id=frame_id,
                    track_id=track_id,
                    old_id=old_id,
                    xtl=clipped_xtl,
                    ytl=clipped_ytl,
                    xbr=clipped_xbr,
                    ybr=clipped_ybr,
                    width=w,
                    height=h,
                    occluded=occluded,
                    occlusion_level=occlusion_level,
                    outside=outside,
                    keyframe=keyframe,
                    action=action,
                    cross=cross,
                    look=look,
                    raw_age=raw_age,
                    raw_gender=raw_gender,
                    raw_designated=raw_designated,
                    harmonized_label=harmonized_label,
                    harmonization_confidence=conf,
                    harmonization_reason=reason,
                    is_valid_geometry=True,
                    is_low_res=is_low_res,
                    validation_flags=val_flags,
                )
                parsed_boxes.append(bbox_record)

        return parsed_boxes, report

