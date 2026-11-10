#!/usr/bin/env python3
"""Validates JAAD and child detection dataset annotations for syntactic and geometric integrity.

Checks:
- XML syntax and element hierarchy.
- Frame and track identifier consistency.
- Bounding box geometry (x2 > x1, y2 > y1, positive dimensions).
- Frame boundary compliance (clipping alerts).
- Duplicate (frame_id, track_id) annotations.
- Demographic attribute integrity and harmonization coverage.
- Exports structured validation report.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safekid_ai.datasets.jaad_parser import JAADAnnotationParser, JAADValidationReport
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.validate_annotations")


def validate_dataset_annotations(
    annotations_dir: Path,
    attributes_dir: Path,
    max_files: int = 0,
) -> Dict[str, Any]:
    """Validates XML annotations in the given directory."""
    parser = JAADAnnotationParser()
    xml_files = sorted(list(annotations_dir.glob("*.xml")))
    if max_files > 0:
        xml_files = xml_files[:max_files]

    total_files = len(xml_files)
    logger.info(f"Validating {total_files} XML annotation files...")

    aggregate_report = JAADValidationReport(total_xml_files=total_files)

    for xml_path in xml_files:
        vid_id = xml_path.stem
        attr_path = attributes_dir / f"{vid_id}_attributes.xml"

        boxes, report = parser.parse_video_annotations(
            ann_xml_path=xml_path,
            attr_xml_path=attr_path if attr_path.exists() else None,
        )

        aggregate_report.total_tracks += report.total_tracks
        aggregate_report.total_boxes += report.total_boxes
        aggregate_report.valid_boxes += report.valid_boxes
        aggregate_report.outside_boxes_filtered += report.outside_boxes_filtered
        aggregate_report.invalid_geometry_boxes += report.invalid_geometry_boxes
        aggregate_report.duplicate_annotations_detected += report.duplicate_annotations_detected
        aggregate_report.low_resolution_boxes += report.low_resolution_boxes

        for cls, count in report.class_counts.items():
            aggregate_report.class_counts[cls] = aggregate_report.class_counts.get(cls, 0) + count

        for raw_age, count in report.raw_age_counts.items():
            aggregate_report.raw_age_counts[raw_age] = (
                aggregate_report.raw_age_counts.get(raw_age, 0) + count
            )

        aggregate_report.errors.extend(report.errors)
        aggregate_report.warnings.extend(report.warnings[:5])  # Cap warnings per file to prevent log spam

    logger.info(
        f"Validation complete! {aggregate_report.valid_boxes}/{aggregate_report.total_boxes} valid boxes, "
        f"{aggregate_report.invalid_geometry_boxes} invalid geometry, {aggregate_report.duplicate_annotations_detected} duplicates."
    )
    return aggregate_report.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Validate JAAD annotations")
    parser.add_argument("--ann-dir", default="data/external/jaad/repo/annotations")
    parser.add_argument("--attr-dir", default="data/external/jaad/repo/annotations_attributes")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output", default="outputs/prompt_27_annotation_validation_summary.json")
    args = parser.parse_args()

    ann_path = Path(args.ann_dir)
    attr_path = Path(args.attr_dir)

    if not ann_path.exists():
        print(f"Error: Annotations directory {ann_path} does not exist.")
        sys.exit(1)

    summary = validate_dataset_annotations(ann_path, attr_path, args.max_files)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

