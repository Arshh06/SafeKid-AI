#!/usr/bin/env python3
"""Prepares real-world child detection dataset from acquired surveillance sequences.

Applies:
- JAADAnnotationParser to parse official XML annotations and demographic attributes.
- SafeKid AI label harmonization (CHILD, ADULT, UNKNOWN) adhering to docs/child_label_harmonization.md.
- PersonCropExtractor for aspect-ratio preserved, symmetric letterbox-padded 128x64 crops.
- Perceptual deduplication via dHash (Hamming distance <= 2).
- Group-aware sequence partitioning to guarantee ZERO train/val/test leakage.
- Generates structured split manifests, statistics, and audit CSVs.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safekid_ai.child_detection.crop_extractor import PersonCropExtractor
from safekid_ai.child_detection.dataset_ingestion import compute_dhash, hamming_distance
from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.datasets.jaad_parser import JAADAnnotationParser, JAADBoundingBox
from safekid_ai.datasets.video_split_manager import VideoSplitManager
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.prepare_dataset")


class RealWorldChildDatasetPreparer:
    """Prepares and audits real-world child detection dataset crops and manifests."""

    def __init__(
        self,
        jaad_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        target_height: int = 128,
        target_width: int = 64,
        min_crop_height: int = 40,
        min_crop_width: int = 16,
        low_res_height_threshold: int = 80,
        seed: int = 42,
    ):
        self.jaad_root = jaad_root or Path("data/external/jaad")
        self.output_root = output_root or Path("data/child_dataset/real_world/jaad")
        self.crops_dir = self.output_root / "crops"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        self.parser = JAADAnnotationParser(
            min_crop_height=min_crop_height,
            min_crop_width=min_crop_width,
            low_res_height_threshold=low_res_height_threshold,
        )
        self.extractor = PersonCropExtractor(
            target_height=target_height,
            target_width=target_width,
            min_crop_height=min_crop_height,
            min_crop_width=min_crop_width,
            low_res_height_threshold=low_res_height_threshold,
        )
        self.split_manager = VideoSplitManager(seed=seed)

    def prepare_dataset(
        self,
        max_crops_per_track: int = 25,
        frame_sample_stride: int = 3,
        use_official_splits: bool = False,
    ) -> Dict[str, Any]:
        """Processes acquired JAAD videos and annotations into standardized crops.
        
        Args:
            max_crops_per_track: Cap on crops per track to avoid frame redundancy.
            frame_sample_stride: Temporal sub-sampling stride across frames.
            use_official_splits: Whether to use JAAD official split files or stratified group split.
            
        Returns:
            Dictionary containing preparation summary and paths.
        """
        videos_dir = self.jaad_root / "videos"
        ann_dir = self.jaad_root / "repo" / "annotations"
        attr_dir = self.jaad_root / "repo" / "annotations_attributes"

        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found at {videos_dir}")

        available_videos = sorted([p.stem for p in videos_dir.glob("*.mp4")])
        if not available_videos:
            logger.warning(f"No video files found in {videos_dir}.")
            return {"status": "NO_VIDEOS", "crops_extracted": 0}

        logger.info(f"Found {len(available_videos)} local video files: {available_videos}")

        # Check which videos contain children
        child_video_ids = []
        for vid in available_videos:
            attr_f = attr_dir / f"{vid}_attributes.xml"
            if attr_f.exists():
                attrs = self.parser.parse_track_attributes(attr_f)
                if any(p.get("age") == "child" for p in attrs.values()):
                    child_video_ids.append(vid)

        # Create split mapping
        if use_official_splits and (self.jaad_root / "repo" / "split_ids" / "all_videos").exists():
            logger.info("Using official JAAD all_videos split assignments...")
            split_dir = self.jaad_root / "repo" / "split_ids" / "all_videos"
            raw_splits = self.split_manager.load_official_splits(split_dir)
            splits_mapping = {
                s: sorted([v for v in available_videos if v in vids])
                for s, vids in raw_splits.items()
            }
        else:
            logger.info("Creating stratified group-aware sequence split...")
            splits_mapping = self.split_manager.create_stratified_group_split(
                video_ids=available_videos,
                child_video_ids=child_video_ids,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )

        video_to_split: Dict[str, str] = {}
        for s_name, v_list in splits_mapping.items():
            for v in v_list:
                video_to_split[v] = s_name

        # Create split folders
        for s in ("train", "val", "test"):
            (self.crops_dir / s).mkdir(parents=True, exist_ok=True)

        accepted_samples: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []
        seen_hashes: Dict[str, Dict[int, str]] = {"train": {}, "val": {}, "test": {}}
        near_dupes_removed = 0
        track_sample_counts: Dict[str, int] = {}

        total_frames_processed = 0

        for vid in available_videos:
            video_path = videos_dir / f"{vid}.mp4"
            ann_path = ann_dir / f"{vid}.xml"
            attr_path = attr_dir / f"{vid}_attributes.xml"
            split_name = video_to_split.get(vid, "train")

            if not ann_path.exists():
                logger.warning(f"Missing annotation file for video {vid}")
                continue

            # Parse bounding boxes
            boxes, val_report = self.parser.parse_video_annotations(
                ann_xml_path=ann_path,
                attr_xml_path=attr_path if attr_path.exists() else None,
            )

            # Group boxes by frame
            boxes_by_frame: Dict[int, List[JAADBoundingBox]] = {}
            for b in boxes:
                boxes_by_frame.setdefault(b.frame_id, []).append(b)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video file {video_path}")
                continue

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames_processed += 1

                # Check if this frame has annotations and satisfies stride
                if frame_idx in boxes_by_frame and (frame_idx % frame_sample_stride == 0):
                    frame_boxes = boxes_by_frame[frame_idx]
                    for bbox in frame_boxes:
                        full_track_key = f"{vid}_{bbox.track_id}"
                        current_track_count = track_sample_counts.get(full_track_key, 0)
                        if current_track_count >= max_crops_per_track:
                            # Skip excess temporal redundancy
                            audit_records.append({
                                "video_id": vid,
                                "frame_id": frame_idx,
                                "track_id": bbox.track_id,
                                "raw_age": bbox.raw_age,
                                "harmonized_label": bbox.harmonized_label.value,
                                "action": "EXCLUDED",
                                "reason": f"Max crops per track reached ({max_crops_per_track})",
                            })
                            continue

                        # Extract padded crop
                        ext_res = self.extractor.extract_crop(
                            frame=frame,
                            bbox_xyxy=(bbox.xtl, bbox.ytl, bbox.xbr, bbox.ybr),
                            metadata={"video_id": vid, "frame_id": frame_idx, "track_id": bbox.track_id},
                        )

                        if not ext_res.is_valid:
                            audit_records.append({
                                "video_id": vid,
                                "frame_id": frame_idx,
                                "track_id": bbox.track_id,
                                "raw_age": bbox.raw_age,
                                "harmonized_label": bbox.harmonized_label.value,
                                "action": "REJECTED",
                                "reason": ext_res.rejection_reason or "Invalid crop geometry",
                            })
                            continue

                        # Perceptual hash deduplication within split
                        dh = compute_dhash(ext_res.crop_image)
                        is_dup = False
                        for prev_h, prev_fname in seen_hashes[split_name].items():
                            if hamming_distance(dh, prev_h) <= 2:
                                is_dup = True
                                near_dupes_removed += 1
                                audit_records.append({
                                    "video_id": vid,
                                    "frame_id": frame_idx,
                                    "track_id": bbox.track_id,
                                    "raw_age": bbox.raw_age,
                                    "harmonized_label": bbox.harmonized_label.value,
                                    "action": "DEDUPLICATED",
                                    "reason": f"Near-duplicate collision with {prev_fname}",
                                })
                                break

                        if is_dup:
                            continue

                        # Save crop
                        sample_id = len(accepted_samples) + 1
                        crop_filename = f"{vid}_f{frame_idx:05d}_t{bbox.track_id}_s{sample_id:06d}.png"
                        crop_path = self.crops_dir / split_name / crop_filename
                        cv2.imwrite(str(crop_path), ext_res.crop_image)

                        seen_hashes[split_name][dh] = crop_filename
                        track_sample_counts[full_track_key] = current_track_count + 1

                        sample_record = {
                            "sample_id": sample_id,
                            "video_id": vid,
                            "frame_id": frame_idx,
                            "track_id": bbox.track_id,
                            "crop_path": str(crop_path).replace("\\", "/"),
                            "crop_filename": crop_filename,
                            "split": split_name,
                            "raw_age": bbox.raw_age,
                            "raw_gender": bbox.raw_gender,
                            "harmonized_label": bbox.harmonized_label.value,
                            "harmonization_confidence": bbox.harmonization_confidence,
                            "harmonization_reason": bbox.harmonization_reason,
                            "is_low_res": bbox.is_low_res or ext_res.is_low_res,
                            "occluded": bbox.occluded,
                            "occlusion_level": bbox.occlusion_level,
                            "action": bbox.action,
                            "cross": bbox.cross,
                            "original_bbox": [bbox.xtl, bbox.ytl, bbox.xbr, bbox.ybr],
                            "crop_dimensions": [ext_res.target_height, ext_res.target_width],
                            "original_crop_dimensions": [ext_res.original_height, ext_res.original_width],
                            "dhash": hex(dh),
                        }
                        accepted_samples.append(sample_record)

                        audit_records.append({
                            "video_id": vid,
                            "frame_id": frame_idx,
                            "track_id": bbox.track_id,
                            "raw_age": bbox.raw_age,
                            "harmonized_label": bbox.harmonized_label.value,
                            "action": "ACCEPTED",
                            "reason": "Passed all quality, resolution, geometry, and deduplication gates",
                        })

                frame_idx += 1

            cap.release()

        # Audit split leakage
        leakage_audit = self.split_manager.audit_splits(accepted_samples, splits_mapping)

        # Write split manifest
        split_manifest_payload = {
            "metadata": {
                "manifest_version": "1.0",
                "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "dataset_source": "JAAD",
                "total_samples": len(accepted_samples),
                "total_frames_processed": total_frames_processed,
                "near_duplicates_removed": near_dupes_removed,
                "is_leakage_free": leakage_audit.is_leakage_free,
            },
            "splits": leakage_audit.split_counts,
            "classes_by_split": leakage_audit.class_distribution,
            "leakage_audit": leakage_audit.to_dict(),
            "samples": accepted_samples,
        }

        manifest_out_path = Path("outputs/prompt_27_split_manifest.json")
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_out_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest_payload, f, indent=2)

        local_manifest_path = self.output_root / "manifest.json"
        with open(local_manifest_path, "w", encoding="utf-8") as f:
            json.dump(split_manifest_payload, f, indent=2)

        # Write audit CSV
        audit_csv_path = Path("outputs/prompt_27_annotation_audit.csv")
        with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["video_id", "frame_id", "track_id", "raw_age", "harmonized_label", "action", "reason"]
            )
            writer.writeheader()
            for r in audit_records:
                writer.writerow(r)

        logger.info(
            f"Preparation complete! Extracted {len(accepted_samples)} crops from {total_frames_processed} frames. "
            f"Duplicates purged: {near_dupes_removed}. Leakage-free: {leakage_audit.is_leakage_free}"
        )

        return {
            "status": "SUCCESS",
            "total_samples": len(accepted_samples),
            "total_frames_processed": total_frames_processed,
            "near_duplicates_removed": near_dupes_removed,
            "is_leakage_free": leakage_audit.is_leakage_free,
            "manifest_path": str(manifest_out_path),
            "audit_csv_path": str(audit_csv_path),
            "splits": leakage_audit.split_counts,
            "classes": leakage_audit.class_distribution,
        }


def main():
    parser = argparse.ArgumentParser(description="Prepare real-world child dataset")
    parser.add_argument("--max-crops-per-track", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--official-splits", action="store_true")
    args = parser.parse_args()

    preparer = RealWorldChildDatasetPreparer()
    summary = preparer.prepare_dataset(
        max_crops_per_track=args.max_crops_per_track,
        frame_sample_stride=args.stride,
        use_official_splits=args.official_splits,
    )
    print("Dataset preparation summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

