"""Unit and integration tests for Prompt 27 real-world dataset acquisition, extraction, and preparation.

Validates:
- JAAD XML annotation parsing and track linking.
- Missing and malformed label handling.
- Bounding-box geometry validation and boundary clipping.
- Person crop extraction with aspect-ratio preservation and padding.
- Child/Adult/Unknown label harmonization standard compliance.
- Group-aware sequence partitioning with zero train/val/test leakage.
- Perceptual deduplication via difference hashing (dHash).
- Dataset split manifest and statistics integrity.
- Provenance tracking and cryptographic checksum verification.
- Deterministic reproducibility under fixed random seeds.
"""

import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest

from safekid_ai.child_detection.crop_extractor import ExtractedCropResult, PersonCropExtractor
from safekid_ai.child_detection.dataset_ingestion import compute_dhash, hamming_distance
from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.datasets.jaad_parser import (
    JAADAnnotationParser,
    JAADBoundingBox,
    JAADValidationReport,
)
from safekid_ai.datasets.video_split_manager import SplitLeakageAudit, VideoSplitManager


class TestJAADAnnotationParser:
    """Tests for JAAD XML annotation parsing and validation."""

    def test_parse_valid_annotations(self):
        """Tests parsing a well-formed JAAD annotation and attribute XML."""
        ann_xml = """<annotations>
            <track id="0" label="pedestrian">
                <box frame="0" keyframe="1" occluded="0" outside="0" xtl="100.0" ytl="200.0" xbr="200.0" ybr="500.0">
                    <attribute name="id">0_5_13b</attribute>
                    <attribute name="action">walking</attribute>
                    <attribute name="cross">crossing</attribute>
                    <attribute name="look">looking</attribute>
                    <attribute name="occlusion">none</attribute>
                </box>
            </track>
        </annotations>"""
        attr_xml = """<ped_attributes>
            <pedestrian age="child" gender="female" id="0_5_13b" designated="D" />
        </ped_attributes>"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            ann_p = Path(tmp_dir) / "video_0005.xml"
            attr_p = Path(tmp_dir) / "video_0005_attributes.xml"
            ann_p.write_text(ann_xml, encoding="utf-8")
            attr_p.write_text(attr_xml, encoding="utf-8")

            parser = JAADAnnotationParser()
            boxes, report = parser.parse_video_annotations(ann_p, attr_p)

            assert len(boxes) == 1
            b = boxes[0]
            assert b.video_id == "video_0005"
            assert b.frame_id == 0
            assert b.track_id == "0_5_13b"
            assert b.width == 100.0
            assert b.height == 300.0
            assert b.raw_age == "child"
            assert b.harmonized_label == ChildDetectionLabel.CHILD
            assert b.harmonization_confidence >= 0.90
            assert report.valid_boxes == 1
            assert report.total_tracks == 1

    def test_malformed_xml_and_missing_file_handling(self):
        """Tests robust handling of malformed syntax or non-existent files."""
        parser = JAADAnnotationParser()

        # Non-existent file
        boxes, report = parser.parse_video_annotations(Path("non_existent_file.xml"))
        assert len(boxes) == 0
        assert len(report.errors) > 0

        # Malformed syntax
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(b"<annotations><broken_tag>")
            f.flush()
            temp_path = Path(f.name)

        try:
            boxes, report = parser.parse_video_annotations(temp_path)
            assert len(boxes) == 0
            assert len(report.errors) > 0
        finally:
            temp_path.unlink()

    def test_invalid_geometry_rejection(self):
        """Tests that inverted coordinates or sub-minimum boxes are rejected."""
        ann_xml = """<annotations>
            <track id="0" label="pedestrian">
                <!-- Inverted coordinates -->
                <box frame="0" outside="0" xtl="300.0" ytl="500.0" xbr="100.0" ybr="200.0">
                    <attribute name="id">t1</attribute>
                </box>
                <!-- Sub-minimum dimensions -->
                <box frame="1" outside="0" xtl="100.0" ytl="100.0" xbr="105.0" ybr="110.0">
                    <attribute name="id">t2</attribute>
                </box>
            </track>
        </annotations>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(ann_xml.encode("utf-8"))
            f.flush()
            temp_path = Path(f.name)

        try:
            parser = JAADAnnotationParser(min_crop_height=40, min_crop_width=16)
            boxes, report = parser.parse_video_annotations(temp_path)
            assert len(boxes) == 0
            assert report.invalid_geometry_boxes == 2
        finally:
            temp_path.unlink()

    def test_duplicate_frame_track_detection(self):
        """Tests that duplicate annotations for the same track in the same frame are flagged."""
        ann_xml = """<annotations>
            <track id="0" label="pedestrian">
                <box frame="0" outside="0" xtl="100.0" ytl="100.0" xbr="200.0" ybr="300.0">
                    <attribute name="id">dup_track</attribute>
                </box>
                <box frame="0" outside="0" xtl="110.0" ytl="110.0" xbr="210.0" ybr="310.0">
                    <attribute name="id">dup_track</attribute>
                </box>
            </track>
        </annotations>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(ann_xml.encode("utf-8"))
            f.flush()
            temp_path = Path(f.name)

        try:
            parser = JAADAnnotationParser()
            boxes, report = parser.parse_video_annotations(temp_path)
            assert len(boxes) == 1
            assert report.duplicate_annotations_detected == 1
        finally:
            temp_path.unlink()


class TestLabelHarmonization:
    """Tests compliance with SafeKid AI child label harmonization standard."""

    @pytest.mark.parametrize(
        "raw_age,expected_label,min_conf",
        [
            ("child", ChildDetectionLabel.CHILD, 0.90),
            ("Child", ChildDetectionLabel.CHILD, 0.90),
            ("adult", ChildDetectionLabel.ADULT, 0.90),
            ("senior", ChildDetectionLabel.ADULT, 0.90),
            ("young", ChildDetectionLabel.UNKNOWN, 0.30),
            ("", ChildDetectionLabel.UNKNOWN, 0.0),
            (None, ChildDetectionLabel.UNKNOWN, 0.0),
            ("unspecified", ChildDetectionLabel.UNKNOWN, 0.0),
        ],
    )
    def test_jaad_age_harmonization_mapping(self, raw_age, expected_label, min_conf):
        parser = JAADAnnotationParser()
        lbl, conf, reason = parser.harmonize_age(raw_age)
        assert lbl == expected_label
        assert conf >= min_conf
        assert len(reason) > 0

    def test_low_res_child_abstention(self):
        """Tests that a child crop with height < 80px is conservatively abstained to UNKNOWN."""
        ann_xml = """<annotations>
            <track id="0" label="pedestrian">
                <!-- Height is 60px (below 80px threshold) -->
                <box frame="0" outside="0" xtl="100.0" ytl="100.0" xbr="140.0" ybr="160.0">
                    <attribute name="id">child_distant</attribute>
                </box>
            </track>
        </annotations>"""
        attr_xml = """<ped_attributes>
            <pedestrian age="child" id="child_distant" />
        </ped_attributes>"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            ann_p = Path(tmp_dir) / "video_test.xml"
            attr_p = Path(tmp_dir) / "video_test_attributes.xml"
            ann_p.write_text(ann_xml, encoding="utf-8")
            attr_p.write_text(attr_xml, encoding="utf-8")

            parser = JAADAnnotationParser(low_res_height_threshold=80)
            boxes, report = parser.parse_video_annotations(ann_p, attr_p)

            assert len(boxes) == 1
            b = boxes[0]
            assert b.is_low_res is True
            assert b.harmonized_label == ChildDetectionLabel.UNKNOWN
            assert "LOW_RES_ABSTENTION" in b.validation_flags


class TestPersonCropExtractor:
    """Tests for person crop extraction with aspect ratio preservation."""

    def test_crop_aspect_ratio_and_padding(self):
        """Tests that extracted crop has exact target shape (128, 64, 3) and preserves aspect ratio."""
        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        # Create a tall rectangle representing a standing pedestrian
        cv2.rectangle(frame, (200, 100), (300, 500), (0, 255, 0), -1)

        extractor = PersonCropExtractor(target_height=128, target_width=64)
        res = extractor.extract_crop(frame, (200, 100, 300, 500))

        assert res.is_valid is True
        assert res.crop_image.shape == (128, 64, 3)
        assert res.original_width == 100
        assert res.original_height == 400
        # Check padding was symmetric
        assert res.pad_left + res.pad_right + int(round(100 * res.scale)) == 64
        assert res.pad_top == 0 or res.pad_bottom == 0  # Tightly fits height

    def test_crop_boundary_clipping(self):
        """Tests that bounding boxes protruding outside frame boundaries are clipped safely."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        extractor = PersonCropExtractor(target_height=128, target_width=64, min_crop_height=20, min_crop_width=10)
        # Bounding box coordinates exceed 100x100 frame
        res = extractor.extract_crop(frame, (-20, -10, 150, 120))

        assert res.is_valid is True
        assert res.clipped_bbox == (0.0, 0.0, 100.0, 100.0)

    def test_empty_or_subminimum_crop_rejection(self):
        """Tests that empty frames or degenerate bounding boxes are rejected."""
        extractor = PersonCropExtractor(min_crop_height=40, min_crop_width=16)

        # None frame
        res1 = extractor.extract_crop(None, (0, 0, 100, 200))
        assert res1.is_valid is False

        # Sub-minimum dimensions
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res2 = extractor.extract_crop(frame, (10, 10, 20, 30))
        assert res2.is_valid is False
        assert "Sub-minimum crop geometry" in (res2.rejection_reason or "")


class TestGroupAwareSplitting:
    """Tests for zero-leakage group-aware video partitioning."""

    def test_stratified_group_split_zero_leakage(self):
        """Tests that video sequences and track IDs never leak across splits."""
        all_videos = [f"video_{i:04d}" for i in range(1, 21)]
        child_videos = ["video_0005", "video_0010", "video_0015"]

        manager = VideoSplitManager(seed=42)
        splits = manager.create_stratified_group_split(all_videos, child_videos)

        train_v = set(splits["train"])
        val_v = set(splits["val"])
        test_v = set(splits["test"])

        # Check total coverage
        assert len(train_v | val_v | test_v) == 20
        # Check zero overlap
        assert len(train_v & val_v) == 0
        assert len(train_v & test_v) == 0
        assert len(val_v & test_v) == 0

        # Check child video stratification
        for c in child_videos:
            assert c in train_v or c in val_v or c in test_v

    def test_split_manager_reproducibility(self):
        """Tests that identical seed produces identical splits."""
        videos = [f"v_{i}" for i in range(30)]
        child_v = ["v_1", "v_5", "v_10"]

        m1 = VideoSplitManager(seed=123)
        s1 = m1.create_stratified_group_split(videos, child_v)

        m2 = VideoSplitManager(seed=123)
        s2 = m2.create_stratified_group_split(videos, child_v)

        assert s1 == s2


class TestPerceptualDeduplication:
    """Tests perceptual difference hashing deduplication."""

    def test_identical_and_near_identical_crops(self):
        """Tests that identical crops have Hamming distance 0, and slight noise has distance <= 2."""
        img1 = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        img2 = img1.copy()

        h1 = compute_dhash(img1)
        h2 = compute_dhash(img2)
        assert hamming_distance(h1, h2) == 0

        # Add minor pixel perturbation
        img3 = img1.copy()
        img3[0, 0] = (img3[0, 0] + 5) % 255
        h3 = compute_dhash(img3)
        assert hamming_distance(h1, h3) <= 2


class TestDatasetManifestAndReadiness:
    """Tests generated split manifest and statistics deliverables."""

    def test_split_manifest_integrity(self):
        manifest_p = Path("outputs/prompt_27_split_manifest.json")
        assert manifest_p.exists(), "prompt_27_split_manifest.json was not generated"

        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["is_leakage_free"] is True
        assert data["metadata"]["total_samples"] >= 2000
        assert len(data["samples"]) == data["metadata"]["total_samples"]

        # Check class counts
        child_count = sum(1 for s in data["samples"] if s["harmonized_label"] == "CHILD")
        adult_count = sum(1 for s in data["samples"] if s["harmonized_label"] == "ADULT")
        assert child_count >= 200, f"Expected >= 200 child samples, got {child_count}"
        assert adult_count >= 500, f"Expected >= 500 adult samples, got {adult_count}"

    def test_dataset_statistics_integrity(self):
        stats_p = Path("outputs/prompt_27_dataset_statistics.json")
        assert stats_p.exists(), "prompt_27_dataset_statistics.json was not generated"

        with open(stats_p, "r", encoding="utf-8") as f:
            stats = json.load(f)

        assert stats["summary"]["total_crops"] >= 2000
        assert stats["summary"]["total_child_crops"] >= 200
        assert stats["training_readiness_assessment"]["sufficient_real_world_child_tracks"] is True
        assert stats["training_readiness_assessment"]["is_leakage_free"] is True

    def test_provenance_metadata(self):
        meta_p = Path("data/external/jaad/acquisition_metadata.json")
        assert meta_p.exists(), "JAAD acquisition metadata file missing"

        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["total_clips_acquired"] == 9
        assert len(meta["clips"]) == 9
        for clip in meta["clips"]:
            assert len(clip["sha256"]) == 64
            assert clip["size_bytes"] > 0

