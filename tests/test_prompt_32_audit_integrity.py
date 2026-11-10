"""Unit tests validating Prompt 32 audit tooling and mathematical integrity."""

import csv
import json
from pathlib import Path
import pytest
import psutil

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_reconciliation_csv_math():
    """Validates mathematical integrity of Prompt 32 latency reconciliation."""
    csv_path = ROOT_DIR / "outputs" / "prompt_32_latency_reconciliation.csv"
    assert csv_path.exists(), "Reconciliation CSV does not exist."

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 8, "Expected at least 8 configurations in reconciliation CSV."

    for row in reader:
        total_frames = int(row["Total_Input_Frames"])
        fresh = int(row["Fresh_Detections"])
        extrapolated = int(row["Extrapolated_Boxes"])
        reported = int(row["Total_Reported_Boxes"])
        invocations = int(row["Detector_Invocations"])
        interval = int(row["Interval"])

        # Check box accounting: fresh + extrapolated must equal total reported
        assert fresh + extrapolated == reported, f"Mismatch in {row['Configuration']}: {fresh} + {extrapolated} != {reported}"

        # Check detector invocations
        expected_invocations = (total_frames + interval - 1) // interval
        assert invocations == expected_invocations, f"Mismatch in {row['Configuration']}: {invocations} != {expected_invocations}"


def test_detection_extrapolation_audit_json():
    """Validates that extrapolated ratio is correctly computed and non-zero for intervals > 1."""
    json_path = ROOT_DIR / "outputs" / "prompt_32_detection_extrapolation_audit.json"
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cfg_name, rec in data.items():
        if rec["interval"] == 1:
            assert rec["extrapolated_ratio"] == 0.0
            assert rec["extrapolated_boxes"] == 0
        else:
            assert rec["extrapolated_ratio"] > 0.0
            assert rec["extrapolated_boxes"] > 0


def test_ground_truth_status_unmeasurable():
    """Validates that ground truth annotations lack spatial trajectories and event logs."""
    anno_path = ROOT_DIR / "configs" / "annotations" / "demo_pool_track_annotations.json"
    assert anno_path.exists()

    with open(anno_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    # Check that annotations only contain track-level classifications, not events or bbox trajectories
    for tid, tdata in anno["annotations"].items():
        assert "classification" in tdata
        assert "events" not in tdata
        assert "trajectory" not in tdata
        assert "bbox_xyxy" not in tdata


def test_cpu_affinity_audit():
    """Verifies that process CPU affinity is unpinned across logical cores."""
    proc = psutil.Process()
    if hasattr(proc, "cpu_affinity"):
        affinity = proc.cpu_affinity()
        # On commodity Windows, default affinity encompasses all logical cores
        assert len(affinity) == psutil.cpu_count(logical=True)

