"""SafeKid AI — VIRAT Surveillance Dataset Inspection Tool.

Inspects local VIRAT dataset files, validates format integrity, and reports:
- Video availability and metadata (resolution, FPS, frame count, duration)
- Annotation integrity (objects, events, mapping file alignment)
- Ground-truth object and person counts
- Dataset pairing status (complete vs orphan sequences)
- Acquisition checklist and guidelines if data has not yet been placed
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safekid_ai.datasets.virat_loader import VIRATDatasetLoader


def print_acquisition_checklist(loader: VIRATDatasetLoader) -> None:
    """Prints a clear, actionable guide for acquiring and placing VIRAT data."""
    print("\n" + "=" * 80)
    print("VIRAT DATASET ACQUISITION & INTEGRITY CHECKLIST")
    print("=" * 80)
    print("""
Status: No complete VIRAT video + annotation pairs were detected in:
        data/virat/videos/
        data/virat/annotations/

To evaluate SafeKid AI against genuine surveillance footage:

1. OFFICIAL DATASET INFORMATION:
   - Source: VIRAT Video Dataset (Release 2.0 Ground Surveillance)
   - Official Portal: https://viratdata.org/
   - Alternate Kitware Mirror: https://data.kitware.com/#collection/56f151598d297f6e3c03554e

2. RECOMMENDED SMALL-SCALE PILOT SUBSET:
   For initial validation, download 1 to 2 stationary camera clips (e.g., from Site 0000 or 0102):
   - Example Video: 'VIRAT_S_000001.mp4' (or 'VIRAT_S_000201_*.mp4')
   - Example Objects Annotation: 'VIRAT_S_000001.viratdata.objects.txt'
   - Example Events Annotation: 'VIRAT_S_000001.viratdata.events.txt' (optional)
   - Example Mapping Annotation: 'VIRAT_S_000001.viratdata.mapping.txt' (optional)

3. TARGET DIRECTORY PLACEMENT:
   Place the downloaded files into the respective SafeKid AI directories:
   - Video files:       C:\\projs\\SafeKid AI\\data\\virat\\videos\\
   - Annotation files:  C:\\projs\\SafeKid AI\\data\\virat\\annotations\\

4. ETHICAL & COMPLIANCE REQUIREMENTS:
   - Dataset Agreement: Use strictly in compliance with the VIRAT dataset research license.
   - Child-Safety Guard: VIRAT surveillance clips feature general public pedestrians.
     DO NOT perform automated demographic inference, face recognition, or child labeling
     on VIRAT footage. All tracks must remain UNCLASSIFIED in SafeKid AI.
================================================================================
""")


def inspect_dataset(virat_dir: Optional[str] = None) -> Dict[str, Any]:
    print("=" * 80)
    print("SAFEKID AI — VIRAT SURVEILLANCE DATASET INSPECTOR")
    print("=" * 80)

    loader = VIRATDatasetLoader(virat_dir)
    summary = loader.get_summary()

    print(f"Dataset Root: {summary['root_dir']}")
    print(f"Total Sequences Discovered: {summary['total_sequences_found']}")
    print(f"Complete Sequences (Video + Annotations): {summary['complete_sequences']}")
    print(f"Total Video Frames: {summary['total_video_frames']}")
    print(f"Total Annotated Frames: {summary['total_annotated_frames']}")
    print(f"Total Ground-Truth Person Detections: {summary['total_person_instances']}")
    print(f"Total Unique Person Tracks: {summary['total_unique_person_tracks']}")

    if summary["total_sequences_found"] == 0 or summary["complete_sequences"] == 0:
        print_acquisition_checklist(loader)
    else:
        print("\n" + "-" * 90)
        print("SEQUENCE-LEVEL INVENTORY:")
        print("-" * 90)
        print(f"{'Sequence ID':<25} {'Tier':<10} {'Resolution':<12} {'Frames':<8} {'FPS':<6} {'Persons':<10} {'Tracks':<8} {'Status':<10}")
        print("-" * 90)

        for seq in summary["sequences"]:
            status = "COMPLETE" if seq["is_complete"] else ("NO-VIDEO" if not seq["has_video"] else "NO-ANNOT")
            res = f"{seq['width']}x{seq['height']}" if seq['width'] > 0 else "N/A"
            tier = seq.get("tier", "fixture").upper()
            print(
                f"{seq['sequence_id']:<25} {tier:<10} {res:<12} {seq['total_frames']:<8} "
                f"{seq['fps']:<6.1f} {seq['person_instance_count']:<10} {seq['unique_person_tracks']:<8} {status:<10}"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect local VIRAT dataset files and metadata.")
    parser.add_argument("--dir", type=str, default=None, help="Custom path to VIRAT dataset directory.")
    parser.add_argument("--json", action="store_true", help="Output summary in JSON format.")
    parser.add_argument("--validate", action="store_true", help="Run full integrity validation and export JSON reports.")
    args = parser.parse_args()

    summary = inspect_dataset(args.dir)
    if args.validate:
        from safekid_ai.datasets.virat_validator import VIRATDatasetValidator
        loader = VIRATDatasetLoader(args.dir)
        val = VIRATDatasetValidator(loader)
        inv_path, val_path = val.generate_inventory_and_validation_reports(Path("outputs"))
        print(f"\n[Generated Inventory Report] {inv_path}")
        print(f"[Generated Validation Report] {val_path}")

    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
