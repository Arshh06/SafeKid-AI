# VIRAT Video Dataset Release 2.0 — Acquisition & Verification Guide

**Project:** SafeKid AI — AI-Based Child Safety Monitoring for Gated Residential Communities  
**Scope:** Dataset Acquisition, Directory Hierarchy, Sequence Synchronization & Integrity Verification  

---

## 1. Overview & Dataset Governance

The **VIRAT Video Dataset (Release 2.0 Ground Dataset)** is a realistic high-resolution surveillance benchmark collected from elevated stationary cameras in real-world facilities, parking lots, and building perimeters.

### Research Licensing & Permitted Usage:
- **Distribution Agreement:** Public academic research dataset distributed under the VIRAT dataset agreement for non-commercial computer vision evaluation.
- **Privacy & Ethical Mandate:** The footage contains public individuals in surveillance environments without prior demographic consent.
- **SafeKid AI Policy Invariant:**
  - Automated age estimation or child classification on unconsented surveillance footage is strictly prohibited.
  - All surveillance tracks are unconditionally assigned `TrackClassification.UNCLASSIFIED`.
  - Zero child-safety alarms are promoted from unclassified surveillance tracks.

---

## 2. Acquisition Sources & Download Instructions

Because the official dataset archives are hosted behind academic distribution portals and require explicit terms acceptance, automatic programmatic scraping without access control exchange is prohibited.

Users must obtain the official files through one of the verified official repositories:

1. **Official Portal:** [https://viratdata.org/](https://viratdata.org/)
   - Access the *Download* section for Release 2.0 Ground Camera Subsets.
2. **Kitware Data Platform (Girder):**
   - Collection URL: `https://data.kitware.com/#collection/56f56db28d777f753209ba9f`
   - Documentation Folder: `VIRAT_Video_Dataset_Download_Instruction_Release2.pdf`
3. **Official Annotation Repository:**
   - [GitLab viratannotations](https://gitlab.com/viratdata/viratannotations)
   - IARPA DIVA annotation updates in Kitware Packet Format (KPF).

---

## 3. Required Files & Sequence Naming Conventions

### Video Streams:
- **File Format:** High-Definition `.mp4` (H.264/AVC encoded)
- **Native Resolution:** 1920×1080 (HD) or 1280×720
- **Frame Rate:** Typically 24.0 or 29.97 FPS
- **Sequence Naming:** `VIRAT_S_<scene_id>_<sequence_id>_<timestamp>.mp4` (e.g. `VIRAT_S_000002.mp4` or `VIRAT_S_010000_00_000000_000170.mp4`)

### Ground-Truth Annotation Files:
Each video sequence requires matching `.txt` annotation files:
1. `VIRAT_S_<sequence>.viratdata.objects.txt`:
   - 8-token spatio-temporal format: `<object_id> <duration> <current_frame> <x> <y> <width> <height> <object_type>`
   - Object Type Mapping: `1 = PERSON`, `2 = CAR`, `3 = OTHER_VEHICLE`, `4 = OTHER_OBJECT`, `5 = BIKE`
2. `VIRAT_S_<sequence>.viratdata.events.txt` (optional, for action recognition)
3. `VIRAT_S_<sequence>.viratdata.mapping.txt` (optional, for event-object linking)

---

## 4. Destination Directory Hierarchy

To ensure strict separation between synthetic validation fixtures and genuine public surveillance footage, organize files into the following directory tree:

```
c:\projs\SafeKid AI\data\virat\
│
├── genuine\                     <-- [GENUINE REAL FOOTAGE TIER]
│   ├── videos\
│   │   ├── VIRAT_S_000002.mp4
│   │   └── ...
│   ├── annotations\
│   │   ├── VIRAT_S_000002.viratdata.objects.txt
│   │   └── ...
│   └── metadata\
│
├── fixtures\                    <-- [SYNTHETIC REFERENCE FIXTURE TIER]
│   ├── videos\
│   │   └── VIRAT_S_000001_sample.mp4
│   └── annotations\
│       ├── VIRAT_S_000001_sample.viratdata.objects.txt
│       ├── VIRAT_S_000001_sample.viratdata.events.txt
│       └── VIRAT_S_000001_sample.viratdata.mapping.txt
│
├── videos\                      <-- [ROOT SCAN FALLBACK / BACKWARD COMPATIBILITY]
└── annotations\
```

---

## 5. Automated Validation & Integrity Check

Once you have placed `.mp4` video files and `.viratdata.objects.txt` files into `data/virat/genuine/`, run the SafeKid AI automated validation suite:

```powershell
# Scan and inspect discovered genuine and fixture sequences
python scripts/inspect_virat_dataset.py

# Run the validation suite to generate inventory and validation audit logs
python -c "from pathlib import Path; from safekid_ai.datasets.virat_loader import VIRATDatasetLoader; from safekid_ai.datasets.virat_validator import VIRATDatasetValidator; val = VIRATDatasetValidator(VIRATDatasetLoader()); val.generate_inventory_and_validation_reports(Path('outputs'))"
```

The validator verifies:
1. Video file container readability and decodability across multiple frame samples.
2. Bounding-box coordinate validity ($x \ge 0, y \ge 0, w > 0, h > 0, x+w \le W, y+h \le H$).
3. Temporal frame synchronization between video frame count and annotation max frame.
4. Absence of duplicate `(track_id, frame_id)` entries.
5. Export of `outputs/virat_dataset_inventory.json` and `outputs/virat_dataset_validation.json`.

