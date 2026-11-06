# SafeKid AI: Child Safety Monitoring System for Gated Communities

**SafeKid AI** is an edge-optimized computer vision and safety intelligence pipeline engineered specifically for **gated residential communities** (residential layouts, apartment enclaves, gated villa developments). It monitors high-risk perimeters—including swimming pool gates, vehicular entrance/exit boom barriers, and playground boundaries—to safeguard unaccompanied children while upholding child privacy and ethical AI principles.

---

## 1. System Architecture & Modular Design

The pipeline operates via an asynchronous, decoupled multi-stage architecture:

```
[Camera / Video Stream]
         │
         ▼
 1. VideoInput (OpenCV Ingestion, Resizing, Hardware Decoupling)
         │
         ▼
 2. PersonDetector Adapter (Modular: YOLOv8 Baseline | D-FINE Candidate | RT-DETR Experimental)
         │
         ▼
 3. PersonTracker (ByteTrack Multi-Object Tracking with Kalman Trajectory Vectors)
         │
         ▼
 4. ROIEngine (Configurable Camera Polygons & Ground-Contact Bottom-Center Evaluation)
         │
         ▼
 5. ZoneStateManager (Entry, Exit, Dwell-Time Exceeded Events)
         │
         ▼
 6. ChildSafetyEventEngine (Dedicated Child Classification & Adult Exclusion Layer)
         │
         ▼
 7. IncidentManager (Alert Lifecycle, Correlation, Debouncing, Disconnection Safeguards)
         │
         ▼
 8. Visualization (Alpha-Blended HUD, Ground-Contact Markers, Safety Alerts)
```

### Key Architectural Tenets
1. **Zero Hardcoded Coordinates:** Camera polygon boundaries are loaded strictly from external JSON configurations ([`configs/cameras/`](configs/cameras/)).
2. **Ground-Contact Geometry:** Occupancy is evaluated using the bottom-center contact point $(x_{\text{bc}}, y_{\text{bc}}) = (\frac{x_1+x_2}{2}, y_2)$ rather than bounding box overlap, preventing false alarms from individuals standing outside fenced areas.
3. **Pluggable Detector Backend:** Detectors adhere to the [`BasePersonDetector`](src/safekid_ai/detectors/base.py) abstract interface and can be hot-swapped via configuration or the [`create_detector()`](src/safekid_ai/detectors/__init__.py) factory.
4. **Dedicated Child Safety Intelligence:** Generic human detection is strictly decoupled from child safety classification. Downstream child classification ([`ChildDetector`](src/safekid_ai/child_detection/detector.py)) utilizes multi-gate abstentions with temporal smoothing, classifying tracks into `CHILD`, `ADULT`, or `UNKNOWN`.

---

## 2. Directory Structure & Project Inventory

```text
SafeKid AI/
├── configs/                          # System configurations
│   ├── child_detector.yaml           # Child classification hyperparameters
│   ├── annotations/                  # Ground-truth track annotation JSONs
│   ├── cameras/                      # Camera-specific polygon ROI configurations
│   ├── detectors/                    # Modular detector configuration templates
│   └── trackers/                     # ByteTrack tracker configurations
├── data/                             # Datasets and synthetic reference fixtures
│   ├── child_dataset/fixtures/       # 200 balanced pedestrian crops + annotations.json
│   ├── demo/                         # 4 synthetic surveillance demonstration videos
│   ├── sample_benchmark/             # Offline validation crops
│   └── virat/                        # VIRAT Release 2.0 reference fixture & schema
├── dfine_repo/                       # Upstream D-FINE (ICLR 2025 Spotlight) repository
├── docs/                             # Data governance and acquisition guides
├── models/                           # Trained and pretrained model checkpoints
│   ├── child_classifier_v1.pt        # Trained child/adult classifier network (0.23M params)
│   ├── dfine_n_coco.pth              # Official D-FINE-N weights (3.81M params)
│   └── dfine_s_coco.pth              # Official D-FINE-S weights (10.39M params)
├── outputs/                          # Milestone reports, benchmark CSVs, JSONs & contact sheets
├── runs/                             # Execution summaries, confusion matrices, scatter plots
├── scripts/                          # 23 evaluation, benchmarking, and demo runners
├── src/safekid_ai/                   # Core Python package
│   ├── child_detection/              # Dedicated child classification subsystem
│   ├── datasets/                     # VIRAT dataset loader and evaluator
│   ├── detectors/                    # Base interface and detector adapters (YOLOv8, D-FINE, RT-DETR)
│   ├── roi/                          # Camera-specific polygon ROI engine
│   ├── benchmarking/                 # Offline evaluation metrics and plots
│   ├── utils/                        # Centralized structured logging
│   ├── detector.py                   # Core DetectionBox and FrameDetections containers
│   ├── tracker.py                    # ByteTrack tracking integration
│   ├── events.py                     # Safety event definitions and ZoneStateManager
│   ├── child_safety.py               # ChildSafetyEventEngine and dwell monitoring
│   ├── incidents.py                  # IncidentManager alert lifecycle correlation
│   ├── pipeline.py                   # Unified SafeKidPipeline orchestrator
│   └── visualization.py              # OpenCV visual overlays and HUD
├── tests/                            # Complete pytest test suite (21 test files, 203 tests)
├── weight/                           # Pretrained backbone weights (HGNetv2-B0)
├── yolov8n.pt                        # Selected baseline person detector (3.15M params)
├── rtdetr-l.pt                       # Audited comparative model (32.97M params)
├── pyproject.toml                    # Editable build and dependency specification
├── requirements.txt                  # Python 3.11 pip dependency list
└── README.md                         # Project documentation
```

---

## 3. Installation & Environment Setup

### Prerequisites
- **Operating System:** Windows 10/11, Ubuntu 22.04+, or macOS
- **Python:** 3.11.x (Strictly recommended)
- **Hardware Profile:** Optimized for CPU-only execution (Intel Core i3 or equivalent edge gateway)

### Step-by-Step Setup

```bash
# 1. Clone or navigate to the repository root
cd "SafeKid AI"

# 2. Create and activate a Python 3.11 virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# 4. Install all required dependencies
pip install -r requirements.txt

# 5. Install SafeKid AI in editable development mode
pip install -e .
```

---

## 4. Configuration

SafeKid AI uses structured YAML and JSON configuration files located under `configs/`:

### 4.1 Camera ROI JSON (`configs/cameras/community_001/camera_pool_01.json`)
Defines the camera view, resolution, and monitored safety zones:
```json
{
  "community_id": "community_001",
  "camera_id": "camera_pool_01",
  "camera_name": "Swimming Pool Deep End & Perimeter Gate",
  "reference_resolution": [1920, 1080],
  "zones": [
    {
      "zone_id": "pool_basin_critical",
      "zone_name": "Pool Water Basin",
      "zone_type": "swimming_pool_gate",
      "severity": "CRITICAL",
      "polygon": [[520, 680], [1400, 680], [1520, 1040], [400, 1040]],
      "dwell_time_seconds": 0.0,
      "requires_child_confirmation": true
    }
  ]
}
```

### 4.2 Detector Backend Configuration (`configs/detectors/`)
Choose between available backends:
- `yolov8_baseline.yaml`: Production CNN baseline (fastest, zero false positives).
- `dfine_n_candidate.yaml`: Primary Transformer candidate (ICLR 2025 Spotlight, 3.81M params).
- `dfine_s_candidate.yaml`: High-capacity Transformer candidate (10.39M params).
- `rtdetrv2_candidate.yaml`: Audited experimental Transformer model (RT-DETR-L v1, 32.97M params).

---

## 5. How to Select the Detector Backend

You can select the detector backend programmatically in Python or via configuration:

### Programmatic Selection via Factory
```python
import cv2
from safekid_ai.detectors import create_detector

# 1. Instantiate selected detector backend:
detector = create_detector("yolov8")       # Production YOLOv8n baseline
# detector = create_detector("dfine_n")    # D-FINE Nano candidate
# detector = create_detector("dfine_s")    # D-FINE Small candidate
# detector = create_detector("rtdetr")     # RT-DETR-L experimental model

# 2. Run detection on an OpenCV BGR frame:
frame = cv2.imread("sample.jpg")
detections = detector.detect(frame)
print(f"Detected {len(detections)} persons.")

# 3. Run detection + ByteTrack multi-object tracking:
tracked_persons = detector.track(frame, frame_id=1, timestamp_ms=0.0)
for p in tracked_persons:
    print(f"Track ID: {p.track_id}, BBox: {p.bbox_xyxy}, Conf: {p.confidence:.2f}")
```

### Pipeline Integration via Configuration
```python
from safekid_ai.pipeline import SafeKidPipeline
from safekid_ai.config import SafeKidPipelineConfig, DetectorConfig

config = SafeKidPipelineConfig(
    community_id="community_001",
    camera_id="camera_pool_01",
    roi_config_path="configs/cameras/community_001/camera_pool_01.json",
    detector=DetectorConfig(backend="dfine", model_variant="dfine_n"),
    enable_automated_child_detection=True
)

pipeline = SafeKidPipeline(config)
```

---

## 6. How to Run Demonstrations & Simulations

SafeKid AI includes ready-to-run demo scripts that execute complete detection, tracking, ROI monitoring, and child-safety alert pipelines:

```bash
# Run the Swimming Pool Safety Demo (Single Camera):
python scripts/run_pool_safety_demo.py

# Run the Multi-Environment Simulation (Pool, Playground, Driveway, Boom Barrier):
python scripts/run_multi_environment_simulation.py

# Run the Real-Time Community Video Monitoring Demo:
python scripts/run_demo.py --video data/demo/pool_safety_demo.mp4 --camera-config configs/cameras/community_001/camera_pool_01.json
```

---

## 7. How to Execute Comparative Benchmarks

Reproduce all empirical research studies published in Prompts 17–24:

```bash
# 1. Execute Four-Way Comparative Study (YOLOv8n vs RT-DETR-L vs D-FINE-N vs D-FINE-S):
python scripts/run_prompt_24_comparative_study.py

# 2. Execute RT-DETR vs YOLOv8 Study (Prompt 23):
python scripts/run_prompt_23_comparative_study.py

# 3. Execute VIRAT Dataset Benchmark (Prompt 18/20):
python scripts/run_virat_benchmark.py

# 4. Execute Dedicated Child Detector Evaluation (Prompt 22):
python scripts/evaluate_child_detector.py
```

Benchmark results, tabular CSVs, JSON metrics, and visual comparison contact sheets are automatically exported to `outputs/`.

---

## 8. Empirical Benchmark Summary (Prompt 24 Four-Way Study)

*Evaluated on 72 identical stationary surveillance frames (`VIRAT_S_000001_sample.mp4`) at 640x640 resolution, $\tau=0.35$, $\text{IoU}=0.45$, Intel Core i3-1215U CPU (4 threads).*

| Metric | YOLOv8n (Selected) | RT-DETR-L (Audited) | D-FINE-N (Candidate) | D-FINE-S (Candidate) |
| :--- | :---: | :---: | :---: | :---: |
| **Architecture** | CNN | Hybrid Transformer | FDR-DETR | FDR-DETR |
| **Parameters** | **3.15M** | 32.97M | 3.81M | 10.39M |
| **Checkpoint Size** | **6.23 MB** | 63.43 MB | 14.77 MB | 39.90 MB |
| **Mean Latency (ms)** | **109.48 ms** | 1,426.62 ms | **268.71 ms** | 552.04 ms |
| **Throughput (FPS)** | **9.13 FPS** | 0.70 FPS | **3.72 FPS** | 1.81 FPS |
| **Relative Speedup vs RT-DETR-L**| 13.0x | 1.0x (Baseline) | **5.31x** | 2.58x |
| **Precision** | **1.0000 (100%)** | 0.6667 (66.7%) | 0.1724 (17.2%) | 0.1316 (13.2%) |
| **Recall** | **0.2778 (27.8%)** | 0.0417 (4.2%) | 0.0347 (3.5%) | 0.0347 (3.5%) |
| **F1-Score** | **0.4348** | 0.0784 | 0.0578 | 0.0549 |
| **False Positives (FP)** | **0** | 3 | 24 | 33 |
| **ByteTrack MOTA** | **+0.2569** | 0.0000 | **-0.1389** | **-0.1528** |

### Key Takeaway
While **D-FINE-N** is a major architectural breakthrough that cuts Transformer latency by **5.3x** over RT-DETR-L, **YOLOv8n remains the selected production detector** due to its superior throughput (9.13 FPS), zero background false alarms, and positive tracking MOTA (+0.2569) on edge CPU hardware.

---

## 9. How to Run the Test Suite

SafeKid AI maintains a comprehensive unit, integration, and reliability test suite covering failure injection, polygon geometries, alert debouncing, and model adapters:

```bash
# Run the complete test suite:
pytest tests/

# Run with verbose output:
pytest -v tests/

# Run a specific subsystem test suite:
pytest tests/test_dfine_detector.py
pytest tests/test_child_safety.py
pytest tests/test_roi_engine.py
```

**Current Test Status:** `203 passed, 7 warnings in 165.48s (100% pass rate)`.

---

## 10. Model Checkpoint Sources & Licenses

| Model | Path | Origin / Source | License |
| :--- | :--- | :--- | :--- |
| **YOLOv8n** | `yolov8n.pt` | Official Ultralytics Release | AGPL-3.0 |
| **RT-DETR-L (v1)** | `rtdetr-l.pt` | Baidu PaddleDetection / Ultralytics | Apache-2.0 |
| **D-FINE-N** | `models/dfine_n_coco.pth` | Peterande/D-FINE (ICLR 2025 Spotlight) | Apache-2.0 |
| **D-FINE-S** | `models/dfine_s_coco.pth` | Peterande/D-FINE (ICLR 2025 Spotlight) | Apache-2.0 |
| **Child Classifier** | `models/child_classifier_v1.pt` | SafeKid AI Trained Architecture (`ChildAdultClassifierNet`) | Proprietary |
| **HGNetv2 Backbone** | `weight/hgnetv2/PPHGNetV2_B0_stage1.pth` | PaddleClas / Peterande | Apache-2.0 |

---

## 11. Dataset Provenance, Privacy & Research Ethics

> [!IMPORTANT]
> **Dataset Provenance & Research Disclosure:**
> 1. **Synthetic Reference Fixtures:** The video sequences evaluated in `data/demo/` and `data/virat/` are **locally generated synthetic reference fixtures** adhering to official VIRAT Release 2.0 schemas. They are designed for reproducible algorithmic verification.
> 2. **Unvalidated Real-World Residential Surveillance:** Real-world performance in live residential gated communities remains **unmeasured**. SafeKid AI must not be deployed in real residential facilities until genuine, authorized surveillance video is captured and evaluated under formal institutional review (IRB) and community consent.
> 3. **Child Privacy & Ethical AI Safeguards:**
>    - SafeKid AI **does not perform biometric facial recognition** or child identity tracking.
>    - Adult exclusion: Adults detected in swimming pool or gate areas are logged as situational guardians; alerts are triggered exclusively when an unaccompanied child enters a restricted hazard zone.
>    - Abstention Gating: Distant or degraded pedestrian crops are classified as `UNKNOWN`, preventing false child safety alarms.

---

*SafeKid AI is maintained by the SafeKid AI Core Engineering Team.*
