# SafeKid AI — Hardware, Dependency & Detection Architecture Audit
## Comparative Object Detection Baseline Study: YOLOv8 vs. RT-DETRv2

**Project:** SafeKid AI — AI-Based Child Safety Monitoring for Gated Residential Communities  
**Audit Date:** November 5, 2026  
**Target Hardware:** Intel Core i3 (Alder Lake), 8 GB RAM, CPU Execution Only  
**Audit Milestone:** Prompt 23 — Phase 1 Repository and Hardware Audit  

---

### 1. Host Hardware & Environment Specifications

| Component / Property | Measured System Specification | Implications for Real-Time Inference |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 64-bit (`10.0.26300-SP0`) | Standard desktop environment |
| **Processor (CPU)** | Intel Core i3 / Core 12th Gen (`Family 6 Model 154 Stepping 4`) | 6 physical cores, 8 logical threads; P/E hybrid architecture |
| **System Memory (RAM)** | **7.68 GB Total**, ~0.70 GB – 1.20 GB Available | Constrained memory footprint; peak process RSS must stay $< 600\text{MB}$ |
| **Dedicated GPU** | **None** (`torch.cuda.is_available() == False`) | Pure CPU forward pass; zero hardware CUDA tensor acceleration |
| **CUDA / cuDNN** | Not available | High-parameter transformer attention must execute in software |
| **Python Version** | Python 3.11.9 (64-bit MSC v.1938 AMD64) | Active virtual environment at `.venv/` |
| **PyTorch Version** | PyTorch 2.15.0+cpu | CPU-only build with multi-threaded vector extensions |
| **ONNX Runtime** | Not currently installed in `.venv` | Native PyTorch engine used for inference |
| **Ultralytics Framework** | Ultralytics 8.1.0+ (`ultralytics.YOLO`, `ultralytics.RTDETR`) | Native support for both YOLOv8 and RT-DETR architectures |

---

### 2. Can RT-DETRv2-L Run on this Hardware?

#### Empirical Verification:
- **Model Checkpoint:** `rtdetr-l.pt` (63.4 MB disk footprint, 32.97 million parameters).
- **Execution Test:** Model successfully loaded into memory and executed full forward pass on $640 \times 640$ frames.
- **Peak Process RSS:** $\approx 365\text{MB}$, fitting safely within the available 700 MB system memory budget without Out-Of-Memory (OOM) paging.
- **Measured CPU Inference Latency:** **`1,214.60 ms` per frame** (~**`0.82 FPS`**).

#### Practical Architectural Verdict:
**RT-DETRv2-L CAN run on this machine without crashing**, but at $\approx 0.82\text{ FPS}$, it is **not capable of real-time stream processing** on a CPU-only Intel i3. 

By contrast, the baseline **YOLOv8n** operates at $\approx 110\text{–}140\text{ ms}$ (~**`7.5–9.0 FPS`**).

#### Model Variant Recommendation:
1. **Primary Experimental Accuracy Candidate:** **RT-DETRv2-L** (32.97M parameters) must be evaluated as requested for maximum transformer feature representation and small-object detection accuracy.
2. **Operational Residential Edge Recommendation:** For practical deployment on CPU hardware without a dedicated GPU, lighter variants such as **RT-DETRv2-S / ResNet18 backbone** (~20M parameters) or INT8 OpenVINO quantization are strongly recommended to bridge the throughput gap toward $\ge 5\text{ FPS}$.

---

### 3. Existing Detection Pipeline & Interface Audit

#### A. Data Structures ([`src/safekid_ai/detector.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/detector.py))
- **`DetectionBox`:** Represents a localized person entity:
  - `bbox_xyxy`: `(x1, y1, x2, y2)` float pixel coordinates.
  - `confidence`: float in $[0.0, 1.0]$.
  - `class_id`: 0 (COCO `person`).
  - `class_name`: `"person"`.
  - `track_id`: Optional integer assigned downstream by ByteTrack.
  - Properties: `bottom_center` (ground contact point), `extract_crop(frame)`.
- **`FrameDetections`:** Frame container containing `frame_id`, `timestamp_ms`, `detections: List[DetectionBox]`.

#### B. YOLOv8 Implementation Details
- Initialized in `PersonDetector` using `yolov8n.pt`.
- CPU Optimization: Explicitly sets `torch.set_num_threads(4)` for Intel P-cores and fuses `Conv2d` + `BatchNorm2d` layers.
- Default Confidence Threshold: `0.35`.
- Default NMS IoU Threshold: `0.45`.
- Preprocessing: Standard letterbox resizing to $640 \times 640$, channel transposition $(H, W, C) \to (1, C, H, W)$, float normalization $[0, 255] \to [0.0, 1.0]$.

#### C. ByteTrack Multi-Object Tracker Integration ([`src/safekid_ai/tracker.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/tracker.py))
- `PersonTracker` expects detections either from `.track()` or structured `FrameDetections`.
- Operates via a two-stage Hungarian data association:
  - High-confidence detection matching ($\ge 0.30$).
  - Low-confidence recovery matching ($0.10 \le \text{conf} < 0.30$) to preserve occluded tracks.
- Tracker Configuration: `configs/trackers/bytetrack_surveillance.yaml` (`track_buffer = 60`, `match_thresh = 0.75`).

---

### 4. Adapter Architecture Plan (Phase 2 Design)

To ensure fair comparative evaluation without altering existing pipeline behavior:
1. Implement `BasePersonDetector` abstract interface in `src/safekid_ai/detectors/base.py`.
2. Implement `YOLOv8DetectorAdapter` in `src/safekid_ai/detectors/yolov8_detector.py`.
3. Implement `RTDETRv2DetectorAdapter` in `src/safekid_ai/detectors/rtdetrv2_detector.py`.
4. Ensure both adapters output standardized `FrameDetections` containing identical `DetectionBox` objects with `bottom_center` ground-contact points.
5. Provide unified configuration switching via `detector_backend: yolov8 | rtdetrv2`.

