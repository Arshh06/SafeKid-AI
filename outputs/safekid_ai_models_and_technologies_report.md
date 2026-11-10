# SafeKid AI — Comprehensive Models & Technologies Master Report

**Document Identifier:** `outputs/safekid_ai_models_and_technologies_report.md`  
**Project:** SafeKid AI — AI-Based Child Safety Monitoring for Gated Residential Communities  
**Scope:** Complete technical registry of all artificial intelligence models, computer-vision algorithms, software libraries, web frameworks, hardware adapters, IoT devices, and dataset benchmarks utilized across the SafeKid AI repository.  
**Last Updated:** November 2026 (Reflecting Prompts 01 through 40)  

---

## 1. Executive Summary Table

| Category | Primary Technology / Model | Secondary / Comparative Models | Operational Role |
| :--- | :--- | :--- | :--- |
| **Object Detection** | **YOLOv8n** (`yolov8n.pt`) | **D-FINE-N**, **D-FINE-S**, **RT-DETR-L** | Real-time pedestrian localization on edge CPU |
| **Multi-Object Tracking** | **ByteTrack** (`bytetrack.yaml`) | Kalman Filter Trajectory Extrapolation | Continuous pedestrian trajectory association |
| **Child Classification** | **Child Classifier Net JAAD v1** (`models/child_classifier_jaad_v1.pt`) | Custom CNN v1 (`models/child_classifier_v1.pt`) | Autonomous `CHILD` vs `ADULT` vs `UNKNOWN` classification |
| **Demographic Research** | **MiVOLO** (ViT + YOLOv8x) | Disabled in operational mode | Ethical research benchmark age estimation |
| **Spatial Geofencing** | **Bottom-Center Foot-Contact Ray Casting** | Bounding box overlap (rejected for false positives) | Danger zone intrusion evaluation |
| **Incident Lifecycle** | **IncidentManager** State Machine | Observation decoupling (`TRACK_LOST`, `DISCONNECTED`) | Audit trail (`ACTIVE` -> `ACK` -> `RESOLVED`) |
| **Web Application Server** | **Werkzeug WSGI** (`safekid_ai.server`) | Asynchronous `VideoJobRunner` thread queue | REST API, video streaming, upload handling |
| **Frontend UI / UX** | **Vanilla ES6+ SPA & HTML5 Canvas** | Native 2D Canvas Charting Engine | Real-time control room dashboard & research portal |
| **External Alerting** | **Telegram Bot API** & **HTTP Webhooks** | `MockNotificationProvider` | Community security central messaging dispatch |
| **Hardware Actuation** | **ESP32 Microcontroller** (GPIO 18) | Piezo Buzzer / Siren (90dB) via HTTP REST | Physical audible alarm at gated hazard zones |
| **Datasets & Evaluation** | **JAAD** & **VIRAT Surveillance** | Pre-packaged residential CCTV demo fixtures | Controlled ground-truth validation |

---

## 2. Artificial Intelligence & Deep Learning Models

### 2.1. Object Detectors (Pedestrian Localization)

#### A. YOLOv8n — Primary Production Detector
- **Architecture**: Single-stage Anchor-Free Convolutional Neural Network (CNN).
  - **Backbone**: Modified CSPDarknet53 with Cross-Stage Partial bottleneck blocks (C2f modules).
  - **Neck**: Path Aggregation Network (PAN-FPN) multi-scale feature pyramid.
  - **Head**: Decoupled anchor-free classification and bounding-box regression heads with Task-Aligned Assigner.
- **Checkpoint File**: [`yolov8n.pt`](file:///c:/projs/SafeKid%20AI/yolov8n.pt) (6.23 MB)
- **Parameters**: 3,157,200 (3.16M)
- **Computational Complexity**: 8.7 GFLOPs (at 640×640 input resolution)
- **Inference Latency**: **106.43 ms** (mean) on Intel Core i3 edge CPU (~9.4 FPS throughput)
- **Key Empirical Benchmark Metrics** (Reconciled on 2,010 JAAD frames / 13,161 ground-truth annotations):
  - **Precision**: 87.80% (low false alarm rate: 0.42 FP/frame)
  - **Recall**: 46.64% overall (49.02% on children)
  - **F1-Score**: 60.92%
  - **AP@0.50**: 54.47% | **mAP@50:95**: 24.97%
- **Software Implementation**: [`src/safekid_ai/detector.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/detector.py) via `ultralytics` 8.1.0.

#### B. D-FINE-N — Primary Transformer Comparative Candidate
- **Architecture**: Fine-grained Distribution Refinement DEtection TRansformer (FDR-DETR; ICLR 2025 Spotlight).
  - **Backbone**: PP-HGNetV2-B0 lightweight hybrid CNN backbone.
  - **Encoder**: Efficient Hybrid Encoder with Intra-scale Feature Interaction (AIFI) and Cross-scale Feature Fusion (CCFM).
  - **Decoder**: Weight-Decoupled Query Selection with Dynamic Decoder Layers (DDL) and Global Boundary Regressor modeling bounding box coordinates as continuous probability distributions.
- **Checkpoint File**: [`models/dfine_n_coco.pth`](file:///c:/projs/SafeKid%20AI/models/dfine_n_coco.pth) (14.77 MB)
- **Pretrained Backbone**: [`weight/hgnetv2/PPHGNetV2_B0_stage1.pth`](file:///c:/projs/SafeKid%20AI/weight/hgnetv2/PPHGNetV2_B0_stage1.pth) (7.21 MB)
- **Parameters**: 3,809,074 (3.81M)
- **Computational Complexity**: 7.03 GFLOPs (at 640×640 input resolution)
- **Inference Latency**: **289.42 ms** (mean) on Intel Core i3 edge CPU (~3.45 FPS throughput)
- **Key Empirical Benchmark Metrics**:
  - **Precision**: 41.71% (higher background sensitivity: 1.83 FP/frame)
  - **Recall**: 56.60% overall (**+9.96% higher than YOLOv8n**)
  - **Small Pedestrian (<32px height) Recall**: **23.09%** (vs YOLOv8n's 3.27% — **7.06x higher recovery of distant figures**)
  - **F1-Score**: 48.03% | **AP@0.50**: 53.02% | **mAP@50:95**: 23.18%
- **Software Implementation**: [`src/safekid_ai/detectors/dfine_detector.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/detectors/dfine_detector.py) and upstream repository in [`dfine_repo/`](file:///c:/projs/SafeKid%20AI/dfine_repo/).

#### C. D-FINE-S — High-Capacity Transformer Baseline
- **Architecture**: FDR-DETR with PP-HGNetV2-B1 backbone.
- **Checkpoint File**: [`models/dfine_s_coco.pth`](file:///c:/projs/SafeKid%20AI/models/dfine_s_coco.pth) (39.90 MB)
- **Parameters**: 10,392,158 (10.39M) | **GFLOPs**: 24.87 GFLOPs.
- **Role**: Explored during research model scaling studies.

#### D. RT-DETR-L — Audited Hybrid Transformer Baseline
- **Architecture**: Baidu Real-Time DEtection TRansformer Large (v1) with ResNet-50 backbone.
- **Checkpoint File**: [`rtdetr-l.pt`](file:///c:/projs/SafeKid%20AI/rtdetr-l.pt) (63.43 MB)
- **Parameters**: 32,970,476 (32.97M) | **GFLOPs**: 110 GFLOPs.
- **Software Implementation**: [`src/safekid_ai/detectors/rtdetrv2_detector.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/detectors/rtdetrv2_detector.py).

---

### 2.2. Child Classification Networks

#### A. Child-Adult Classifier Net JAAD v1 (`models/child_classifier_jaad_v1.pt`)
- **Architecture**: Custom 4-stage Convolutional Neural Network:
  - Input: Normalized 128×64 RGB pedestrian crop (standard pedestrian aspect ratio 2:1).
  - Stage 1: Conv2d(3 -> 32, kernel=3, stride=1, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2×2).
  - Stage 2: Conv2d(32 -> 64, kernel=3, stride=1, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2×2).
  - Stage 3: Conv2d(64 -> 128, kernel=3, stride=1, pad=1) + BatchNorm2d + ReLU + MaxPool2d(2×2).
  - Stage 4: Conv2d(128 -> 256, kernel=3, stride=1, pad=1) + BatchNorm2d + ReLU + AdaptiveAvgPool2d((4, 2)).
  - Classifier Head: Flatten (2048) -> Linear(2048 -> 256) + ReLU + Dropout(p=0.4) -> Linear(256 -> 2) -> Softmax.
- **Training Provenance**: Trained on 11,480 extracted pedestrian crops from JAAD ground-truth video sequences with data augmentation (random horizontal flips, color jitter, affine scale perturbations).
- **Test Set Performance**: Precision 88.46%, Recall 86.88%, F1-Score 87.67%.
- **Safety Governance & Abstention**:
  - Automatically abstains to `UNKNOWN` when crop height < 36px, aspect ratio < 1.0 or > 4.8, or softmax probability margin < 0.20.
  - Software modules: [`src/safekid_ai/child_detection/classifier_net.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/child_detection/classifier_net.py) and [`detector.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/child_detection/detector.py).

#### B. Child-Adult Classifier Net v1 Baseline (`models/child_classifier_v1.pt`)
- **Checkpoint File**: [`models/child_classifier_v1.pt`](file:///c:/projs/SafeKid%20AI/models/child_classifier_v1.pt) (0.87 MB, 225,894 params). Initial prototype trained on synthetic and seed crop fixtures.

---

### 2.3. Research Demographic Module (Ethics-Restricted)

#### MiVOLO (Multi-Input Vision Transformer for Age & Gender Estimation)
- **Architecture**: Dual-branch Vision Transformer (ViT) taking face crops and person full-body crops simultaneously with Cross-Attention fusion.
- **Weights Repository**: `typorch/mivolo-imdb_cross_person-yolov8x_person_face` (`model_imdb_cross_person_4.22_99.46.pth.tar`).
- **Ethical Governance**: Strictly gated behind `allow_research_inference=True` and `consent_verified=True`. **Hard-disabled in community monitoring operational mode** to comply with privacy policies and avoid biometrics/facial recognition.
- **Software Implementation**: [`src/safekid_ai/age_estimator.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/age_estimator.py).

---

## 3. Computer Vision & Geofencing Algorithms

### 3.1. Multi-Object Tracking: ByteTrack
- **Algorithm**: ByteTrack associates bounding boxes by matching both high-score and low-score detection candidates through two-stage association using Kalman Filter motion prediction and Hungarian assignment.
- **Configuration**: [`configs/bytetrack.yaml`](file:///c:/projs/SafeKid%20AI/configs/bytetrack.yaml).
- **Benefits for Gated Communities**:
  - Overcomes temporary occlusions behind fences, bushes, or passing vehicles.
  - Maintains track identity across skipped keyframes without requiring heavy Re-ID neural embeddings.
  - Software implementation: [`src/safekid_ai/tracker.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/tracker.py).

### 3.2. Bottom-Center Ground Contact Foot Projection
- **Formula**:
  $$x_{\text{ground}} = \frac{x_1 + x_2}{2}, \quad y_{\text{ground}} = y_2$$
- **Purpose**: Eliminates false positives when a walking child's upper body, outstretched arms, or head lean over a fence boundary while their feet remain safely on the sidewalk.
- **Software implementation**: [`src/safekid_ai/roi/engine.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/roi/engine.py).

### 3.3. Ray-Casting Point-in-Polygon (PIP) Geofencing
- **Mathematical Principle**: Tests spatial inclusion by casting a semi-infinite ray along the positive x-axis and counting edge intersections (Jordan Curve Theorem).
- **Validation**: Enforces non-self-intersecting simple polygons; supports arbitrary concave boundaries for pool aprons, gate swing radii, and driveway bends.
- **Software implementation**: [`src/safekid_ai/roi/models.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/roi/models.py).

### 3.4. Temporal Zone State Buffering & Incident Lifecycle
- **ZoneStateManager** ([`src/safekid_ai/events.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/events.py)): 3-frame entry and exit confirmation buffers to suppress boundary jitter.
- **IncidentManager** ([`src/safekid_ai/incidents.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/incidents.py)): Correlates events into lifecycle states: `ACTIVE`, `ACKNOWLEDGED`, `RESOLVED`, `EXPIRED`.
- **Decoupled Observation**: States `OBSERVING`, `OCCLUDED`, `TRACK_LOST`, `DISCONNECTED` ensure camera disconnects are never confused with safe incident resolution.

---

## 4. Software Stack & Engineering Technologies

### 4.1. Core Runtime & Backend

| Technology / Library | Version | Role in SafeKid AI |
| :--- | :---: | :--- |
| **Python** | 3.11.9 (64-bit) | Primary development and runtime language |
| **PyTorch (`torch`)** | 2.15.0 | Deep learning framework, tensor compute, model evaluation |
| **`torchvision`** | 0.29.2 | Image transformations, crop normalization, dataset loaders |
| **OpenCV (`opencv-python`)** | 5.0.0.93 | Video container decoding, VideoWriter encoding, HUD rendering, geometric operations |
| **Ultralytics** | 8.1.0 | YOLOv8 model loading, layer fusion, anchor-free inference |
| **Werkzeug** | 3.1.9 | Production WSGI HTTP server, REST API routing, multipart uploads, static file server |
| **NumPy** | 2.4.6 | Fast array manipulation, coordinate math, spatial bounding box calculations |
| **Pillow (`PIL`)** | 13.0.0 | High-quality image crop extraction and snapshot serialization |
| **Pytest** | 9.2.0 | Automated unit, regression, and failure-injection test suite (270 passing tests) |
| **Thop & Calflops** | 0.1.1 / 0.3.2 | Computational complexity, parameter counting, and GFLOPs profiling |
| **Pandas & Scikit-learn** | 3.1.0 / 1.9.1 | Metric aggregation, confusion matrix calculation, demographic data tables |
| **Matplotlib & Seaborn** | 3.11.2 / 0.13.2 | Qualitative error visualization, PR curves, and contact sheets |

### 4.2. Asynchronous Job Architecture
- Multi-threaded worker queue implemented in [`src/safekid_ai/server/job_runner.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/server/job_runner.py).
- Offloads heavy computer-vision inference to background threads so video uploads never block operator web interactions.
- Produces atomic progress tracking (`progress_percent`, `current_frame`, `fps`, `incident_count`).

---

## 5. Web Frontend & Operator Interface

- **Pure ES6+ JavaScript**: Lightweight single-page application without heavy external frameworks (zero Node/npm build dependencies required).
- **Interactive Geofence Canvas**: Native HTML5 `<canvas>` in [`website/js/control_room.js`](file:///c:/projs/SafeKid%20AI/website/js/control_room.js) allowing operators to click on video frames and define polygonal danger boundaries.
- **Custom Charting Engine**: Lightweight canvas rendering engine ([`website/js/charts.js`](file:///c:/projs/SafeKid%20AI/website/js/charts.js)) rendering grouped bar charts, horizontal comparisons, and latency percentiles.
- **Modern Responsive Dark Theme**: Custom CSS ([`website/css/style.css`](file:///c:/projs/SafeKid%20AI/website/css/style.css)) using CSS Grid, Flexbox, and CSS Variables tailored for security control rooms.
- **REST API Endpoints**: 18 documented endpoints ([`outputs/prompt_40_api_and_security_report.md`](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_api_and_security_report.md)).

---

## 6. Hardware & External Integrations

### 6.1. ESP32 Physical Buzzer Device Adapter
- **Microcontroller Hardware**: Espressif ESP32 (NodeMCU / DevKit v1, Tensilica Xtensa 32-bit dual-core, 2.4 GHz Wi-Fi).
- **Physical Actuator**: 90dB Piezo / Magnetic Buzzer connected to GPIO 18 via a transistor driver circuit.
- **Communication Protocol**: HTTP REST (`GET /health`, `POST /buzzer/trigger`, `POST /buzzer/silence`).
- **Software Adapter**: [`src/safekid_ai/hardware/esp32_client.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/hardware/esp32_client.py).
- **Integrity Policy**: Automatically enforces a 10-second cooldown; gracefully marks status as `NOT CONNECTED` when offline; never fakes hardware activation.

### 6.2. Community Messaging Gateway
- **HTTP Webhook Adapter**: Generic JSON payload delivery to community Slack, Discord, or Mattermost security channels.
- **Telegram Bot API Adapter**: Direct HTML-formatted incident card delivery to private community Telegram groups via `https://api.telegram.org/bot<token>/sendMessage`.
- **Software Adapter**: [`src/safekid_ai/notifications/provider.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/notifications/provider.py) and [`service.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/notifications/service.py).
- **Security**: Tokens loaded securely via environment variables (`SAFEKID_WEBHOOK_URL`, `SAFEKID_TELEGRAM_BOT_TOKEN`, `SAFEKID_TELEGRAM_CHAT_ID`) and masked in API responses.

---

## 7. Datasets & Benchmarks

1. **JAAD (Joint Attention for Autonomous Driving)**:
   - 346 high-resolution (1920×1080) pedestrian video sequences.
   - 13,161 pedestrian ground-truth bounding box annotations with demographic age labels (`child`, `adult`).
2. **VIRAT Video Surveillance Dataset**:
   - High-angle elevated surveillance video fixtures for top-down perspective validation.
3. **Calibrated Residential CCTV Demo Fixtures**:
   - `pool_safety_demo.mp4` (848×480, 240 frames, swimming pool boundary)
   - `driveway_demo.mp4` (848×480, 240 frames, vehicle entrance driveway)
   - `playground_demo.mp4` (848×480, 240 frames, playground perimeter fence)
   - `boom_barrier_demo.mp4` (848×480, 240 frames, automated vehicular gate)

---

## 8. Directory Reference for Related Artifacts

- [Operational Application Architecture](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_operational_application_architecture.md)
- [Video Upload & Processing Report](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_video_upload_and_processing_report.md)
- [Hazard Zone Geofencing & Event Engine Report](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_hazard_zone_event_engine_report.md)
- [Community Notification Integration Report](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_notification_integration_report.md)
- [ESP32 Physical Buzzer Integration Report](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_esp32_buzzer_integration_report.md)
- [REST API & Security Governance Report](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_api_and_security_report.md)
- [End-to-End Test & Verification Report](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_end_to_end_test_report.md)
- [Hardware & Credential Readiness Inventory](file:///c:/projs/SafeKid%20AI/outputs/prompt_40_remaining_hardware_and_credentials.md)
- [Final Repository Audit & Checkpoint Inventory (Prompt 25)](file:///c:/projs/SafeKid%20AI/outputs/final_repository_audit.md)
- [YOLOv8 vs D-FINE Architectural Comparison (Prompt 33)](file:///c:/projs/SafeKid%20AI/outputs/prompt_33_architecture_comparison.md)

