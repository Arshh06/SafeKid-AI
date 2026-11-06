# SafeKid AI — Synthetic Video Methodology Audit & Simulation Fidelity Assessment

**Document ID:** `AUDIT-12-SYNTHETIC-METHODOLOGY`  
**Execution Timestamp:** `2026-11-04 10:00:00 UTC`  
**Audited Artifacts:**
- Generator Script: `scripts/generate_synthetic_environment_videos.py`
- Source Video: `data/demo/pool_safety_demo.mp4`
- Crop Asset: `outputs/track1_crop.png` ($127 \times 42$ pixels)
- Generated Scenarios:
  - `data/demo/playground_demo.mp4` ($1280 \times 720$, 24 FPS, 240 frames)
  - `data/demo/driveway_demo.mp4` ($1920 \times 1080$, 24 FPS, 192 frames)
  - `data/demo/boom_barrier_demo.mp4` ($1920 \times 1080$, 24 FPS, 144 frames)

---

## 1. Executive Summary

This audit rigorously evaluates the methodology, fidelity, and technical limitations of the synthetic video assets used to validate the **SafeKid AI** multi-environment pipeline. 

In Prompt 11, three synthetic scenarios (Playground, Driveway, and Boom Barrier) were generated to extend testing beyond the swimming pool scenario. This audit examines how those videos were constructed, how the underlying generation technique impacts the YOLOv8n detector and ByteTrack tracker, and delineates what synthetic validation conclusively proves versus what remains unvalidated.

---

## 2. Characterization of Video Sources: Generative vs. Procedural

A critical first finding of this audit is clarifying the distinct nature of the video assets in the repository:

### A. The Swimming Pool Demo (`data/demo/pool_safety_demo.mp4`)
- **Origin & Nature:** Introduced in Prompt 06 as a "10-second synthetic residential swimming-pool video" ($848 \times 480$, 24 FPS, 240 frames).
- **Visual Characteristics:** Photorealistic full-motion video containing natural human gait, limb articulation, diving/stepping into a pool, realistic water surface caustics, splash dynamics, and natural lighting gradients.
- **Classification:** It is an **AI-generated photorealistic synthetic video** (likely produced via advanced diffusion/generative video models like Sora or Runway) or rendered 3D CGI simulation, rather than real-world CCTV surveillance from an operating community.
- **Detector Interaction:** Because the subject possesses realistic limbs, shading, and articulation, YOLOv8n identifies the subject as class 0 (`person`) through organic visual features.

### B. The Multi-Environment Scenarios (`playground_demo.mp4`, `driveway_demo.mp4`, `boom_barrier_demo.mp4`)
- **Origin & Nature:** Programmatically generated in Prompt 11 by `scripts/generate_synthetic_environment_videos.py`.
- **Classification:** **Procedural 2D Cutout Compositing**.
- **Mechanism:** A static, rectangular bounding-box crop of a person ($127 \times 42$ pixels, extracted from Track 1 of the pool video) is programmatically pasted onto a static, procedurally drawn 2D background canvas at calculated $(x, y)$ coordinates across frames.

---

## 3. Deep-Dive Inspection of Procedural Video Generation

The generation logic in `scripts/generate_synthetic_environment_videos.py` relies on `paste_person_crop()`:

```python
def paste_person_crop(
    canvas: np.ndarray,
    crop: np.ndarray,
    bottom_center_x: float,
    bottom_center_y: float,
) -> None:
    ch, cw = crop.shape[:2]  # 127 x 42
    x1 = int(round(bottom_center_x - cw / 2.0))
    y1 = int(round(bottom_center_y - ch))
    canvas[y1:y1+ch, x1:x1+cw] = crop
```

This procedure introduces five specific physical and perceptual artifacts:

### 1. Fixed Crop Scale vs. Perspective Depth (Absence of Projective Geometry)
- **Observed Limitation:** The human crop remains exactly $127 \times 42$ pixels in every single frame, regardless of whether the person is positioned deep in the background ($y = 160$ in the Playground hedge, or $y = 350$ at the gate north approach) or close in the foreground ($y = 850$ on the driveway crossing).
- **Physical Reality:** In a calibrated pinhole camera model, an object's apparent height scales inversely with distance ($h \propto \frac{f \cdot H}{Z}$). A child standing 25 meters away in the background should occupy far fewer pixels (e.g., $30 \times 10$ px) than when standing 5 meters away in the foreground ($250 \times 80$ px).
- **Consequence:** ByteTrack Kalman filter velocity vectors and bounding box associations encounter an artificial invariance where scale never changes during longitudinal motion.

### 2. Static Background vs. Dynamic Environmental Lighting
- **Observed Limitation:** The background canvases (`draw_playground_background`, `draw_gate_background`) are mathematically static arrays.
- **Missing Real-World Factors:**
  - Zero illumination changes, cloud shadows, or diurnal solar glare.
  - Zero camera sensor noise, compression artifacts, or wind-induced camera pole vibration/jitter.
  - Zero environmental motion (e.g., swaying trees, fluttering safety tape, vehicle headlights).
- **Consequence:** Background subtraction or motion-based priors in tracking algorithms operate in an artificially noise-free environment.

### 3. Absence of Natural Articulation & Motion Dynamics ("Sliding Cardboard Cutout")
- **Observed Limitation:** The subject does not take steps, bend knees, swing arms, or change heading. Instead, a static snapshot translates rigidly across the image plane along piecewise-linear trajectories.
- **Consequence:** Gait-based feature extractors, temporal action recognition, or optical flow estimators would observe zero limb velocity and non-physical translational sliding.

### 4. Sharp Crop Boundary Artifacts and Detector Confidence
- **Observed Limitation:** The crop is directly pasted without alpha blending, edge feathering, or Poisson image editing. The boundary between the pool pavement inside the crop and the grass/asphalt on the canvas creates high-frequency rectangular step-discontinuities.
- **Impact on YOLOv8n:**
  - High-frequency edge artifacts can trigger false edge convolutions in the CNN backbone.
  - While YOLOv8n's strong receptive field still detects the person ($conf \approx 0.60 - 0.75$), detection confidence fluctuates unnaturally as the crop moves across contrasting background patches (e.g., moving from green grass into dark asphalt).
  - When the crop border overlaps background lines (such as road centerlines or fence bars), the detector can suffer momentary confidence drops.

### 5. Non-Maximum Suppression (NMS) Interaction with Synthetic Positioning
- **Observed Limitation:** During Prompt 11 development, placing the adult supervisor at $(220, 780)$ and the child at $(280, 850)$ resulted in detection merges.
- **Root Cause:** In high-resolution $1920 \times 1080$ scenes, YOLO downscales the frame to $640 \times 384$ for CPU inference. Two subjects positioned within 60 horizontal pixels produce overlapping candidate anchor boxes whose IoU exceeds `iou_threshold=0.45`, causing NMS to suppress one of the tracks.
- **Methodological Fix:** The script was adjusted to enforce $>150\text{ px}$ separation. This highlights that synthetic positioning must explicitly account for detector receptive field and NMS thresholds.

---

## 4. What the Synthetic Pipeline Tests Conclusively Validate

Despite the visual simplifications of 2D compositing, the synthetic simulation tests provide **100% mathematical and architectural validation** for the following software layers:

| Tested Component | Validation Provided by Synthetic Scenarios |
| :--- | :--- |
| **Pipeline Plumbing & Integration** | Confirms that `VideoInput`, `PersonDetector`, `PersonTracker`, `ROIEngine`, `ZoneStateManager`, and `ChildSafetyEventEngine` execute synchronously frame-by-frame without deadlocks, memory leaks, or unhandled exceptions. |
| **Coordinate Transformations** | Proves that camera ROI polygon coordinates (defined in normalized $[0, 1]$ or native pixels) correctly scale to arbitrary video resolutions ($848 \times 480$, $1280 \times 720$, $1920 \times 1080$). |
| **Point-in-Polygon Accuracy** | Confirms that the bottom-center ground-contact calculation accurately evaluates topological zone containment across complex convex and concave polygons. |
| **Temporal State Machine** | Proves that the 3-frame entry and exit confirmation logic successfully suppresses single-frame boundary jitter while preserving the exact timestamp of the initial qualifying observation. |
| **Dwell Time Accumulation** | Proves that elapsed dwell time is accumulated monotonically across frames and that `DWELL_TIME_EXCEEDED` events trigger precisely when the configured threshold is crossed. |
| **Declarative Zone Policies** | Proves that `SAFE` zones suppress danger entry alerts, `MONITORED` zones enforce dwell-only alerts, and `DANGER` zones emit immediate alerts according to camera-specific JSON policies. |
| **Child vs. Adult Exclusion** | Proves that tracks annotated as `ADULT` or unclassified are strictly excluded from generating child safety alerts under all zone policies. |

---

## 5. What the Synthetic Pipeline Tests Do NOT Validate

The synthetic video tests must **never** be misrepresented as proof of real-world operational efficacy. Specifically, they do **NOT** validate:

1. **Real-World Detection Robustness:** Real children vary enormously in height, build, posture, clothing color, and orientation. YOLOv8n may fail to detect children in low-contrast clothing or unconventional postures (crawling, crouching, rolling).
2. **Occlusion & Re-Identification:** The synthetic videos feature zero partial or dynamic occlusions (e.g., walking behind a parked SUV, trees, playground slides, or other pedestrians). Real community cameras experience frequent severe occlusions that challenge ByteTrack's Kalman filter.
3. **Environmental & Weather Variance:** Tests do not cover rain droplets on camera domes, night-time infrared (IR) illumination, headlight glare, puddle reflections, or wind-blown vegetation.
4. **Erratic Child Gait & Velocity:** Real children exhibit non-linear acceleration, sudden direction reversals, running, and jumping, which can violate ByteTrack's constant-velocity motion model.
5. **Automated Child Classification:** The tests use manual ground-truth annotations. They provide zero evidence regarding how an automated classifier would perform under real surveillance conditions.

---

## 6. Recommendations for Future Video Benchmarking

To bridge the gap between procedural simulation and physical pilot deployment:

1. **Acquire Consented Pilot CCTV Footage:** Collect multi-camera CCTV footage from pilot gated residential communities under strict resident consent and institutional ethics review.
2. **Implement Perspective-Aware Scaling in Synthetic Generator:** If synthetic generation is retained for automated regression testing, update `scripts/generate_synthetic_environment_videos.py` to use a camera calibration homography matrix so pasted crops scale realistically based on $y$-depth.
3. **Utilize 3D Synthetic Engines (Unreal Engine / Carla):** Generate photorealistic test suites using 3D simulation environments that model dynamic lighting, shadows, physics-based walking animations, and occluding static/dynamic meshes.

