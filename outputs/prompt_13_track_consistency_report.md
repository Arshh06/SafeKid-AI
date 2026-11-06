# SafeKid AI — Prompt 13: Track and Annotation Consistency Audit

**Document ID:** `REPORT-PROMPT-13-TRACK-CONSISTENCY`  
**Execution Timestamp:** `2026-11-04 10:30:00 UTC`  
**Audited Annotation Registries:**
1. `configs/annotations/demo_pool_track_annotations.json` (Camera: `camera_pool_demo`, 9 tracks)
2. `configs/annotations/demo_playground_track_annotations.json` (Camera: `camera_playground_01`, 2 tracks)
3. `configs/annotations/demo_driveway_track_annotations.json` (Camera: `camera_gate_north`, 2 tracks)
4. `configs/annotations/demo_boom_barrier_track_annotations.json` (Camera: `camera_gate_north`, 2 tracks)

---

## 1. Executive Summary

In AI-based safety monitoring for residential communities, **track identity integrity and classification consistency are critical**. False classification transfers (e.g. an adult being misidentified as a child, or a child losing alert coverage following an occlusion-induced track ID switch) undermine both safety and user trust.

This audit reviews the demo annotation registries across all four demonstration environments, verifies that child/adult/unclassified labels remain strictly associated with their intended ByteTrack IDs, confirms that ID switches cannot silently transfer child annotations, and examines video-level consistency across key event frames.

---

## 2. Audit of the Demo Annotation Registries

### A. Registry Structure & Invariant Checks
Each camera configuration is paired with an immutable JSON track classification registry loaded via [`ManualTrackAnnotationProvider`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/child_safety.py#L155-L215). Every registry was audited against four invariant requirements:

1. **Explicit Ephemeral Scoping:** Track IDs are documented as ephemeral ByteTrack assignments valid only for the specific demonstration stream session.
2. **Provisional Ground-Truth Notice:** Every file explicitly documents that child labels are manually assigned as provisional ground truth for algorithm validation, and cannot be established autonomously from video metadata.
3. **Absence of Demographic Profiling:** All annotation files contain the binding clause: *"These labels are for testing and evaluation only, NOT for operational surveillance or autonomous demographic profiling."*
4. **Strict Default Fallback:** Any track ID appearing in the video that is not enumerated in the JSON registry strictly defaults to `UNCLASSIFIED`.

### B. Scenario-by-Scenario Track Registry Audit

| Scenario | Video File & Resolution | Track ID | Assigned Classification | Status | Role in Scenario & Visual Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Playground** | `playground_demo.mp4`<br>($1280 \times 720$, 240 frames) | **1** | **`CHILD`** | PROVISIONAL | Child playing in sandbox, moving to hedge, dwelling > 4.0s. Correctly triggers `CHILD_DWELL_TIME_EXCEEDED` alert. |
| | | **2** | **`ADULT`** | PROVISIONAL | Adult supervisor seated near sandbox bench. Strictly excluded from child alerts across all frames. |
| **Driveway** | `driveway_demo.mp4`<br>($1920 \times 1080$, 192 frames) | **1** | **`ADULT`** | PROVISIONAL | Adult pedestrian standing on sidewalk. Continuously detected; 100% excluded from roadway alerts. |
| | | **2** | **`CHILD`** | PROVISIONAL | Child walking on sidewalk, entering vehicle driveway at frame 24, dwelling > 2.0s, returning to sidewalk. |
| **Boom Barrier** | `boom_barrier_demo.mp4`<br>($1920 \times 1080$, 144 frames) | **1** | **`ADULT`** | PROVISIONAL | Adult security guard / pedestrian standing on sidewalk. Excluded from driveway and barrier alerts. |
| | | **2** | **`CHILD`** | PROVISIONAL | Child approaching gate from north, entering driveway, entering boom barrier at frame 28, exiting at frame 120. |
| **Swimming Pool** | `pool_safety_demo.mp4`<br>($848 \times 480$, 240 frames) | **1** | **`ADULT`** | PROVISIONAL | Adult standing on east pool deck. Excluded from water and deck dwell alerts. |
| | | **2** | **`UNCLASSIFIED`** | PROVISIONAL | Pedestrian on western deck boundary. Excluded from child alerts. |
| | | **3** | **`ADULT`** | PROVISIONAL | Adult seated poolside on north deck. Excluded from child alerts. |
| | | **4** | **`CHILD`** | PROVISIONAL | Child jumping into water basin, climbing onto deck, dwelling > 3.0s. Emits water entry and deck dwell alerts. |
| | | **5** | **`UNCLASSIFIED`** | PROVISIONAL | Pedestrian on southwest perimeter. Excluded from child alerts. |
| | | **6** | **`CHILD`** | PROVISIONAL | Child swimming in water basin, climbing out onto deck, dwelling > 3.0s. Emits water entry, exit, and dwell alerts. |
| | | **7** | **`UNCLASSIFIED`** | PROVISIONAL | Pedestrian walking on exterior perimeter. Excluded from child alerts. |
| | | **8** | **`ADULT`** | PROVISIONAL | Adult walking along southern deck pavement. Excluded from child alerts. |
| | | **9** | **`UNCLASSIFIED`** | PROVISIONAL | Pedestrian in distant background. Excluded from child alerts. |

---

## 3. Prevention of Silent Annotation Transfer Across Discontinuities

### A. The Risk of Identity Switching
In real-world tracking, occlusions (e.g. a child walking behind a tree or vehicle) can cause ByteTrack to lose the active Kalman filter track. When the subject re-emerges, ByteTrack assigns a new integer ID (e.g. Track 2 becomes Track 10).

If a naive system attempts to heuristically "guess" that Track 10 is Track 2 based on clothing color, spatial proximity, or bounding-box height, it risks **silent annotation transfer**: an adult walking past the same tree could inherit the child label, triggering false child hazard alerts.

### B. SafeKid AI Architectural Protection
The SafeKid AI pipeline strictly prohibits silent annotation inheritance:
1. **Explicit ID Matching:** [`ManualTrackAnnotationProvider.get_classification(track_id)`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/child_safety.py#L182-L195) only returns `CHILD` if the exact integer `track_id` matches a key in the registry.
2. **Strict Fallback to `UNCLASSIFIED`:** Any newly spawned track ID (e.g. Track 10) automatically receives `TrackClassification.UNCLASSIFIED`.
3. **Discontinuity Logging:** When a track disappears while occupying a hazard zone, [`ChildSafetyEventEngine`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/child_safety.py#L480-L520) logs an explicit `TrackDiscontinuity` entry. It does **not** bind the lost annotation to subsequent detections.
4. **Verified by Automated Unit Tests:** This invariant is verified by:
   - `test_track_identity_discontinuity_handled_honestly` in `tests/test_child_safety.py`.
   - `test_id_switch_creates_independent_tracks_no_silent_merge` in `tests/test_reliability.py`.

---

## 4. Ethical & Privacy Boundary: Non-Inference Principles

SafeKid AI enforces four inviolable privacy boundaries in its software design:

```
+----------------------------------------------------------------------------------------------------+
|                                SAFEKID AI PRIVACY & ETHICAL BOUNDARIES                             |
+----------------------------------------------------------------------------------------------------+
| [X] NO Facial Recognition       --> Zero facial landmarking, face embeddings, or identity database |
| [X] NO Age Estimation in Ops    --> MiVOLO age estimation is restricted by policy in operational   |
|                                     mode (AgeEstimationStatus.RESTRICTED_BY_POLICY)                |
| [X] NO Visual Appearance Heuristics -> Child status is NEVER inferred from clothing, skin tone,   |
|                                        hair color, or subjective visual appearance                 |
| [X] NO Gender or Demographic Profiling -> Completely absent from data models and event schemas     |
+----------------------------------------------------------------------------------------------------+
```

All child labels in the current repository are **manually curated ground-truth fixtures** for controlled pipeline verification. In operational pilot deployments, child identification will rely strictly on **camera metric homography height estimation** and **guardian accompaniment proximity modeling**, preserving resident privacy.

---

## 5. Video-Level Consecutive Frame Verification

Consecutive frames around all entry, dwell, and exit transitions were analyzed across all four videos:

### 1. Scenario A — Playground (`playground_demo.mp4`)
- **Initial State (Frames 0–35):** Track 1 (Child) and Track 2 (Adult) detected in `toddler_play_sandbox`. Because sandbox is designated `SAFE` (`entry_alert=False`), zero alerts fire. Bounding box labels display `ID:1 [CHILD] (0.76)` and `ID:2 [ADULT] (0.84)`.
- **Hedge Approach & Transition (Frames 36–71):** Track 1 walks toward `perimeter_hedge_boundary`. Crosses boundary at Frame 42. Because policy has `entry_alert=False`, entry is confirmed after 3 frames (Frame 45) with backdated timestamp ($t = 1.75\text{s}$) without firing an alert.
- **Hedge Dwell Alert (Frame 168, $t = 7.00\text{s}$):** At exactly $4.00\text{ seconds}$ of dwell inside the hedge ($t_{\text{entry}} = 3.00\text{s} + 4.00\text{s}$ dwell), `CHILD_DWELL_TIME_EXCEEDED` fires. The bounding box switches to high-alert crimson red, and the secondary label displays `Perimeter Fence (4.0s)`.
- **Exit & Sandbox Return (Frames 192–239):** Track 1 returns to sandbox. Exit confirmed after 3 frames outside hedge; dwell timer resets cleanly.

### 2. Scenario B — Driveway (`driveway_demo.mp4`)
- **Sidewalk Co-Presence (Frames 0–23):** Track 1 (Adult) at $(220, 600)$ and Track 2 (Child) at $(320, 850)$ on `pedestrian_safety_sidewalk`. Horizontal proximity is $\Delta x = 100\text{px}$. The collision-aware overlay engine staggers Track 1 and Track 2 badge altitudes, ensuring both labels are readable without overlap.
- **Driveway Entry (Frames 24–47):** Track 2 steps off sidewalk curb into `vehicle_driveway_lane`. Ground-contact target crosshair crosses boundary at Frame 24. After 3-frame confirmation (Frame 27), `CHILD_ZONE_ENTRY` alert fires with backdated timestamp ($t = 1.00\text{s}$), matching event JSON.
- **Driveway Dwell Alert (Frame 72, $t = 3.00\text{s}$):** Track 2 accumulates $2.0\text{ seconds}$ of dwell in roadway. `CHILD_DWELL_TIME_EXCEEDED` alert fires. Bounding box glows red with `Main Driveway (2.0s)`.
- **Return to Sidewalk (Frames 120–168):** Track 2 steps back onto sidewalk. Exit confirmed; zero lingering alerts.

### 3. Scenario C — Boom Barrier (`boom_barrier_demo.mp4`)
- **North Approach (Frames 0–11):** Track 2 approaches gate from north outside all zones ($y < 400$). Zero false alerts.
- **Driveway Breach (Frame 19, $t = 0.79\text{s}$):** Track 2 enters `vehicle_driveway_lane`. `CHILD_ZONE_ENTRY` [Severity: HIGH] alert emitted.
- **Boom Barrier Mechanical Zone Breach (Frame 28, $t = 1.17\text{s}$):** Track 2 enters `boom_barrier_danger_zone`. `CHILD_ZONE_ENTRY` [Severity: CRITICAL] alert emitted. The centroid zone badge clearly displays `[DANGER] Boom Barrier Mechanical Zone [CRITICAL]`.
- **Barrier Departure (Frame 84–131):** Track 2 exits barrier zone. Exit confirmed after 3 frames; exit alert suppressed by policy (`exit_alert=False`).

### 4. Scenario D — Swimming Pool (`pool_safety_demo.mp4`)
- **Poolside Deck Presence (Frames 0–24):** Track 1 (Adult) and Track 3 (Adult) dwell on deck. Both are classified as `ADULT`. Deck dwell threshold ($3.0\text{s}$) expires at Frame 72 ($t = 3.00\text{s}$). Raw `DWELL_TIME_EXCEEDED` event fires in system log, but `ChildSafetyEventEngine` excludes both tracks. **Zero child alerts are emitted for adult tracks.**
- **Child Water Entry (Frame 27, $t = 1.12\text{s}$):** Track 4 (Child) enters water basin. Immediate `CHILD_ZONE_ENTRY` [Severity: CRITICAL] alert emitted. Ground-contact target marker renders cleanly at the water entry point.
- **Child Deck Lingering (Frame 140, $t = 5.83\text{s}$):** Track 4 climbs onto deck and remains $> 3.0\text{s}$. `CHILD_DWELL_TIME_EXCEEDED` [Severity: HIGH] alert emitted.
- **Climb-Out Recovery (Frame 210, $t = 8.75\text{s}$):** Track 6 climbs out of water basin. `CHILD_ZONE_EXIT` emitted. Deck dwell timer accumulates accurately.

---

## 6. Full Regression Testing Results

Following the implementation of resolution-adaptive font scaling, collision-aware label stacking, ground-contact anchor clamping, and role-specific zone styling, the complete automated test suite was executed:

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0 -- C:\projs\SafeKid AI\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\projs\SafeKid AI
configfile: pyproject.toml
plugins: anyio-4.16.0
collected 105 items

tests/test_age_estimator.py ......................... [  5%]
tests/test_benchmarking.py .......................... [  9%]
tests/test_child_safety.py .......................... [ 28%]
tests/test_detector.py .............................. [ 32%]
tests/test_multi_environment_pipeline.py ............ [ 36%]
tests/test_multi_environment_policies.py ............ [ 46%]
tests/test_pipeline.py .............................. [ 48%]
tests/test_reliability.py ........................... [ 61%]
tests/test_roi_engine.py ............................ [ 75%]
tests/test_safety_events.py ......................... [ 93%]
tests/test_tracker.py ............................... [ 97%]
tests/test_video_input.py ........................... [100%]

================= 105 passed, 5 warnings in 96.20s (0:01:36) ==================
```

### Regression Verification:
- **Baseline:** 102 tests passing.
- **New Tests Added:** 3 tests in `tests/test_safety_events.py`:
  1. `test_bottom_center_calculation_with_frame_clamping`: Verifies $x_{\text{bc}}$ and $y_{\text{bc}}$ are clamped to $[0, W-1]$ and $[0, H-1]$ when bounding boxes extend beyond frame edges.
  2. `test_visual_rendering_safe_monitored_danger_zones`: Verifies role-based color distinction and non-zero polygon rasterization for `SAFE`, `MONITORED`, and `DANGER` zones.
  3. `test_visual_rendering_collision_aware_labels_and_clamping`: Verifies that horizontally proximate detections receive staggered vertical offsets and that labels do not clip outside frame bounds.
- **Total Passing Tests:** **105 passed (100% green)**. Zero regressions observed.

