# SafeKid AI — Simulation Evaluation Report: Residential Driveway

**Scenario Identifier:** `scenario_b_driveway`  
**Environment Category:** `Residential Driveway`  
**Execution Timestamp:** `2026-11-04 13:46:56 UTC`  
**Camera Configuration:** `camera_gate_north` (`community_001`)  
**Source Video:** `1920x1080 @ 24.0 FPS` (192 frames, 8.0s)  
**Evaluation Status:** `PASSED`  

---

## 1. Scenario Description & Operational Context

Scenario B simulates a residential community main gate driveway. Track 2 (Provisional Child) starts on the dedicated pedestrian sidewalk (SAFE), steps off into the main vehicular driveway lane (DANGER, entry_alert=True, dwell_alert=True @ 2.0s), dwells for 3.0s, and returns to the sidewalk. Track 1 (Adult) walks on the sidewalk. Verifies immediate entry alert, dwell alert after 2.0s, configured HIGH severity, and complete adult exclusion.

### Configured Safety Zones & Policies

| Zone ID | Zone Name | Zone Type | Role | Entry Alert | Exit Alert | Dwell Alert | Dwell Threshold | Severity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `boom_barrier_danger_zone` | Boom Barrier Mechanical Zone | `RESTRICTED_INFRASTRUCTURE` | **DANGER** | `True` | `False` | `False` | N/A | `CRITICAL` |
| `vehicle_driveway_lane` | Main Vehicular Driveway | `VEHICULAR_GATEWAY` | **DANGER** | `True` | `False` | `True` | 2.0s | `HIGH` |
| `pedestrian_safety_sidewalk` | Dedicated Pedestrian Sidewalk | `GENERAL_COMMON_AREA` | **SAFE** | `False` | `False` | `False` | N/A | `LOW` |

---

## 2. Quantitative Performance & Tracking Metrics

- **Total Frames Processed:** `192`
- **Execution Duration:** `36.89 s`
- **Processing Throughput:** `5.2 FPS`
- **Unique Track IDs Assigned:** `[1, 2]`
- **Total Raw Zone Events:** `7`
- **Total Child Safety Alerts:** `2`

### Track Classification Registry

| Track ID | Classification | Status | Ground-Truth Operational Role | Child Alerts Generated | Exclusion Status |
| :---: | :---: | :---: | :--- | :---: | :---: |
| ID: 1 | `ADULT` | `PROVISIONAL` | Adult Supervisor / Pedestrian | **0** | `STRICTLY EXCLUDED (PASS)` |
| ID: 2 | `CHILD` | `PROVISIONAL` | Provisional Child Subject | **2** | `EVALUATED` |

---

## 3. Child Safety Policy Evaluation Matrix

| Metric | Expected Count | Observed Count | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Target Child Alerts** | 2 | 2 | 0 | `PASS` |
| **Missing Expected Alerts** | 0 | 0 | +0 | `PASS` |
| **Unexpected Alerts** | 0 | 0 | +0 | `PASS` |
| **Severity Mismatches** | 0 | 0 | +0 | `PASS` |
| **Adult / Unclassified Violations** | 0 | 0 | +0 | `PASS` |

### Detailed Event Verification Log

| Frame | Timestamp (s) | Track ID | Zone ID | Event Type | Severity | Dwell (s) | Alert Message |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 33 | 1.38s | ID: 2 | `vehicle_driveway_lane` | **CHILD_ZONE_ENTRY** | `HIGH` | 0.0s | CHILD SAFETY ALERT: Child (Track ID 2) entered configured danger zone 'Main Vehicular Driveway' (VEHICULAR_GATEWAY) [Severity: HIGH] |
| 81 | 3.38s | ID: 2 | `vehicle_driveway_lane` | **CHILD_DWELL_TIME_EXCEEDED** | `HIGH` | 2.0s | CHILD DWELL ALERT: Child (Track ID 2) remained in danger zone 'Main Vehicular Driveway' beyond configured threshold (2.0s) [Severity: HIGH] |

---

## 4. Policy Compliance Analysis

- **Track 2 Alert:** `CHILD SAFETY ALERT: Child (Track ID 2) entered configured danger zone 'Main Vehicular Driveway' (VEHICULAR_GATEWAY) [Severity: HIGH]` (keyword: `entered configured danger zone`) -> **PASS**
- **Track 2 Alert:** `CHILD DWELL ALERT: Child (Track ID 2) remained in danger zone 'Main Vehicular Driveway' beyond configured threshold (2.0s) [Severity: HIGH]` (keyword: `remained in danger zone`) -> **PASS**
- **Adult Exclusion Policy:** Track [1] produced zero child alerts -> **PASS**

---

## 5. Scope & Scientific Integrity Statement

> [!IMPORTANT]
> **Notice on Testing Scope and Evaluation Semantics:**
> This evaluation evaluates deterministic synthetic simulation video sequences under controlled testing conditions.
> It validates the logical correctness of the detection, ByteTrack tracking, ROI boundary resolution, monotonic dwell timing, and declarative policy filtering pipeline.
> **These results do NOT represent real-world deployment accuracy, precision, recall, or operational false-positive rates.**
> In production deployments, real human behavior, lighting variations, occlusions, and camera vibrations introduce variables outside this controlled evaluation harness.
