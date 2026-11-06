# SafeKid AI — Simulation Evaluation Report: Boom Barrier

**Scenario Identifier:** `scenario_c_boom_barrier`  
**Environment Category:** `Boom Barrier`  
**Execution Timestamp:** `2026-11-04 13:47:21 UTC`  
**Camera Configuration:** `camera_gate_north` (`community_001`)  
**Source Video:** `1920x1080 @ 24.0 FPS` (144 frames, 6.0s)  
**Evaluation Status:** `PASSED`  

---

## 1. Scenario Description & Operational Context

Scenario C simulates an automated mechanical boom barrier entrance nested within the vehicle driveway lane. Track 2 (Provisional Child) enters the vehicle driveway (HIGH) and mechanical drop zone (CRITICAL). The child dwells for 2.0s, triggering the driveway dwell threshold while the boom barrier zone correctly suppresses dwell alerts (dwell_alert=False). On exit, the boom barrier zone correctly suppresses exit alert (exit_alert=False). Track 1 (Adult) stands on the sidewalk. Verifies immediate CRITICAL entry alert, suppression of boom barrier dwell and exit alerts, and complete adult exclusion.

### Configured Safety Zones & Policies

| Zone ID | Zone Name | Zone Type | Role | Entry Alert | Exit Alert | Dwell Alert | Dwell Threshold | Severity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `boom_barrier_danger_zone` | Boom Barrier Mechanical Zone | `RESTRICTED_INFRASTRUCTURE` | **DANGER** | `True` | `False` | `False` | N/A | `CRITICAL` |
| `vehicle_driveway_lane` | Main Vehicular Driveway | `VEHICULAR_GATEWAY` | **DANGER** | `True` | `False` | `True` | 2.0s | `HIGH` |
| `pedestrian_safety_sidewalk` | Dedicated Pedestrian Sidewalk | `GENERAL_COMMON_AREA` | **SAFE** | `False` | `False` | `False` | N/A | `LOW` |

---

## 2. Quantitative Performance & Tracking Metrics

- **Total Frames Processed:** `144`
- **Execution Duration:** `22.76 s`
- **Processing Throughput:** `6.3 FPS`
- **Unique Track IDs Assigned:** `[1, 2]`
- **Total Raw Zone Events:** `5`
- **Total Child Safety Alerts:** `3`

### Track Classification Registry

| Track ID | Classification | Status | Ground-Truth Operational Role | Child Alerts Generated | Exclusion Status |
| :---: | :---: | :---: | :--- | :---: | :---: |
| ID: 1 | `ADULT` | `PROVISIONAL` | Adult Supervisor / Pedestrian | **0** | `STRICTLY EXCLUDED (PASS)` |
| ID: 2 | `CHILD` | `PROVISIONAL` | Provisional Child Subject | **3** | `EVALUATED` |

---

## 3. Child Safety Policy Evaluation Matrix

| Metric | Expected Count | Observed Count | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Target Child Alerts** | 3 | 3 | 0 | `PASS` |
| **Missing Expected Alerts** | 0 | 0 | +0 | `PASS` |
| **Unexpected Alerts** | 0 | 0 | +0 | `PASS` |
| **Severity Mismatches** | 0 | 0 | +0 | `PASS` |
| **Adult / Unclassified Violations** | 0 | 0 | +0 | `PASS` |

### Detailed Event Verification Log

| Frame | Timestamp (s) | Track ID | Zone ID | Event Type | Severity | Dwell (s) | Alert Message |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 19 | 0.79s | ID: 2 | `vehicle_driveway_lane` | **CHILD_ZONE_ENTRY** | `HIGH` | 0.0s | CHILD SAFETY ALERT: Child (Track ID 2) entered configured danger zone 'Main Vehicular Driveway' (VEHICULAR_GATEWAY) [Severity: HIGH] |
| 28 | 1.17s | ID: 2 | `boom_barrier_danger_zone` | **CHILD_ZONE_ENTRY** | `CRITICAL` | 0.0s | CHILD SAFETY ALERT: Child (Track ID 2) entered configured danger zone 'Boom Barrier Mechanical Zone' (RESTRICTED_INFRASTRUCTURE) [Severity: CRITICAL] |
| 67 | 2.79s | ID: 2 | `vehicle_driveway_lane` | **CHILD_DWELL_TIME_EXCEEDED** | `HIGH` | 2.0s | CHILD DWELL ALERT: Child (Track ID 2) remained in danger zone 'Main Vehicular Driveway' beyond configured threshold (2.0s) [Severity: HIGH] |

---

## 4. Policy Compliance Analysis

- **Track 2 Alert:** `CHILD SAFETY ALERT: Child (Track ID 2) entered configured danger zone 'Main Vehicular Driveway' (VEHICULAR_GATEWAY) [Severity: HIGH]` (keyword: `entered configured danger zone`) -> **PASS**
- **Track 2 Alert:** `CHILD SAFETY ALERT: Child (Track ID 2) entered configured danger zone 'Boom Barrier Mechanical Zone' (RESTRICTED_INFRASTRUCTURE) [Severity: CRITICAL]` (keyword: `entered configured danger zone`) -> **PASS**
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
