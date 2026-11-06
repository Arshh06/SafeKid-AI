# SafeKid AI — Simulation Evaluation Report: Playground

**Scenario Identifier:** `scenario_a_playground`  
**Environment Category:** `Playground`  
**Execution Timestamp:** `2026-11-04 13:46:13 UTC`  
**Camera Configuration:** `camera_playground_01` (`community_002`)  
**Source Video:** `1280x720 @ 24.0 FPS` (240 frames, 10.0s)  
**Evaluation Status:** `PASSED`  

---

## 1. Scenario Description & Operational Context

Scenario A simulates a residential playground with a designated safe toddler sandbox and a hazardous perimeter hedge/fence. Track 1 (Provisional Child) starts in the sandbox (SAFE, entry_alert=False), walks to the perimeter hedge (DANGER, entry_alert=False, dwell_alert=True @ 4.0s), lingers beyond 4.0s, and walks away. Track 2 (Adult) is stationary on the bench supervising. Verifies safe zone entry suppression, danger zone entry alert suppression when disabled, and dwell alert emission at 4.0s.

### Configured Safety Zones & Policies

| Zone ID | Zone Name | Zone Type | Role | Entry Alert | Exit Alert | Dwell Alert | Dwell Threshold | Severity |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `toddler_play_sandbox` | Toddler Sandbox & Swings Area | `PLAYGROUND` | **SAFE** | `False` | `False` | `False` | N/A | `LOW` |
| `perimeter_hedge_boundary` | North Perimeter Fence & Dense Hedge | `PERIMETER_FENCE` | **DANGER** | `False` | `False` | `True` | 4.0s | `HIGH` |

---

## 2. Quantitative Performance & Tracking Metrics

- **Total Frames Processed:** `240`
- **Execution Duration:** `51.56 s`
- **Processing Throughput:** `4.7 FPS`
- **Unique Track IDs Assigned:** `[1, 2]`
- **Total Raw Zone Events:** `7`
- **Total Child Safety Alerts:** `1`

### Track Classification Registry

| Track ID | Classification | Status | Ground-Truth Operational Role | Child Alerts Generated | Exclusion Status |
| :---: | :---: | :---: | :--- | :---: | :---: |
| ID: 1 | `CHILD` | `PROVISIONAL` | Provisional Child Subject | **1** | `EVALUATED` |
| ID: 2 | `ADULT` | `PROVISIONAL` | Adult Supervisor / Pedestrian | **0** | `STRICTLY EXCLUDED (PASS)` |

---

## 3. Child Safety Policy Evaluation Matrix

| Metric | Expected Count | Observed Count | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Target Child Alerts** | 1 | 1 | 0 | `PASS` |
| **Missing Expected Alerts** | 0 | 0 | +0 | `PASS` |
| **Unexpected Alerts** | 0 | 0 | +0 | `PASS` |
| **Severity Mismatches** | 0 | 0 | +0 | `PASS` |
| **Adult / Unclassified Violations** | 0 | 0 | +0 | `PASS` |

### Detailed Event Verification Log

| Frame | Timestamp (s) | Track ID | Zone ID | Event Type | Severity | Dwell (s) | Alert Message |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 155 | 6.46s | ID: 1 | `perimeter_hedge_boundary` | **CHILD_DWELL_TIME_EXCEEDED** | `HIGH` | 4.0s | CHILD DWELL ALERT: Child (Track ID 1) remained in danger zone 'North Perimeter Fence & Dense Hedge' beyond configured threshold (4.0s) [Severity: HIGH] |

---

## 4. Policy Compliance Analysis

- **Track 1 Alert:** `CHILD DWELL ALERT: Child (Track ID 1) remained in danger zone 'North Perimeter Fence & Dense Hedge' beyond configured threshold (4.0s) [Severity: HIGH]` (keyword: `remained in danger zone`) -> **PASS**
- **Adult Exclusion Policy:** Track [2] produced zero child alerts -> **PASS**

---

## 5. Scope & Scientific Integrity Statement

> [!IMPORTANT]
> **Notice on Testing Scope and Evaluation Semantics:**
> This evaluation evaluates deterministic synthetic simulation video sequences under controlled testing conditions.
> It validates the logical correctness of the detection, ByteTrack tracking, ROI boundary resolution, monotonic dwell timing, and declarative policy filtering pipeline.
> **These results do NOT represent real-world deployment accuracy, precision, recall, or operational false-positive rates.**
> In production deployments, real human behavior, lighting variations, occlusions, and camera vibrations introduce variables outside this controlled evaluation harness.
