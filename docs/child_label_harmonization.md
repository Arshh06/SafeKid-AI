# SafeKid AI — Child Label Harmonization & Annotation Specification

**Project:** SafeKid AI — AI-Based Child Safety Monitoring for Gated Residential Communities  
**Document:** Child Label Harmonization Standard  
**Version:** 1.0  
**Date:** 2026-11-06  
**Audience:** SafeKid AI Data Engineering, Research & Subsystem Integrators  

---

## 1. Ethical Governance & Core Operational Principles

The SafeKid AI child classification subsystem operates under strict ethical and technical boundaries designed to protect minor privacy while delivering reliable safety alerts:

1. **No Facial Recognition or Biometric Identity Tracking:**  
   Child detection must operate entirely on full-body pedestrian crops, spatial context, body proportions, gait, and posture. Facial feature extraction, facial landmarking, and biometric identification are strictly forbidden.
2. **No Continuous Chronological Age Regression:**  
   Predicting continuous ages (e.g., predicting an exact age of 6.8 vs. 7.9 years) from distant surveillance CCTV is ungrounded, prone to high variance, and ethically hazardous. SafeKid AI discretizes age into **three operational semantic states**: `CHILD`, `ADULT`, and `UNKNOWN`.
3. **No Single-Attribute Inferences:**  
   Age must never be inferred from a single weak visual cue (such as wearing a backpack, carrying an umbrella, or wearing bright colors). Adult pedestrians frequently carry backpacks; teenagers often dress similarly to children.
4. **Mandatory Conservative Abstention:**  
   When features are ambiguous, resolution is degraded ($h < 80$ px), or subject age is in the adolescent transition bracket (13–17 years), the subsystem **must abstain to `UNKNOWN`**. Under no circumstances should ambiguous cases be forced into `CHILD` or `ADULT`.

---

## 2. Operational Class Definitions

### 2.1 The `CHILD` Class
- **Operational Safety Definition:** Pre-pubescent minors and young children (estimated chronological age $< 13$ years).
- **Physical Characteristics:** High head-to-body ratio ($> 0.20$), smaller absolute bounding box height relative to adult scale baselines at identical camera depth, distinct limb proportions, and typical child walking/running dynamics.
- **Safety Policy Impact:** An unescorted track classified as `CHILD` inside a restricted hazard zone (swimming pool basin, deep water deck, vehicular driveway, boom barrier roadway) triggers an immediate emergency safety alert and dispatch notification.

### 2.2 The `ADULT` Class
- **Operational Safety Definition:** Mature individuals (chronological age $\ge 18$ years) capable of exercising independent physical care and supervising minors.
- **Physical Characteristics:** Standard adult stature ($> 155$ cm in calibrated perspective), mature head-to-body ratio ($0.12 - 0.16$), and adult gait.
- **Safety Policy Impact:** An adult detected within a monitored zone satisfies supervision policies. The presence of an adult in proximity to a child suppresses unaccompanied child hazard alerts.

### 2.3 The `UNKNOWN` Class (Abstention State)
- **Operational Safety Definition:** Any pedestrian track whose age classification cannot be established with high statistical certainty.
- **Mandatory UNKNOWN Inclusions:**
  1. **Independent Teenagers (Ages 13–17):** Teenagers possess near-adult physical stature and adult legal capacity in many community common areas, but do not possess adult supervision authority over young children. They must **never** be silently relabeled as `CHILD` or `ADULT`.
  2. **Distant or Low-Resolution Targets:** Any pedestrian crop with height $< 80$ pixels or width $< 32$ pixels.
  3. **High Occlusion:** Any person instance where more than $40\%$ of the body is occluded by fences, vehicles, foliage, or furniture.
  4. **Non-Upright or Distorted Postures:** Persons sitting on benches, riding bicycles, crouching, or leaning heavily.
  5. **Ambiguous Source Dataset Brackets:** Datasets that bundle teenagers with children into coarse categories (e.g., `AgeLess16` in RAP v2 or `personalLess15` in PETA) without fine-grained age confirmation.
- **Safety Policy Impact:** Tracks classified as `UNKNOWN` remain tracked for situational awareness and perimeter boundary security, but **do not trigger emergency unaccompanied-child alarms**.

---

## 3. Dataset-Specific Label Harmonization Mapping

Academic and public surveillance datasets employ divergent, non-standardized age thresholds. Below is the official SafeKid AI harmonization mapping matrix:

| Source Dataset | Original Dataset Attribute | Original Dataset Definition | SafeKid AI Harmonized Label | Confidence | Harmonization Rationale & Constraints |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **JAAD** | `age="child"` | Visually apparent child | **`CHILD`** | 0.95 | High semantic alignment. Annotators explicitly distinguished children from older youth. |
| **JAAD** | `age="adult"` / `age="senior"` | Mature adult / elderly | **`ADULT`** | 0.95 | Clean mapping to adult supervisor class. |
| **JAAD** | `age="young"` | Teenager / young adult | **`UNKNOWN`** | 0.40 | Conservative abstention. Covers the ambiguous 13–24 age bracket. |
| **PIE** | `age="child"` | Visually apparent child | **`CHILD`** | 0.95 | High semantic alignment from York University visual lab. |
| **PIE** | `age="adult"` | Mature adult | **`ADULT`** | 0.95 | Clean mapping to adult supervisor class. |
| **PIE** | `age="young"` | Young adult / teen | **`UNKNOWN`** | 0.40 | Conservative abstention to prevent false child alarms. |
| **FA-INRIA** | `label="child"` | Age 0–14 years | **`CHILD`** | 0.85 | Good alignment, with minor teen leakage (13–14yo). Allowed for static crop training. |
| **FA-INRIA** | `label="adult"` | Age 15+ years | **`ADULT`** | 0.90 | Adults and older teens combined; suitable for supervisor negative baseline. |
| **RAP v2** | `AgeLess16 == 1` | Estimated age $< 16$ years | **`UNKNOWN`** | 0.50 | **Semantic Mismatch:** Bundles 13–15yo teenagers with children. Cannot be mapped to `CHILD` without manual sub-filtering. |
| **RAP v2** | `Age31_45 == 1` / `Age46_60 == 1`| Mature adults 31–60 | **`ADULT`** | 0.95 | Verified mature adult supervisors. |
| **RAP v2** | `Age17_30 == 1` | Ages 17–30 | **`UNKNOWN`** | 0.50 | High stylistic overlap between young adults and older adolescents. |
| **PETA** | `personalLess15 == 1` | Estimated age $< 15$ years | **`UNKNOWN`** | 0.55 | **Semantic Mismatch:** Bundles young teens. Requires conservative UNKNOWN mapping unless accompanied by height gating. |
| **PETA** | `Age30_45 == 1` / `Age45_60 == 1`| Adults 30–60 | **`ADULT`** | 0.95 | Clean adult supervisory baseline. |
| **PA-100K** | `AgeLess18 == 1` | Age $< 18$ years | **`UNKNOWN`** | 0.20 | **Incompatible:** Bundles high school seniors with infants; unviable for child detection. |
| **Authorized Residential**| `verified_age < 13` | Verified via parental opt-in | **`CHILD`** | 1.00 | Ground truth established with verified guardian consent. |
| **Authorized Residential**| `verified_age >= 18` | Verified resident adult | **`ADULT`** | 1.00 | Ground truth verified adult resident. |
| **Authorized Residential**| `13 <= verified_age < 18`| Verified minor adolescent | **`UNKNOWN`** | 0.70 | Teenager resident; logged for awareness, exempt from child alarms. |

---

## 4. Multi-Stage Gating & Quality Control Pipeline

When ingesting pedestrian crops from any real-world source, the ingestion engine ([`ChildDatasetIngestionEngine`](src/safekid_ai/child_detection/dataset_ingestion.py)) must apply the following sequential gates before assigning a training or evaluation label:

```
[Raw Pedestrian Crop & Metadata]
               │
               ▼
   [Gate 1: Corrupted & Empty Check] ──(Invalid)──> REJECT / DISCARD
               │
               ▼
   [Gate 2: Minimum Resolution Check] ──(<80px)───> Map to UNKNOWN (Low Resolution)
               │
               ▼
   [Gate 3: Bounding Box Geometry] ──(Aspect/Bound Error)──> REJECT / DISCARD
               │
               ▼
   [Gate 4: Perceptual Deduplication] ──(dHash Hamming <= 2)──> REMOVE DUPLICATE
               │
               ▼
   [Gate 5: Semantic Attribute Harmonization]
               ├─ Confirmed < 13yo ───────────────> CHILD
               ├─ Confirmed Mature Adult (>=18yo) ─> ADULT
               └─ Adolescent / Ambiguous / Teen ───> UNKNOWN
```

### Deduplication Policy
- All pedestrian crops undergo 64-bit difference hashing (`compute_dhash`).
- Any pair of crops exhibiting a bitwise Hamming distance $\le 2$ is treated as a near-duplicate collision.
- The secondary duplicate is purged from the dataset to prevent split leakage between training, validation, and test partitions.

---

## 5. Summary Declaration

By standardizing on this harmonization protocol:
1. SafeKid AI eliminates semantic ambiguity across disparate academic datasets.
2. The operational system will not trigger false child emergency alerts on teenage residents or adult students.
3. Model evaluation benchmarks can be compared honestly against verified ground-truth boundaries.

