# SafeKid AI — Real-World Child Dataset Availability & Legal Governance Report

**Project:** SafeKid AI — AI-Based Child Safety Monitoring for Gated Residential Communities  
**Report Date:** November 5, 2026  
**Status Assessment:** **REAL-WORLD SURVEILLANCE TRAINING MILESTONE: BLOCKED**  
*(Awaiting Academic EULA Authorization & On-Site Residential Consent Deployment)*

---

## 1. Executive Assessment

SafeKid AI requires supervised training data that can reliably distinguish vulnerable children (approx. $< 13$ years old) from adults ($\ge 18$ years old) across diverse CCTV viewing angles, distances, and lighting conditions. 

An exhaustive investigation of candidate surveillance datasets was conducted. **No publicly accessible real-world pedestrian surveillance dataset is currently cleared for operational commercial training or immediate local download within our sandboxed development environment.**

Furthermore, existing academic attribute datasets exhibit a **significant semantic mismatch**: they define youth thresholds at 15 or 16 years old, grouping independent teenagers with vulnerable pre-teens. Using these datasets naively would induce unacceptable false alarm rates in residential communities.

Therefore, in accordance with the project's strict engineering ethics rules, the real-world child training milestone is formally designated as **BLOCKED / AWAITING_EULA_ACQUISITION**.

---

## 2. In-Depth Audit of Candidate Real-World Datasets

### A. RAP v2 (Richly Annotated Pedestrian Dataset v2)

| Criteria | Audit Assessment |
| :--- | :--- |
| **Custodian & Origin** | Institute of Automation, Chinese Academy of Sciences (CASIA) & CUHK |
| **Dataset Scale** | 84,928 pedestrian images captured from 26 surveillance camera streams |
| **Explicit Child Attribute** | Attribute #1: `AgeLess16` (Binary attribute: 1 = Yes, 0 = No) |
| **Semantic Meaning & Age Threshold** | Estimated age $< 16$ years. **Critical Semantic Gap:** This attribute bundles 13-, 14-, and 15-year-old teenagers with young children ($< 13$ years). In a residential community, a 15-year-old walking along an access road is normal behavior and should not trigger a child intrusion alarm. Naive supervised training on `AgeLess16` would teach the model to flag teenagers as vulnerable children. |
| **Label Level** | Bounding-box-level pedestrian crops extracted from 1080p surveillance video. |
| **Visible Body Information** | Full-body pedestrian crops. Resolutions vary widely from low ($32 \times 80\text{px}$) to high ($140 \times 320\text{px}$). |
| **Surveillance Realism** | High: 26 indoor/outdoor viewpoints, high-angle camera mounting ($2.5\text{m} - 4.5\text{m}$), varying distances, motion blur, and natural partial occlusions. |
| **Permitted Usage** | **Strictly Non-Commercial Academic Research Only.** Commercial deployment, SaaS monitoring, and embedding into commercial products are expressly prohibited by the EULA. |
| **Redistribution & Model Weights** | Redistribution of raw images or derived datasets is prohibited. |
| **Required Acquisition Steps** | 1. Submit a formal application from an accredited academic/research institutional email to the CASIA/CUHK dataset administrators.<br>2. Sign and return the official RAP End User License Agreement (EULA).<br>3. Obtain custodian approval and download credentials via secure server. |
| **Current Workspace Status** | **NOT ACQUIRED** (Blocked by sandbox network isolation and institutional EULA requirement). |

---

### B. PETA (PEdesTrian Attribute Dataset)

| Criteria | Audit Assessment |
| :--- | :--- |
| **Custodian & Origin** | Deng et al., 2014, Multimedia Laboratory, The Chinese University of Hong Kong (CUHK) |
| **Dataset Scale** | 19,000 pedestrian images aggregated from 10 benchmark datasets (VIPeR, CUHK, PRID, GRID, MIT, SARC3D, TownCentre, etc.) |
| **Explicit Child Attribute** | Attribute: `personalLess15` / `AgeLess15` |
| **Semantic Meaning & Age Threshold** | Estimated age $< 15$ years. **Semantic Gap:** Includes 13- and 14-year-olds; lacks fine-grained differentiation for young toddlers and primary school children ($< 10$ years). |
| **Label Level** | Bounding-box-level pedestrian crops. |
| **Visible Body Information** | Highly heterogeneous. Sub-datasets like VIPeR possess low resolution ($48 \times 128\text{px}$) with heavy sensor noise; TownCentre features high-angle outdoor surveillance. |
| **Surveillance Realism** | Moderate: Aggregated across older academic datasets with inconsistent camera heights and compression artifacts. |
| **Permitted Usage** | **Non-Commercial Academic Research Only.** |
| **Redistribution & Licensing** | Redistribution restricted to authorized researchers. |
| **Required Acquisition Steps** | 1. Access CUHK MMLab PETA project repository.<br>2. Agree to non-commercial academic research terms.<br>3. Download and extract dataset archives out-of-band. |
| **Current Workspace Status** | **NOT ACQUIRED** (Blocked by sandbox network isolation and licensing constraints). |

---

### C. Other Academic Pedestrian Attribute Datasets

1. **PA-100K (100,000 images from 598 cameras):**
   - Attributes: `AgeLess18`, `Age18-60`, `AgeOver60`.
   - **Assessment: REJECTED.** The `AgeLess18` threshold groups 17-year-old adults with infants, rendering it semantically useless for child safety alerting.
2. **Market-1501 Attribute / DukeMTMC Attribute:**
   - Attributes: `teenager` ($< 18$), `young-adult` (18–30), `adult` (30–60), `senior` ($> 60$).
   - **Assessment: REJECTED.** Conflates teenagers with children; DukeMTMC has been officially retracted by Duke University due to human subject privacy concerns.

---

### D. Authorized Residential Community CCTV (Consented Deployment Tier)

| Criteria | Deployment Specification |
| :--- | :--- |
| **Environment** | Gated residential perimeter fences, pool access gates, vehicular transit lanes, and playgrounds. |
| **Legal Framework** | COPPA / GDPR Article 8 compliance. |
| **Consent Protocol** | 1. Verified written opt-in consent from parents/guardians for participating children.<br>2. Prominent public on-site signage notifying residents of AI-assisted safety monitoring.<br>3. Facial recognition and biometric re-identification strictly prohibited.<br>4. Ephemeral spatial Kalman tracking only; no permanent biometric profiles stored. |
| **Data Flow** | Model predictions produce provisional candidate alerts (`PROVISIONAL_MODEL_PREDICTION`) requiring security staff confirmation before identity enrollment. |
| **Current Workspace Status** | **AWAITING ON-SITE RESIDENTIAL PILOT DEPLOYMENT.** |

---

## 3. Semantic Mapping & Uncertainty Protocol

To safely ingest academic datasets when authorized, SafeKid AI defines the following strict attribute mapping rules:

```
+---------------------+-------------------------+------------------------------------------------------+
| Source Dataset Label| SafeKid AI Class        | Governance & Confidence Handling                     |
+---------------------+-------------------------+------------------------------------------------------+
| AgeLess16 == 1 (RAP)| UNKNOWN (Flagged Youth) | Mark as UNKNOWN with high margin uncertainty         |
|                     |                         | (to prevent false alarms on 13-15yo teenagers).      |
+---------------------+-------------------------+------------------------------------------------------+
| Age31-45 == 1 (RAP) | ADULT                   | High confidence adult supervisor candidate.          |
+---------------------+-------------------------+------------------------------------------------------+
| Age46-60 == 1 (RAP) | ADULT                   | High confidence adult supervisor candidate.          |
+---------------------+-------------------------+------------------------------------------------------+
| personalLess15 (PETA| UNKNOWN (Flagged Youth) | Mark as UNKNOWN; requires corroborating perspective.  |
+---------------------+-------------------------+------------------------------------------------------+
| Consented < 13 yo   | CHILD                   | Direct operational CHILD candidate.                  |
+---------------------+-------------------------+------------------------------------------------------+
| Consented >= 18 yo  | ADULT                   | Direct operational ADULT candidate.                  |
+---------------------+-------------------------+------------------------------------------------------+
```

---

## 4. Requirements to Unblock Real-World Training

To transition the real-world training milestone from **BLOCKED** to **ACTIVE**:

1. **Institutional Research Partnership:** Partner with an accredited academic institution to execute the RAP v2 EULA with CUHK/NLPR.
2. **Out-of-Band Download & Checksum Verification:** Deposit verified archives into `data/child_dataset/genuine/rap_v2/` and verify SHA-256 integrity.
3. **Execution of Ingestion Pipeline:** Execute `scripts/ingest_real_world_dataset.py` to map attributes, deduplicate near-identical crops, and partition train/val/test splits strictly by camera/subject groups.
4. **On-Site Residential Pilot:** Deploy SafeKid AI under authorized community IRB/ethics board approval with signed parental consent in a controlled gated enclave.
