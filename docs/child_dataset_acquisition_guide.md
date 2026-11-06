# SafeKid AI — Child Detection Dataset Acquisition & Annotation Guide

This document defines the data acquisition protocols, dataset inventory, ethical governance, and annotation specifications for training and benchmarking supervised child detection in SafeKid AI.

---

## 1. Ethical Governance & Privacy Principles

SafeKid AI operates in gated residential communities and shared public spaces. Applying computer vision to minors requires the highest standard of ethical safeguards:

1. **Strict Consent Mandate:**
   - Public surveillance footage (such as VIRAT, CityFlow, or urban traffic CCTV) must **NEVER** be used to infer child identities, ages, or demographic profiles without explicit authorization.
   - For real-world residential community deployments, child detection features must operate exclusively under verified parental or guardian opt-in consent (complying with COPPA and GDPR Article 8).

2. **Provisional Model Predictions vs. Verified Child Profiles:**
   - A model prediction is strictly a **heuristic candidate** (`CHILD_CANDIDATE`).
   - A model output alone must never automatically enroll, index, or store a child's profile without human security review and parental authorization.

3. **No Biometric Identification or Facial Recognition:**
   - Child detection models in SafeKid AI operate exclusively on overall body proportions, gait, clothing, spatial perspective, and context.
   - Facial recognition, facial landmarking, and biometric templates are strictly prohibited.

4. **No Continuous Age Regression:**
   - Continuous age estimation (e.g., predicting an exact age of 7.4 vs. 8.1 years) is brittle at CCTV distances and ethically problematic.
   - SafeKid AI standardizes on three discrete operational states:
     - `CHILD` (approx. age < 13 years; high vulnerability in vehicular or pool zones)
     - `ADULT` (approx. age >= 18 years; verified adults authorized to supervise)
     - `UNKNOWN` (ambiguous features, teenagers 13-17, distant pedestrians, or heavy occlusions; requires conservative abstention)

---

## 2. Benchmark & Research Dataset Inventory

| Dataset | Modality | Samples | Relevant Annotations | License / Access Restrictions | Suitability for SafeKid AI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RAP v2** *(Richly Annotated Pedestrian v2)* | Multi-camera indoor/outdoor CCTV | 84,928 images from 26 cameras | `AgeLess16`, body attributes, occlusion, viewpoints | Academic non-commercial research; requires signed EULA with CUHK/NLPR | **High:** Realistic CCTV angles, diverse lighting, explicit `AgeLess16` attribute. |
| **PETA** *(PEdesTrian Attribute)* | Outdoor surveillance pedestrian crops | 19,000 images (VIPeR, CUHK, PRID, etc.) | `personalLess15` / `AgeLess15`, clothing, orientation | Academic research only; non-commercial | **High:** Standard pedestrian benchmark with explicit child tag (< 15 years). |
| **PA-100K** | Outdoor surveillance | 100,000 images from 598 cameras | `AgeLess18`, clothing, carrying | Academic research only | **Medium:** Lacks fine-grained young-child distinction (bundles teens with children). |
| **Gated Residential Reference Fixture** | Controlled multi-perspective video fixtures | 300+ annotated crops (pool, playground, gate, driveway) | `CHILD`, `ADULT`, `AMBIGUOUS` with occlusion & distance metadata | Open internal benchmark (SafeKid AI) | **High:** Zero privacy risk, reproducible, verified ground-truth for regression tests. |

---

## 3. Recommended Manual Acquisition Protocols

### A. Academic Pedestrian Datasets (PETA & RAP v2)
1. **Official Research Repositories:**
   - **RAP Dataset:** [http://www.rapdataset.com/](http://www.rapdataset.com/) (Maintained by Institute of Automation, Chinese Academy of Sciences).
   - **PETA Dataset:** [https://mmlab.ie.cuhk.edu.hk/projects/PETA.html](https://mmlab.ie.cuhk.edu.hk/projects/PETA.html) (Multimedia Laboratory, CUHK).
2. **Access Protocol:**
   - Submit the institutional academic license agreement to the dataset coordinators.
   - Once authorized, extract the image archives into:
     ```
     data/child_dataset/genuine/peta/
     data/child_dataset/genuine/rap_v2/
     ```
3. **Attribute Filter:**
   - For PETA: filter for `personalLess15 == 1` as `CHILD`, and `Age30-45 == 1` or `Age45-60 == 1` as `ADULT`.
   - For RAP v2: filter for `AgeLess16 == 1` as `CHILD`, and `Age31-45 == 1` as `ADULT`.

### B. Authorized Residential Community CCTV
1. **Consent Protocol:**
   - Residential management must collect signed consent forms from residents with registered children.
   - Camera streams designated for child safety monitoring (e.g. playground, pool gate) must display public signage informing residents of AI-assisted safety monitoring.
2. **Directory Placement:**
   ```
   data/child_dataset/authorized_residential/
     metadata/consent_registry.json
     crops/
     annotations/
   ```

---

## 4. Standard Annotation Specification (Release 1.0)

All child-detection datasets used in SafeKid AI must conform to the following schema:

```json
{
  "annotation_spec_version": "1.0",
  "samples": [
    {
      "image_id": "ped_000102.jpg",
      "bbox_xyxy": [120.0, 45.0, 185.0, 210.0],
      "classification": "CHILD",
      "confidence": 1.0,
      "metadata": {
        "distance_tier": "MEDIUM",
        "occlusion_ratio": 0.15,
        "head_to_body_ratio": 0.28,
        "environment": "PLAYGROUND",
        "consent_status": "CONSENTED_RESEARCH"
      }
    }
  ]
}
```

### Classification Labels:
- `CHILD`: Pedestrian estimated < 13 years (larger head-to-body proportion, shorter stature relative to scene landmarks).
- `ADULT`: Pedestrian estimated $\ge$ 18 years.
- `UNKNOWN`: Ambiguous stature, teenagers (13-17), distant pedestrians (< 40px bounding box height), or severe occlusion (> 50%).

---

## 5. Model Training Best Practices

1. **Sequence-Level Split Isolation:**
   - Never split frames of the same video or camera track between training and testing sets.
   - All evaluation must be performed on held-out sequences or independent subjects.
2. **Handling Small and Distant Pedestrians:**
   - Enforce minimum bounding box height ($H_{min} \ge 32\text{px}$). Pedestrians below this threshold must automatically yield `UNKNOWN` to avoid false child alarms.
3. **Temporal Filtering:**
   - Standalone single-frame predictions must be aggregated via the ByteTrack track smoother before triggering safety alert escalations.

