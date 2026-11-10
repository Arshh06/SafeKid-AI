# SafeKid AI — Real-World Child Detection Dataset Access & Acquisition Guide

**Project:** SafeKid AI — AI-Based Child Safety Monitoring for Gated Residential Communities  
**Document:** Official Data Access & Licensing Guide  
**Version:** 1.0  
**Date:** 2026-11-06  

---

## 1. Objective & Ethical Acquisition Mandate

This guide establishes the legitimate, authorized procedures for acquiring real-world pedestrian and child detection datasets for the SafeKid AI project.

### Core Acquisition Policy
- **Zero Web Scraping:** Scraping public video platforms, social media, or unauthorized web mirrors is strictly prohibited.
- **Zero Access Control Bypasses:** Do not bypass registration portals, paywalls, or EULA verification forms.
- **Academic & Non-Commercial Compliance:** All candidate datasets are governed by academic licenses. They must be utilized strictly within non-commercial research parameters.
- **Parental Consent for Residential CCTV:** Real residential camera data must **never** be gathered without signed, verified guardian opt-in consent complying with COPPA and GDPR Article 8.

---

## 2. Dataset Access Profiles & Procedures

### 2.1 JAAD (Joint Attention in Autonomous Driving)
- **Host Institution:** York University, Department of Electrical Engineering & Computer Science (EECS), Vision Lab.
- **Authors:** Iva Kotseruba, Amir Rasouli, John K. Tsotsos.
- **Access Status:** **`AVAILABLE` (Direct Public Academic Access)**
- **License:** **MIT License** for annotations; **CC BY-NC-SA 4.0** for video clips.
- **Official GitHub:** `https://github.com/ykotseruba/JAAD`
- **Video Server:** `http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip` (Verified HTTP 200, 2.87 GB).
- **Step-by-Step Acquisition Procedure:**
  1. Clone the official annotation repository into a working directory:
     ```bash
     git clone https://github.com/ykotseruba/JAAD.git data/external/JAAD
     ```
  2. Download the official video archive directly from York University:
     ```bash
     cd data/external/JAAD
     curl -L -o JAAD_clips.zip http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
     unzip JAAD_clips.zip
     rm JAAD_clips.zip
     ```
  3. Verify checksum and extract frame crops using SafeKid AI's ingestion parser:
     - Annotations: `data/external/JAAD/annotations/` (per-frame bounding boxes).
     - Attributes: `data/external/JAAD/annotations_attributes/` (demographic labels: `child`, `young`, `adult`, `senior`).

---

### 2.2 PIE (Pedestrian Intent Estimation)
- **Host Institution:** York University, Computer Science & Engineering.
- **Authors:** Amir Rasouli, Iva Kotseruba, Toni Kunic, John K. Tsotsos.
- **Access Status:** **`AVAILABLE` (Direct Public Academic Access)**
- **License:** **CC BY-NC-SA 4.0 (Non-Commercial Research)**
- **Official GitHub:** `https://github.com/aras62/PIEPredict`
- **Official Website:** `http://data.nvision2.eecs.yorku.ca/PIE_dataset/`
- **Step-by-Step Acquisition Procedure:**
  1. Clone the official PIE interface repository:
     ```bash
     git clone https://github.com/aras62/PIEPredict.git data/external/PIE
     ```
  2. Access the video clips from the official York University server directory:
     - Download instructions and split IDs are managed through `pie_data.py`.
  3. Extract XML attributes:
     - Query pedestrian demographic attribute dictionary (`'age': {'child': 0, 'young': 1, 'adult': 2}`).

---

### 2.3 PETA (PEdesTrian Attribute Dataset)
- **Host Institution:** The Chinese University of Hong Kong (CUHK), Multimedia Laboratory (MMLab).
- **Authors:** Yutian Lin, Liang Zheng, Zhedong Zheng, Yu Wu, Shengjin Wang, Yi Yang, Xiaogang Wang.
- **Access Status:** **`AVAILABLE` (Open Academic Research Mirrors)**
- **License:** **Academic Research Use Only (Non-Commercial)**
- **Official Website:** `https://mmlab.ie.cuhk.edu.hk/projects/PETA.html`
- **Official Mirror:** `https://opendatalab.com/OpenDataLab/PETA` / `https://hyper.ai/datasets/15147`
- **Step-by-Step Acquisition Procedure:**
  1. Register a verified academic account on OpenDataLab or access via CUHK MMLab project portal.
  2. Download `PETA.zip` (approx. 19,000 cropped surveillance images, ~350 MB).
  3. Extract into SafeKid AI target directory:
     ```bash
     mkdir -p data/child_dataset/genuine/peta
     unzip PETA.zip -d data/child_dataset/genuine/peta/
     ```
  4. Parse `PETA.mat` or `annotation.mat` using SafeKid AI ingestion engine:
     - Filter for `personalLess15 == 1` and apply teenage gating / uncertainty mapping.

---

### 2.4 RAP v2 (Richly Annotated Pedestrian Dataset v2)
- **Host Institution:** National Laboratory of Pattern Recognition (NLPR), Institute of Automation, Chinese Academy of Sciences (CASIA).
- **Authors:** Dangwei Li, Zhang Zhang, Xiaotang Chen, Kaiqi Huang.
- **Access Status:** **`REQUIRES_AGREEMENT` (Formal Institutional EULA Required)**
- **License:** **Academic Non-Commercial Research EULA**
- **Official Project Website:** `http://www.rapdataset.com/`
- **Step-by-Step Acquisition Procedure:**
  1. Navigate to the official RAP dataset request page: `http://www.rapdataset.com/`
  2. Download the official **RAP Dataset Release Agreement** (PDF/DOC).
  3. Fill out the application form with institutional details:
     - Principal Investigator name and academic email (`.edu` / `.ac`).
     - Affiliated university or research lab.
     - Formal project description (e.g., "Child safety monitoring and demographic fairness in residential surveillance vision systems").
  4. Sign the agreement and submit via email to the primary CASIA coordinators (`dangwei.li@nlpr.ia.ac.cn` / `zzhang@nlpr.ia.ac.cn`).
  5. Upon approval and receipt of the secure download credentials / BaiduYun link, extract the archive into:
     ```bash
     mkdir -p data/child_dataset/genuine/rap_v2
     ```
  6. **Warning:** Direct anonymous downloading without an approved EULA violates terms and is legally prohibited.

---

### 2.5 FA-INRIA (Fairness-Aware INRIA Person Dataset)
- **Host Institution:** Keio University (Aoki Media Lab) & National Institute of Advanced Industrial Science and Technology (AIST), Japan.
- **Authors:** Shunsuke Kogure, Kai Watabe, Ryosuke Yamada, Yoshimitsu Aoki, Akio Nakamura, Hirokatsu Kataoka.
- **Access Status:** **`REQUIRES_REGISTRATION` (Available Upon Academic Request to Authors)**
- **License:** **Academic Research Use Only**
- **Official Publication:** MDPI Computer Sciences & Mathematics Forum, 2022 (`https://www.mdpi.com/2813-0324/3/1/11`).
- **Step-by-Step Acquisition Procedure:**
  1. Download the standard base **INRIA Person Dataset** (public benchmark):
     ```bash
     wget https://pascal.inrialpes.fr/data/human/INRIAPerson.tar
     tar -xvf INRIAPerson.tar -C data/external/
     ```
  2. Contact the corresponding authors at Keio University (Aoki Media Laboratory: `aoki-medialab@keio.jp`) or Dr. Hirokatsu Kataoka at AIST.
  3. Request the supplementary age annotation files (`fairness_inria_annotations.json` / VOC XMLs).
  4. Place the age annotations alongside base INRIA images in:
     ```bash
     data/child_dataset/genuine/fa_inria/
     ```

---

### 2.6 CrowdHuman
- **Host Institution:** Megvii Technology (Face++).
- **Authors:** Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, Jian Sun.
- **Access Status:** **`NOT_COMPATIBLE` (for Child Classification)**
- **Official Portal:** `https://www.crowdhuman.net/`
- **Assessment:** Contains 24,370 dense crowd images with full-body (`fbox`), visible (`vbox`), and head (`hbox`) bounding boxes. **Zero age or child annotations exist in the official release.** Suitable strictly for pretraining generic person detectors or benchmarking occlusion handling.

---

### 2.7 Authorized Gated Residential Community Data
- **Deployment Category:** Live residential enclave pilot (villas, gated apartments).
- **Access Status:** **`REQUIRES_AGREEMENT` (Opt-in Parental Consent Mandate)**
- **Compliance Protocols:**
  1. **Legal Framework:** Must comply with COPPA (Children's Online Privacy Protection Act) and GDPR Article 8 (conditions applicable to child's consent in relation to information society services).
  2. **Opt-in Documentation:** Explicit written consent signed by parents or legal guardians authorizing CCTV capture of minors within designated common areas (e.g. resident swimming pool or playground).
  3. **Data Retention & Encryption:**
     - All residential video recordings must be AES-256 encrypted at rest.
     - Retention period clamped to 30 days maximum, with automated purging.
     - Ingestion into SafeKid AI target path:
       ```
       data/child_dataset/authorized_residential/
       ```

---

## 3. Recommended Dataset Integration Strategy for Prompt 27

| Dataset | Priority | Role in Next Stage (Prompt 27) | Action Required |
| :--- | :---: | :--- | :--- |
| **JAAD** | **Primary (Priority 1)** | Real-world child vs adult temporal classification and crop training | Download official clips and parse open XML annotations. |
| **PIE** | **Secondary (Priority 2)** | Independent test split for continuous video validation | Ingest child tracks for validation of multi-frame temporal smoothing. |
| **PETA** | **Secondary (Priority 3)** | Large-scale diverse pedestrian crop training baseline | Download via academic mirror and apply conservative teenage abstention gating. |
| **RAP v2** | **Awaiting Approval** | CCTV-specific perspective fine-tuning | Execute and submit institutional EULA to CASIA. |
| **FA-INRIA**| **Supplementary** | Static pedestrian crop evaluation | Send academic request to Keio University authors. |

---

*Guide maintained by SafeKid AI Compliance & Engineering Team.*

