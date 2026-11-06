# SafeKid AI — Visual Validation & Rendering Quality Audit

**Document ID:** `AUDIT-12-VISUAL-VALIDATION`  
**Execution Timestamp:** `2026-11-04 10:45:00 UTC`  
**Audited Artifacts:**
- Contact Sheets: `outputs/visual_contact_sheets/*.jpg`
- Milestone Frames: `outputs/visual_contact_sheets/<scenario>/frame_*.png`
- Annotated Videos: `outputs/*_annotated.mp4`

---

## 1. Executive Summary

Visual feedback is a critical operational component of SafeKid AI. When a child safety alert is dispatched to a residential guard console or community property manager, the annotated video stream must immediately and unequivocally communicate:
1. Who is in danger (Child vs. Adult).
2. Where the hazard is (Zone boundary geometry).
3. Why the alert fired (Immediate breach vs. lingering dwell).
4. Ground-contact point positioning (Feet on ground vs. bounding box perspective overlap).

This audit inspects representative milestone frames across all four demonstration environments and documents visual rendering quality, color-coding efficacy, and UI layout stability.

---

## 2. Visual Contact Sheets Inventory

Four comprehensive 4-panel visual contact sheets were generated in `outputs/visual_contact_sheets/`:

| Environment | Contact Sheet File | Extracted Milestones | Video Resolution |
| :--- | :--- | :--- | :---: |
| **Playground** | `playground_contact_sheet.jpg` | Frame 15 (Sandbox play)<br>Frame 70 (Perimeter approach)<br>Frame 140 (Dwell alert triggered)<br>Frame 220 (Safe sandbox return) | $1280 \times 720$ |
| **Driveway** | `driveway_contact_sheet.jpg` | Frame 15 (Sidewalk presence)<br>Frame 40 (Roadway entry alert)<br>Frame 80 (Roadway dwell alert)<br>Frame 180 (Sidewalk safe return) | $1920 \times 1080$ |
| **Boom Barrier** | `boom_barrier_contact_sheet.jpg` | Frame 08 (North approach)<br>Frame 25 (Driveway entry alert)<br>Frame 45 (Boom barrier critical alert)<br>Frame 120 (Safe gate exit) | $1920 \times 1080$ |
| **Swimming Pool** | `pool_contact_sheet.jpg` | Frame 20 (Poolside monitoring)<br>Frame 60 (Water basin critical entry)<br>Frame 140 (Deck dwell alert)<br>Frame 210 (Climb-out recovery) | $848 \times 480$ |

---

## 3. Detailed Inspection of Visual Elements

### A. Track Bounding Boxes & Classification Badges
- **Color Coding:**
  - Standard tracked persons in non-alert states are rendered in **Lime Green** (`#00FF00`).
  - Tracked children triggering active danger alerts are rendered in **Vivid Red** (`#0000FF` BGR).
- **Classification Badges:**
  - Above each bounding box, an explicit text pill displays:
    * `[CHILD] ID: 2 (0.78)` for verified children.
    * `[ADULT] ID: 1 (0.85)` for verified adults.
    * `[UNKNOWN] ID: X (0.65)` for unclassified tracks.
- **Verification:** The badge makes adult exclusion immediately obvious to monitoring guards—adult guards standing on the sidewalk or parents sitting on benches never display red danger boxes.

### B. Ground-Contact Point (Bottom-Center Anchor)
- **Visual Representation:** A bright yellow filled circle with a dark target outline is anchored at $(x_{\text{center}}, y_2)$.
- **Critical Function:**
  - Distinguishes ground contact from bounding box perspective overlap.
  - In elevated surveillance perspectives, a tall person standing outside an ROI often has their upper torso or head projecting over the ROI boundary. By rendering the yellow ground-contact anchor at the feet, guards can visually confirm that containment evaluation is geometrically accurate.

### C. Zone Boundary Overlay Clarity
- **Rendering Technique:** Semi-transparent polygon fill (`alpha = 0.22`) with anti-aliased 2px boundary contours (`cv2.polylines`).
- **Zone Role Color Hierarchy:**
  - `SAFE` zones (e.g., Toddler Sandbox, Pedestrian Sidewalk): Light Blue / Cyan.
  - `MONITORED` zones (e.g., Pool Deck Enclosure): Amber / Gold.
  - `DANGER` zones (e.g., Pool Basin, Vehicle Driveway, Boom Barrier): Crimson / Red.
- **Observation:** Background asphalt, pavers, and water remain clearly visible through the semi-transparent polygon fills without obscuring pedestrian footing.

### D. Alert Banners & Header HUD
- **Top HUD:** Displays community name, camera ID, active frame counter, and presentation timestamp in high-contrast white text against a darkened header bar.
- **Alert Banners:**
  - Rendered dynamically across the upper viewport when safety events fire.
  - `CHILD ALERT: Child (Track 2) entered configured danger zone 'Main Vehicular Driveway' [Severity: HIGH]`
  - `CHILD DWELL ALERT: Child (Track 1) remained in danger zone 'North Perimeter Fence' beyond configured threshold (4.0s) [Severity: HIGH]`
- **Duration:** Banners persist for $2.5\text{ seconds}$ post-event to allow human operators adequate reading time.

---

## 4. Documented Visual Rendering Glitches & UI Limitations

While overall rendering quality is strong, this audit identified three specific visual defects that should be resolved in the next iteration:

### 1. Label Collisions When Tracks Are Spatially Proximate
- **Issue:** When two people stand within 80 horizontal pixels of each other, their classification badges and dwell-time text tags overlap, rendering the text illegible.
- **Remedy:** Implement dynamic label altitude offsetting (staggering badge heights based on neighboring track proximity).

### 2. Fixed Font Scale Across Divergent Resolutions
- **Issue:** OpenCV `cv2.putText()` uses a static font scale ($0.55 - 0.70$).
  - On $848 \times 480$ (Pool Demo), text is large and easily readable.
  - On $1920 \times 1080$ (Driveway / Boom Barrier), text appears noticeably smaller relative to the frame height.
- **Remedy:** Dynamically scale font size, padding, and line thickness proportional to frame height ($\text{scale} = \max(0.5, H / 1080.0)$).

### 3. Vertical Stacking Overflow During Alert Bursts
- **Issue:** When a child triggers both driveway and boom barrier alerts within 375 ms, multiple banner boxes stack vertically from $y = 30$ down to $y = 120$. On lower-resolution streams ($480p$), 3 active banners cover nearly 25% of the vertical frame.
- **Remedy:** Integrate the proposed Alert Correlation Engine to display a single, consolidated incident banner.

