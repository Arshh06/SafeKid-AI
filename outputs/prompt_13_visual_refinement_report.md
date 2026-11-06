# SafeKid AI — Prompt 13: Visual Overlay Refinement & Readability Report

**Document ID:** `REPORT-PROMPT-13-VISUAL-REFINEMENT`  
**Execution Timestamp:** `2026-11-04 10:15:00 UTC`  
**Audited Environments:**
1. Scenario A: Playground (`outputs/playground_annotated.mp4`, $1280 \times 720$)
2. Scenario B: Residential Driveway (`outputs/driveway_annotated.mp4`, $1920 \times 1080$)
3. Scenario C: Boom Barrier (`outputs/boom_barrier_annotated.mp4`, $1920 \times 1080$)
4. Scenario D: Swimming Pool (`outputs/pool_safety_annotated.mp4`, $848 \times 480$)

---

## 1. Executive Summary & Improvement Scope

This report documents the visual refinement pass executed on the SafeKid AI monitoring pipeline under Prompt 13. The objective was to eliminate visual overlay clutter, prevent label text collisions, enhance ground-contact point legibility, visually differentiate zone policies, and ensure consistent rendering across disparate camera resolutions ($480p$ to $1080p$).

All visual enhancements were implemented purely in the presentation layer ([`src/safekid_ai/visualization.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/visualization.py) and bounding-box anchor clamping in [`src/safekid_ai/roi/engine.py`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/roi/engine.py)). **Underlying event generation, zone transition state machines, and child safety policy logic were strictly preserved.**

---

## 2. Bounding-Box & Label Readability Improvements

### A. The Baseline Visual Deficiencies
An audit of Prompt 12 visual outputs identified several key readability flaws:
1. **Long Monolithic Labels:** Labels combined Track ID, Adult/Child classification, full zone name, dwell duration, and confidence into a single 50+ character string (e.g. `ID:1 [ADULT] [Dedicated Pedestrian Sidewalk] | 1.2s (0.85)`). At 400px wide, this covered neighboring pedestrians and zone lines.
2. **Horizontal Text Collisions:** When two pedestrians stood within 100px of each other, their top label text blocks rendered at the exact same $y$-coordinate, colliding and becoming illegible.
3. **Fixed Font Scaling:** A constant OpenCV font scale ($0.45$) was previously hardcoded. On $1080p$ streams, text was disproportionately small; on $480p$ streams, text was overly bulky.
4. **Frame Clipping:** Labels for pedestrians standing near the right or top edges clipped beyond frame boundaries.

### B. Implemented Solutions
1. **Resolution-Adaptive Stroke & Font Engine (`get_resolution_scale`):**
   ```python
   def get_resolution_scale(frame_height: int) -> Tuple[float, int, int]:
       scale = max(0.42, min(0.75, frame_height / 1400.0 + 0.22))
       font_thick = 1 if scale < 0.60 else 2
       line_thick = max(1, int(round(frame_height / 600.0)))
       return scale, font_thick, line_thick
   ```
   - **$848 \times 480$ (Pool):** Scale: `0.56`, Thickness: `1`, Bbox line: `1px`
   - **$1280 \times 720$ (Playground):** Scale: `0.73`, Thickness: `2`, Bbox line: `2px`
   - **$1920 \times 1080$ (Gate/Driveway):** Scale: `0.75`, Thickness: `2`, Bbox line: `2px`

2. **Two-Tier Compact Badge Architecture:**
   - **Primary Pill (Header):** `ID:{tid} [{CLASS}] ({conf:.2f})`  
     *Guarantees that Track ID, classification, and detection confidence are permanently visible without truncation.*
   - **Secondary Status Tag (Sub-pill):** Rendered only if the person occupies a zone or has accumulated dwell time (e.g. `Main Driveway (2.1s)`). Long zone names are cleanly truncated to 20 characters with ellipses.

3. **Intelligent Collision-Aware Altitude Staggering:**
   - The engine measures horizontal distance $\Delta x = |x_{\text{center}, i} - x_{\text{center}, j}|$ between all detected persons.
   - When $\Delta x < 130.0\text{ px}$, adjacent tracks are assigned alternating vertical altitude offsets. Track $A$ renders at standard height ($y_1 - \text{badge\_h} - 4$), while Track $B$ is elevated by an additional $(\text{badge\_h} + 4)\text{ px}$, preventing horizontal overlap.

4. **Strict Boundary Clamping:**
   - **Horizontal:** Clamped via `bx1 = max(4, min(fw - badge_w - 4, x1))` and `bx2 = bx1 + badge_w`. Labels never clip against the left or right image edge.
   - **Vertical:** If $y_1$ is close to the top header HUD ($y_1 < \text{header\_h} + \text{badge\_h} + 6$), the badge automatically docks inside the top of the bounding box or beneath the feet ($y_2 + 4$), preventing occlusion of top-bar camera metadata.

---

## 3. Ground-Contact Validation & Anchor Clamping

### A. Anchor Calculation Audit
SafeKid AI enforces a foundational design rule: **Zone containment must be evaluated at the bottom-center of the person bounding box $(x_{\text{center}}, y_2)$, never by bounding box area overlap.**

```
          [Person Bounding Box]
          +-------------------+  (x1, y1)
          |      Head/Torso   |
          |                   |
          |       Waist       |
          |                   |
          |       Legs        |
          +---------*---------+  (x2, y2)
                    ^
             Bottom-Center Anchor: ((x1 + x2)/2, y2)
```

In Prompt 13, [`calculate_bottom_center()`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/roi/engine.py#L20-L40) was enhanced to accept an optional `frame_resolution=(W, H)` parameter. Coordinates are now clamped to valid image pixels:
$$x_{\text{bc}} = \max(0.0, \min(W - 1.0, \frac{x_1 + x_2}{2.0}))$$
$$y_{\text{bc}} = \max(0.0, \min(H - 1.0, y_2))$$

### B. High-Visibility Multi-Ring Target Marker
To ensure the anchor point is immediately distinguishable from surrounding pavement, water, grass, and red bounding boxes, the ground-contact marker was redesigned into a four-element composite target:
1. **Outer Dark Halo:** Radius $R+1$, thickness 2, black contour `(10, 10, 10)` to guarantee contrast against bright concrete, white road markings, and sunlit water.
2. **High-Vis Neon Ring:** Radius $R$, filled with vibrant neon yellow `(0, 235, 255)`.
3. **Center Bullseye Dot:** Radius $R-3$, filled with crimson `(0, 0, 220)` for child/alert tracks, or dark navy for adult tracks.
4. **Precision Crosshair Ticks:** 4 crosshair ticks extending $3\text{ px}$ beyond the outer ring along North, South, East, and West axes.

### C. Physical Ground-Plane Localization Limitations
> [!WARNING]
> **Limitations of Bottom-Center Approximation:**
> 1. **Occluded Feet:** When a person stands behind pool lounge chairs, sandbox walls, or vehicles, the detector's bottom edge $y_2$ aligns with the occluding obstacle, placing the anchor in the air above actual ground level.
> 2. **Frame Boundary Truncation:** When a person is partially cut off at the bottom edge of the frame, $y_2$ is clamped to $H-1$, which does not represent true foot placement.
> 3. **Perspective Distortion:** Without 4-point camera homography calibration, a 2D image pixel $(x, y)$ cannot be mapped to physical metric coordinates $(X_w, Y_w, Z_w = 0)$ on the ground plane.

---

## 4. Zone Role & Alert Visualization Enhancements

### A. Role-Specific Color Differentiation
Previously, all zones were colored solely by alert severity, obscuring whether a zone was a permitted play area or a restricted zone. The new palette establishes an intuitive hierarchy:

| Zone Role | Color Palette (BGR) | Alpha Fill | Contour Stroke | Semantic Intent |
| :--- | :---: | :---: | :---: | :--- |
| **SAFE** | Emerald Soft Green<br>`(140, 210, 80)` | `0.16` | Standard (`line_thick`) | Designated safe play area (sandbox, sidewalk). Reassuring visual tone; zero danger alarms. |
| **MONITORED** | Warm Amber / Gold<br>`(0, 190, 245)` | `0.22` | Medium (`line_thick + 1`) | Common amenity area (pool deck, lawn). Entry permitted; lingering dwell monitored. |
| **DANGER** | Bold Crimson Red<br>`(30, 30, 235)` | `0.26` | Bold (`line_thick + 2`) | Severe hazard zone (water basin, driveway, barrier). Immediate breach warning. |

### B. Zone Centroid Badges
Each zone now displays a permanent, high-contrast centroid pill:
`"[{ROLE}] {Zone Name} [{SEVERITY}]"`  
*(e.g., `"[SAFE] Toddler Sandbox & Swings Area [LOW]"`, `"[DANGER] Boom Barrier Mechanical Zone [CRITICAL]"`).*

### C. Non-Intrusive Alert Banners
Alert messages are docked directly beneath the top header HUD bar. Instead of opaque rectangles that obscure central action, alerts render as compact, semi-transparent crimson capsules (`(0, 0, 195)` BGR) with white text and 1px borders, clamped to a maximum of 2 active alerts to prevent screen overcrowding.

---

## 5. Video-Level Verification Across All Four Scenarios

| Scenario & Resolution | Key Milestone Frames Inspected | Box & ID Stability | Ground-Contact Accuracy | Alert Timestamp Agreement | Observed Rendering Quality |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Scenario A: Playground**<br>($1280 \times 720$, 24 FPS) | Frames 15, 70, 140, 220 | 100% consistent (Track 1 = Child, Track 2 = Adult) | Precise anchor at sandbox sand and hedge boundary | Frame 168 dwell alert matches JSON ($t = 7.00\text{s}$) | Adult supervisor badge in cyan; child in red upon 4.0s hedge dwell. Zero label overlap. |
| **Scenario B: Driveway**<br>($1920 \times 1080$, 24 FPS) | Frames 15, 40, 80, 180 | 100% consistent (Track 1 = Adult, Track 2 = Child) | Precise anchor on sidewalk curb and road surface | Frame 24 entry alert ($t = 1.00\text{s}$) matches JSON | Two-tier label cleanly displays `ID:2 [CHILD] (0.76)` above and `Main Driveway (2.0s)` below. |
| **Scenario C: Boom Barrier**<br>($1920 \times 1080$, 24 FPS) | Frames 8, 25, 45, 120 | 100% consistent (Track 1 = Adult, Track 2 = Child) | Precise anchor entering mechanical swing zone | Frame 28 barrier entry ($t = 1.17\text{s}$) matches JSON | Crimson DANGER zone outline distinguishes boom barrier from surrounding driveway lane. |
| **Scenario D: Swimming Pool**<br>($848 \times 480$, 24 FPS) | Frames 20, 60, 140, 210 | Consistent track lifecycles (Tracks 1, 3, 4, 6, 8) | Precise contact on deck tiles; in-water anchor at surface | Frame 27 water entry ($t = 1.12\text{s}$) matches JSON | High-vis yellow/red crosshair target clearly visible against blue water and white deck coping. |

---

## 6. Implementation Status Matrix

| Visual / Consistency Improvement | Status | Implementation Details & Architectural Notes |
| :--- | :---: | :--- |
| **Resolution-Adaptive Font & Line Scaling** | **IMPLEMENTED** | Dynamic scaling based on frame height in [`get_resolution_scale()`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/visualization.py#L36-L45); verified at 480p, 720p, and 1080p. |
| **Two-Tier Compact Track Labels** | **IMPLEMENTED** | Primary badge (`ID: X [CLASS] (conf)`) separated from secondary zone status tag; no information hidden. |
| **Collision-Aware Altitude Staggering** | **IMPLEMENTED** | Detects tracks with $\Delta x < 130\text{ px}$ and staggers label altitudes, eliminating text collisions. |
| **Frame Boundary Label & Marker Clamping** | **IMPLEMENTED** | Badges and markers strictly clamped within `[4, W-4]` and `[header_h+2, H-4]`. |
| **High-Visibility Target Ground Marker** | **IMPLEMENTED** | 4-element target marker with dark outer halo, neon yellow ring, center dot, and crosshair ticks. |
| **Ground-Contact Calculation Clamping** | **IMPLEMENTED** | Updated [`calculate_bottom_center()`](file:///c:/projs/SafeKid%20AI/src/safekid_ai/roi/engine.py#L20-L40) with `frame_resolution` bounds clamping. |
| **Zone Role Visual Differentiation** | **IMPLEMENTED** | Distinct color palettes, transparencies, and contour strokes for `SAFE`, `MONITORED`, and `DANGER` zones. |
| **Centroid Zone Role Badges** | **IMPLEMENTED** | Centroid badges display `[SAFE]`, `[MONITORED]`, or `[DANGER]` alongside zone name and severity. |
| **Compact Top-Docked Alert Banners** | **IMPLEMENTED** | Replaced screen-spanning opaque boxes with top-docked crimson alert pills clamped to 2 active alerts. |
| **Automated Metric Homography Localization** | **UNRESOLVED** | Ground-contact points remain 2D image coordinates; 3D real-world localization requires physical camera calibration. |
| **Dynamic Occlusion Height Estimation** | **UNRESOLVED** | When feet are occluded by playground equipment or pool furniture, $y_2$ represents the occluding obstacle top. |

