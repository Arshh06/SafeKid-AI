# SafeKid AI — Prompt 14: Alert Presentation, Visual Quality & HUD Refinement Report

**Project:** SafeKid AI — Child Safety Monitoring for Gated Residential Communities  
**Scope:** Alert Visualization Layer Refinement, Crowded-Scene Label Placement, Adaptive Zone Opacity, and Multi-Resolution Video Audit  
**Date:** November 2026  
**Status:** Validated & Deployed  

---

## 1. Executive Summary

This report documents the architectural and visual refinements implemented under **Prompt 14** of the SafeKid AI initiative. The goal was to elevate the readability, operational ergonomics, and non-obscuring properties of the visual overlay layer across all four verified community scenarios:
1. **Swimming Pool** (`pool_safety_demo.mp4`, $848 \times 480$, 24 FPS)
2. **Playground** (`playground_demo.mp4`, $1280 \times 720$, 24 FPS)
3. **Residential Driveway** (`driveway_demo.mp4`, $1920 \times 1080$, 24 FPS)
4. **Boom Barrier Gateway** (`boom_barrier_demo.mp4`, $1920 \times 1080$, 24 FPS)

### Key Achievements
- **Compact Dedicated Alert Panel:** Replaced screen-spanning stacked alert banners with a self-contained, bounded corner panel. Prioritizes `CRITICAL` over `HIGH` alerts, renders the top 2 active alerts, and presents an explicit overflow indicator (`+ N more active alert(s)`) without silently discarding or overwriting any events.
- **Crowded-Scene Label Optimization:** Implemented proximity detection ($\Delta x < 130\text{px}$). In crowded scenes, secondary dwell and zone strings are suppressed to prioritize Track ID, manual classification (`[CHILD]`, `[ADULT]`, `[UNCLASSIFIED]`), and detection confidence.
- **Non-Obscuring Label Placement:** Bounding-box intersection audits test candidate label positions (above bbox, below feet, inside top) against all adjacent persons, preventing label badges from obscuring other pedestrians.
- **Adaptive Zone Opacity & Perimeter Badges:** Large zones (area $> 15\%$ of frame, e.g., vehicle driveway lane) use reduced fill opacity ($0.12 - 0.16$) and perimeter-anchored badges to preserve background asphalt and pedestrian visibility, while compact danger zones retain high-contrast fill ($0.20 - 0.24$) and crisp 2–3px boundary contours.
- **Full Test Suite Verification:** Expanded test suite with 4 new targeted visual regression tests; all **109 pytest tests pass** with zero failures.

---

## 2. Alert Presentation Layer Redesign

### 2.1 Problems with Legacy Alert Banners
In earlier iterations, active safety alerts were drawn as horizontal banners stretching across the top-left of the viewport. In scenes with multiple concurrent alerts or high-resolution frames:
- Long alert banners stretched across up to $80\%$ of the screen width, partially covering persons walking in the upper-left or center-left of the scene.
- When more than two alerts fired simultaneously (e.g., driveway entry + dwell + barrier entry), lower-priority alerts competed for visual space without strict severity prioritization.

### 2.2 Redesigned Alert Panel Specifications

| Parameter | Specification | Implementation Detail |
| :--- | :--- | :--- |
| **Container Geometry** | Fixed bounded card | `panel_w = max(280, min(int(round(fw * 0.38)), 460))` |
| **Docking Position** | Upper-right corner (below header) | `panel_x = fw - panel_w - 12`, `panel_y = header_h + 6` |
| **Collision Evasion** | Dynamic corner avoidance | Shifts to upper-left (`panel_x = 12`) if a person detection intersects upper-right |
| **Severity Ranking** | Strict priority hierarchy | `CRITICAL (4) > HIGH (3) > MEDIUM (2) > LOW (1)` |
| **Displayed Capacity** | Top 2 active alerts | Top 2 alerts displayed with severity pills (`[CRIT]`, `[HIGH]`, `[MED]`) |
| **Overflow Indication** | Explicit count badge | `+ N more active alert(s)` rendered in high-visibility amber |
| **Event Retention** | 100% event preservation | Underlying events, timestamps, and JSON logs remain completely intact |
| **Styling & Theme** | Dark container with border | Background `(18, 22, 28)`, border color dynamically matching highest active severity |

```
┌─────────────────────────────────────────────────────────────┐
│ SafeKid AI | Community 001 | Cam: camera_gate_north | Tracks: 2 │  <-- Top Header HUD
└─────────────────────────────────────────────────────────────┘
                                  ┌───────────────────────────┐
                                  │ SAFETY ALERTS (3 ACTIVE)  │  <-- Alert Panel
                                  ├───────────────────────────┤
                                  │ [CRIT] Child in gate zone │  <-- Rank 1
                                  ├───────────────────────────┤
                                  │ [HIGH] Driveway dwell >2s │  <-- Rank 2
                                  ├───────────────────────────┤
                                  │ + 1 more active alert(s)  │  <-- Overflow
                                  └───────────────────────────┘
```

---

## 3. Label Collision Handling & Crowded Scenes

### 3.1 Proximity Detection & Dwell Text Suppression
When two or more tracked persons walk side-by-side or cross paths (horizontal center difference $\Delta x < 130.0\text{px}$):
- Displaying multi-line labels (Track ID + Classification + Zone + Dwell Time) created vertical footprints up to 60px tall, which frequently occluded the adjacent person's face or upper body.
- Under Prompt 14, `is_crowded_track` triggers automatic suppression of the secondary zone and dwell text.
- The badge is reduced to a single compact line: `f"ID:{tid} [{cls_name}] ({det.confidence:.2f})"`.
- For isolated tracks, the complete secondary dwell line (`f"{short_zone} ({dwell:.1f}s)"`) remains visible.

### 3.2 Non-Obscuring Label Placement Algorithm
To ensure badges do not obscure other tracked persons, candidate badge coordinates are evaluated against the bounding boxes of all other detections:
1. **Candidate 1 (Above Box):** $y_1 - \text{badge\_h} - 4 - \text{lift}$. Valid if above the header and does not intersect any adjacent detection.
2. **Candidate 2 (Below Feet):** $y_2 + 4 + \text{lift}$. Valid if within frame bounds and does not intersect any adjacent detection.
3. **Candidate 3 (Inside Top):** $\max(\text{header\_h} + 2, y_1 + 4)$. Safely placed inside the person's own bounding box if exterior space is obstructed.

Horizontal coordinates are strictly clamped: $4 \le x_1 \le x_2 \le \text{frame\_width} - 4$.

---

## 4. Zone Overlay Refinement

### 4.1 Adaptive Opacity Scaling
Previously, a single uniform alpha blend factor ($0.22$) was applied across all zone types. In environments like the residential driveway, the vehicular lane covered over $39\%$ of the viewport, visibly darkening the roadway.

The new implementation computes relative area:
$$\text{rel\_area} = \frac{\sum \text{Area}(\text{polygons})}{\text{Frame Width} \times \text{Frame Height}}$$

- **Large Zones ($\text{rel\_area} > 0.15$):**
  - `DANGER`: Alpha reduced to $0.16$ (38% reduction in darkening).
  - `MONITORED`: Alpha reduced to $0.14$.
  - `SAFE`: Alpha reduced to $0.12$.
- **Compact Critical Zones ($\text{rel\_area} \le 0.15$):**
  - `DANGER`: High-contrast alpha $0.24$.
  - `MONITORED`: Alpha $0.20$.
  - `SAFE`: Alpha $0.16$.
- **Boundary Contours:** Rendered on the blended frame at full opacity using anti-aliased polylines with role-proportional thickness (3px for `DANGER`, 2px for `MONITORED`/`SAFE`).

### 4.2 Perimeter Zone Badge Placement
In large zones, centroid placement placed the text badge in the center of the road or walkway. For large polygons ($\text{rel\_area} > 0.15$), the zone badge is now anchored near the upper perimeter:
$$x_{\text{badge}} = \min(x_i) + 40\text{px}, \quad y_{\text{badge}} = \min(y_i) + 36\text{px}$$
This keeps the central travel area clear of static badges.

---

## 5. Multi-Resolution Verification Matrix

| Environment | Video File | Resolution | FPS | Frames | Alert Panel Dimensions | Zone Opacity Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Swimming Pool** | `outputs/pool_safety_annotated.mp4` | $848 \times 480$ | 24.0 | 240 | $322\text{px} \times 96\text{px}$ | High-contrast basin ($0.24$), monitored deck ($0.20$) |
| **Playground** | `outputs/playground_annotated.mp4` | $1280 \times 720$ | 24.0 | 240 | $440\text{px} \times 112\text{px}$ | Perimeter fence ($0.24$), sandbox safe ($0.16$) |
| **Driveway** | `outputs/driveway_annotated.mp4` | $1920 \times 1080$ | 24.0 | 192 | $460\text{px} \times 116\text{px}$ | Driveway lane ($0.16$ adaptive), sidewalk ($0.12$) |
| **Boom Barrier** | `outputs/boom_barrier_annotated.mp4` | $1920 \times 1080$ | 24.0 | 144 | $460\text{px} \times 116\text{px}$ | Barrier gate ($0.24$), roadway ($0.16$) |

All contact sheets have been regenerated in `outputs/visual_contact_sheets/` confirming crisp typography, bounded badges, and clear background visibility.

---

## 6. Regression Testing Summary

The complete test suite was executed via pytest. All 109 tests passed with zero failures:

```
=========================== 109 passed, 5 warnings in 88.31s ===========================
```

### New Targeted Tests Added in `tests/test_safety_events.py`:
1. `test_alert_prioritization_critical_over_high`: Verifies strict ranking: CRITICAL (score 4) > HIGH (score 3) > MEDIUM (score 2) > LOW (score 1).
2. `test_alert_panel_overflow_indication`: Validates `+ N more active alert(s)` indicator when $\ge 3$ alerts fire.
3. `test_crowded_scene_label_compactness`: Verifies suppression of secondary dwell/zone string when tracks are within 130px.
4. `test_large_roi_polygon_adaptive_opacity_and_perimeter_badge`: Verifies adaptive opacity factor ($0.16$) and perimeter-based badge offset for large zones ($> 15\%$ area).

