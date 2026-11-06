"""Visual rendering utilities for SafeKid AI community safety zones and tracked entities."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from safekid_ai.detector import DetectionBox
from safekid_ai.events import SafetyEvent, SafetyEventType
from safekid_ai.roi.models import AlertSeverity, SafetyZone, ZoneRole
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.visualization")

# Color mapping in BGR
SEVERITY_COLORS: Dict[AlertSeverity, Tuple[int, int, int]] = {
    AlertSeverity.LOW: (100, 220, 100),       # Soft green
    AlertSeverity.MEDIUM: (0, 220, 255),      # Bright yellow
    AlertSeverity.HIGH: (0, 140, 255),        # Deep orange
    AlertSeverity.CRITICAL: (0, 0, 235),      # Bold red
}

# Role-specific color mapping for zone polygons and boundaries
ZONE_ROLE_COLORS: Dict[ZoneRole, Tuple[int, int, int]] = {
    ZoneRole.SAFE: (140, 210, 80),          # Emerald soft green
    ZoneRole.MONITORED: (0, 190, 245),      # Warm amber/gold
    ZoneRole.DANGER: (30, 30, 235),         # High-alert crimson red
}

ZONE_ROLE_ALPHA: Dict[ZoneRole, float] = {
    ZoneRole.SAFE: 0.16,
    ZoneRole.MONITORED: 0.20,
    ZoneRole.DANGER: 0.24,
}

# Lighter opacity factors for large area polygons (> 15% frame area) to prevent obscuring scene
LARGE_ZONE_ROLE_ALPHA: Dict[ZoneRole, float] = {
    ZoneRole.SAFE: 0.12,
    ZoneRole.MONITORED: 0.14,
    ZoneRole.DANGER: 0.16,
}

SEVERITY_ORDER: Dict[AlertSeverity, int] = {
    AlertSeverity.CRITICAL: 4,
    AlertSeverity.HIGH: 3,
    AlertSeverity.MEDIUM: 2,
    AlertSeverity.LOW: 1,
}


def extract_alert_info(alert_item: Any) -> Tuple[str, AlertSeverity]:
    """Extracts alert message and severity rank from string, tuple, or structured alert object."""
    if isinstance(alert_item, tuple):
        # Format: (exp, msg, sev) or (msg, sev)
        if len(alert_item) >= 3:
            sev = alert_item[2] if isinstance(alert_item[2], AlertSeverity) else AlertSeverity.HIGH
            return str(alert_item[1]), sev
        elif len(alert_item) == 2:
            if isinstance(alert_item[1], AlertSeverity):
                return str(alert_item[0]), alert_item[1]
            return str(alert_item[0]), AlertSeverity.HIGH

    if hasattr(alert_item, "message") and hasattr(alert_item, "severity"):
        sev = alert_item.severity if isinstance(alert_item.severity, AlertSeverity) else AlertSeverity.HIGH
        return str(alert_item.message), sev

    msg = str(alert_item)
    upper = msg.upper()
    if "CRITICAL" in upper:
        return msg, AlertSeverity.CRITICAL
    if "HIGH" in upper or "DANGER" in upper:
        return msg, AlertSeverity.HIGH
    if "MEDIUM" in upper or "WARN" in upper or "MONITORED" in upper or "DWELL" in upper:
        return msg, AlertSeverity.MEDIUM
    if "LOW" in upper or "SAFE" in upper:
        return msg, AlertSeverity.LOW
    return msg, AlertSeverity.HIGH


def get_resolution_scale(frame_height: int) -> Tuple[float, int, int]:
    """Computes font scale, font thickness, and line stroke thickness proportional to frame resolution.

    Ensures text and lines remain legible at 480p, 720p, and 1080p without visual overcrowding.
    """
    scale = max(0.42, min(0.75, frame_height / 1400.0 + 0.22))
    font_thick = 1 if scale < 0.60 else 2
    line_thick = max(1, int(round(frame_height / 600.0)))
    return scale, font_thick, line_thick


def draw_roi_polygons(
    frame: np.ndarray,
    zones: List[SafetyZone],
    alpha: float = 0.22,
    detections: Optional[List[DetectionBox]] = None,
) -> np.ndarray:
    """Renders configured ROI polygons with distinct visual styling for SAFE, MONITORED, and DANGER zones.

    Enhancements for visual clarity and non-obscuring overlays:
    - Adaptive fill opacity: large zones (e.g. driveways, area > 15% of frame) use lighter opacity
      (0.12-0.16) to avoid obscuring background asphalt and pedestrians, while compact danger zones
      (water basin, boom barrier gate) retain high-contrast opacity (0.20-0.24).
    - Crisp boundary contours: drawn at full opacity with role-proportional line thickness.
    - Intelligent badge placement: large zones place badges near upper boundary edges rather than
      center centroids to prevent blocking the primary pedestrian/vehicle pathway.
    """
    if not zones:
        return frame

    vis = frame.copy()
    fh, fw = frame.shape[:2]
    frame_area = float(fw * fh)
    font_scale, font_thick, line_thick = get_resolution_scale(fh)

    for zone in zones:
        if not zone.enabled:
            continue

        # Determine zone role
        zone_role = zone.policy.zone_role if zone.policy else None
        if zone_role is None:
            if zone.severity == AlertSeverity.CRITICAL:
                zone_role = ZoneRole.DANGER
            elif zone.severity == AlertSeverity.LOW:
                zone_role = ZoneRole.SAFE
            else:
                zone_role = ZoneRole.MONITORED

        color = ZONE_ROLE_COLORS.get(zone_role, SEVERITY_COLORS.get(zone.severity, (0, 200, 255)))
        contour_thick = line_thick + 1 if zone_role == ZoneRole.DANGER else line_thick

        active_polys = zone.polygons if zone.polygons else [zone.polygon]
        total_zone_area = sum(p.area for p in active_polys)
        rel_area = total_zone_area / frame_area if frame_area > 0 else 0.0

        # Size-adaptive opacity: reduce opacity for large regions to prevent obscuring scene
        if rel_area > 0.15:
            zone_alpha = LARGE_ZONE_ROLE_ALPHA.get(zone_role, 0.15)
        else:
            zone_alpha = ZONE_ROLE_ALPHA.get(zone_role, alpha)

        # 1. Fill polygon with adaptive opacity using sub-bounding-box ROI slicing to eliminate multi-megabyte full-frame copies
        pts_list = [np.array(poly.to_coordinates(), dtype=np.int32).reshape((-1, 1, 2)) for poly in active_polys]
        min_bx = max(0, min(pts[:, 0, 0].min() for pts in pts_list))
        min_by = max(0, min(pts[:, 0, 1].min() for pts in pts_list))
        max_bx = min(fw, max(pts[:, 0, 0].max() for pts in pts_list) + 1)
        max_by = min(fh, max(pts[:, 0, 1].max() for pts in pts_list) + 1)

        if max_bx > min_bx and max_by > min_by:
            sub_vis = vis[min_by:max_by, min_bx:max_bx]
            sub_overlay = sub_vis.copy()
            offset = np.array([min_bx, min_by], dtype=np.int32)
            offset_polys = [pts - offset for pts in pts_list]
            cv2.fillPoly(sub_overlay, offset_polys, color)
            cv2.addWeighted(sub_overlay, zone_alpha, sub_vis, 1.0 - zone_alpha, 0, sub_vis)

        # 2. Draw crisp boundary contour on blended frame
        cv2.polylines(vis, pts_list, isClosed=True, color=color, thickness=contour_thick, lineType=cv2.LINE_AA)

        # 3. Intelligent zone badge placement
        primary_poly = max(active_polys, key=lambda p: p.area) if active_polys else zone.polygon

        # In large zones, avoid placing badge in middle where persons move; place near upper perimeter
        if rel_area > 0.15:
            all_pts = primary_poly.to_coordinates()
            min_x = min(pt[0] for pt in all_pts)
            min_y = min(pt[1] for pt in all_pts)
            ix = int(round(min_x + 40))
            iy = int(round(min_y + 36))
        else:
            cx, cy = primary_poly.centroid
            ix, iy = int(round(cx)), int(round(cy))

        badge_text = f"[{zone_role.value}] {zone.name} [{zone.severity.value}]"
        (tw, th), bl = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, font_thick)

        pad_x, pad_y = 5, 3
        bx1 = max(4, min(fw - tw - 2 * pad_x - 4, ix - tw // 2 - pad_x))
        by1 = max(36, min(fh - th - 2 * pad_y - 4, iy - th // 2 - pad_y))
        bx2 = bx1 + tw + 2 * pad_x
        by2 = by1 + th + 2 * pad_y

        # Badge pill with dark background and colored border
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (20, 24, 28), -1)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 1)
        cv2.putText(
            vis,
            badge_text,
            (bx1 + pad_x, by2 - pad_y - bl // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.75,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    return vis


def draw_person_and_events(
    frame: np.ndarray,
    detections: List[DetectionBox],
    occupied_zones_per_track: Dict[Optional[int], List[str]],
    dwell_times_per_track: Dict[Optional[int], Dict[str, float]],
    events: List[SafetyEvent],
    community_name: str,
    camera_id: str,
    active_alerts: Optional[Union[List[str], List[Tuple[Any, ...]]]] = None,
    track_classifications: Optional[Dict[int, str]] = None,
    child_events: Optional[List[Any]] = None,
    incidents: Optional[List[Any]] = None,
    copy: bool = True,
) -> np.ndarray:
    """Renders person bounding boxes, collision-aware labels, ground-contact markers, and alerts.

    Features:
    - Resolution-adaptive text and line metrics for 480p, 720p, and 1080p.
    - Intelligent collision-aware label placement: checks bounding box overlaps with adjacent tracks.
    - Crowded scene handling: prioritises Person ID, classification, and confidence, suppressing
      secondary dwell strings when tracks are proximate (< 130px) to prevent visual occlusion.
    - Compact dedicated Alert Panel: docked in upper corner, prioritizing CRITICAL over HIGH alerts,
      displaying the top 2 alerts with an explicit '+ N more active alert(s)' overflow indicator.
    - High-visibility multi-ring ground-contact target marker anchored to feet.
    """
    vis = frame.copy() if copy else frame
    fh, fw = frame.shape[:2]
    font_scale, font_thick, line_thick = get_resolution_scale(fh)

    header_h = max(30, int(round(fh / 26.0)))

    # Track IDs with active events
    event_track_ids = {e.track_id for e in events if e.track_id is not None}
    child_event_track_ids = {
        getattr(ce, "track_id", None) for ce in (child_events or []) if getattr(ce, "track_id", None) is not None
    }

    # Detect horizontal proximity between tracks for collision-aware label stacking and crowded-scene logic
    det_centers_x = []
    for det in detections:
        x1, _, x2, _ = det.bbox_xyxy
        det_centers_x.append((x1 + x2) / 2.0)

    collision_offset_level: Dict[int, int] = {}
    is_crowded_track: Dict[int, bool] = {}
    for i in range(len(detections)):
        collision_offset_level[i] = 0
        is_crowded_track[i] = False
        for j in range(len(detections)):
            if i != j and abs(det_centers_x[i] - det_centers_x[j]) < 130.0:
                is_crowded_track[i] = True
                if j < i:
                    collision_offset_level[i] = (collision_offset_level[j] + 1) % 2

    def bbox_overlap(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    for det_idx, det in enumerate(detections):
        tid = det.track_id
        x1, y1, x2, y2 = [int(round(coord)) for coord in det.bbox_xyxy]

        # Clamp bounding box display coordinates to valid frame
        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(0, min(fw - 1, x2))
        y2 = max(0, min(fh - 1, y2))

        # Bottom-center ground contact coordinates, strictly clamped to valid frame pixels
        bc_x = max(0, min(fw - 1, int(round((x1 + x2) / 2.0))))
        bc_y = max(0, min(fh - 1, int(round(y2))))

        zones = occupied_zones_per_track.get(tid, [])
        is_in_zone = len(zones) > 0
        has_event = tid in event_track_ids
        has_child_alert = tid in child_event_track_ids

        # Classification label tag
        cls_name = "UNCLASSIFIED"
        is_child = False
        is_adult = False
        if track_classifications and tid in track_classifications:
            cls_name = track_classifications[tid].upper()
            is_child = (cls_name == "CHILD")
            is_adult = (cls_name == "ADULT")

        # Check dwell threshold
        dwell_map = dwell_times_per_track.get(tid, {})
        max_dwell = max(dwell_map.values()) if dwell_map else 0.0
        is_dwell_alert = any(
            e.track_id == tid and e.event_type == SafetyEventType.DWELL_TIME_EXCEEDED for e in events
        ) or any(
            getattr(ce, "track_id", None) == tid and getattr(ce, "event_type", None) == "CHILD_DWELL_TIME_EXCEEDED"
            for ce in (child_events or [])
        )

        # Color scheme
        if has_child_alert or (is_child and is_in_zone and not is_adult):
            box_color = (0, 0, 245)       # High-alert red
            text_bg_color = (0, 0, 200)
            box_thick = line_thick + 2
        elif is_adult:
            box_color = (220, 180, 60)    # Soft teal / cyan (designated safe adult)
            text_bg_color = (30, 40, 45)
            box_thick = line_thick
        elif has_event or is_dwell_alert:
            box_color = (0, 110, 255)     # Deep orange
            text_bg_color = (0, 90, 220)
            box_thick = line_thick + 1
        elif is_in_zone:
            box_color = (0, 205, 255)     # Warm amber
            text_bg_color = (25, 30, 35)
            box_thick = line_thick
        else:
            box_color = (180, 180, 180)   # Neutral slate
            text_bg_color = (30, 30, 30)
            box_thick = line_thick

        # 1. Draw Bounding Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thick)

        # 2. Draw High-Visibility Multi-Ring Ground-Contact Marker
        r_outer = max(6, int(round(fh / 130.0)))
        r_inner = max(3, r_outer - 3)
        # Outer dark ring for contrast on bright surfaces
        cv2.circle(vis, (bc_x, bc_y), r_outer + 1, (10, 10, 10), 2, lineType=cv2.LINE_AA)
        # High-visibility neon yellow ring
        cv2.circle(vis, (bc_x, bc_y), r_outer, (0, 235, 255), -1, lineType=cv2.LINE_AA)
        # Center target dot (crimson if child/alert, navy if adult)
        target_dot_color = (0, 0, 220) if is_child or has_child_alert else (140, 80, 20)
        cv2.circle(vis, (bc_x, bc_y), r_inner, target_dot_color, -1, lineType=cv2.LINE_AA)
        # Crosshair ticks (4px extending N, S, E, W)
        tick_len = max(3, int(round(fh / 250.0)))
        cv2.line(vis, (bc_x, bc_y - r_outer - tick_len), (bc_x, bc_y - r_outer), (10, 10, 10), 2, cv2.LINE_AA)
        cv2.line(vis, (bc_x, bc_y + r_outer), (bc_x, bc_y + r_outer + tick_len), (10, 10, 10), 2, cv2.LINE_AA)
        cv2.line(vis, (bc_x - r_outer - tick_len, bc_y), (bc_x - r_outer, bc_y), (10, 10, 10), 2, cv2.LINE_AA)
        cv2.line(vis, (bc_x + r_outer, bc_y), (bc_x + r_outer + tick_len, bc_y), (10, 10, 10), 2, cv2.LINE_AA)

        # 3. Intelligent Collision-Aware Track Label Badges
        tid_str = f"ID:{tid}" if tid is not None else "Person"
        primary_label = f"{tid_str} [{cls_name}] ({det.confidence:.2f})"

        (pw, ph), p_bl = cv2.getTextSize(primary_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad_x, pad_y = 6, 4
        badge_w = pw + 2 * pad_x
        badge_h = ph + 2 * pad_y

        # In crowded scenes, prioritize Person ID and Classification over optional dwell/zone text
        secondary_label = ""
        if zones and not is_crowded_track.get(det_idx, False):
            short_zone = zones[0]
            if len(short_zone) > 20:
                short_zone = short_zone[:18] + "..."
            if max_dwell > 0.4:
                secondary_label = f"{short_zone} ({max_dwell:.1f}s)"
            else:
                secondary_label = f"{short_zone}"

        sw, sh = 0, 0
        if secondary_label:
            (sw, sh), _ = cv2.getTextSize(secondary_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, font_thick)
            badge_w = max(badge_w, sw + 2 * pad_x)
            badge_h += sh + pad_y

        # Horizontal clamping: prevent label extending beyond left or right frame edge
        bx1 = max(4, min(fw - badge_w - 4, x1))
        bx2 = bx1 + badge_w

        # Vertical placement with collision-aware offset and alternative placement
        offset_multiplier = collision_offset_level.get(det_idx, 0)
        extra_lift = offset_multiplier * (badge_h + 4)

        cand_above_y1 = y1 - badge_h - 4 - extra_lift
        cand_above_y2 = cand_above_y1 + badge_h
        cand_below_y1 = y2 + 4 + extra_lift
        cand_below_y2 = cand_below_y1 + badge_h
        cand_inside_y1 = max(header_h + 2, y1 + 4)
        cand_inside_y2 = cand_inside_y1 + badge_h

        # Check collision with other person bounding boxes to avoid obscuring other people
        other_boxes = [d.bbox_xyxy for j, d in enumerate(detections) if j != det_idx]
        above_collides = cand_above_y1 <= header_h or any(
            bbox_overlap((bx1, cand_above_y1, bx2, cand_above_y2), ob) for ob in other_boxes
        )
        below_collides = cand_below_y2 >= fh - 35 or any(
            bbox_overlap((bx1, cand_below_y1, bx2, cand_below_y2), ob) for ob in other_boxes
        )

        if not above_collides and cand_above_y1 > header_h:
            by1, by2 = cand_above_y1, cand_above_y2
        elif not below_collides and cand_below_y2 < fh - 35:
            by1, by2 = cand_below_y1, cand_below_y2
        else:
            by1, by2 = cand_inside_y1, cand_inside_y2

        # Draw Badge Pill
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), text_bg_color, -1)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_color, 1)

        # Primary Text
        cv2.putText(
            vis,
            primary_label,
            (bx1 + pad_x, by1 + pad_y + ph),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

        # Secondary Text (if present and not suppressed)
        if secondary_label:
            sec_color = (0, 230, 255) if is_child or is_dwell_alert else (200, 210, 220)
            cv2.putText(
                vis,
                secondary_label,
                (bx1 + pad_x, by1 + pad_y + ph + sh + pad_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.85,
                sec_color,
                font_thick,
                cv2.LINE_AA,
            )

    # 4. Top Status Header HUD
    header_text = f"SafeKid AI | {community_name} | Cam: {camera_id} | Tracks: {len(detections)}"
    cv2.rectangle(vis, (0, 0), (fw, header_h), (18, 22, 26), -1)
    cv2.line(vis, (0, header_h), (fw, header_h), (50, 60, 70), 1)
    cv2.putText(
        vis,
        header_text,
        (12, int(round(header_h * 0.68))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.95,
        (240, 245, 250),
        font_thick,
        cv2.LINE_AA,
    )

    # 5. Compact Dedicated Safety Alert / Incident Panel
    if incidents is not None and len(incidents) > 0:
        # Structured incident lifecycle presentation
        open_incidents = [
            inc for inc in incidents
            if (hasattr(inc, "is_open") and inc.is_open())
            or (getattr(inc, "state", None) in ("ACTIVE", "ACKNOWLEDGED"))
            or (hasattr(getattr(inc, "state", None), "value") and inc.state.value in ("ACTIVE", "ACKNOWLEDGED"))
        ]
        sorted_incidents = sorted(
            open_incidents,
            key=lambda x: (
                -SEVERITY_ORDER.get(getattr(x, "severity", AlertSeverity.HIGH), 0),
                getattr(x, "created_at_ms", 0.0),
            ),
        )
        total_inc_count = len(sorted_incidents)
        if total_inc_count > 0:
            top_incidents = sorted_incidents[:2]
            excess_inc_count = total_inc_count - len(top_incidents)
            highest_sev = getattr(top_incidents[0], "severity", AlertSeverity.HIGH)

            # Resolution-adaptive panel dimensions
            panel_w = max(310, min(int(round(fw * 0.40)), 470))
            panel_x = fw - panel_w - 12
            panel_y = header_h + 6

            # Avoid occluding any person bounding box located in upper-right quadrant
            ur_quadrant = (panel_x - 10, panel_y, fw, panel_y + 160)
            if any(bbox_overlap(det.bbox_xyxy, ur_quadrant) for det in detections):
                ul_quadrant = (12, panel_y, 12 + panel_w, panel_y + 160)
                if not any(bbox_overlap(det.bbox_xyxy, ul_quadrant) for det in detections):
                    panel_x = 12

            # Compute row metrics
            card_pad = 8
            title_h = max(20, int(round(fh / 48.0)))
            row_h = max(24, int(round(fh / 36.0)))
            overflow_h = max(18, int(round(fh / 52.0))) if excess_inc_count > 0 else 0

            panel_h = title_h + len(top_incidents) * (row_h + 4) + (overflow_h + 4 if excess_inc_count > 0 else 0) + card_pad * 2

            # 1. Panel Container with dark background and severity-coded border
            border_color = SEVERITY_COLORS.get(highest_sev, (0, 0, 235))
            cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), -1)
            cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), border_color, 2)

            # 2. Panel Title Strip
            title_text = f"SAFETY INCIDENTS ({total_inc_count} ACTIVE)"
            cv2.putText(
                vis,
                title_text,
                (panel_x + card_pad, panel_y + card_pad + int(round(title_h * 0.72))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.82,
                (240, 245, 250),
                font_thick,
                cv2.LINE_AA,
            )
            # Pulse indicator dot
            dot_color = (0, 0, 245) if highest_sev == AlertSeverity.CRITICAL else (0, 140, 255)
            cv2.circle(
                vis,
                (panel_x + panel_w - card_pad - 6, panel_y + card_pad + title_h // 2),
                max(3, int(round(fh / 200.0))),
                dot_color,
                -1,
                lineType=cv2.LINE_AA,
            )

            # 3. Top 2 Incidents Rendered as Structured Compact Rows
            cur_y = panel_y + card_pad + title_h + 4
            for inc in top_incidents:
                row_bg = (28, 32, 40)
                sev = getattr(inc, "severity", AlertSeverity.HIGH)
                sev_color = SEVERITY_COLORS.get(sev, (0, 140, 255))
                cv2.rectangle(vis, (panel_x + card_pad, cur_y), (panel_x + panel_w - card_pad, cur_y + row_h), row_bg, -1)
                cv2.rectangle(vis, (panel_x + card_pad, cur_y), (panel_x + panel_w - card_pad, cur_y + row_h), sev_color, 1)

                # Incident ID badge
                inc_id_str = getattr(inc, "incident_id", "INC")
                cv2.putText(
                    vis,
                    inc_id_str,
                    (panel_x + card_pad + 4, cur_y + int(round(row_h * 0.68))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.72,
                    (210, 225, 240),
                    font_thick,
                    cv2.LINE_AA,
                )
                (itw, _), _ = cv2.getTextSize(inc_id_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.72, font_thick)

                # Severity tag pill
                sev_tag = f"[{sev.value[:4] if hasattr(sev, 'value') else str(sev)[:4]}]"
                sev_x = panel_x + card_pad + itw + 6
                cv2.putText(
                    vis,
                    sev_tag,
                    (sev_x, cur_y + int(round(row_h * 0.68))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.72,
                    sev_color,
                    font_thick,
                    cv2.LINE_AA,
                )
                (stw, _), _ = cv2.getTextSize(sev_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.72, font_thick)

                # State tag pill ([ACT], [ACK], with observation health indicator)
                obs_val = getattr(inc, "observation_status", "OBSERVING")
                if hasattr(obs_val, "value"):
                    obs_val = obs_val.value
                st_val = inc.state.value if hasattr(getattr(inc, "state", None), "value") else str(getattr(inc, "state", "ACT"))

                if obs_val == "DISCONNECTED":
                    state_tag = "[OFFLINE]"
                    state_color = (0, 0, 255)  # High visibility red: camera stream offline
                elif obs_val == "TRACK_LOST":
                    state_tag = "[LOST]"
                    state_color = (0, 140, 255)  # High visibility orange: track lost while in hazard
                elif obs_val == "OCCLUDED":
                    state_tag = "[OCCL]"
                    state_color = (0, 215, 255)  # Amber: momentary occlusion
                elif "ACK" in st_val:
                    state_tag = "[ACK]"
                    state_color = (0, 230, 255)  # Cyan
                else:
                    state_tag = "[ACT]"
                    state_color = (100, 220, 100)  # Green

                state_x = sev_x + stw + 6
                cv2.putText(
                    vis,
                    state_tag,
                    (state_x, cur_y + int(round(row_h * 0.68))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.72,
                    state_color,
                    font_thick,
                    cv2.LINE_AA,
                )
                (state_w, _), _ = cv2.getTextSize(state_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.72, font_thick)

                # Track ID and Hazard Summary
                zname = getattr(inc, "primary_zone_name", "Hazard")
                info_text = f"T:{getattr(inc, 'track_id', '?')} | {zname}"
                info_x = state_x + state_w + 6
                avail_w = (panel_x + panel_w - card_pad) - info_x - 6

                display_text = info_text
                (mtw, _), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.70, 1)
                while mtw > avail_w and len(display_text) > 8:
                    display_text = display_text[:-4] + "..."
                    (mtw, _), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.70, 1)

                cv2.putText(
                    vis,
                    display_text,
                    (info_x, cur_y + int(round(row_h * 0.68))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.70,
                    (240, 245, 250),
                    1,
                    cv2.LINE_AA,
                )
                cur_y += row_h + 4

            # 4. Overflow Indicator Row
            if excess_inc_count > 0:
                overflow_text = f"+ {excess_inc_count} more active incident(s)"
                cv2.putText(
                    vis,
                    overflow_text,
                    (panel_x + card_pad + 4, cur_y + int(round(overflow_h * 0.72))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.72,
                    (0, 205, 255),
                    font_thick,
                    cv2.LINE_AA,
                )
    else:
        # Fallback to legacy raw alert list
        raw_alerts = active_alerts if active_alerts is not None else [
            (e.message, e.severity) for e in events
        ]
        if raw_alerts:
            # Deduplicate while preserving highest severity
            parsed_alerts: Dict[str, AlertSeverity] = {}
            for item in raw_alerts:
                msg, sev = extract_alert_info(item)
                clean_msg = msg.strip()
                if clean_msg:
                    if clean_msg not in parsed_alerts or SEVERITY_ORDER[sev] > SEVERITY_ORDER[parsed_alerts[clean_msg]]:
                        parsed_alerts[clean_msg] = sev

            # Sort strictly descending by severity: CRITICAL (4) > HIGH (3) > MEDIUM (2) > LOW (1)
            sorted_alerts = sorted(
                parsed_alerts.items(),
                key=lambda item: SEVERITY_ORDER.get(item[1], 0),
                reverse=True,
            )

            total_alerts_count = len(sorted_alerts)
            if total_alerts_count > 0:
                top_alerts = sorted_alerts[:2]
                excess_alerts_count = total_alerts_count - len(top_alerts)
                highest_sev = top_alerts[0][1]

                # Resolution-adaptive panel dimensions
                panel_w = max(280, min(int(round(fw * 0.38)), 460))
                panel_x = fw - panel_w - 12
                panel_y = header_h + 6

                # Avoid occluding any person bounding box located in upper-right quadrant
                ur_quadrant = (panel_x - 10, panel_y, fw, panel_y + 160)
                if any(bbox_overlap(det.bbox_xyxy, ur_quadrant) for det in detections):
                    ul_quadrant = (12, panel_y, 12 + panel_w, panel_y + 160)
                    if not any(bbox_overlap(det.bbox_xyxy, ul_quadrant) for det in detections):
                        panel_x = 12

                # Compute row metrics
                card_pad = 8
                title_h = max(20, int(round(fh / 48.0)))
                row_h = max(24, int(round(fh / 36.0)))
                overflow_h = max(18, int(round(fh / 52.0))) if excess_alerts_count > 0 else 0

                panel_h = title_h + len(top_alerts) * (row_h + 4) + (overflow_h + 4 if excess_alerts_count > 0 else 0) + card_pad * 2

                # 1. Panel Container with dark background and severity-coded border
                border_color = SEVERITY_COLORS.get(highest_sev, (0, 0, 235))
                cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), -1)
                cv2.rectangle(vis, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), border_color, 2)

                # 2. Panel Title Strip
                title_text = f"SAFETY ALERTS ({total_alerts_count} ACTIVE)"
                cv2.putText(
                    vis,
                    title_text,
                    (panel_x + card_pad, panel_y + card_pad + int(round(title_h * 0.72))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.82,
                    (240, 245, 250),
                    font_thick,
                    cv2.LINE_AA,
                )
                # Pulse indicator dot
                dot_color = (0, 0, 245) if highest_sev == AlertSeverity.CRITICAL else (0, 140, 255)
                cv2.circle(
                    vis,
                    (panel_x + panel_w - card_pad - 6, panel_y + card_pad + title_h // 2),
                    max(3, int(round(fh / 200.0))),
                    dot_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

                # 3. Top 2 Alerts Rendered as Compact Rows
                cur_y = panel_y + card_pad + title_h + 4
                for alert_msg, sev in top_alerts:
                    row_bg = (28, 32, 40)
                    sev_color = SEVERITY_COLORS.get(sev, (0, 140, 255))
                    cv2.rectangle(vis, (panel_x + card_pad, cur_y), (panel_x + panel_w - card_pad, cur_y + row_h), row_bg, -1)
                    cv2.rectangle(vis, (panel_x + card_pad, cur_y), (panel_x + panel_w - card_pad, cur_y + row_h), sev_color, 1)

                    # Severity tag pill
                    sev_tag = f"[{sev.value[:4]}]"
                    cv2.putText(
                        vis,
                        sev_tag,
                        (panel_x + card_pad + 4, cur_y + int(round(row_h * 0.68))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.75,
                        sev_color,
                        font_thick,
                        cv2.LINE_AA,
                    )

                    # Message text with truncation to prevent panel overflow
                    (stw, _), _ = cv2.getTextSize(sev_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, font_thick)
                    avail_w = panel_w - 2 * card_pad - stw - 16

                    display_text = alert_msg
                    (mtw, _), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.72, 1)
                    while mtw > avail_w and len(display_text) > 10:
                        display_text = display_text[:-4] + "..."
                        (mtw, _), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.72, 1)

                    cv2.putText(
                        vis,
                        display_text,
                        (panel_x + card_pad + stw + 10, cur_y + int(round(row_h * 0.68))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.72,
                        (230, 235, 240),
                        1,
                        cv2.LINE_AA,
                    )
                    cur_y += row_h + 4

                # 4. Overflow Indicator Row (if > 2 alerts exist)
                if excess_alerts_count > 0:
                    overflow_text = f"+ {excess_alerts_count} more active alert(s)"
                    cv2.putText(
                        vis,
                        overflow_text,
                        (panel_x + card_pad + 4, cur_y + int(round(overflow_h * 0.72))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.72,
                        (0, 205, 255),
                        font_thick,
                        cv2.LINE_AA,
                    )

    return vis
