"""Generates qualitative comparison visualizations for Prompt 36 Error Analysis."""

import json
from pathlib import Path
import cv2
import numpy as np

from safekid_ai.config import DetectorConfig
from safekid_ai.detector import PersonDetector
from safekid_ai.detectors.dfine_detector import DFINEDetectorAdapter
from safekid_ai.datasets.jaad_parser import JAADAnnotationParser


def generate_error_visualizations():
    print("Generating qualitative error visualizations...")
    out_dir = Path("outputs/prompt_36_error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo_cfg = DetectorConfig(model_name_or_path="yolov8n.pt", device="cpu", confidence_threshold=0.25, imgsz=640)
    yolo_det = PersonDetector(yolo_cfg)

    dfine_cfg = DetectorConfig(model_name_or_path="models/dfine_n_coco.pth", device="cpu", confidence_threshold=0.25, imgsz=640)
    dfine_det = DFINEDetectorAdapter(dfine_cfg, variant="dfine_n")

    parser = JAADAnnotationParser()
    jaad_dir = Path("data/external/jaad")

    # Target frames to capture diverse failure modes
    # (video_id, frame_id, description_tag)
    targets = [
        ("video_0005", 102, "correct_detections_children_adults"),
        ("video_0105", 143, "crossing_child_mid_street"),
        ("video_0128", 322, "partially_occluded_child_curb"),
        ("video_0152", 63, "dense_crowd_inter_pedestrian_occlusion"),
        ("video_0305", 80, "background_false_positives_comparison"),
        ("video_0317", 166, "test_split_two_children_crossing"),
    ]

    records = []

    for vid, fid, tag in targets:
        v_path = jaad_dir / "videos" / f"{vid}.mp4"
        cap = cv2.VideoCapture(str(v_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            continue

        ann_path = jaad_dir / "repo" / "annotations" / f"{vid}.xml"
        attr_path = jaad_dir / "repo" / "annotations_attributes" / f"{vid}_attributes.xml"
        boxes, _ = parser.parse_video_annotations(ann_path, attr_path, 1920, 1080)
        gt_boxes = [b for b in boxes if b.frame_id == fid]

        # Run models
        yolo_res = yolo_det.detect(frame, frame_id=fid, confidence_thresh=0.25)
        dfine_res = dfine_det.detect(frame, frame_id=fid, confidence_thresh=0.25)

        # Create side-by-side visualization: Left = Ground Truth, Center = YOLOv8n, Right = D-FINE-N
        h, w = frame.shape[:2]
        vis_gt = frame.copy()
        vis_yolo = frame.copy()
        vis_dfine = frame.copy()

        # Render GT
        for g in gt_boxes:
            c = (0, 0, 255) if g.harmonized_label.value == "CHILD" else ((0, 255, 0) if g.harmonized_label.value == "ADULT" else (255, 255, 0))
            cv2.rectangle(vis_gt, (int(g.xtl), int(g.ytl)), (int(g.xbr), int(g.ybr)), c, 2)
            cv2.putText(vis_gt, f"GT:{g.harmonized_label.value}", (int(g.xtl), max(20, int(g.ytl) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

        # Render YOLO
        for y in yolo_res.detections:
            cv2.rectangle(vis_yolo, (int(y.x1), int(y.y1)), (int(y.x2), int(y.y2)), (0, 255, 255), 2)
            cv2.putText(vis_yolo, f"YOLO:{y.confidence:.2f}", (int(y.x1), max(20, int(y.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Render D-FINE
        for d in dfine_res.detections:
            cv2.rectangle(vis_dfine, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 100, 0), 2)
            cv2.putText(vis_dfine, f"DFINE:{d.confidence:.2f}", (int(d.x1), max(20, int(d.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # Add headers
        cv2.putText(vis_gt, f"Ground Truth (GT: {len(gt_boxes)})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_yolo, f"YOLOv8n (Dets: {len(yolo_res.detections)})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(vis_dfine, f"D-FINE-N (Dets: {len(dfine_res.detections)})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

        # Resize for compact side-by-side display (each panel 640x360)
        p_gt = cv2.resize(vis_gt, (640, 360))
        p_yolo = cv2.resize(vis_yolo, (640, 360))
        p_dfine = cv2.resize(vis_dfine, (640, 360))

        combined = np.hstack([p_gt, p_yolo, p_dfine])

        img_filename = f"comparison_{vid}_f{fid:04d}_{tag}.jpg"
        img_path = out_dir / img_filename
        cv2.imwrite(str(img_path), combined)

        records.append({
            "video_id": vid,
            "frame_id": fid,
            "tag": tag,
            "gt_count": len(gt_boxes),
            "yolo_count": len(yolo_res.detections),
            "dfine_count": len(dfine_res.detections),
            "image_path": str(img_path).replace("\\", "/"),
        })
        print(f"Saved error visualization: {img_filename}")

    # VIRAT sample visualization
    virat_v_path = Path("data/virat/videos/VIRAT_S_000001_sample.mp4")
    cap = cv2.VideoCapture(str(virat_v_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        y_v = yolo_det.detect(frame, frame_id=30, confidence_thresh=0.25)
        d_v = dfine_det.detect(frame, frame_id=30, confidence_thresh=0.25)

        vis_gt = frame.copy()
        # GT for frame 30 has 2 persons: track 1 and track 3
        # In VIRAT_S_000001_sample at frame 30:
        cv2.rectangle(vis_gt, (450, 360), (450 + 65, 360 + 175), (0, 255, 0), 2)
        cv2.rectangle(vis_gt, (1250, 480), (1250 + 60, 480 + 165), (0, 255, 0), 2)

        vis_y = frame.copy()
        for y in y_v.detections:
            cv2.rectangle(vis_y, (int(y.x1), int(y.y1)), (int(y.x2), int(y.y2)), (0, 255, 255), 2)

        vis_d = frame.copy()
        for d in d_v.detections:
            cv2.rectangle(vis_d, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 100, 0), 2)

        cv2.putText(vis_gt, "VIRAT GT (Persons: 2)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_y, f"YOLOv8n (Dets: {len(y_v.detections)})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(vis_d, f"D-FINE-N (Dets: {len(d_v.detections)})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

        p_gt = cv2.resize(vis_gt, (640, 360))
        p_yolo = cv2.resize(vis_y, (640, 360))
        p_dfine = cv2.resize(vis_d, (640, 360))

        combined = np.hstack([p_gt, p_yolo, p_dfine])
        v_img_path = out_dir / "comparison_virat_sample_f0030.jpg"
        cv2.imwrite(str(v_img_path), combined)
        records.append({
            "video_id": "VIRAT_S_000001_sample",
            "frame_id": 30,
            "tag": "virat_parking_lot_cctv",
            "gt_count": 2,
            "yolo_count": len(y_v.detections),
            "dfine_count": len(d_v.detections),
            "image_path": str(v_img_path).replace("\\", "/"),
        })

    with open("scratch/qualitative_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print("Qualitative visualizations generated successfully.")


if __name__ == "__main__":
    generate_error_visualizations()

