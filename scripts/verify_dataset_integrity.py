"""Verification of JAAD and VIRAT dataset and annotation integrity for Prompt 36."""

import os
import random
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from safekid_ai.datasets.jaad_parser import JAADAnnotationParser
from safekid_ai.datasets.virat_loader import VIRATDatasetLoader

def verify_datasets():
    print("=== Phase 1: Verifying JAAD Dataset Integrity ===")
    jaad_dir = Path("data/external/jaad")
    parser = JAADAnnotationParser()
    
    videos = sorted(list((jaad_dir / "videos").glob("*.mp4")))
    jaad_summary = []
    all_boxes_by_video = {}
    
    # Track-level demographic mapping
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        is_open = cap.isOpened()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        ann_path = jaad_dir / "repo" / "annotations" / f"{v.stem}.xml"
        attr_path = jaad_dir / "repo" / "annotations_attributes" / f"{v.stem}_attributes.xml"
        
        boxes, rep = parser.parse_video_annotations(ann_path, attr_path, w, h)
        all_boxes_by_video[v.stem] = boxes
        
        # Check coordinate bounds
        out_of_bounds = 0
        zero_area = 0
        for b in boxes:
            if b.xtl < 0 or b.ytl < 0 or b.xbr > w or b.ybr > h:
                out_of_bounds += 1
            if b.width <= 0 or b.height <= 0:
                zero_area += 1
                
        # Count by demographic class
        child_count = sum(1 for b in boxes if b.harmonized_label.value == "CHILD")
        adult_count = sum(1 for b in boxes if b.harmonized_label.value == "ADULT")
        unk_count = sum(1 for b in boxes if b.harmonized_label.value == "UNKNOWN")
        
        # Count unique tracks
        unique_tracks = len(set(b.track_id for b in boxes))
        child_tracks = len(set(b.track_id for b in boxes if b.harmonized_label.value == "CHILD"))
        
        # Check occlusion breakdown
        none_occl = sum(1 for b in boxes if b.occlusion_level == "none")
        part_occl = sum(1 for b in boxes if b.occlusion_level == "part")
        full_occl = sum(1 for b in boxes if b.occlusion_level == "full")
        
        # Max annotated frame
        max_ann_frame = max(b.frame_id for b in boxes) if boxes else 0
        
        status = "PASS" if is_open and max_ann_frame < total_frames and zero_area == 0 else "FAIL"
        
        jaad_summary.append({
            "video_id": v.stem,
            "filename": v.name,
            "can_open": is_open,
            "width": w,
            "height": h,
            "fps": fps,
            "video_frames": total_frames,
            "max_ann_frame": max_ann_frame,
            "total_boxes": len(boxes),
            "unique_tracks": unique_tracks,
            "child_tracks": child_tracks,
            "child_boxes": child_count,
            "adult_boxes": adult_count,
            "unknown_boxes": unk_count,
            "none_occl": none_occl,
            "part_occl": part_occl,
            "full_occl": full_occl,
            "out_of_bounds": out_of_bounds,
            "zero_area": zero_area,
            "status": status,
        })
        print(f"[{status}] {v.name}: {w}x{h}, {total_frames} frames, {len(boxes)} boxes ({child_count} child, {adult_count} adult), {unique_tracks} tracks ({child_tracks} child tracks)")

    print("\n=== Phase 1: Verifying VIRAT Dataset Integrity ===")
    virat_loader = VIRATDatasetLoader(root_dir="data/virat")
    seq_ids = virat_loader.list_sequences(complete_only=True)
    virat_summary = []
    
    for s_id in seq_ids:
        seq = virat_loader.load_sequence(s_id)
        m = seq.metadata
        print(f"VIRAT sequence: {s_id}, frames: {m.total_frames}, annotations: {m.annotation_frame_count}, persons: {m.person_instance_count}, unique tracks: {m.unique_person_tracks}")
        virat_summary.append(m.to_dict())

    # Visual Inspection of 20 randomly selected annotated frames
    print("\n=== Phase 1: Visual Inspection of 20 Annotated Frames ===")
    random.seed(42)
    visual_audit_dir = Path("outputs/prompt_36_visual_audit")
    visual_audit_dir.mkdir(parents=True, exist_ok=True)
    
    all_candidates = []
    for vid, boxes in all_boxes_by_video.items():
        frames_in_vid = sorted(list(set(b.frame_id for b in boxes)))
        for fid in frames_in_vid:
            all_candidates.append((vid, fid))
            
    selected_20 = random.sample(all_candidates, 20)
    selected_20.sort()
    
    visual_audit_records = []
    for vid, fid in selected_20:
        v_path = jaad_dir / "videos" / f"{vid}.mp4"
        cap = cv2.VideoCapture(str(v_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print(f"Error reading frame {fid} from {vid}")
            continue
            
        frame_boxes = [b for b in all_boxes_by_video[vid] if b.frame_id == fid]
        overlay = frame.copy()
        
        box_details = []
        for b in frame_boxes:
            color = (0, 255, 0) if b.harmonized_label.value == "ADULT" else ((0, 0, 255) if b.harmonized_label.value == "CHILD" else (255, 255, 0))
            cv2.rectangle(overlay, (int(b.xtl), int(b.ytl)), (int(b.xbr), int(b.ybr)), color, 2)
            lbl = f"{b.track_id}:{b.harmonized_label.value}:{b.occlusion_level}"
            cv2.putText(overlay, lbl, (int(b.xtl), max(15, int(b.ytl) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            box_details.append({
                "track_id": b.track_id,
                "label": b.harmonized_label.value,
                "bbox": [round(b.xtl, 1), round(b.ytl, 1), round(b.xbr, 1), round(b.ybr, 1)],
                "occlusion": b.occlusion_level,
                "action": b.action
            })
            
        out_name = f"audit_{vid}_f{fid:04d}.jpg"
        out_path = visual_audit_dir / out_name
        cv2.imwrite(str(out_path), overlay)
        
        visual_audit_records.append({
            "video_id": vid,
            "frame_id": fid,
            "box_count": len(frame_boxes),
            "boxes": box_details,
            "image_path": str(out_path).replace("\\", "/")
        })
        print(f"Verified visual audit frame: {vid} frame {fid} ({len(frame_boxes)} persons)")

    import json
    results = {
        "jaad_summary": jaad_summary,
        "virat_summary": virat_summary,
        "visual_audit_records": visual_audit_records,
    }
    with open("scratch/dataset_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Verification data written to scratch/dataset_verification_results.json")
    return jaad_summary, virat_summary, visual_audit_records

if __name__ == "__main__":
    verify_datasets()
