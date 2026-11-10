"""Extracts and records detector configurations for YOLOv8n and D-FINE-N."""

import hashlib
import json
from pathlib import Path
import torch

from safekid_ai.config import DetectorConfig
from safekid_ai.detector import PersonDetector
from safekid_ai.detectors.dfine_detector import DFINEDetectorAdapter

def hash_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest(), path.stat().st_size

def record_configs():
    print("Recording detector configurations...")
    
    # Checkpoints
    yolo_ckpt = Path("yolov8n.pt")
    dfine_ckpt = Path("models/dfine_n_coco.pth")
    child_ckpt = Path("models/child_classifier_jaad_v1.pt")
    
    yolo_hash, yolo_size = hash_file(yolo_ckpt)
    dfine_hash, dfine_size = hash_file(dfine_ckpt)
    child_hash, child_size = hash_file(child_ckpt)
    
    # Thread settings
    torch.set_num_threads(4)
    active_threads = torch.get_num_threads()
    
    # Load detectors
    yolo_cfg = DetectorConfig(model_name_or_path=str(yolo_ckpt), device="cpu", confidence_threshold=0.25, iou_threshold=0.45, imgsz=640)
    yolo_det = PersonDetector(yolo_cfg)
    
    dfine_cfg = DetectorConfig(model_name_or_path=str(dfine_ckpt), device="cpu", confidence_threshold=0.25, iou_threshold=0.45, imgsz=640)
    dfine_det = DFINEDetectorAdapter(dfine_cfg, variant="dfine_n")
    
    # Parameters
    yolo_params = sum(p.numel() for p in yolo_det.model.model.parameters())
    dfine_params = sum(p.numel() for p in dfine_det.model.parameters())
    
    configs = {
        "execution_environment": {
            "device": "cpu",
            "cpu_architecture": "x86_64",
            "cpu_model": "Intel Core i3-1215U (6 cores / 8 threads: 2P + 4E)",
            "simd_extensions": ["SSE4.1", "SSE4.2", "AVX2"],
            "avx512_supported": False,
            "configured_pytorch_threads": active_threads,
            "interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity": "Unpinned (Windows default dispatching across logical threads 0-7)",
            "total_ram_gb": 7.68
        },
        "preserved_child_classifier_checkpoint": {
            "filename": str(child_ckpt).replace("\\", "/"),
            "sha256": child_hash,
            "size_bytes": child_size,
            "status": "PRESERVED_UNTOUCHED"
        },
        "detectors": {
            "yolov8n": {
                "model_name": "YOLOv8n",
                "architecture_family": "Single-stage Anchor-Free CNN with Distribution Focal Loss",
                "checkpoint_filename": str(yolo_ckpt).replace("\\", "/"),
                "sha256": yolo_hash,
                "size_bytes": yolo_size,
                "parameter_count": yolo_params,
                "gflops": 8.7,
                "input_resolution": [640, 640],
                "color_space": "RGB",
                "normalization": "[0.0, 1.0]",
                "letterbox_padding_value": [114, 114, 114],
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "postprocessing": "Non-Maximum Suppression (Ultralytics torchvision nms)",
                "target_class": "person (class_id=0)",
                "layer_fusion": True
            },
            "dfine_n": {
                "model_name": "D-FINE-N",
                "architecture_family": "Real-time DETR Transformer with Fine-grained Distribution Refinement",
                "checkpoint_filename": str(dfine_ckpt).replace("\\", "/"),
                "sha256": dfine_hash,
                "size_bytes": dfine_size,
                "parameter_count": dfine_params,
                "gflops": 7.0,
                "input_resolution": [640, 640],
                "color_space": "RGB",
                "normalization": "[0.0, 1.0]",
                "letterbox_padding_value": [114, 114, 114],
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "postprocessing": "Query score ranking + D-FINE FDR coordinate decoding + NMS duplicate suppression",
                "target_class": "person (class_id=0)",
                "layer_fusion": False
            }
        }
    }
    
    out_path = Path("outputs/prompt_36_detector_configuration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    print(f"Configurations saved to {out_path}")

if __name__ == "__main__":
    record_configs()
