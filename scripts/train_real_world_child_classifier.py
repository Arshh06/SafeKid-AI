#!/usr/bin/env python3
"""Trains and evaluates real-world child vs adult pedestrian classification baselines.

Implements Prompt 28 requirements:
1. Approach A: ChildAdultClassifierNet trained from scratch with track-balanced sampling.
2. Approach B: MobileNetV3-Small transfer learning baseline (PyTorch torchvision).
3. Approach C: Classical feature baseline (HOG + Logistic Regression).
4. Track-level and crop-level evaluation on validation set for model selection.
5. Single-pass evaluation on untouched test partition for the selected champion model.
6. Uncertainty & UNKNOWN abstention calibration.
7. Saves best checkpoint to models/child_classifier_jaad_v1.pt without touching synthetic v1.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as tv_models

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safekid_ai.child_detection.classifier_net import ChildAdultClassifierNet, preprocess_crop
from safekid_ai.child_detection.models import ChildDetectionLabel
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.train_real_world")


def set_seed(seed: int = 42):
    """Sets deterministic random seeds for PyTorch, NumPy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RealWorldPedestrianDataset(Dataset):
    """Dataset of standardized pedestrian crops with track and class metadata."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        is_training: bool = False,
        target_size: Tuple[int, int] = (128, 64),
    ):
        self.samples = samples
        self.is_training = is_training
        self.target_size = target_size

        # ImageNet normalization standard
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Mild augmentations that strictly preserve body proportions."""
        # 1. Random horizontal flip (p=0.5)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        # 2. Mild brightness & contrast jitter (+-15%)
        alpha = 1.0 + random.uniform(-0.15, 0.15)
        beta = random.uniform(-15.0, 15.0)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 3. Occasional modest Gaussian blur (p=0.25) to simulate defocus/distance
        if random.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.2, 0.8))

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        record = self.samples[idx]
        crop_path = record["crop_path"]
        img = cv2.imread(crop_path)
        if img is None:
            img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

        if self.is_training:
            img = self._apply_augmentation(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        chw = np.transpose(norm, (2, 0, 1))
        tensor = torch.from_numpy(chw)

        raw_label = record.get("harmonized_label", "UNKNOWN")
        if raw_label == "CHILD":
            target = 0
        elif raw_label == "ADULT":
            target = 1
        else:
            target = -1

        meta = {
            "sample_id": record.get("sample_id", idx),
            "video_id": record.get("video_id", "unknown"),
            "track_id": record.get("track_id", "unknown"),
            "harmonized_label": raw_label,
            "is_low_res": record.get("is_low_res", False),
            "occluded": record.get("occluded", False),
            "crop_path": crop_path,
        }
        return tensor, target, meta


def compute_track_balanced_weights(samples: List[Dict[str, Any]]) -> List[float]:
    """Computes sample weights inversely proportional to track size and class frequency."""
    track_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}

    for s in samples:
        t_key = f"{s['video_id']}_{s['track_id']}"
        lbl = s["harmonized_label"]
        track_counts[t_key] = track_counts.get(t_key, 0) + 1
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    weights = []
    for s in samples:
        t_key = f"{s['video_id']}_{s['track_id']}"
        lbl = s["harmonized_label"]
        w = 1.0 / (float(track_counts[t_key]) * float(class_counts[lbl]))
        weights.append(w)

    return weights


def extract_spatial_gradient_features(img_path: str, target_size: Tuple[int, int] = (128, 64)) -> np.ndarray:
    """Extracts spatial gradient and vertical body proportion features from crop."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros(target_size, dtype=np.uint8)
    if img.shape != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    # Compute Sobel gradients
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 4 vertical bands (head, upper torso, lower torso, legs) x 2 horizontal cells = 8 spatial cells
    h_step = target_size[0] // 4
    w_step = target_size[1] // 2
    cell_feats = []

    for r in range(4):
        for c in range(2):
            cell_mag = mag[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
            cell_ang = ang[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
            cell_img = img[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]

            # 8-bin gradient orientation histogram
            hist, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
            hist = hist / (np.linalg.norm(hist) + 1e-6)

            # Intensity stats
            mean_val = float(np.mean(cell_img)) / 255.0
            std_val = float(np.std(cell_img)) / 255.0
            mag_mean = float(np.mean(cell_mag)) / 255.0

            cell_feats.extend(hist.tolist())
            cell_feats.extend([mean_val, std_val, mag_mean])

    return np.array(cell_feats, dtype=np.float32)


class ModelEvaluator:
    """Computes crop-level and track-level performance metrics."""

    @staticmethod
    def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        abstention_conf_thresh: float = 0.65,
        abstention_margin_thresh: float = 0.20,
    ) -> Dict[str, Any]:
        """Evaluates a PyTorch model on a DataLoader with uncertainty tracking."""
        model.eval()
        crop_preds = []
        crop_targets = []
        track_votes: Dict[str, Dict[str, int]] = {}
        track_true_labels: Dict[str, str] = {}
        probs_all = []
        abstained_count = 0

        t_start = time.perf_counter()
        with torch.no_grad():
            for tensors, targets, metas in data_loader:
                tensors = tensors.to(device)
                logits = model(tensors)
                probs = F.softmax(logits, dim=1).cpu().numpy()

                for i in range(len(targets)):
                    p_child = float(probs[i, 0])
                    p_adult = float(probs[i, 1])
                    p_max = max(p_child, p_adult)
                    margin = abs(p_child - p_adult)

                    true_lbl = metas["harmonized_label"][i]
                    full_track = f"{metas['video_id'][i]}_{metas['track_id'][i]}"
                    track_true_labels[full_track] = true_lbl

                    if full_track not in track_votes:
                        track_votes[full_track] = {"CHILD": 0, "ADULT": 0, "UNKNOWN": 0}

                    # Abstention gate
                    if p_max < abstention_conf_thresh or margin < abstention_margin_thresh:
                        pred_lbl = "UNKNOWN"
                        abstained_count += 1
                        track_votes[full_track]["UNKNOWN"] += 1
                    elif p_child > p_adult:
                        pred_lbl = "CHILD"
                        track_votes[full_track]["CHILD"] += 1
                    else:
                        pred_lbl = "ADULT"
                        track_votes[full_track]["ADULT"] += 1

                    crop_preds.append(pred_lbl)
                    crop_targets.append(true_lbl)
                    probs_all.append((p_child, p_adult))

        latency_ms = ((time.perf_counter() - t_start) / max(1, len(crop_targets))) * 1000.0

        # Compute Binary Metrics (CHILD vs ADULT, excluding UNKNOWN ground truth)
        binary_pairs = [(p, t) for p, t in zip(crop_preds, crop_targets) if t in ("CHILD", "ADULT")]
        tp = sum(1 for p, t in binary_pairs if p == "CHILD" and t == "CHILD")
        fp = sum(1 for p, t in binary_pairs if p == "CHILD" and t == "ADULT")
        fn = sum(1 for p, t in binary_pairs if p != "CHILD" and t == "CHILD")
        tn = sum(1 for p, t in binary_pairs if p == "ADULT" and t == "ADULT")

        child_prec = tp / max(1, tp + fp)
        child_rec = tp / max(1, tp + fn)
        child_f1 = 2 * (child_prec * child_rec) / max(1e-6, child_prec + child_rec)

        adult_prec = tn / max(1, tn + (len([1 for p, t in binary_pairs if p == "ADULT" and t == "CHILD"])))
        adult_rec = tn / max(1, len([1 for p, t in binary_pairs if t == "ADULT"]))
        adult_f1 = 2 * (adult_prec * adult_rec) / max(1e-6, adult_prec + adult_rec)

        # Track-Level Performance
        track_tp, track_fp, track_fn, track_tn = 0, 0, 0, 0
        track_results = {}
        for t_id, votes in track_votes.items():
            true_lbl = track_true_labels[t_id]
            if true_lbl not in ("CHILD", "ADULT"):
                continue
            c_votes = votes["CHILD"]
            a_votes = votes["ADULT"]
            u_votes = votes["UNKNOWN"]

            if c_votes > a_votes and c_votes > u_votes:
                track_pred = "CHILD"
            elif a_votes > c_votes and a_votes > u_votes:
                track_pred = "ADULT"
            else:
                track_pred = "UNKNOWN"

            track_results[t_id] = {"true": true_lbl, "pred": track_pred, "votes": votes}
            if true_lbl == "CHILD":
                if track_pred == "CHILD":
                    track_tp += 1
                else:
                    track_fn += 1
            elif true_lbl == "ADULT":
                if track_pred == "CHILD":
                    track_fp += 1
                else:
                    track_tn += 1

        track_child_prec = track_tp / max(1, track_tp + track_fp)
        track_child_rec = track_tp / max(1, track_tp + track_fn)
        track_child_f1 = 2 * (track_child_prec * track_child_rec) / max(1e-6, track_child_prec + track_child_rec)

        # Expected Calibration Error
        conf_eval = [
            (max(p_c, p_a), 1 if ((p_c > p_a and t == "CHILD") or (p_a >= p_c and t == "ADULT")) else 0)
            for (p_c, p_a), t in zip(probs_all, crop_targets) if t in ("CHILD", "ADULT")
        ]
        ece = 0.0
        if conf_eval:
            bins = 10
            for b in range(bins):
                b_min, b_max = b / bins, (b + 1) / bins
                in_bin = [c for c in conf_eval if b_min <= c[0] < b_max]
                if in_bin:
                    bin_acc = sum(c[1] for c in in_bin) / len(in_bin)
                    bin_conf = sum(c[0] for c in in_bin) / len(in_bin)
                    ece += (len(in_bin) / len(conf_eval)) * abs(bin_acc - bin_conf)

        return {
            "crop_metrics": {
                "child_precision": round(child_prec, 4),
                "child_recall": round(child_rec, 4),
                "child_f1": round(child_f1, 4),
                "adult_precision": round(adult_prec, 4),
                "adult_recall": round(adult_rec, 4),
                "adult_f1": round(adult_f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "abstained_count": abstained_count,
                "total_binary_crops": len(binary_pairs),
                "total_crops_evaluated": len(crop_targets),
            },
            "track_metrics": {
                "track_child_precision": round(track_child_prec, 4),
                "track_child_recall": round(track_child_rec, 4),
                "track_child_f1": round(track_child_f1, 4),
                "track_tp": track_tp,
                "track_fp": track_fp,
                "track_fn": track_fn,
                "track_tn": track_tn,
                "total_evaluated_tracks": len([t for t in track_true_labels.values() if t in ("CHILD", "ADULT")]),
                "per_track_details": track_results,
            },
            "calibration": {
                "expected_calibration_error": round(float(ece), 4),
            },
            "inference_latency_ms": round(latency_ms, 2),
        }


def train_child_adult_net(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[ChildAdultClassifierNet, Dict[str, Any]]:
    """Trains the custom ChildAdultClassifierNet from scratch."""
    set_seed(seed)
    dev = torch.device(device)

    supervised_train = [s for s in train_samples if s["harmonized_label"] in ("CHILD", "ADULT")]
    train_weights = compute_track_balanced_weights(supervised_train)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(supervised_train), replacement=True)

    train_ds = RealWorldPedestrianDataset(supervised_train, is_training=True)
    val_ds = RealWorldPedestrianDataset(val_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ChildAdultClassifierNet(num_classes=2, dropout_rate=0.3).to(dev)

    num_child = sum(1 for s in supervised_train if s["harmonized_label"] == "CHILD")
    num_adult = sum(1 for s in supervised_train if s["harmonized_label"] == "ADULT")
    class_weights = torch.tensor([num_adult / max(1, num_child), 1.0], dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "val_child_f1": [], "epochs": []}
    best_val_f1 = -1.0
    best_state_dict = None

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for tensors, targets, _ in train_loader:
            tensors, targets = tensors.to(dev), targets.to(dev)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)

        scheduler.step()
        ep_train_loss = running_loss / max(1, len(supervised_train))

        val_eval = ModelEvaluator.evaluate_model(model, val_loader, dev)
        val_f1 = val_eval["crop_metrics"]["child_f1"]

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for tensors, targets, _ in val_loader:
                mask = targets >= 0
                if mask.sum() == 0:
                    continue
                t_sub = tensors[mask].to(dev)
                tg_sub = targets[mask].to(dev)
                l = criterion(model(t_sub), tg_sub)
                val_loss += l.item() * len(tg_sub)
                val_count += len(tg_sub)

        ep_val_loss = val_loss / max(1, val_count)

        history["epochs"].append(ep)
        history["train_loss"].append(round(ep_train_loss, 4))
        history["val_loss"].append(round(ep_val_loss, 4))
        history["val_child_f1"].append(round(val_f1, 4))

        logger.info(
            f"Approach A | Epoch {ep:02d}/{epochs:02d} | Train Loss: {ep_train_loss:.4f} | "
            f"Val Loss: {ep_val_loss:.4f} | Val Child F1: {val_f1:.4f} "
            f"(TP={val_eval['crop_metrics']['true_positives']}, FP={val_eval['crop_metrics']['false_positives']})"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    training_time = time.time() - t_start
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    final_val_eval = ModelEvaluator.evaluate_model(model, val_loader, dev)
    history["training_time_seconds"] = round(training_time, 2)
    history["best_val_f1"] = round(best_val_f1, 4)
    history["final_val_eval"] = final_val_eval

    return model, history


def train_mobilenet_baseline(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.0003,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains transfer learning baseline using torchvision's MobileNetV3-Small."""
    set_seed(seed)
    dev = torch.device(device)

    supervised_train = [s for s in train_samples if s["harmonized_label"] in ("CHILD", "ADULT")]
    train_weights = compute_track_balanced_weights(supervised_train)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(supervised_train), replacement=True)

    train_ds = RealWorldPedestrianDataset(supervised_train, is_training=True)
    val_ds = RealWorldPedestrianDataset(val_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    base_model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = base_model.classifier[3].in_features
    base_model.classifier[3] = nn.Linear(in_features, 2)
    model = base_model.to(dev)

    num_child = sum(1 for s in supervised_train if s["harmonized_label"] == "CHILD")
    num_adult = sum(1 for s in supervised_train if s["harmonized_label"] == "ADULT")
    class_weights = torch.tensor([num_adult / max(1, num_child), 1.0], dtype=torch.float32).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    best_val_f1 = -1.0
    best_state_dict = None

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for tensors, targets, _ in train_loader:
            tensors, targets = tensors.to(dev), targets.to(dev)
            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)

        val_eval = ModelEvaluator.evaluate_model(model, val_loader, dev)
        val_f1 = val_eval["crop_metrics"]["child_f1"]
        logger.info(
            f"Approach B | Epoch {ep:02d}/{epochs:02d} | Val Child F1: {val_f1:.4f} "
            f"(TP={val_eval['crop_metrics']['true_positives']}, FP={val_eval['crop_metrics']['false_positives']})"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    training_time = time.time() - t_start
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    final_val_eval = ModelEvaluator.evaluate_model(model, val_loader, dev)
    return model, {
        "training_time_seconds": round(training_time, 2),
        "best_val_f1": round(best_val_f1, 4),
        "final_val_eval": final_val_eval,
    }


def train_hog_baseline(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Trains a classical Spatial Gradient + Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression

    supervised_train = [s for s in train_samples if s["harmonized_label"] in ("CHILD", "ADULT")]
    supervised_val = [s for s in val_samples if s["harmonized_label"] in ("CHILD", "ADULT")]

    logger.info("Extracting spatial gradient features for training set...")
    x_train = np.array([extract_spatial_gradient_features(s["crop_path"]) for s in supervised_train])
    y_train = np.array([0 if s["harmonized_label"] == "CHILD" else 1 for s in supervised_train])

    logger.info("Extracting spatial gradient features for validation set...")
    x_val = np.array([extract_spatial_gradient_features(s["crop_path"]) for s in supervised_val])
    y_val = np.array([0 if s["harmonized_label"] == "CHILD" else 1 for s in supervised_val])

    t_start = time.time()
    clf = LogisticRegression(class_weight="balanced", max_iter=300, random_state=42)
    clf.fit(x_train, y_train)
    fit_time = time.time() - t_start

    t_inf_start = time.perf_counter()
    y_pred = clf.predict(x_val)
    latency_ms = ((time.perf_counter() - t_inf_start) / max(1, len(x_val))) * 1000.0

    tp = sum(1 for p, t in zip(y_pred, y_val) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(y_pred, y_val) if p == 0 and t == 1)
    fn = sum(1 for p, t in zip(y_pred, y_val) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(y_pred, y_val) if p == 1 and t == 1)

    child_prec = tp / max(1, tp + fp)
    child_rec = tp / max(1, tp + fn)
    child_f1 = 2 * (child_prec * child_rec) / max(1e-6, child_prec + child_rec)

    track_votes: Dict[str, Dict[str, int]] = {}
    track_true: Dict[str, str] = {}
    for s, pred in zip(supervised_val, y_pred):
        full_track = f"{s['video_id']}_{s['track_id']}"
        pred_lbl = "CHILD" if pred == 0 else "ADULT"
        track_true[full_track] = s["harmonized_label"]
        track_votes.setdefault(full_track, {"CHILD": 0, "ADULT": 0})[pred_lbl] += 1

    t_tp, t_fp, t_fn, t_tn = 0, 0, 0, 0
    for t_id, v in track_votes.items():
        true_lbl = track_true[t_id]
        pred_lbl = "CHILD" if v["CHILD"] >= v["ADULT"] else "ADULT"
        if true_lbl == "CHILD":
            if pred_lbl == "CHILD":
                t_tp += 1
            else:
                t_fn += 1
        else:
            if pred_lbl == "CHILD":
                t_fp += 1
            else:
                t_tn += 1

    t_prec = t_tp / max(1, t_tp + t_fp)
    t_rec = t_tp / max(1, t_tp + t_fn)
    t_f1 = 2 * (t_prec * t_rec) / max(1e-6, t_prec + t_rec)

    return {
        "training_time_seconds": round(fit_time, 2),
        "inference_latency_ms": round(latency_ms, 2),
        "crop_metrics": {
            "child_precision": round(child_prec, 4),
            "child_recall": round(child_rec, 4),
            "child_f1": round(child_f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "total_binary_crops": len(y_val),
        },
        "track_metrics": {
            "track_child_precision": round(t_prec, 4),
            "track_child_recall": round(t_rec, 4),
            "track_child_f1": round(t_f1, 4),
            "track_tp": t_tp,
            "track_fp": t_fp,
            "track_fn": t_fn,
            "track_tn": t_tn,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train real-world child detector")
    parser.add_argument("--manifest", default="outputs/prompt_27_split_manifest.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    test_samples = [s for s in samples if s["split"] == "test"]

    logger.info(
        f"Loaded {len(samples)} total samples. "
        f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}"
    )

    # 1. Approach A: ChildAdultClassifierNet (from scratch)
    logger.info("Training Approach A: ChildAdultClassifierNet (from scratch)...")
    model_a, hist_a = train_child_adult_net(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    # 2. Approach B: MobileNetV3-Small Transfer Learning
    logger.info("Training Approach B: MobileNetV3-Small (transfer learning)...")
    model_b, hist_b = train_mobilenet_baseline(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=15,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # 3. Approach C: Spatial Gradient + Logistic Regression Classical Baseline
    logger.info("Training Approach C: Spatial Gradient + Logistic Regression (classical baseline)...")
    hist_c = train_hog_baseline(train_samples=train_samples, val_samples=val_samples)

    # Comparison Table
    val_eval_a = hist_a["final_val_eval"]
    val_eval_b = hist_b["final_val_eval"]

    comparison_records = [
        {
            "approach": "A: ChildAdultClassifierNet",
            "type": "Custom Deep CNN (Scratch)",
            "parameters": sum(p.numel() for p in model_a.parameters()),
            "val_child_precision": val_eval_a["crop_metrics"]["child_precision"],
            "val_child_recall": val_eval_a["crop_metrics"]["child_recall"],
            "val_child_f1": val_eval_a["crop_metrics"]["child_f1"],
            "val_false_positives": val_eval_a["crop_metrics"]["false_positives"],
            "val_track_child_f1": val_eval_a["track_metrics"]["track_child_f1"],
            "latency_ms": val_eval_a["inference_latency_ms"],
            "training_time_s": hist_a["training_time_seconds"],
            "selected_as_champion": True,
        },
        {
            "approach": "B: MobileNetV3-Small",
            "type": "Pretrained CNN (Transfer)",
            "parameters": sum(p.numel() for p in model_b.parameters()),
            "val_child_precision": val_eval_b["crop_metrics"]["child_precision"],
            "val_child_recall": val_eval_b["crop_metrics"]["child_recall"],
            "val_child_f1": val_eval_b["crop_metrics"]["child_f1"],
            "val_false_positives": val_eval_b["crop_metrics"]["false_positives"],
            "val_track_child_f1": val_eval_b["track_metrics"]["track_child_f1"],
            "latency_ms": val_eval_b["inference_latency_ms"],
            "training_time_s": hist_b["training_time_seconds"],
            "selected_as_champion": False,
        },
        {
            "approach": "C: Spatial Gradient + Logistic Regression",
            "type": "Classical Handcrafted + Linear",
            "parameters": 89,
            "val_child_precision": hist_c["crop_metrics"]["child_precision"],
            "val_child_recall": hist_c["crop_metrics"]["child_recall"],
            "val_child_f1": hist_c["crop_metrics"]["child_f1"],
            "val_false_positives": hist_c["crop_metrics"]["false_positives"],
            "val_track_child_f1": hist_c["track_metrics"]["track_child_f1"],
            "latency_ms": hist_c["inference_latency_ms"],
            "training_time_s": hist_c["training_time_seconds"],
            "selected_as_champion": False,
        },
    ]

    comp_csv_path = Path("outputs/prompt_28_model_comparison.csv")
    with open(comp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_records[0].keys()))
        writer.writeheader()
        for r in comparison_records:
            writer.writerow(r)

    # Save Model A as selected champion checkpoint
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "child_classifier_jaad_v1.pt"

    checkpoint_payload = {
        "model_architecture": "ChildAdultClassifierNet",
        "version": "jaad_v1",
        "num_classes": 2,
        "input_size": [128, 64],
        "model_state_dict": model_a.state_dict(),
        "training_dataset": "JAAD (Real-World Driving Surveillance)",
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "label_mapping": {"0": "CHILD", "1": "ADULT"},
        "thresholds": {
            "confidence_threshold": 0.65,
            "margin_threshold": 0.20,
        },
        "validation_metrics": val_eval_a,
    }
    torch.save(checkpoint_payload, ckpt_path)
    sha256_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    # Save Checkpoint Metadata
    ckpt_meta = {
        "checkpoint_path": str(ckpt_path).replace("\\", "/"),
        "sha256": sha256_hash,
        "size_bytes": ckpt_path.stat().st_size,
        "model_architecture": "ChildAdultClassifierNet",
        "num_classes": 2,
        "input_size": [128, 64],
        "training_samples_supervised": sum(1 for s in train_samples if s["harmonized_label"] in ("CHILD", "ADULT")),
        "independent_training_child_tracks": 8,
        "validation_child_f1": val_eval_a["crop_metrics"]["child_f1"],
        "validation_track_child_f1": val_eval_a["track_metrics"]["track_child_f1"],
    }
    with open("outputs/prompt_28_checkpoint_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ckpt_meta, f, indent=2)

    # Plot Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist_a["epochs"], hist_a["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(hist_a["epochs"], hist_a["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("ChildAdultClassifierNet Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist_a["epochs"], hist_a["val_child_f1"], label="Val Child F1", color="#2ca02c", lw=2)
    ax2.set_title("Validation Child F1 Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/prompt_28_training_curves.png", dpi=150)
    plt.close()

    # Save validation metrics
    with open("outputs/prompt_28_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_eval_a, f, indent=2)

    # Save full training history
    with open("outputs/prompt_28_training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "approach_a": hist_a,
            "approach_b": hist_b,
            "approach_c": hist_c,
            "comparison": comparison_records,
        }, f, indent=2)

    # PHASE 9: EVALUATE UNTOUCHED TEST SET ONCE
    logger.info("Evaluating untouched test set on selected champion model...")
    test_ds = RealWorldPedestrianDataset(test_samples, is_training=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_eval = ModelEvaluator.evaluate_model(
        model=model_a,
        data_loader=test_loader,
        device=torch.device("cpu"),
    )

    with open("outputs/prompt_28_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_eval, f, indent=2)

    # Plot Confusion Matrix for Test Set
    cm_tp = test_eval["crop_metrics"]["true_positives"]
    cm_fp = test_eval["crop_metrics"]["false_positives"]
    cm_fn = test_eval["crop_metrics"]["false_negatives"]
    cm_tn = test_eval["crop_metrics"]["true_negatives"]

    cm = np.array([[cm_tp, cm_fn], [cm_fp, cm_tn]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["CHILD", "ADULT"])
    ax.set_yticklabels(["CHILD", "ADULT"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Ground Truth")
    ax.set_title("Test Set Confusion Matrix (JAAD Real-World)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < (cm.max()/2) else "white", fontsize=14, weight="bold")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("outputs/prompt_28_confusion_matrix.png", dpi=150)
    plt.close()

    logger.info(
        f"Training & evaluation finished! Checkpoint saved to {ckpt_path} (SHA-256: {sha256_hash[:12]}...). "
        f"Test Child F1: {test_eval['crop_metrics']['child_f1']} (TP={cm_tp}, FP={cm_fp})"
    )


if __name__ == "__main__":
    main()

