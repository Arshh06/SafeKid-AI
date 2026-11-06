"""Supervised training script for the SafeKid AI ChildDetector neural network."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from safekid_ai.child_detection.classifier_net import (
    ChildAdultClassifierNet,
    preprocess_crop,
)
from safekid_ai.child_detection.config import ChildDetectorConfig
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.train_child_detector")


class PedestrianCropDataset(Dataset):
    """PyTorch Dataset for pedestrian silhouette crops."""

    def __init__(self, samples: List[Dict], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img_path = sample["filepath"]
        crop = cv2.imread(img_path)
        if crop is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Label encoding: 0 = CHILD, 1 = ADULT
        label_str = sample["ground_truth"]
        label = 0 if label_str == "CHILD" else 1

        # Data augmentation on train split
        if self.augment:
            # 1. Random horizontal flip
            if np.random.rand() > 0.5:
                crop = cv2.flip(crop, 1)

            # 2. Random slight brightness/contrast perturbation
            if np.random.rand() > 0.3:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-15, 15)
                crop = np.clip(alpha * crop.astype(np.float32) + beta, 0, 255).astype(np.uint8)

            # 3. Random small cutout / occlusion patch (simulate occluding pole/post)
            if np.random.rand() > 0.6:
                ch, cw = crop.shape[:2]
                rw = max(2, int(cw * np.random.uniform(0.1, 0.25)))
                rh = max(2, int(ch * np.random.uniform(0.1, 0.25)))
                rx = np.random.randint(0, max(1, cw - rw))
                ry = np.random.randint(0, max(1, ch - rh))
                crop[ry:ry+rh, rx:rx+rw] = 40

        tensor = preprocess_crop(crop, target_size=(128, 64), device="cpu").squeeze(0)
        return tensor, label


def train_child_classifier(
    annotations_path: str = "data/child_dataset/fixtures/annotations.json",
    output_checkpoint_path: str = "models/child_classifier_v1.pt",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> Dict:
    """Trains the ChildAdultClassifierNet and saves weights."""
    ann_file = Path(annotations_path).resolve()
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations file missing: {ann_file}")

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_samples = data["samples"]
    
    # Train only on clear CHILD and ADULT samples for sharp representation learning
    train_samples = [s for s in all_samples if s["split"] == "train" and s["ground_truth"] in ("CHILD", "ADULT")]
    val_samples = [s for s in all_samples if s["split"] == "val" and s["ground_truth"] in ("CHILD", "ADULT")]

    print(f"Training Samples: {len(train_samples)} (CHILD: {sum(1 for s in train_samples if s['ground_truth']=='CHILD')}, ADULT: {sum(1 for s in train_samples if s['ground_truth']=='ADULT')})")
    print(f"Validation Samples: {len(val_samples)} (CHILD: {sum(1 for s in val_samples if s['ground_truth']=='CHILD')}, ADULT: {sum(1 for s in val_samples if s['ground_truth']=='ADULT')})")

    train_ds = PedestrianCropDataset(train_samples, augment=True)
    val_ds = PedestrianCropDataset(val_samples, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = ChildAdultClassifierNet(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_acc = 0.0
    history = []

    print("\nStarting Training Run...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for tensors, labels in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tensors.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for tensors, labels in val_loader:
                tensors, labels = tensors.to(device), labels.to(device)
                outputs = model(tensors)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * tensors.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            out_path = Path(output_checkpoint_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "history": history,
            }, str(out_path))

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc*100:.1f}%.")
    print(f"Model checkpoint saved to: {output_checkpoint_path}")
    return {"best_val_acc": best_val_acc, "history": history}


if __name__ == "__main__":
    train_child_classifier()

