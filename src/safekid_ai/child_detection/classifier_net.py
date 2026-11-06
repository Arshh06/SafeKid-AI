"""Lightweight PyTorch neural network for supervised child vs. adult pedestrian classification."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChildAdultClassifierNet(nn.Module):
    """Deep convolutional neural network for distinguishing children from adults in CCTV crops.
    
    Operates on full-body pedestrian silhouette crops (128x64 standard pedestrian aspect ratio).
    Focuses on holistic body proportions, head-to-torso ratios, and limb posture without
    relying on facial features, continuous age regression, or biometric identifiers.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: Initial feature extraction and spatial downsampling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 128x64 -> 64x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                            # 64x32 -> 32x16

        # Block 2: Intermediate body proportion & texture representations
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False) # 32x16 -> 32x16
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                            # 32x16 -> 16x8

        # Block 3: Higher-level semantic abstraction
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)# 16x8 -> 16x8
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 2))                                     # Output: 128x4x2 = 1024

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),  # Class 0: CHILD, Class 1: ADULT
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 128, 64), normalized to range [0, 1] or standard normal.
            
        Returns:
            Logits of shape (B, 2) where column 0 is CHILD and column 1 is ADULT.
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def preprocess_crop(
    crop: np.ndarray,
    target_size: Tuple[int, int] = (128, 64),
    device: str = "cpu"
) -> torch.Tensor:
    """Preprocesses an OpenCV BGR crop into a normalized PyTorch tensor (1, 3, H, W)."""
    import cv2
    if crop is None or crop.size == 0:
        raise ValueError("Invalid empty crop provided.")
    
    # Resize to target size (height, width)
    resized = cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    # BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Scale to [0, 1]
    norm = rgb.astype(np.float32) / 255.0
    # Standard normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (norm - mean) / std
    
    # Transpose (H, W, C) -> (C, H, W)
    chw = np.transpose(norm, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
    return tensor

