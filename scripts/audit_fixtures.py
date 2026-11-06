"""Audit script for analyzing dataset provenance, duplicate detection, and visual diversity."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) for perceptual similarity comparison."""
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(h1: int, h2: int) -> int:
    """Calculates bitwise Hamming distance between two hashes."""
    return bin(h1 ^ h2).count("1")


def audit_fixture_dataset() -> Dict:
    ann_path = Path("data/child_dataset/fixtures/annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Total fixture samples analyzed: {len(samples)}")

    hashes: Dict[int, Dict] = {}
    near_duplicates = []

    for s in samples:
        img = cv2.imread(s["filepath"])
        if img is None:
            continue
        dh = compute_dhash(img)
        hashes[s["sample_id"]] = {
            "sample_id": s["sample_id"],
            "ground_truth": s["ground_truth"],
            "split": s["split"],
            "filename": s["filename"],
            "hash": dh,
            "shape": img.shape,
        }

    # Compare pairwise distances
    sample_ids = list(hashes.keys())
    split_leakage_candidates = []
    for i in range(len(sample_ids)):
        for j in range(i + 1, len(sample_ids)):
            id1, id2 = sample_ids[i], sample_ids[j]
            s1, s2 = hashes[id1], hashes[id2]
            dist = hamming_distance(s1["hash"], s2["hash"])
            if dist <= 2:  # Perceptually near-identical
                near_duplicates.append((id1, id2, dist, s1["split"], s2["split"]))
                if s1["split"] != s2["split"]:
                    split_leakage_candidates.append((id1, id2, dist, s1["split"], s2["split"]))

    print(f"Near-duplicate pairs (Hamming dist <= 2): {len(near_duplicates)}")
    print(f"Cross-split near-duplicate candidates: {len(split_leakage_candidates)}")

    return {
        "total_samples": len(samples),
        "near_duplicate_count": len(near_duplicates),
        "cross_split_near_duplicates": len(split_leakage_candidates),
        "split_leakage_candidates": split_leakage_candidates[:10],
    }


if __name__ == "__main__":
    audit_fixture_dataset()
