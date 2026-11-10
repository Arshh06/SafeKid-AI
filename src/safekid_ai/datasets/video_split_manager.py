"""Group-aware dataset partitioning and leakage audit for video surveillance sequences.

Ensures strict zero-leakage partitions:
1. Videos are assigned exclusively to a single partition (train, val, or test).
2. All frames and track IDs from a video belong strictly to that partition.
3. Intersecting video or track IDs across splits are flagged and forbidden.
4. Deterministic random seeds ensure reproducible stratification.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import random

from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.split_manager")


@dataclass
class SplitLeakageAudit:
    """Audit summary verifying zero leakage across dataset partitions."""
    train_videos: List[str] = field(default_factory=list)
    val_videos: List[str] = field(default_factory=list)
    test_videos: List[str] = field(default_factory=list)
    train_tracks: List[str] = field(default_factory=list)
    val_tracks: List[str] = field(default_factory=list)
    test_tracks: List[str] = field(default_factory=list)
    video_overlap_train_val: List[str] = field(default_factory=list)
    video_overlap_train_test: List[str] = field(default_factory=list)
    video_overlap_val_test: List[str] = field(default_factory=list)
    track_overlap_train_val: List[str] = field(default_factory=list)
    track_overlap_train_test: List[str] = field(default_factory=list)
    track_overlap_val_test: List[str] = field(default_factory=list)
    is_leakage_free: bool = True
    split_counts: Dict[str, int] = field(default_factory=dict)
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes leakage audit to dictionary."""
        return {
            "is_leakage_free": self.is_leakage_free,
            "video_counts": {
                "train": len(self.train_videos),
                "val": len(self.val_videos),
                "test": len(self.test_videos),
                "total": len(self.train_videos) + len(self.val_videos) + len(self.test_videos),
            },
            "track_counts": {
                "train": len(self.train_tracks),
                "val": len(self.val_tracks),
                "test": len(self.test_tracks),
                "total": len(self.train_tracks) + len(self.val_tracks) + len(self.test_tracks),
            },
            "video_overlaps": {
                "train_val": self.video_overlap_train_val,
                "train_test": self.video_overlap_train_test,
                "val_test": self.video_overlap_val_test,
            },
            "track_overlaps": {
                "train_val": self.track_overlap_train_val,
                "train_test": self.track_overlap_train_test,
                "val_test": self.track_overlap_val_test,
            },
            "split_sample_counts": self.split_counts,
            "class_distribution": self.class_distribution,
        }


class VideoSplitManager:
    """Manages group-aware sequence splitting and audit for video datasets."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def load_official_splits(self, split_dir: Path) -> Dict[str, Set[str]]:
        """Loads official train.txt, val.txt, test.txt from a split directory."""
        splits: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        for s in ("train", "val", "test"):
            fpath = split_dir / f"{s}.txt"
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    splits[s] = set(line.strip() for line in f if line.strip())
        return splits

    def create_stratified_group_split(
        self,
        video_ids: List[str],
        child_video_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[str]]:
        """Creates a group-aware split stratified by minority child video sequences.
        
        Guarantees:
        - Child video sequences are distributed proportionally across splits.
        - Zero video overlap between train, val, and test partitions.
        - Fully deterministic partition given self.seed.
        """
        rng = random.Random(self.seed)

        child_set = set(child_video_ids)
        other_vids = sorted([v for v in video_ids if v not in child_set])
        child_vids = sorted(list(child_set))

        rng.shuffle(child_vids)
        rng.shuffle(other_vids)

        def partition_list(items: List[str]) -> Tuple[List[str], List[str], List[str]]:
            n = len(items)
            n_train = max(1, int(round(n * train_ratio))) if n >= 3 else (1 if n >= 1 else 0)
            n_val = max(1, int(round(n * val_ratio))) if n >= 3 else (1 if n >= 2 else 0)
            # Adjust if sum exceeds n
            if n_train + n_val >= n and n >= 3:
                n_train = n - 2
                n_val = 1
            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:]
            return train_items, val_items, test_items

        c_train, c_val, c_test = partition_list(child_vids)
        o_train, o_val, o_test = partition_list(other_vids)

        return {
            "train": sorted(c_train + o_train),
            "val": sorted(c_val + o_val),
            "test": sorted(c_test + o_test),
        }

    def audit_splits(
        self,
        samples: List[Dict[str, Any]],
        splits_mapping: Dict[str, List[str]],
    ) -> SplitLeakageAudit:
        """Audits an assigned set of samples against partition assignments."""
        train_vids = set(splits_mapping.get("train", []))
        val_vids = set(splits_mapping.get("val", []))
        test_vids = set(splits_mapping.get("test", []))

        # Check video overlaps
        v_tv = sorted(list(train_vids & val_vids))
        v_tt = sorted(list(train_vids & test_vids))
        v_vt = sorted(list(val_vids & test_vids))

        # Collect tracks per split
        tracks_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
        counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        classes_by_split: Dict[str, Dict[str, int]] = {
            "train": {}, "val": {}, "test": {},
        }

        for s in samples:
            split_name = s.get("split")
            if split_name not in tracks_by_split:
                continue
            counts[split_name] += 1
            track_id = s.get("track_id")
            if track_id:
                # Disambiguate track by prepending video_id
                vid = s.get("video_id", "")
                full_track = f"{vid}_{track_id}"
                tracks_by_split[split_name].add(full_track)

            label = s.get("operational_label", s.get("harmonized_label", "UNKNOWN"))
            classes_by_split[split_name][label] = classes_by_split[split_name].get(label, 0) + 1

        t_tv = sorted(list(tracks_by_split["train"] & tracks_by_split["val"]))
        t_tt = sorted(list(tracks_by_split["train"] & tracks_by_split["test"]))
        t_vt = sorted(list(tracks_by_split["val"] & tracks_by_split["test"]))

        is_leak_free = (
            len(v_tv) == 0 and len(v_tt) == 0 and len(v_vt) == 0 and
            len(t_tv) == 0 and len(t_tt) == 0 and len(t_vt) == 0
        )

        return SplitLeakageAudit(
            train_videos=sorted(list(train_vids)),
            val_videos=sorted(list(val_vids)),
            test_videos=sorted(list(test_vids)),
            train_tracks=sorted(list(tracks_by_split["train"])),
            val_tracks=sorted(list(tracks_by_split["val"])),
            test_tracks=sorted(list(tracks_by_split["test"])),
            video_overlap_train_val=v_tv,
            video_overlap_train_test=v_tt,
            video_overlap_val_test=v_vt,
            track_overlap_train_val=t_tv,
            track_overlap_train_test=t_tt,
            track_overlap_val_test=t_vt,
            is_leakage_free=is_leak_free,
            split_counts=counts,
            class_distribution=classes_by_split,
        )

