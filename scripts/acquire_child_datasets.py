#!/usr/bin/env python3
"""Acquires and verifies real-world datasets for SafeKid AI child classification.

Supports:
- JAAD (Joint Attention in Autonomous Driving):
  * Official GitHub XML annotation repository (MIT License).
  * Video clips archive from York University EECS vision lab (CC BY-NC-SA 4.0).
  * Remote ZIP stream inspection and direct range-based clip extraction.
  * SHA-256 checksum verification and provenance logging.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request
import zlib
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.acquire")

JAAD_ZIP_URL = "http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip"
JAAD_REPO_URL = "https://github.com/ykotseruba/JAAD.git"

# Videos known to contain verified child pedestrian tracks in JAAD
JAAD_CHILD_VIDEOS = [
    "video_0005", "video_0105", "video_0128", "video_0152", "video_0175",
    "video_0305", "video_0317", "video_0327", "video_0344",
]

# Standard control videos providing adult and youth baseline diversity across splits
JAAD_CONTROL_VIDEOS = [
    "video_0001", "video_0002", "video_0003", "video_0004", "video_0006",
    "video_0007", "video_0008", "video_0009", "video_0010", "video_0011",
]


class JAADAcquisitionManager:
    """Manages acquisition and verification of JAAD dataset clips and annotations."""

    def __init__(self, target_dir: Optional[Path] = None):
        self.target_dir = target_dir or Path("data/external/jaad")
        self.videos_dir = self.target_dir / "videos"
        self.annotations_dir = self.target_dir / "annotations"
        self.attributes_dir = self.target_dir / "annotations_attributes"
        self.repo_dir = self.target_dir / "repo"

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self._cd_index: Optional[Dict[str, Dict[str, Any]]] = None

    def acquire_annotations(self) -> bool:
        """Clones or verifies the official JAAD annotation repository."""
        logger.info(f"Checking JAAD annotation repository at {self.repo_dir}...")
        if (self.repo_dir / "annotations").exists() and (self.repo_dir / "annotations_attributes").exists():
            ann_count = len(list((self.repo_dir / "annotations").glob("*.xml")))
            attr_count = len(list((self.repo_dir / "annotations_attributes").glob("*.xml")))
            logger.info(f"JAAD repository already present with {ann_count} annotations and {attr_count} attributes.")
            return True

        logger.info(f"Cloning JAAD annotations from {JAAD_REPO_URL} (branch JAAD_2.0)...")
        import subprocess
        cmd = ["git", "clone", "--depth", "1", "--branch", "JAAD_2.0", JAAD_REPO_URL, str(self.repo_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"Failed to clone JAAD annotations: {res.stderr}")
            return False

        logger.info("Successfully cloned JAAD annotation repository.")
        return True

    def get_remote_zip_index(self) -> Dict[str, Dict[str, Any]]:
        """Parses central directory of remote JAAD_clips.zip using HTTP range requests."""
        if self._cd_index is not None:
            return self._cd_index

        logger.info(f"Inspecting remote ZIP index on {JAAD_ZIP_URL}...")
        # Step 1: Probe file length via HEAD
        head_req = urllib.request.Request(JAAD_ZIP_URL, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=15) as resp:
            total_size = int(resp.headers.get("Content-Length", 3084055074))

        # Step 2: Read EOCD from tail
        tail_len = min(65536, total_size)
        tail_req = urllib.request.Request(
            JAAD_ZIP_URL,
            headers={"Range": f"bytes={total_size - tail_len}-{total_size - 1}"},
        )
        with urllib.request.urlopen(tail_req, timeout=15) as resp:
            tail_data = resp.read()

        eocd_pos = tail_data.rfind(b"PK\x05\x06")
        if eocd_pos == -1:
            raise RuntimeError("Could not find ZIP End of Central Directory signature.")

        eocd = tail_data[eocd_pos:]
        disk_no, cd_disk, entries_disk, total_entries, cd_size, cd_offset = struct.unpack(
            "<HHHHII", eocd[4:20]
        )

        # Step 3: Fetch Central Directory
        cd_req = urllib.request.Request(
            JAAD_ZIP_URL,
            headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
        )
        with urllib.request.urlopen(cd_req, timeout=20) as resp:
            cd_data = resp.read()

        ptr = 0
        index: Dict[str, Dict[str, Any]] = {}
        while ptr < len(cd_data):
            if cd_data[ptr:ptr + 4] != b"PK\x01\x02":
                break
            header = cd_data[ptr:ptr + 46]
            (sig, ver_made, ver_ext, flags, comp_method, mod_time, mod_date,
             crc32, comp_size, uncomp_size, name_len, extra_len, comment_len,
             disk_start, int_attr, ext_attr, local_offset) = struct.unpack(
                "<IHHHHHHIIIHHHHHII", header
            )
            name = cd_data[ptr + 46:ptr + 46 + name_len].decode("latin1")
            index[name] = {
                "name": name,
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "offset": local_offset,
                "crc32": crc32,
                "comp_method": comp_method,
            }
            ptr += 46 + name_len + extra_len + comment_len

        logger.info(f"Successfully indexed {len(index)} files from remote JAAD ZIP.")
        self._cd_index = index
        return index

    def download_and_extract_clip(self, video_id: str) -> Optional[Path]:
        """Downloads and decompresses an individual MP4 video clip directly from the remote archive."""
        dest_path = self.videos_dir / f"{video_id}.mp4"
        if dest_path.exists() and dest_path.stat().st_size > 0:
            logger.info(f"Clip {video_id}.mp4 already present locally ({dest_path.stat().st_size} bytes).")
            return dest_path

        index = self.get_remote_zip_index()
        key = f"JAAD_clips/{video_id}.mp4"
        if key not in index:
            logger.error(f"Video {video_id} not found in remote JAAD archive index.")
            return None

        entry = index[key]
        offset = entry["offset"]
        comp_size = entry["comp_size"]
        crc32 = entry["crc32"]
        comp_method = entry["comp_method"]

        logger.info(f"Downloading {video_id}.mp4 ({entry['uncomp_size'] / 1024 / 1024:.2f} MB)...")

        # Read local header to determine exact data offset
        loc_req = urllib.request.Request(
            JAAD_ZIP_URL,
            headers={"Range": f"bytes={offset}-{offset + 64}"},
        )
        with urllib.request.urlopen(loc_req, timeout=15) as resp:
            loc_data = resp.read()

        loc_nlen, loc_xlen = struct.unpack("<HH", loc_data[26:30])
        data_start = offset + 30 + loc_nlen + loc_xlen
        data_end = data_start + comp_size - 1

        # Fetch compressed payload
        pay_req = urllib.request.Request(
            JAAD_ZIP_URL,
            headers={"Range": f"bytes={data_start}-{data_end}"},
        )
        with urllib.request.urlopen(pay_req, timeout=60) as resp:
            payload = resp.read()

        if len(payload) != comp_size:
            logger.error(f"Payload size mismatch for {video_id}: got {len(payload)}, expected {comp_size}")
            return None

        # Decompress
        if comp_method == 8:
            decompressed = zlib.decompress(payload, -15)
        else:
            decompressed = payload

        # Check CRC32
        actual_crc = zlib.crc32(decompressed)
        if actual_crc != crc32:
            logger.error(f"CRC32 mismatch for {video_id}: got {actual_crc}, expected {crc32}")
            return None

        with open(dest_path, "wb") as f:
            f.write(decompressed)

        logger.info(f"Successfully saved and verified {video_id}.mp4 ({len(decompressed)} bytes).")
        return dest_path

    def acquire_target_clips(
        self,
        include_child_videos: bool = True,
        include_controls: bool = True,
        extra_videos: Optional[List[str]] = None,
    ) -> List[Path]:
        """Acquires a verified subset of JAAD video clips."""
        vids_to_fetch: List[str] = []
        if include_child_videos:
            vids_to_fetch.extend(JAAD_CHILD_VIDEOS)
        if include_controls:
            vids_to_fetch.extend(JAAD_CONTROL_VIDEOS)
        if extra_videos:
            vids_to_fetch.extend(extra_videos)

        vids_to_fetch = sorted(list(set(vids_to_fetch)))
        acquired_paths: List[Path] = []
        meta_records: List[Dict[str, Any]] = []

        for vid in vids_to_fetch:
            p = self.download_and_extract_clip(vid)
            if p and p.exists():
                acquired_paths.append(p)
                # Compute SHA-256
                sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
                meta_records.append({
                    "video_id": vid,
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256,
                    "retrieval_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_url": f"{JAAD_ZIP_URL}#{vid}.mp4",
                    "contains_children": vid in JAAD_CHILD_VIDEOS,
                })

        meta_path = self.target_dir / "acquisition_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": "JAAD (Joint Attention in Autonomous Driving)",
                "source_archive_url": JAAD_ZIP_URL,
                "annotation_repo_url": JAAD_REPO_URL,
                "license_annotations": "MIT License",
                "license_videos": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
                "total_clips_acquired": len(acquired_paths),
                "total_bytes_acquired": sum(r["size_bytes"] for r in meta_records),
                "clips": meta_records,
            }, f, indent=2)

        logger.info(f"Acquisition metadata saved to {meta_path}.")
        return acquired_paths


def main():
    parser = argparse.ArgumentParser(description="SafeKid AI Dataset Acquisition Tool")
    parser.add_argument("--dataset", default="jaad", choices=["jaad"], help="Target dataset")
    parser.add_argument("--skip-videos", action="store_true", help="Acquire annotations only, skip video download")
    parser.add_argument("--child-only", action="store_true", help="Download only videos with verified children")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to download")
    args = parser.parse_args()

    manager = JAADAcquisitionManager()
    ok = manager.acquire_annotations()
    if not ok:
        sys.exit(1)

    if not args.skip_videos:
        include_controls = not args.child_only
        vids = manager.acquire_target_clips(
            include_child_videos=True,
            include_controls=include_controls,
        )
        print(f"Successfully acquired {len(vids)} video clips.")


if __name__ == "__main__":
    main()

