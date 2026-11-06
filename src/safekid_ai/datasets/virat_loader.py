"""VIRAT surveillance dataset loader and sequence management module."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2

from safekid_ai.datasets.virat_annotations import (
    VIRATAnnotationParser,
    VIRATEvent,
    VIRATFrameAnnotation,
    VIRATObjectDetection,
    VIRATObjectType,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.datasets.virat_loader")


@dataclass
class VIRATSequenceMetadata:
    """Metadata describing a single VIRAT surveillance video sequence and its annotations."""
    sequence_id: str
    video_path: Optional[Path] = None
    objects_path: Optional[Path] = None
    events_path: Optional[Path] = None
    mapping_path: Optional[Path] = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    total_frames: int = 0
    duration_seconds: float = 0.0
    has_video: bool = False
    has_annotations: bool = False
    annotation_frame_count: int = 0
    total_object_instances: int = 0
    person_instance_count: int = 0
    unique_person_tracks: int = 0
    unique_all_tracks: int = 0
    max_annotation_frame: int = 0
    is_genuine: bool = False
    tier: str = "fixture"

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}" if self.width > 0 else "Unknown"

    @property
    def is_complete(self) -> bool:
        """True if both video file and objects annotation file exist."""
        return self.has_video and self.has_annotations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "has_video": self.has_video,
            "has_annotations": self.has_annotations,
            "is_complete": self.is_complete,
            "is_genuine": self.is_genuine,
            "tier": self.tier,
            "video_path": str(self.video_path) if self.video_path else None,
            "objects_path": str(self.objects_path) if self.objects_path else None,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "annotation_frame_count": self.annotation_frame_count,
            "person_instance_count": self.person_instance_count,
            "unique_person_tracks": self.unique_person_tracks,
            "unique_all_tracks": self.unique_all_tracks,
        }


@dataclass
class VIRATSequence:
    """Active loaded sequence containing metadata and parsed frame annotations."""
    metadata: VIRATSequenceMetadata
    frame_annotations: Dict[int, VIRATFrameAnnotation] = field(default_factory=dict)

    def get_frame_annotation(self, frame_id: int) -> Optional[VIRATFrameAnnotation]:
        return self.frame_annotations.get(frame_id)

    def get_ground_truth_persons(self, frame_id: int) -> List[VIRATObjectDetection]:
        fa = self.get_frame_annotation(frame_id)
        if fa:
            return fa.person_detections
        return []


class VIRATDatasetLoader:
    """Filesystem loader and sequence manager for the VIRAT Video Dataset.

    Standard folder convention:
    data/virat/
      videos/        <- contains VIRAT_S_*.mp4
      annotations/   <- contains VIRAT_S_*.viratdata.objects.txt
      metadata/      <- optional metadata files
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        if root_dir is not None:
            self.root_dir = Path(root_dir).resolve()
        else:
            self.root_dir = (Path.cwd() / "data" / "virat").resolve()

        self.videos_dir = self.root_dir / "videos"
        self.annotations_dir = self.root_dir / "annotations"
        self.metadata_dir = self.root_dir / "metadata"

        # Ensure directories exist
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._sequences_cache: Dict[str, VIRATSequenceMetadata] = {}
        self.scan_dataset()

    def scan_dataset(self) -> Dict[str, VIRATSequenceMetadata]:
        """Scans the dataset directories, matching video files with annotation files."""
        self._sequences_cache.clear()

        # 1. Discover all video files
        video_files: Dict[str, Path] = {}
        if self.videos_dir.exists():
            for p in self.videos_dir.glob("*"):
                if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
                    seq_id = p.stem
                    video_files[seq_id] = p

        # 2. Discover all objects annotation files
        objects_files: Dict[str, Path] = {}
        events_files: Dict[str, Path] = {}
        mapping_files: Dict[str, Path] = {}

        if self.annotations_dir.exists():
            for p in self.annotations_dir.glob("*.txt"):
                name = p.name
                if "objects" in name.lower():
                    # Strip .viratdata.objects.txt or .objects.txt
                    seq_id = name.replace(".viratdata.objects.txt", "").replace(".objects.txt", "")
                    objects_files[seq_id] = p
                elif "events" in name.lower():
                    seq_id = name.replace(".viratdata.events.txt", "").replace(".events.txt", "")
                    events_files[seq_id] = p
                elif "mapping" in name.lower():
                    seq_id = name.replace(".viratdata.mapping.txt", "").replace(".mapping.txt", "")
                    mapping_files[seq_id] = p

        # 3. Combine discovered sequence identifiers
        all_seq_ids = set(video_files.keys()) | set(objects_files.keys())

        for seq_id in sorted(all_seq_ids):
            v_path = video_files.get(seq_id)
            o_path = objects_files.get(seq_id)
            e_path = events_files.get(seq_id)
            m_path = mapping_files.get(seq_id)

            # Determine whether sequence is genuine surveillance footage or reference test fixture
            is_genuine = False
            tier = "fixture"
            if v_path and "genuine" in str(v_path).lower():
                is_genuine = True
                tier = "genuine"
            elif "_sample" not in seq_id.lower() and "fixture" not in str(v_path or "").lower():
                # External VIRAT sequences (e.g. VIRAT_S_000002, VIRAT_S_010000)
                is_genuine = True
                tier = "genuine"

            meta = VIRATSequenceMetadata(
                sequence_id=seq_id,
                video_path=v_path,
                objects_path=o_path,
                events_path=e_path,
                mapping_path=m_path,
                has_video=(v_path is not None and v_path.is_file()),
                has_annotations=(o_path is not None and o_path.is_file()),
                is_genuine=is_genuine,
                tier=tier,
            )

            # Probe video metadata if video is available
            if meta.has_video and v_path is not None:
                try:
                    cap = cv2.VideoCapture(str(v_path))
                    if cap.isOpened():
                        meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        meta.fps = float(cap.get(cv2.CAP_PROP_FPS))
                        meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        meta.duration_seconds = (
                            meta.total_frames / meta.fps if meta.fps > 0 else 0.0
                        )
                    cap.release()
                except Exception as e:
                    logger.warning(f"Could not read video metadata for {v_path.name}: {e}")

            # Quick inspect annotation summary if available
            if meta.has_annotations and o_path is not None:
                try:
                    # Light scan to count records without loading full memory
                    unique_persons: Set[int] = set()
                    unique_all: Set[int] = set()
                    frame_set: Set[int] = set()
                    person_count = 0
                    total_objs = 0
                    max_f = 0

                    with open(o_path, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            tokens = line.strip().split()
                            if not tokens or tokens[0].startswith("#"):
                                continue
                            if len(tokens) >= 8:
                                oid = int(tokens[0])
                                fid = int(tokens[2])
                                otype = int(tokens[7])
                            elif len(tokens) >= 6:
                                oid = int(tokens[0])
                                fid = int(tokens[1])
                                otype = int(tokens[6]) if len(tokens) >= 7 else 1
                            else:
                                continue

                            total_objs += 1
                            unique_all.add(oid)
                            frame_set.add(fid)
                            if fid > max_f:
                                max_f = fid
                            if otype == 1:
                                person_count += 1
                                unique_persons.add(oid)

                    meta.annotation_frame_count = len(frame_set)
                    meta.total_object_instances = total_objs
                    meta.person_instance_count = person_count
                    meta.unique_person_tracks = len(unique_persons)
                    meta.unique_all_tracks = len(unique_all)
                    meta.max_annotation_frame = max_f

                except Exception as e:
                    logger.warning(f"Could not scan annotations for {o_path.name}: {e}")

            self._sequences_cache[seq_id] = meta

        logger.info(
            f"VIRATDatasetLoader scanned '{self.root_dir}': found {len(self._sequences_cache)} sequences "
            f"({sum(1 for s in self._sequences_cache.values() if s.is_complete)} complete with video+annotations)."
        )
        return self._sequences_cache

    def list_sequences(self, complete_only: bool = False) -> List[str]:
        """Returns sequence IDs available in the dataset."""
        if complete_only:
            return [s.sequence_id for s in self._sequences_cache.values() if s.is_complete]
        return list(self._sequences_cache.keys())

    def list_genuine_sequences(self, complete_only: bool = True) -> List[str]:
        """Returns sequence IDs of verified genuine external surveillance sequences."""
        return [
            s.sequence_id
            for s in self._sequences_cache.values()
            if s.is_genuine and (not complete_only or s.is_complete)
        ]

    def list_fixture_sequences(self, complete_only: bool = True) -> List[str]:
        """Returns sequence IDs of synthetic reference testbed fixtures."""
        return [
            s.sequence_id
            for s in self._sequences_cache.values()
            if not s.is_genuine and (not complete_only or s.is_complete)
        ]

    def get_sequence_metadata(self, sequence_id: str) -> Optional[VIRATSequenceMetadata]:
        return self._sequences_cache.get(sequence_id)

    def load_sequence(self, sequence_id: str) -> VIRATSequence:
        """Loads a VIRAT sequence and parses its object and event annotations."""
        meta = self._sequences_cache.get(sequence_id)
        if meta is None:
            raise KeyError(f"Sequence '{sequence_id}' not found in dataset at {self.root_dir}")

        if not meta.has_annotations or meta.objects_path is None:
            raise FileNotFoundError(f"Objects annotation file missing for sequence '{sequence_id}'")

        # Parse objects
        raw_objects = VIRATAnnotationParser.parse_objects_file(meta.objects_path)
        events_map: Dict[int, VIRATEvent] = {}
        if meta.events_path and meta.events_path.is_file():
            events_map = VIRATAnnotationParser.parse_events_file(meta.events_path)

        mapping: Dict[int, List[int]] = {}
        if meta.mapping_path and meta.mapping_path.is_file():
            mapping = VIRATAnnotationParser.parse_mapping_file(meta.mapping_path)
            # Associate object IDs with events
            for eid, obj_ids in mapping.items():
                if eid in events_map:
                    events_map[eid].associated_object_ids = obj_ids

        # Build FrameAnnotation structures
        frame_annotations: Dict[int, VIRATFrameAnnotation] = {}
        for fid, dets in raw_objects.items():
            frame_annotations[fid] = VIRATFrameAnnotation(
                frame_id=fid,
                detections=dets,
                events=[],
            )

        # Associate active events with frames
        for evt in events_map.values():
            for fid in range(evt.start_frame, evt.end_frame + 1):
                if fid in frame_annotations:
                    frame_annotations[fid].events.append(evt)

        return VIRATSequence(
            metadata=meta,
            frame_annotations=frame_annotations,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Provides an aggregate summary of available dataset files and annotations."""
        total_seqs = len(self._sequences_cache)
        complete_seqs = sum(1 for s in self._sequences_cache.values() if s.is_complete)
        total_video_frames = sum(s.total_frames for s in self._sequences_cache.values())
        total_annotated_frames = sum(s.annotation_frame_count for s in self._sequences_cache.values())
        total_person_instances = sum(s.person_instance_count for s in self._sequences_cache.values())
        total_person_tracks = sum(s.unique_person_tracks for s in self._sequences_cache.values())

        return {
            "root_dir": str(self.root_dir),
            "total_sequences_found": total_seqs,
            "complete_sequences": complete_seqs,
            "total_video_frames": total_video_frames,
            "total_annotated_frames": total_annotated_frames,
            "total_person_instances": total_person_instances,
            "total_unique_person_tracks": total_person_tracks,
            "sequences": [s.to_dict() for s in self._sequences_cache.values()],
        }

