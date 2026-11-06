"""VIRAT dataset annotation parser and data structures for SafeKid AI."""

from dataclasses import dataclass, field
from enum import IntEnum
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.datasets.virat_annotations")


class VIRATObjectType(IntEnum):
    """VIRAT Ground Dataset Release 2.0 object categorical labels."""
    UNKNOWN = 0
    PERSON = 1
    CAR = 2
    OTHER_VEHICLE = 3
    OTHER_OBJECT = 4
    BIKE = 5

    @classmethod
    def from_id(cls, type_id: int) -> "VIRATObjectType":
        try:
            return cls(int(type_id))
        except (ValueError, TypeError):
            return cls.UNKNOWN

    @property
    def label_name(self) -> str:
        names = {
            VIRATObjectType.UNKNOWN: "unknown",
            VIRATObjectType.PERSON: "person",
            VIRATObjectType.CAR: "car",
            VIRATObjectType.OTHER_VEHICLE: "other_vehicle",
            VIRATObjectType.OTHER_OBJECT: "object",
            VIRATObjectType.BIKE: "bike",
        }
        return names.get(self, "unknown")


@dataclass
class VIRATObjectDetection:
    """Individual object detection or ground-truth bounding box at a specific video frame."""
    track_id: int
    frame_id: int
    bbox_xywh: Tuple[float, float, float, float]  # (x, y, w, h)
    object_type: VIRATObjectType
    duration: int = 1
    confidence: float = 1.0  # Ground truth is 1.0

    @property
    def x(self) -> float:
        return self.bbox_xywh[0]

    @property
    def y(self) -> float:
        return self.bbox_xywh[1]

    @property
    def width(self) -> float:
        return self.bbox_xywh[2]

    @property
    def height(self) -> float:
        return self.bbox_xywh[3]

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """(x1, y1, x2, y2) bounding box."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Bottom-centre ground-contact marker (feet contact point)."""
        return (self.x + self.width / 2.0, self.y + self.height)

    @property
    def is_person(self) -> bool:
        return self.object_type == VIRATObjectType.PERSON


@dataclass
class VIRATEvent:
    """Temporal activity/event definition in VIRAT Ground Dataset."""
    event_id: int
    event_type_id: int
    duration: int
    start_frame: int
    end_frame: int
    associated_object_ids: List[int] = field(default_factory=list)
    bbox_xywh: Optional[Tuple[float, float, float, float]] = None


@dataclass
class VIRATFrameAnnotation:
    """All ground truth annotations occurring at a specific frame index."""
    frame_id: int
    detections: List[VIRATObjectDetection] = field(default_factory=list)
    events: List[VIRATEvent] = field(default_factory=list)

    @property
    def person_detections(self) -> List[VIRATObjectDetection]:
        """Filters ground-truth objects strictly to person category."""
        return [d for d in self.detections if d.is_person]

    @property
    def vehicle_detections(self) -> List[VIRATObjectDetection]:
        """Filters ground-truth objects to vehicles (car + other vehicle)."""
        return [
            d for d in self.detections
            if d.object_type in (VIRATObjectType.CAR, VIRATObjectType.OTHER_VEHICLE)
        ]


class VIRATAnnotationParser:
    """Parser for official VIRAT Video Dataset Release 2.0 text annotation files.

    Handles:
    - `<video_name>.viratdata.objects.txt` or `<video_name>.objects.txt`
    - `<video_name>.viratdata.events.txt` or `<video_name>.events.txt`
    - `<video_name>.viratdata.mapping.txt` or `<video_name>.mapping.txt`

    Robustly handles formatting variations and provides clear diagnostic error messages.
    """

    @classmethod
    def parse_objects_file(cls, file_path: Union[str, Path]) -> Dict[int, List[VIRATObjectDetection]]:
        """Parses a VIRAT objects file and returns a mapping from frame_id -> List[VIRATObjectDetection].

        Standard VIRAT 2.0 line format:
        `<object_id> <duration> <current_frame> <x> <y> <width> <height> <object_type>` (8 tokens)

        Fallback formats supported:
        - 7 tokens: `<object_id> <current_frame> <x> <y> <width> <height> <object_type>`
        - 6 tokens: `<object_id> <current_frame> <x> <y> <width> <height>` (defaults to PERSON)

        Returns:
            Dict mapping frame_id to list of object detections.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VIRAT objects annotation file not found: {path}")

        frame_map: Dict[int, List[VIRATObjectDetection]] = {}
        line_num = 0
        parsed_count = 0
        error_count = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_num += 1
                text = line.strip()
                if not text or text.startswith("#"):
                    continue

                tokens = text.split()
                try:
                    if len(tokens) >= 8:
                        # Official VIRAT 2.0 format:
                        # obj_id, duration, frame_id, x, y, w, h, obj_type
                        obj_id = int(tokens[0])
                        duration = int(tokens[1])
                        frame_id = int(tokens[2])
                        x = float(tokens[3])
                        y = float(tokens[4])
                        w = float(tokens[5])
                        h = float(tokens[6])
                        obj_type = VIRATObjectType.from_id(int(tokens[7]))
                    elif len(tokens) == 7:
                        # Alternate format: obj_id, frame_id, x, y, w, h, obj_type
                        obj_id = int(tokens[0])
                        duration = 1
                        frame_id = int(tokens[1])
                        x = float(tokens[2])
                        y = float(tokens[3])
                        w = float(tokens[4])
                        h = float(tokens[5])
                        obj_type = VIRATObjectType.from_id(int(tokens[6]))
                    elif len(tokens) == 6:
                        # Minimal format: obj_id, frame_id, x, y, w, h
                        obj_id = int(tokens[0])
                        duration = 1
                        frame_id = int(tokens[1])
                        x = float(tokens[2])
                        y = float(tokens[3])
                        w = float(tokens[4])
                        h = float(tokens[5])
                        obj_type = VIRATObjectType.PERSON
                    else:
                        logger.warning(
                            f"Skipping line {line_num} in {path.name}: expected >=6 tokens, got {len(tokens)}: '{text}'"
                        )
                        error_count += 1
                        continue

                    # Validate bounding box coordinates
                    if w < 0 or h < 0:
                        logger.warning(f"Invalid negative dimensions on line {line_num}: w={w}, h={h}")
                        error_count += 1
                        continue

                    det = VIRATObjectDetection(
                        track_id=obj_id,
                        frame_id=frame_id,
                        bbox_xywh=(x, y, w, h),
                        object_type=obj_type,
                        duration=duration,
                    )

                    if frame_id not in frame_map:
                        frame_map[frame_id] = []
                    frame_map[frame_id].append(det)
                    parsed_count += 1

                except (ValueError, IndexError) as e:
                    logger.warning(f"Parse error on line {line_num} in {path.name}: {e} -> '{text}'")
                    error_count += 1

        logger.info(
            f"Parsed {parsed_count} object detections across {len(frame_map)} frames from {path.name} "
            f"({error_count} malformed lines skipped)."
        )
        return frame_map

    @classmethod
    def parse_events_file(cls, file_path: Union[str, Path]) -> Dict[int, List[VIRATEvent]]:
        """Parses a VIRAT events file and returns a mapping from event_id -> VIRATEvent.

        Standard VIRAT format:
        `<event_id> <event_type> <duration> <start_frame> <end_frame> <current_frame> <x> <y> <width> <height>`
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            return {}

        events_map: Dict[int, VIRATEvent] = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                tokens = text.split()
                if len(tokens) < 5:
                    continue
                try:
                    eid = int(tokens[0])
                    etype = int(tokens[1])
                    dur = int(tokens[2])
                    sf = int(tokens[3])
                    ef = int(tokens[4])
                    bbox = None
                    if len(tokens) >= 10:
                        bbox = (float(tokens[6]), float(tokens[7]), float(tokens[8]), float(tokens[9]))
                    
                    if eid not in events_map:
                        events_map[eid] = VIRATEvent(
                            event_id=eid,
                            event_type_id=etype,
                            duration=dur,
                            start_frame=sf,
                            end_frame=ef,
                            bbox_xywh=bbox,
                        )
                except (ValueError, IndexError):
                    continue

        return events_map

    @classmethod
    def parse_mapping_file(cls, file_path: Union[str, Path]) -> Dict[int, List[int]]:
        """Parses VIRAT mapping file associating event_id -> list of object_ids."""
        path = Path(file_path).resolve()
        if not path.is_file():
            return {}

        mapping: Dict[int, List[int]] = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                tokens = text.split()
                if len(tokens) < 3:
                    continue
                try:
                    eid = int(tokens[0])
                    num_objs = int(tokens[1])
                    obj_ids = [int(tok) for tok in tokens[2:2 + num_objs]]
                    mapping[eid] = obj_ids
                except (ValueError, IndexError):
                    continue

        return mapping

