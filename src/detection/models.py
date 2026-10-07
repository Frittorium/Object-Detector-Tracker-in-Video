from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class TrackObservation:
    tracker_id: int
    detection: Detection
    association_confidence: float | None
    state: str