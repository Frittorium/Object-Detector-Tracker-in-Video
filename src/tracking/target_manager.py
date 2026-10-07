import logging
from dataclasses import dataclass
from enum import Enum, auto

from detection.models import Detection, TrackObservation

log = logging.getLogger(__name__)


class Lifecycle(Enum):
    CANDIDATE = auto()
    ACTIVE = auto()
    TEMPORARILY_LOST = auto()
    REACQUIRED = auto()
    PERMANENTLY_LOST = auto()


class Visibility(Enum):
    VISIBLE = auto()
    NOT_VISIBLE = auto()


@dataclass
class TargetObject:
    target_id: int
    class_name: str
    tracker_id: int | None
    latest_detection: Detection | None
    last_bbox: tuple[int, int, int, int] | None
    last_seen_frame: int
    lifecycle_state: Lifecycle
    visibility_state: Visibility
    association_confidence: float | None


def _bbox(d: Detection):
    return (d.x1, d.y1, d.x2, d.y2)


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TargetManager:
    def __init__(self, max_targets, reassoc_min_iou, ambiguity_margin, lost_timeout_frames):
        self._max = max_targets
        self._min_iou = reassoc_min_iou
        self._margin = ambiguity_margin
        self._timeout = lost_timeout_frames
        self._targets: dict[int, TargetObject] = {}
        self._by_tracker: dict[int, int] = {}
        self._next_id = 1

    def target_count(self): return len(self._targets)
    def slots_left(self): return self._max - len(self._targets)   # lost targets keep their slot
    def contains_tracker(self, tracker_id): return tracker_id in self._by_tracker
    def targets(self): return sorted(self._targets.values(), key=lambda t: t.target_id)

    def create_target(self, obs: TrackObservation, frame_index: int) -> TargetObject:
        d = obs.detection
        t = TargetObject(self._next_id, d.class_name, obs.tracker_id, d, _bbox(d), frame_index,
                         Lifecycle.ACTIVE, Visibility.VISIBLE, obs.association_confidence)
        self._targets[t.target_id] = t
        self._by_tracker[obs.tracker_id] = t.target_id
        self._next_id += 1
        log.info("Target %d selected from tracker %d", t.target_id, obs.tracker_id)
        return t

    def _apply(self, t, obs, frame_index):
        t.tracker_id = obs.tracker_id
        t.latest_detection = obs.detection
        t.last_bbox = _bbox(obs.detection)
        t.last_seen_frame = frame_index
        t.visibility_state = Visibility.VISIBLE
        t.association_confidence = obs.association_confidence

    def update_existing(self, observations, frame_index):
        """Updates targets whose tracker ID is still mapped; returns unmapped observations."""
        unmapped = []
        for o in observations:
            tid = self._by_tracker.get(o.tracker_id)
            if tid is None:
                unmapped.append(o)
                continue
            t = self._targets[tid]
            if t.lifecycle_state is Lifecycle.PERMANENTLY_LOST:
                continue                      # never revive / duplicate a lost identity
            if t.lifecycle_state is Lifecycle.TEMPORARILY_LOST:
                log.info("Target %d recovered with same tracker %d", tid, o.tracker_id)
            self._apply(t, o, frame_index)
            t.lifecycle_state = Lifecycle.ACTIVE
        return unmapped

    def reassociate(self, observations, frame_index):
        """Conservative IoU re-association to temporarily lost targets; returns the rest."""
        lost = [t for t in self._targets.values()
                if t.lifecycle_state is Lifecycle.TEMPORARILY_LOST
                and frame_index - t.last_seen_frame <= self._timeout and t.last_bbox]
        used, remaining = set(), []
        for o in sorted(observations, key=lambda o: o.tracker_id):
            box = _bbox(o.detection)
            scored = sorted(((_iou(t.last_bbox, box), t.target_id) for t in lost
                             if t.target_id not in used), reverse=True)
            if scored and scored[0][0] >= self._min_iou:
                if len(scored) == 1 or scored[0][0] - scored[1][0] >= self._margin:
                    t = self._targets[scored[0][1]]
                    self._by_tracker.pop(t.tracker_id, None)
                    self._by_tracker[o.tracker_id] = t.target_id
                    self._apply(t, o, frame_index)
                    t.lifecycle_state = Lifecycle.REACQUIRED
                    used.add(t.target_id)
                    log.info("Target %d reacquired with tracker %d", t.target_id, o.tracker_id)
                    continue
                log.warning("Ambiguous re-association for tracker %d", o.tracker_id)
            remaining.append(o)
        return remaining

    def finalize_visibility(self, frame_index):
        for t in self._targets.values():
            if t.lifecycle_state is Lifecycle.PERMANENTLY_LOST or t.last_seen_frame == frame_index:
                continue
            t.visibility_state = Visibility.NOT_VISIBLE
            if t.lifecycle_state is not Lifecycle.TEMPORARILY_LOST:
                t.lifecycle_state = Lifecycle.TEMPORARILY_LOST
                log.warning("Target %d temporarily lost", t.target_id)
            elif frame_index - t.last_seen_frame > self._timeout:
                t.lifecycle_state = Lifecycle.PERMANENTLY_LOST
                log.warning("Target %d permanently lost (slot stays reserved)", t.target_id)