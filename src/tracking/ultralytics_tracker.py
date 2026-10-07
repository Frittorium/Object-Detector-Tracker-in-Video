import logging

import numpy as np

from detection.models import TrackObservation
from errors import TrackerError

log = logging.getLogger(__name__)


class _Dets:
    """Minimal results-like object the Ultralytics trackers expect."""
    def __init__(self, xyxy, xywh, conf, cls):
        self.xyxy, self.xywh, self.conf, self.cls = xyxy, xywh, conf, cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return _Dets(self.xyxy[i], self.xywh[i], self.conf[i], self.cls[i])


class UltralyticsTracker:
    def __init__(self, tracker_type, reid_enabled, track_buffer):
        self._type = tracker_type
        self._reid = reid_enabled
        self._buffer = track_buffer
        self._tracker = None
        self._frame_rate = 30

    def _build(self):
        from ultralytics.trackers import BOTSORT, BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        cls = {"botsort": BOTSORT, "bytetrack": BYTETracker}.get(self._type)
        if cls is None:
            raise ValueError(f"Unknown tracker type: {self._type}")
        cfg = YAML.load(check_yaml(f"{self._type}.yaml"))
        cfg["track_buffer"] = self._buffer
        if self._type == "botsort":
            cfg["with_reid"] = bool(self._reid)
            if self._reid:
                cfg["model"] = "yolo11n-cls.pt"  # "auto" needs detector features, which we don't have here
        return cls(IterableSimpleNamespace(**cfg))

    def initialize(self, frame_rate: int = 30):
        try:
            self._frame_rate = frame_rate
            self._tracker = self._build()
            log.info("Tracker: %s, ReID: %s", self._type, "enabled" if self._reid else "disabled")
        except Exception as e:
            raise TrackerError(f"Tracker initialization failed: {e}") from e

    def update(self, frame, detections):
        try:
            xyxy = np.array([(d.x1, d.y1, d.x2, d.y2) for d in detections], dtype=np.float32).reshape(-1, 4)
            conf = np.array([d.confidence for d in detections], dtype=np.float32)
            cls = np.arange(len(detections), dtype=np.float32)  # carries original detection index
            xywh = np.column_stack([
                (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
            ])
            tracks = self._tracker.update(_Dets(xyxy, xywh, conf, cls), frame)
            # rows: x1, y1, x2, y2, track_id, score, cls (= original detection index), det_index
            return [
                TrackObservation(int(t[4]), detections[int(t[6])], None, "tracked")
                for t in tracks
            ]
        except Exception as e:
            raise TrackerError(f"Tracker update failed: {e}") from e

    def reset(self):
        if self._tracker is not None:
            self._tracker.reset()

    def close(self):
        self._tracker = None