import math
from pathlib import Path

import cv2
import numpy as np

import config
from errors import InputValidationError
from video.metadata import VideoMetadata


def is_valid_frame(frame) -> bool:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    if frame.ndim not in (2, 3):
        return False
    h, w = frame.shape[:2]
    return h > 0 and w > 0


class OpenCVVideoReader:
    def __init__(self):
        self._cap = None
        self._pending = None
        self._meta = None

    def open(self, path):
        path = Path(path)
        if not path.exists():
            raise InputValidationError("The selected file does not exist.")
        if not path.is_file():
            raise InputValidationError("The selected path is not a file.")
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            raise InputValidationError(
                "Unsupported file type. Supported: " + ", ".join(config.SUPPORTED_EXTENSIONS))
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise InputValidationError(
                "The video could not be opened. Its format or codec may not be supported.")
        ok, frame = cap.read()
        if not ok or not is_valid_frame(frame):
            cap.release()
            raise InputValidationError("The video contains no readable frames.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_ok = bool(fps) and math.isfinite(fps) and 1.0 <= fps <= 240.0
        count_ok = bool(count) and math.isfinite(count) and count > 0
        h, w = frame.shape[:2]
        self._meta = VideoMetadata(
            width=w, height=h,
            source_fps=fps if fps_ok else None,
            frame_count=int(count) if count_ok else None,
            duration=(count / fps) if (fps_ok and count_ok) else None,
            frame_count_known=count_ok, fps_reliable=fps_ok)
        self._cap, self._pending = cap, frame

    def metadata(self) -> VideoMetadata:
        return self._meta

    def read(self):
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return True, frame
        if self._cap is None:
            return False, None
        return self._cap.read()

    def looks_complete(self, frames_read: int) -> bool:
        """Used after repeated failed reads to decide EOF vs. read error."""
        n = self._meta.frame_count
        return n is None or frames_read >= 0.95 * n

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None