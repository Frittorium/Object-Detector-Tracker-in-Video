import logging
import shutil
import tempfile
from pathlib import Path

import cv2

import config
from errors import VideoWriteError
from video.reader import is_valid_frame

log = logging.getLogger(__name__)


class TemporaryOutput:
    def __init__(self):
        self._dir = None

    def create(self) -> Path:
        self.cleanup()
        self._dir = Path(tempfile.mkdtemp(prefix="dogdetect-session-"))
        return self.path()

    def path(self):
        return self._dir / f"processed{config.OUTPUT_EXTENSION}" if self._dir else None

    def cleanup(self):
        if self._dir is None:
            return
        try:
            shutil.rmtree(self._dir)
        except Exception as e:
            log.warning("Temporary cleanup failed: %s", e)
        self._dir = None


def validate_output(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoWriteError("Output missing or empty")
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok or not is_valid_frame(frame):
            raise VideoWriteError("Output cannot be reopened")
    finally:
        cap.release()