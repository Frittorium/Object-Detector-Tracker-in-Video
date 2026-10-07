import cv2

import config
from errors import VideoWriteError


class OpenCVVideoWriter:
    def __init__(self):
        self._w = None
        self._size = None

    def open(self, path, metadata):
        fps = metadata.source_fps if metadata.fps_reliable and metadata.source_fps else config.FALLBACK_FPS
        self._size = (metadata.width, metadata.height)
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC)
        self._w = cv2.VideoWriter(str(path), fourcc, fps, self._size)
        if not self._w.isOpened():
            self._w = None
            raise VideoWriteError("VideoWriter failed to open",
                                  "Could not create the output video (codec unavailable?).")

    def write(self, frame):
        if self._w is None:
            raise VideoWriteError("Writer is not open")
        if (frame.shape[1], frame.shape[0]) != self._size:
            raise VideoWriteError("Frame size does not match writer size")
        self._w.write(frame)

    def close(self):
        if self._w is not None:
            self._w.release()
            self._w = None