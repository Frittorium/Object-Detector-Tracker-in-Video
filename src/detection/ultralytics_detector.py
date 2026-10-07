import logging
from pathlib import Path

import numpy as np

from detection.models import Detection
from errors import ConfigurationError, ModelLoadError

log = logging.getLogger(__name__)


class UltralyticsDetector:
    def __init__(self, model_path, device, min_confidence, imgsz):
        self._path = Path(model_path)
        self._device_cfg = device
        self._conf = min_confidence
        self._imgsz = imgsz
        self._model = None
        self._device = "cpu"
        self._names = {}

    def load(self):
        if not self._path.is_file():
            log.error("Model not found: %s", self._path)
            raise ModelLoadError(f"Model file missing: {self._path}")
        if self._device_cfg not in ("auto", "cpu", "cuda"):
            raise ConfigurationError(f"Invalid device: {self._device_cfg}")
        try:
            import torch
            from ultralytics import YOLO
            cuda = torch.cuda.is_available()
            if self._device_cfg == "cuda" and not cuda:
                raise ModelLoadError("CUDA requested but unavailable",
                                     "CUDA was requested but is not available.")
            self._device = "cuda:0" if (cuda and self._device_cfg in ("auto", "cuda")) else "cpu"
            self._model = YOLO(str(self._path))
            self._names = self._model.names
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Model load failed: {e}") from e
        log.info("Model loaded: %s on %s", self._path.name, self._device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        r = self._model.predict(frame, device=self._device, imgsz=self._imgsz,
                                conf=self._conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        h, w = frame.shape[:2]
        out = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if not np.isfinite([x1, y1, x2, y2, c]).all() or not 0.0 <= c <= 1.0:
                log.warning("Rejected malformed detection")
                continue
            x1, x2 = int(max(0, min(w - 1, x1))), int(max(0, min(w - 1, x2)))
            y1, y2 = int(max(0, min(h - 1, y1))), int(max(0, min(h - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(int(k), str(self._names[int(k)]), float(c), x1, y1, x2, y2))
        return out

    def close(self):
        self._model = None