import cv2

from tracking.target_manager import Visibility

_PALETTE = [(60, 180, 75), (230, 120, 40), (60, 110, 230), (200, 60, 200), (40, 190, 200)]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class OpenCVRenderer:
    def __init__(self, debug_labels=False):
        self._debug = debug_labels

    def render(self, frame, targets):
        out = frame.copy()
        for t in targets:
            d = t.latest_detection
            if t.visibility_state is not Visibility.VISIBLE or d is None:
                continue
            color = _PALETTE[(t.target_id - 1) % len(_PALETTE)]
            label = f"{d.class_name} {d.confidence:.2f}"
            if self._debug:
                label = f"Target {t.target_id} | {label}"
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            (tw, th), base = cv2.getTextSize(label, _FONT, 0.6, 1)
            y = max(d.y1, th + base + 4)
            cv2.rectangle(out, (d.x1, y - th - base - 4), (d.x1 + tw + 6, y), color, -1)
            cv2.putText(out, label, (d.x1 + 3, y - base - 2), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out