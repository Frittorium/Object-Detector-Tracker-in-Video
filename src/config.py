from pathlib import Path
from errors import ConfigurationError

ROOT = Path(__file__).resolve().parent.parent

TARGET_CLASS = "dog"
MAX_TARGETS = 10
DETECTION_CONFIDENCE = 0.40
TARGET_CONFIRMATION_FRAMES = 3

MODEL_PATH = ROOT / "src" / "models" / "model.pt"
DEVICE = "auto"            # auto | cpu | cuda
INFERENCE_SIZE = 640

TRACKER_TYPE = "botsort"   # botsort | bytetrack
REID_ENABLED = True
TRACK_BUFFER = 30

REASSOC_MIN_IOU = 0.30
REASSOC_AMBIGUITY_MARGIN = 0.15
LOST_TIMEOUT_FRAMES = 90

MAX_CONSECUTIVE_READ_FAILURES = 10
MAX_CONSECUTIVE_INFERENCE_FAILURES = 3
PERFORMANCE_LOG_INTERVAL = 5.0
WARMUP_FRAMES = 5

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
OUTPUT_EXTENSION = ".mp4"
OUTPUT_FOURCC = "mp4v"
FALLBACK_FPS = 30.0

DEBUG_LABELS = False
LOG_LEVEL = "INFO"
WINDOW_TITLE = "Offline Dog Detection"


def validate():
    if MAX_TARGETS < 1:
        raise ConfigurationError("MAX_TARGETS must be >= 1")
    if not 0.0 <= DETECTION_CONFIDENCE <= 1.0:
        raise ConfigurationError("DETECTION_CONFIDENCE must be in [0, 1]")
    if TARGET_CONFIRMATION_FRAMES < 1:
        raise ConfigurationError("TARGET_CONFIRMATION_FRAMES must be >= 1")
    if TRACKER_TYPE not in ("botsort", "bytetrack"):
        raise ConfigurationError(f"Unknown TRACKER_TYPE: {TRACKER_TYPE}")
    if DEVICE not in ("auto", "cpu", "cuda"):
        raise ConfigurationError(f"Unknown DEVICE: {DEVICE}")