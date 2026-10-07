# Object Detector/Tracker in Video

Offline desktop application that detects dogs in a prerecorded video, selects the first N distinct dogs, tracks them persistently, and produces a labelled video for playback and saving.

## Features

- Select a local video (`.mp4`, `.avi`, `.mov`, `.mkv`) through a PySide6 GUI
- Detection with a local Ultralytics YOLO model, restricted to the `dog` class
- First-N target selection: a dog must be seen in several consecutive frames before it is assigned a target slot
- Persistent application-level target IDs, with IoU-based re-association after temporary loss
- Bounding boxes with labels in the form `dog 0.87`
- Processing runs on a background thread, with progress, cancellation and performance stats (FPS, inference time, CPU, memory, GPU)
- Output is written to a temporary file and only saved when you press **Save**

## Requirements

- Python 3.10+
- PySide6, OpenCV (`opencv-python`), NumPy, psutil, PyTorch, Ultralytics
- Optional: CUDA-capable GPU (falls back to CPU)

```bash
pip install PySide6 opencv-python numpy psutil torch ultralytics
```

## Usage

```bash
python src/main.py
```

1. Click **Select Video** and choose a file.
2. Click **Process**. Cancel at any time.
3. When processing completes, use **Play / Pause / Stop** to review the result.
4. Click **Save** to write the labelled video to a location of your choice (`.mp4`).

Note: the BoT-SORT ReID model (`yolo11n-cls.pt`) may be downloaded by Ultralytics on first run. Set `REID_ENABLED = False` in `src/config.py` to avoid this.

## Configuration

All settings live in `src/config.py`.

| Setting | Default | Description |
|---|---|---|
| `TARGET_CLASS` | `"dog"` | Class name to detect |
| `MAX_TARGETS` | `10` | Maximum number of targets selected |
| `DETECTION_CONFIDENCE` | `0.40` | Minimum detection confidence |
| `TARGET_CONFIRMATION_FRAMES` | `3` | Consecutive frames needed to confirm a candidate |
| `MODEL_PATH` | `src/models/model.pt` | YOLO weights |
| `DEVICE` | `"auto"` | `auto`, `cpu` or `cuda` |
| `INFERENCE_SIZE` | `640` | Inference image size |
| `TRACKER_TYPE` | `"botsort"` | `botsort` or `bytetrack` |
| `REID_ENABLED` | `True` | Appearance ReID (BoT-SORT only) |
| `TRACK_BUFFER` | `30` | Tracker lost-track buffer |
| `REASSOC_MIN_IOU` | `0.30` | Minimum IoU to re-associate a lost target |
| `REASSOC_AMBIGUITY_MARGIN` | `0.15` | Required IoU gap to the second-best match |
| `LOST_TIMEOUT_FRAMES` | `90` | Frames before a lost target is permanently lost |
| `DEBUG_LABELS` | `False` | Prefix labels with the target ID |

## Architecture

```
GUI (PySide6) -> Application -> ProcessingWorker (QThread) -> ProcessingPipeline
```

Per frame, the pipeline runs: read -> detect -> filter to target class -> track -> update existing targets -> re-associate lost targets -> select new targets -> render -> write.

```
src/
├── main.py              Entry point
├── application.py       Controller, worker thread, lifecycle
├── config.py            Settings and validation
├── errors.py            Application exception hierarchy
├── applog.py            Logging setup
├── gui/                 Main window and UI state machine
├── video/               Reader, writer, metadata
├── detection/           YOLO detector, data models
├── tracking/            Ultralytics tracker wrapper, target manager
├── selection/           First-N target selector
├── rendering/           Box and label drawing
├── processing/          Frame-processing pipeline
├── output/              Temporary output and save manager
├── monitoring/          Performance metrics
└── models/              YOLO weights
documents/
├── SAD.pdf              System Architecture Document
└── SDP.pdf              Software Design Document
```

Design principle: a detection is not a tracker ID, and a tracker ID is not an application target. The application owns the final target identity.

## Limitations

- Output has no audio track
- Targets keep their slot even when permanently lost, so at most `MAX_TARGETS` dogs are ever selected per video
- Output uses the `mp4v` codec; playback depends on your Qt multimedia backend

## Documentation

See `documents/SAD.pdf` and `documents/SDP.pdf` for the full architecture and design specifications.
