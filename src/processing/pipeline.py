import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import config
from errors import InferenceError, VideoReadError
from video.reader import is_valid_frame

log = logging.getLogger(__name__)


def _noop(*a, **k):
    pass


@dataclass
class PipelineEvents:
    on_started: Callable = field(default=_noop)
    on_status: Callable = field(default=_noop)
    on_progress: Callable = field(default=_noop)      # (frames, total_or_-1)
    on_target: Callable = field(default=_noop)        # (target_id, tracker_id)
    on_performance: Callable = field(default=_noop)   # (text)


@dataclass(frozen=True)
class ProcessingResult:
    success: bool
    cancelled: bool
    output_path: Path | None
    frames_processed: int
    processing_time: float
    average_processing_fps: float | None
    targets_selected: int


class ProcessingPipeline:
    def __init__(self, *, reader, detector, tracker, selector, manager, renderer, writer,
                 monitor, input_path, output_path, cancel_event: threading.Event, events):
        self.reader, self.detector, self.tracker = reader, detector, tracker
        self.selector, self.manager, self.renderer = selector, manager, renderer
        self.writer, self.monitor = writer, monitor
        self.input_path, self.output_path = input_path, output_path
        self.cancel, self.ev = cancel_event, events

    def process(self) -> ProcessingResult:
        frame_index, cancelled, t0 = 0, False, time.perf_counter()
        try:
            self.ev.on_status("Validating video...")
            self.reader.open(self.input_path)
            meta = self.reader.metadata()
            log.info("Input %s: %dx%d, fps=%s, frames=%s", self.input_path,
                     meta.width, meta.height, meta.source_fps, meta.frame_count)
            self.ev.on_status("Loading model...")
            self.detector.load()
            self.tracker.initialize(round(meta.source_fps) if meta.fps_reliable else 30)
            self.writer.open(self.output_path, meta)
            self.monitor.start()
            self.ev.on_started()
            self.ev.on_status("Processing...")
            total = meta.frame_count if meta.frame_count_known else -1

            fails = 0
            while True:
                if self.cancel.is_set():
                    cancelled = True
                    break
                ok, frame = self.reader.read()
                if not ok or not is_valid_frame(frame):
                    fails += 1
                    if fails == 1:
                        log.warning("Frame read failure")
                    if fails >= config.MAX_CONSECUTIVE_READ_FAILURES:
                        if self.reader.looks_complete(frame_index):
                            break                               # normal EOF
                        log.error("Maximum consecutive frame-read failures exceeded")
                        raise VideoReadError("Too many consecutive read failures")
                    continue
                fails = 0
                frame_index += 1
                self._process_frame(frame, frame_index)
                if frame_index % 5 == 0:
                    self.ev.on_progress(frame_index, total)

            self.writer.close()                                 # finalize before validation
            elapsed = time.perf_counter() - t0
            snap = self.monitor.finish()
            log.info("Final: %s", snap.format())
            return ProcessingResult(not cancelled, cancelled, self.output_path, frame_index,
                                    elapsed, frame_index / elapsed if elapsed > 0 else None,
                                    self.manager.target_count())
        finally:
            self._cleanup()

    def _process_frame(self, frame, frame_index):
        detections, dt = self._detect(frame)
        dogs = [d for d in detections
                if d.class_name == config.TARGET_CLASS and d.confidence >= config.DETECTION_CONFIDENCE]
        obs = self.tracker.update(frame, dogs)
        unmapped = self.manager.update_existing(obs, frame_index)
        unmapped = self.manager.reassociate(unmapped, frame_index)
        for c in self.selector.evaluate(unmapped, frame_index, self.manager.slots_left()):
            t = self.manager.create_target(c.latest_observation, frame_index)
            self.selector.discard(c.tracker_id)
            self.ev.on_target(t.target_id, c.tracker_id)
        self.manager.finalize_visibility(frame_index)
        self.writer.write(self.renderer.render(frame, self.manager.targets()))
        self.monitor.record_frame(dt)
        if self.monitor.due():
            text = self.monitor.snapshot().format()
            log.info("Performance: %s", text)
            self.ev.on_performance(text)

    def _detect(self, frame):
        last = None
        for attempt in range(1, config.MAX_CONSECUTIVE_INFERENCE_FAILURES + 1):
            try:
                t = time.perf_counter()
                dets = self.detector.detect(frame)
                return dets, time.perf_counter() - t
            except Exception as e:
                last = e
                log.warning("Inference failure %d/%d: %s", attempt,
                            config.MAX_CONSECUTIVE_INFERENCE_FAILURES, e)
        raise InferenceError("Inference failure threshold exceeded") from last

    def _cleanup(self):
        for c in (self.writer, self.tracker, self.detector, self.reader):
            try:
                c.close()
            except Exception:
                log.exception("Cleanup failure")