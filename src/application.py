import logging
import sys
import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication

import config
from detection.ultralytics_detector import UltralyticsDetector
from errors import ApplicationError
from gui.main_window import MainWindow
from monitoring.performance import PerformanceMonitor
from output.save_manager import SaveManager
from output.temporary_output import TemporaryOutput, validate_output
from processing.pipeline import PipelineEvents, ProcessingPipeline
from rendering.renderer import OpenCVRenderer
from selection.target_selector import TargetSelector
from tracking.target_manager import TargetManager
from tracking.ultralytics_tracker import UltralyticsTracker
from video.reader import OpenCVVideoReader
from video.writer import OpenCVVideoWriter

log = logging.getLogger(__name__)


class ProcessingWorker(QObject):
    started_processing = Signal()
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    target_selected = Signal(int, int)
    performance_updated = Signal(str)
    processing_completed = Signal(object)
    processing_failed = Signal(str)
    processing_cancelled = Signal()

    def __init__(self, pipeline_factory, output_path):
        super().__init__()
        self._factory, self._out = pipeline_factory, output_path

    @Slot()
    def run(self):
        try:
            events = PipelineEvents(
                on_started=self.started_processing.emit, on_status=self.status_changed.emit,
                on_progress=self.progress_changed.emit, on_target=self.target_selected.emit,
                on_performance=self.performance_updated.emit)
            result = self._factory(events).process()
            if result.cancelled:
                self.processing_cancelled.emit()
                return
            validate_output(self._out)
            self.processing_completed.emit(result)
        except ApplicationError as e:
            log.error("Processing failed: %s", e, exc_info=True)
            self.processing_failed.emit(e.user_message)
        except Exception:
            log.exception("Unexpected error")
            self.processing_failed.emit(
                "An unexpected error stopped processing. See the application log for details.")


class Application(QObject):
    def __init__(self):
        super().__init__()
        self._temp, self._saver = TemporaryOutput(), SaveManager()
        self._cancel = threading.Event()
        self._thread = self._worker = self._window = None

    def run(self) -> int:
        config.validate()
        qt = QApplication.instance() or QApplication(sys.argv)
        self._window = MainWindow(self)
        self._window.show()
        return qt.exec()

    def start_processing(self, input_path: str):
        self.discard_output()
        out = self._temp.create()
        self._cancel = threading.Event()
        cancel = self._cancel

        def factory(events):
            return ProcessingPipeline(
                reader=OpenCVVideoReader(),
                detector=UltralyticsDetector(config.MODEL_PATH, config.DEVICE,
                                             config.DETECTION_CONFIDENCE, config.INFERENCE_SIZE),
                tracker=UltralyticsTracker(config.TRACKER_TYPE, config.REID_ENABLED, config.TRACK_BUFFER),
                selector=TargetSelector(config.TARGET_CONFIRMATION_FRAMES),
                manager=TargetManager(config.MAX_TARGETS, config.REASSOC_MIN_IOU,
                                      config.REASSOC_AMBIGUITY_MARGIN, config.LOST_TIMEOUT_FRAMES),
                renderer=OpenCVRenderer(config.DEBUG_LABELS),
                writer=OpenCVVideoWriter(),
                monitor=PerformanceMonitor(config.PERFORMANCE_LOG_INTERVAL, config.WARMUP_FRAMES),
                input_path=input_path, output_path=out, cancel_event=cancel, events=events)

        w, win = ProcessingWorker(factory, out), self._window
        t = QThread()
        w.moveToThread(t)
        t.started.connect(w.run)
        w.started_processing.connect(win.on_started)
        w.status_changed.connect(win.show_status)
        w.progress_changed.connect(win.on_progress)
        w.target_selected.connect(win.on_target)
        w.performance_updated.connect(win.on_performance)
        w.processing_completed.connect(self._on_completed)
        w.processing_failed.connect(self._on_failed)
        w.processing_cancelled.connect(self._on_cancelled)
        for sig in (w.processing_completed, w.processing_failed, w.processing_cancelled):
            sig.connect(t.quit)
        t.finished.connect(w.deleteLater)
        self._thread, self._worker = t, w
        t.start()

    @Slot(object)
    def _on_completed(self, result):
        log.info("Done: %d frames, %d targets, %.1f fps", result.frames_processed,
                 result.targets_selected, result.average_processing_fps or 0)
        self._window.on_completed(result.output_path, result.targets_selected)

    @Slot(str)
    def _on_failed(self, message):
        self._temp.cleanup()
        self._window.on_failed(message)

    @Slot()
    def _on_cancelled(self):
        self._temp.cleanup()
        self._window.on_cancelled()

    def cancel_processing(self):
        self._cancel.set()

    def save(self, destination):
        return self._saver.save(self._temp.path(), destination, overwrite=True)  # dialog already confirmed

    def discard_output(self):
        if self._window:
            self._window.release_player()
        self._temp.cleanup()

    def shutdown(self):
        self._cancel.set()
        try:
            if self._thread is not None and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(15000)
        except RuntimeError:
            pass
        self.discard_output()