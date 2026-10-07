import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Slot
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMainWindow, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

import config
from errors import ApplicationError
from gui.state import ENABLED, TRANSITIONS, ApplicationState as S

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self._c = controller
        self._state = S.IDLE
        self._path = None
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(900, 700)

        self._btn = {name: QPushButton(label) for name, label in [
            ("select", "Select Video"), ("process", "Process"), ("cancel", "Cancel"),
            ("play", "Play"), ("pause", "Pause"), ("stop", "Stop"), ("save", "Save")]}
        self._file = QLabel("Selected: none")
        self._status = QLabel("Select a video to begin.")
        self._status.setWordWrap(True)
        self._perf = QLabel("")
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._video = QVideoWidget()
        self._video.setMinimumHeight(360)
        self._player = QMediaPlayer(self)
        self._player.setVideoOutput(self._video)
        self._player.errorOccurred.connect(self._on_player_error)

        def row(*names):
            h = QHBoxLayout()
            for n in names:
                h.addWidget(self._btn[n])
            h.addStretch()
            return h

        root = QVBoxLayout()
        root.addLayout(row("select"))
        root.addWidget(self._file)
        root.addLayout(row("process", "cancel"))
        root.addWidget(self._status)
        root.addWidget(self._bar)
        root.addWidget(self._perf)
        root.addWidget(self._video, 1)
        root.addLayout(row("play", "pause", "stop"))
        root.addLayout(row("save"))
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._btn["select"].clicked.connect(self._select)
        self._btn["process"].clicked.connect(self._process)
        self._btn["cancel"].clicked.connect(self._cancel)
        self._btn["play"].clicked.connect(self._play)
        self._btn["pause"].clicked.connect(self._player.pause)
        self._btn["stop"].clicked.connect(self._stop)
        self._btn["save"].clicked.connect(self._save)
        self._apply_state()

    # ---- state -------------------------------------------------------
    def set_state(self, new: S):
        if new is not self._state and new not in TRANSITIONS[self._state]:
            raise RuntimeError(f"Invalid state transition {self._state.name} -> {new.name}")
        self._state = new
        self._apply_state()

    def _apply_state(self):
        on = ENABLED[self._state]
        for name, b in self._btn.items():
            b.setEnabled(name in on)

    # ---- user actions ------------------------------------------------
    def _select(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "", "Video files (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self._c.discard_output()
        self._path = path
        self._file.setText(f"Selected: {Path(path).name}")
        self._status.setText("Ready to process.")
        self._perf.setText("")
        self._bar.setValue(0)
        self.set_state(S.FILE_SELECTED)

    def _process(self):
        self.set_state(S.VALIDATING)
        self._status.setText("Validating video...")
        self._bar.setRange(0, 0)
        self._c.start_processing(self._path)

    def _cancel(self):
        self.set_state(S.CANCELLING)
        self._status.setText("Cancelling...")
        self._c.cancel_processing()

    def _play(self):
        self.set_state(S.PLAYBACK)
        self._player.play()

    def _stop(self):
        self._player.stop()
        self.set_state(S.COMPLETED)

    def _save(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save processed video", "processed.mp4",
                                              "MP4 video (*.mp4)")
        if not dest:
            return
        self.set_state(S.SAVING)
        self._status.setText("Saving...")
        try:
            saved = self._c.save(Path(dest))
            self.set_state(S.COMPLETED)
            self._status.setText(f"Saved to {saved}")
        except ApplicationError as e:
            log.error("Save failed: %s", e)
            self.set_state(S.SAVE_FAILED)
            self._status.setText(f"Save failed. {e.user_message} You can retry.")

    # ---- worker events -----------------------------------------------
    @Slot()
    def on_started(self):
        self.set_state(S.PROCESSING)

    @Slot(str)
    def show_status(self, text):
        self._status.setText(text)

    @Slot(int, int)
    def on_progress(self, frames, total):
        if total > 0:
            self._bar.setRange(0, 100)
            self._bar.setValue(min(100, int(frames / total * 100)))
        else:
            self._bar.setRange(0, 0)          # indeterminate: frame count unreliable

    @Slot(int, int)
    def on_target(self, target_id, tracker_id):
        self._status.setText(f"Processing... Target {target_id} selected")

    @Slot(str)
    def on_performance(self, text):
        self._perf.setText(text)

    # ---- called by Application ----------------------------------------
    def on_completed(self, output_path, targets):
        self._bar.setRange(0, 100)
        self._bar.setValue(100)
        self._status.setText("Processing complete." if targets else
                             "Processing complete. No qualifying dogs were detected.")
        self._player.setSource(QUrl.fromLocalFile(str(output_path)))
        self.set_state(S.COMPLETED)

    def on_failed(self, message):
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._status.setText(f"Video processing failed.\n{message}")
        self.set_state(S.FAILED)

    def on_cancelled(self):
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._path = None
        self._file.setText("Selected: none")
        self._status.setText("Processing cancelled.")
        self.set_state(S.IDLE)

    def release_player(self):
        self._player.stop()
        self._player.setSource(QUrl())

    def _on_player_error(self, error, text):
        log.error("Playback error: %s", text)
        self._status.setText("Playback failed. The processed video can still be saved.")

    def closeEvent(self, event):
        self._c.shutdown()
        super().closeEvent(event)