from enum import Enum, auto


class ApplicationState(Enum):
    IDLE = auto()
    FILE_SELECTED = auto()
    VALIDATING = auto()
    PROCESSING = auto()
    CANCELLING = auto()
    FAILED = auto()
    COMPLETED = auto()
    PLAYBACK = auto()
    SAVING = auto()
    SAVE_FAILED = auto()


S = ApplicationState
TRANSITIONS = {
    S.IDLE: {S.FILE_SELECTED},
    S.FILE_SELECTED: {S.FILE_SELECTED, S.VALIDATING},
    S.VALIDATING: {S.PROCESSING, S.FAILED},
    S.PROCESSING: {S.CANCELLING, S.FAILED, S.COMPLETED},
    S.CANCELLING: {S.IDLE, S.FAILED},
    S.FAILED: {S.FILE_SELECTED, S.VALIDATING},
    S.COMPLETED: {S.FILE_SELECTED, S.PLAYBACK, S.SAVING},
    S.PLAYBACK: {S.COMPLETED, S.FILE_SELECTED, S.SAVING},
    S.SAVING: {S.COMPLETED, S.SAVE_FAILED},
    S.SAVE_FAILED: {S.FILE_SELECTED, S.SAVING, S.COMPLETED, S.PLAYBACK},
}
_PLAY = {"play", "pause", "stop"}
ENABLED = {
    S.IDLE: {"select"},
    S.FILE_SELECTED: {"select", "process"},
    S.VALIDATING: set(),
    S.PROCESSING: {"cancel"},
    S.CANCELLING: set(),
    S.FAILED: {"select", "process"},
    S.COMPLETED: {"select", "save"} | _PLAY,
    S.PLAYBACK: {"select", "save"} | _PLAY,
    S.SAVING: set(),
    S.SAVE_FAILED: {"select", "save"} | _PLAY,
}