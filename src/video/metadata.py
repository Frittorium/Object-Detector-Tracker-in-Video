from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    source_fps: float | None
    frame_count: int | None
    duration: float | None
    frame_count_known: bool
    fps_reliable: bool