import time
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class PerformanceSnapshot:
    frames: int
    elapsed: float
    processing_fps: float
    avg_inference_ms: float | None
    cpu_percent: float
    memory_mb: float
    gpu_percent: float | None

    def format(self) -> str:
        inf = f"{self.avg_inference_ms:.1f}" if self.avg_inference_ms is not None else "n/a"
        gpu = f"{self.gpu_percent:.0f}%" if self.gpu_percent is not None else "unavailable"
        return (f"frames={self.frames} processing_fps={self.processing_fps:.1f} "
                f"avg_inference_ms={inf} cpu={self.cpu_percent:.0f}% "
                f"memory={self.memory_mb:.0f}MB gpu={gpu}")


def _gpu_percent():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.utilization())
    except Exception:
        pass
    return None


class PerformanceMonitor:
    def __init__(self, interval: float, warmup_frames: int):
        self._interval, self._warmup = interval, warmup_frames
        self._proc = psutil.Process()
        self._t0 = self._last = 0.0
        self._frames = self._inf_n = 0
        self._inf_sum = 0.0

    def start(self):
        self._t0 = self._last = time.perf_counter()
        self._frames = self._inf_n = 0
        self._inf_sum = 0.0
        self._proc.cpu_percent(None)       # prime the CPU counter

    def record_frame(self, inference_time: float):
        self._frames += 1
        if self._frames > self._warmup:    # warm-up excluded from steady-state average
            self._inf_sum += inference_time
            self._inf_n += 1

    def due(self) -> bool:
        return time.perf_counter() - self._last >= self._interval

    def snapshot(self) -> PerformanceSnapshot:
        now = time.perf_counter()
        self._last = now
        elapsed = now - self._t0
        return PerformanceSnapshot(
            self._frames, elapsed, self._frames / elapsed if elapsed > 0 else 0.0,
            (self._inf_sum / self._inf_n * 1000) if self._inf_n else None,
            self._proc.cpu_percent(None), self._proc.memory_info().rss / 1e6, _gpu_percent())

    finish = snapshot