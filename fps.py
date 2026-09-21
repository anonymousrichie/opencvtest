"""Rolling-window FPS counter."""
from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Tracks instantaneous FPS averaged over a small sliding window.

    A sliding window (rather than a single-frame delta) smooths out the
    per-frame jitter that would otherwise make the on-screen counter flicker.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a frame and return the current smoothed FPS estimate."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span
