"""Screen-space payoff for the FER+ emotion recognizer.

Confetti falls from a tracked face's bounding box while it reads as
"Happy", and the screen edges flash red (a vignette, easier to make legible
than actually shaking the frame) while one reads as "Angry". Purely
reactive to Track.emotion/emotion_score each frame -- no gesture or hand
involvement, unlike the other effects modules.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from config import EmotionFxConfig

_CONFETTI_COLORS = [
    (60, 60, 230),  # red
    (50, 200, 230),  # yellow
    (60, 200, 60),  # green
    (230, 160, 50),  # blue
    (200, 70, 200),  # magenta
]


class _EmotionTrack(Protocol):
    bbox: tuple[int, int, int, int]
    emotion: str
    emotion_score: float


@dataclass
class _Confetto:
    x: float
    y: float
    vx: float
    vy: float
    size: int
    color: tuple[int, int, int]
    angle: float
    spin: float


def _build_vignette(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = np.clip(dist - 0.5, 0.0, 1.0) / 0.5  # 0 near center, 1 at the far corners
    vignette = np.zeros((h, w, 3), dtype=np.uint8)
    vignette[..., 2] = (mask * 255).astype(np.uint8)  # red channel only (BGR)
    return vignette


class EmotionEffects:
    """Confetti-on-smile + angry-vignette overlay driven by tracked faces' emotions."""

    def __init__(self, cfg: EmotionFxConfig) -> None:
        self._cfg = cfg
        self._confetti: list[_Confetto] = []
        self._anger = 0.0  # smoothed 0..1 vignette strength
        self._vignette: np.ndarray | None = None
        self._vignette_shape: tuple[int, int] | None = None

    def update(self, tracks: list[_EmotionTrack]) -> None:
        cfg = self._cfg
        any_angry = False
        for track in tracks:
            if track.emotion == "Happy" and track.emotion_score >= cfg.happy_threshold:
                self._spawn_confetti(track.bbox)
            if track.emotion == "Angry" and track.emotion_score >= cfg.angry_threshold:
                any_angry = True

        target = 1.0 if any_angry else 0.0
        self._anger += (target - self._anger) * cfg.anger_fade_step

        for c in self._confetti:
            c.vy += 0.15  # gravity
            c.x += c.vx
            c.y += c.vy
            c.angle += c.spin
        self._confetti = [c for c in self._confetti if c.y < 2000][-self._cfg.max_confetti :]

    def _spawn_confetti(self, bbox: tuple[int, int, int, int]) -> None:
        if len(self._confetti) >= self._cfg.max_confetti:
            return
        x, y, w, _h = bbox
        for _ in range(self._cfg.confetti_spawn_rate):
            self._confetti.append(
                _Confetto(
                    x=x + random.uniform(0, w),
                    y=y - random.uniform(0, 20),
                    vx=random.uniform(-1.5, 1.5),
                    vy=random.uniform(-1.0, 1.0),
                    size=random.randint(3, 6),
                    color=random.choice(_CONFETTI_COLORS),
                    angle=random.uniform(0, 360),
                    spin=random.uniform(-8, 8),
                )
            )

    def render(self, frame: np.ndarray) -> None:
        if self._anger > 0.01:
            self._render_anger_vignette(frame)
        for c in self._confetti:
            self._draw_confetto(frame, c)

    def _draw_confetto(self, frame: np.ndarray, c: _Confetto) -> None:
        h, w = frame.shape[:2]
        if not (0 <= c.x < w and 0 <= c.y < h):
            return
        box = cv2.boxPoints(((c.x, c.y), (c.size, c.size * 2), c.angle)).astype(np.int32)
        cv2.fillConvexPoly(frame, box, c.color, cv2.LINE_AA)

    def _render_anger_vignette(self, frame: np.ndarray) -> None:
        shape = frame.shape[:2]
        if self._vignette is None or self._vignette_shape != shape:
            self._vignette = _build_vignette(shape)
            self._vignette_shape = shape
        alpha = self._anger * self._cfg.anger_max_alpha
        cv2.addWeighted(frame, 1.0, self._vignette, alpha, 0, dst=frame)
