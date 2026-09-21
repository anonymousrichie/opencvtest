"""Iron-Man-style repulsor blast, drawn over an open palm.

Same trigger and per-hand state pattern as shield.py/rasengan.py: an open
palm (HandResult.gesture in ("Open Palm", "Four")) anchors the effect to the
palm centroid, with a smoothed fade-in/out strength. Visually it's a small,
bright arc-reactor-like core with rotating light spokes that periodically
fires an expanding shockwave ring outward, like a charged energy blast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import RepulsorConfig
from hand_gestures import HandResult

_WRIST = 0
_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + all four finger MCPs -> palm centroid
_MIDDLE_MCP = 9

_CORE_COLOR = (255, 255, 255)  # white-hot center (BGR)
_RING_COLOR = (255, 220, 120)  # bright cyan-blue
_OUTER_COLOR = (255, 150, 40)  # deep blue


@dataclass
class _Pulse:
    radius_mult: float  # current radius, as a multiple of the core radius


@dataclass
class _RepulsorState:
    phase: float = 0.0
    strength: float = 0.0  # smoothed 0..1 activation
    frames_active: int = 0
    pulses: list[_Pulse] = field(default_factory=list)


def _scaled(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


class RepulsorEffect:
    """Renders a charging/firing energy blast anchored to each open-palm hand."""

    def __init__(self, cfg: RepulsorConfig) -> None:
        self._cfg = cfg
        self.enabled = cfg.enabled_by_default
        self._states: dict[str, _RepulsorState] = {}

    def render(self, frame: np.ndarray, hands: list[HandResult]) -> None:
        if not self.enabled:
            return

        active_labels = {h.handedness for h in hands}
        for label in list(self._states):
            if label not in active_labels:
                state = self._states[label]
                state.strength = max(0.0, state.strength - self._cfg.fade_step)
                if state.strength <= 0.0 and not state.pulses:
                    del self._states[label]

        if not hands and not self._states:
            return

        glow = np.zeros_like(frame)
        sharp = np.zeros_like(frame)
        any_visible = False

        for hand in hands:
            state = self._states.setdefault(hand.handedness, _RepulsorState())
            target = 1.0 if hand.gesture in ("Open Palm", "Four") else 0.0
            state.strength += (target - state.strength) * self._cfg.fade_step
            if state.strength < 0.01 and target == 0.0:
                continue

            any_visible = True
            center, radius = self._palm_anchor(hand.landmarks)
            state.phase += self._cfg.rotation_speed
            if target > 0.5:
                state.frames_active += 1
            else:
                state.frames_active = 0
            self._update_pulses(state)
            self._draw_blast(glow, sharp, center, radius, state)

        if not any_visible:
            return

        glow = cv2.GaussianBlur(glow, (0, 0), self._cfg.glow_sigma)
        cv2.add(frame, glow, dst=frame)
        cv2.add(frame, sharp, dst=frame)

    def _palm_anchor(self, pts: np.ndarray) -> tuple[np.ndarray, float]:
        center = pts[list(_PALM_LANDMARKS)].mean(axis=0)
        scale = float(np.linalg.norm(pts[_MIDDLE_MCP] - pts[_WRIST])) or 1.0
        return center, scale * self._cfg.radius_scale

    def _update_pulses(self, state: _RepulsorState) -> None:
        cfg = self._cfg
        for pulse in state.pulses:
            pulse.radius_mult += cfg.pulse_speed
        state.pulses = [p for p in state.pulses if p.radius_mult < cfg.pulse_max_radius]

        if state.strength > 0.6 and state.frames_active > 0 and state.frames_active % cfg.pulse_interval == 0:
            state.pulses.append(_Pulse(radius_mult=1.0))

    def _draw_blast(
        self, glow: np.ndarray, sharp: np.ndarray, center: np.ndarray, radius: float, state: _RepulsorState,
    ) -> None:
        s = state.strength
        cx, cy = int(center[0]), int(center[1])

        # Ambient glow -- warm halo behind the crisp core.
        cv2.circle(glow, (cx, cy), int(radius * 1.4), _scaled(_OUTER_COLOR, s * 0.6), -1, cv2.LINE_AA)
        cv2.circle(glow, (cx, cy), int(radius * 0.7), _scaled(_RING_COLOR, s * 0.6), -1, cv2.LINE_AA)

        # Rotating light spokes, like a lens flare/star burst.
        for i in range(self._cfg.num_spokes):
            a = state.phase + i * (2 * math.pi / self._cfg.num_spokes)
            inner = radius * 0.3
            outer = radius * 1.6
            p1 = (int(center[0] + math.cos(a) * inner), int(center[1] + math.sin(a) * inner))
            p2 = (int(center[0] + math.cos(a) * outer), int(center[1] + math.sin(a) * outer))
            cv2.line(glow, p1, p2, _scaled(_RING_COLOR, s * 0.5), 1, cv2.LINE_AA)

        # Crisp arc-reactor rim + white-hot core.
        cv2.circle(sharp, (cx, cy), int(radius), _scaled(_RING_COLOR, s), max(1, int(2 * s)), cv2.LINE_AA)
        cv2.circle(sharp, (cx, cy), max(1, int(radius * 0.45)), _scaled(_CORE_COLOR, s), -1, cv2.LINE_AA)

        # Expanding shockwave pulses fired outward while fully charged.
        for pulse in state.pulses:
            life = 1.0 - pulse.radius_mult / self._cfg.pulse_max_radius
            r = int(radius * pulse.radius_mult)
            thickness = max(1, int(3 * life * s))
            cv2.circle(glow, (cx, cy), r, _scaled(_RING_COLOR, life * s), thickness, cv2.LINE_AA)
