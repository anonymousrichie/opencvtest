"""Doctor-Strange-style rotating fire shield, drawn over an open palm.

A screen-space visual effect: an open palm (HandResult.gesture == "Open
Palm") anchors a layered, rotating mandala -- a serrated flame-spike rim, a
rotating dashed ring, counter-rotating tick marks, a sacred-geometry star
pattern, a slow outer containment band with rivet-like dots, a pulsating
core, and comet-tailed embers flying outward -- rendered with an additive
Gaussian-blurred glow pass and a per-frame brightness flicker so it reads as
fiery rather than flat, static vector art.

State (rotation phase, flicker, embers, and a fade-in/out strength) is
tracked per hand, keyed by handedness, so a gesture flickering for a frame
doesn't pop the shield in and out, and two hands can each carry their own.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import ShieldConfig
from hand_gestures import HandResult

_WRIST = 0
_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + all four finger MCPs -> palm centroid
_MIDDLE_MCP = 9

_CORE_COLOR = (210, 245, 255)  # near-white amber (BGR) -- hottest part of the flame
_RING_COLOR = (30, 170, 255)  # amber
_OUTER_COLOR = (0, 90, 255)  # deep red-orange
_EMBER_COLOR = (60, 200, 255)


@dataclass
class _Ember:
    angle: float
    radius_frac: float
    drift: float
    size: float
    life: float  # 1.0 (born) -> 0.0 (dead)
    prev_angle: float = field(init=False)
    prev_radius_frac: float = field(init=False)

    def __post_init__(self) -> None:
        self.prev_angle = self.angle
        self.prev_radius_frac = self.radius_frac


@dataclass
class _ShieldState:
    phase: float = 0.0
    strength: float = 0.0  # smoothed 0..1 activation
    flicker: float = 1.0  # slow random walk around 1.0, for fire-like brightness variance
    embers: list[_Ember] = field(default_factory=list)


def _scaled(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def _draw_flame_rim(
    img: np.ndarray, center: np.ndarray, radius: float, phase: float,
    n_spikes: int, color: tuple[int, int, int],
) -> None:
    """A ring of small outward-pointing flame spikes instead of a plain
    circle -- each spike's height flickers on its own desynced frequency so
    the rim reads as a corona of fire rather than a smooth gear edge."""
    step = 2 * math.pi / n_spikes
    half_spread = step * 0.28
    for i in range(n_spikes):
        base_angle = phase * 0.3 + i * step
        flicker = 0.5 + 0.5 * math.sin(phase * (2.3 + 0.13 * i) + i * 1.7)
        h = radius * (0.10 + 0.16 * flicker)
        a0, a1 = base_angle - half_spread, base_angle + half_spread
        p0 = (int(center[0] + math.cos(a0) * radius), int(center[1] + math.sin(a0) * radius))
        p1 = (int(center[0] + math.cos(a1) * radius), int(center[1] + math.sin(a1) * radius))
        tip_r = radius + h
        tip = (int(center[0] + math.cos(base_angle) * tip_r), int(center[1] + math.sin(base_angle) * tip_r))
        cv2.fillPoly(img, [np.array([p0, tip, p1], dtype=np.int32)], color)


def _draw_dashed_ring(
    img: np.ndarray, center: np.ndarray, radius: float, phase: float,
    n_segments: int, arc_deg: float, color: tuple[int, int, int], thickness: int,
) -> None:
    step = 360.0 / n_segments
    base = math.degrees(phase)
    pt = (int(center[0]), int(center[1]))
    axes = (int(radius), int(radius))
    for i in range(n_segments):
        start = base + i * step
        cv2.ellipse(img, pt, axes, 0, start, start + arc_deg, color, thickness, cv2.LINE_AA)


def _draw_ticks(
    img: np.ndarray, center: np.ndarray, r_inner: float, r_outer: float, phase: float,
    n: int, color: tuple[int, int, int], thickness: int,
) -> None:
    for i in range(n):
        a = phase + i * (2 * math.pi / n)
        cos_a, sin_a = math.cos(a), math.sin(a)
        p1 = (int(center[0] + cos_a * r_inner), int(center[1] + sin_a * r_inner))
        p2 = (int(center[0] + cos_a * r_outer), int(center[1] + sin_a * r_outer))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_star_polygon(
    img: np.ndarray, center: np.ndarray, radius: float, phase: float,
    n_points: int, skip: int, color: tuple[int, int, int], thickness: int,
) -> None:
    pts = []
    for i in range(n_points):
        a = phase + i * (2 * math.pi / n_points)
        pts.append((center[0] + math.cos(a) * radius, center[1] + math.sin(a) * radius))
    for i in range(n_points):
        p1 = (int(pts[i][0]), int(pts[i][1]))
        p2 = (int(pts[(i + skip) % n_points][0]), int(pts[(i + skip) % n_points][1]))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_riveted_ring(
    img: np.ndarray, center: np.ndarray, radius: float, phase: float,
    n_dots: int, color: tuple[int, int, int], thickness: int,
) -> None:
    """A slow outer containment band with small rivet-like dots -- the outer
    boundary of the mandala, like the frame around the film's shield glyph."""
    cv2.circle(img, (int(center[0]), int(center[1])), int(radius), color, thickness, cv2.LINE_AA)
    for i in range(n_dots):
        a = phase + i * (2 * math.pi / n_dots)
        x = int(center[0] + math.cos(a) * radius)
        y = int(center[1] + math.sin(a) * radius)
        cv2.circle(img, (x, y), max(1, thickness + 1), color, -1, cv2.LINE_AA)


class ShieldEffect:
    """Renders a fiery rotating shield anchored to each open-palm hand."""

    def __init__(self, cfg: ShieldConfig) -> None:
        self._cfg = cfg
        self.enabled = cfg.enabled_by_default
        self._states: dict[str, _ShieldState] = {}

    def render(self, frame: np.ndarray, hands: list[HandResult]) -> None:
        if not self.enabled:
            return

        active_labels = {h.handedness for h in hands}
        for label in list(self._states):
            if label not in active_labels:
                state = self._states[label]
                state.strength = max(0.0, state.strength - self._cfg.fade_step)
                if state.strength <= 0.0 and not state.embers:
                    del self._states[label]

        if not hands and not self._states:
            return

        glow = np.zeros_like(frame)  # blurred ambient halo -- outer rim, core haze, embers
        sharp = np.zeros_like(frame)  # crisp mandala linework, drawn unblurred so detail reads
        any_visible = False

        for hand in hands:
            state = self._states.setdefault(hand.handedness, _ShieldState())
            target = 1.0 if hand.gesture == "Open Palm" else 0.0
            state.strength += (target - state.strength) * self._cfg.fade_step
            if state.strength < 0.01 and target == 0.0:
                continue

            any_visible = True
            center, radius = self._palm_anchor(hand.landmarks)
            state.phase += self._cfg.rotation_speed
            state.flicker += (random.uniform(0.82, 1.18) - state.flicker) * 0.25
            self._update_embers(state)
            self._draw_shield(glow, sharp, center, radius, state)

        if not any_visible:
            return

        glow = cv2.GaussianBlur(glow, (0, 0), self._cfg.glow_sigma)
        cv2.add(frame, glow, dst=frame)
        cv2.add(frame, sharp, dst=frame)

    def _palm_anchor(self, pts: np.ndarray) -> tuple[np.ndarray, float]:
        center = pts[list(_PALM_LANDMARKS)].mean(axis=0)
        scale = float(np.linalg.norm(pts[_MIDDLE_MCP] - pts[_WRIST])) or 1.0
        return center, scale * self._cfg.radius_scale

    def _update_embers(self, state: _ShieldState) -> None:
        cfg = self._cfg
        for ember in state.embers:
            ember.prev_angle = ember.angle
            ember.prev_radius_frac = ember.radius_frac
            ember.angle += ember.drift
            ember.radius_frac += cfg.ember_outflow
            ember.life -= cfg.ember_decay
        state.embers = [e for e in state.embers if e.life > 0 and e.radius_frac < 1.7]

        if state.strength > 0.5 and len(state.embers) < cfg.max_embers:
            for _ in range(cfg.ember_spawn_rate):
                angle = random.uniform(0, 2 * math.pi)
                radius_frac = random.uniform(0.95, 1.05)
                state.embers.append(
                    _Ember(
                        angle=angle,
                        radius_frac=radius_frac,
                        drift=random.uniform(-0.04, 0.04),
                        size=random.uniform(1.5, 3.5),
                        life=1.0,
                    )
                )

    def _draw_shield(
        self, glow: np.ndarray, sharp: np.ndarray, center: np.ndarray, radius: float, state: _ShieldState,
    ) -> None:
        s = state.strength * (0.85 + 0.15 * state.flicker)
        cx, cy = int(center[0]), int(center[1])

        # Ambient halo -- soft rim light and a warm haze at the center; this
        # layer gets blurred, so keep it to broad shapes, not fine linework.
        cv2.circle(glow, (cx, cy), int(radius), _scaled(_OUTER_COLOR, s * 0.85), max(3, int(5 * s)), cv2.LINE_AA)
        cv2.circle(glow, (cx, cy), max(1, int(radius * 0.22)), _scaled(_CORE_COLOR, s * 0.5), -1, cv2.LINE_AA)

        # Serrated flame-spike corona around the rim.
        _draw_flame_rim(glow, center, radius * 1.02, state.phase, 28, _scaled(_OUTER_COLOR, s * 0.8))

        # Crisp mandala linework -- drawn unblurred so the rings/spokes stay
        # sharp instead of dissolving into the glow.
        cv2.circle(sharp, (cx, cy), int(radius), _scaled(_OUTER_COLOR, s), max(1, int(2 * s)), cv2.LINE_AA)

        # Rotating dashed middle ring.
        _draw_dashed_ring(
            sharp, center, radius * 0.85, state.phase, 12, 14, _scaled(_RING_COLOR, s), max(1, int(2 * s))
        )

        # Counter-rotating inner tick marks (different speed reads as
        # independently-spinning rings, like the film's shield).
        _draw_ticks(sharp, center, radius * 0.58, radius * 0.72, -state.phase * 1.3, 16, _scaled(_RING_COLOR, s), 1)

        # Sacred-geometry star polygon.
        _draw_star_polygon(sharp, center, radius * 0.5, state.phase * 0.5, 8, 3, _scaled(_OUTER_COLOR, s), 1)

        # Slow outer containment band with rivet-like dots.
        _draw_riveted_ring(sharp, center, radius * 1.12, -state.phase * 0.15, 20, _scaled(_RING_COLOR, s * 0.8), 1)

        # Pulsating white-hot core.
        pulse = 0.09 + 0.015 * math.sin(state.phase * 4)
        cv2.circle(sharp, (cx, cy), max(1, int(radius * pulse * s)), _scaled(_CORE_COLOR, s), -1, cv2.LINE_AA)

        # Comet-tailed embers flying outward off the ring.
        for ember in state.embers:
            r0 = radius * ember.prev_radius_frac
            r1 = radius * ember.radius_frac
            p0 = (int(center[0] + math.cos(ember.prev_angle) * r0), int(center[1] + math.sin(ember.prev_angle) * r0))
            p1 = (int(center[0] + math.cos(ember.angle) * r1), int(center[1] + math.sin(ember.angle) * r1))
            size = max(1, int(ember.size * ember.life * s))
            color = _scaled(_EMBER_COLOR, ember.life * s)
            cv2.line(glow, p0, p1, color, max(1, size - 1), cv2.LINE_AA)
            cv2.circle(glow, p1, size, _scaled(_CORE_COLOR, ember.life * s), -1, cv2.LINE_AA)
