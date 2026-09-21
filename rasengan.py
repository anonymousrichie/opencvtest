"""Naruto-style spinning chakra ball ("Rasengan"), drawn over an open palm.

Same trigger and per-hand state pattern as shield.py (see that module for
the rationale): an open palm (HandResult.gesture in ("Open Palm", "Four"),
the latter covering a flat hand held with the thumb tucked in rather than
splayed) anchors the effect to the palm centroid, with a smoothed fade-in/
out strength so a flickery gesture read doesn't pop it in and out.

The ball is rendered as a squashed, rotated ellipse rather than a plain
circle: as the palm turns away from facing the camera (e.g. rotating the
wrist so the palm faces sideways -- "flat"), the on-screen triangle formed
by the wrist and the index/pinky knuckles flattens out the same way a real
disc's silhouette would, and that flattening drives the ball's squash
factor. So instead of the effect vanishing or looking wrong when the palm
isn't square to the camera, it keeps reading as a tilting 3D sphere. A
small offset highlight blob adds further shading cues on top of that.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from config import RasenganConfig
from hand_gestures import HandResult

_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_PINKY_MCP = 17
_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist + all four finger MCPs -> palm centroid

_CORE_COLOR = (255, 250, 220)  # near-white, faint blue tint (BGR) -- hottest part of the chakra
_INNER_COLOR = (255, 210, 60)  # bright cyan-blue
_OUTER_COLOR = (255, 120, 20)  # deep blue
_HIGHLIGHT_COLOR = (255, 255, 255)

_Transform = Callable[[float, float], tuple[int, int]]


@dataclass
class _Mote:
    angle: float
    radius_frac: float
    speed: float
    size: float


@dataclass
class _RasenganState:
    phase: float = 0.0
    strength: float = 0.0  # smoothed 0..1 activation
    motes: list[_Mote] = field(default_factory=list)


def _scaled(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def _ellipse_transform(center: np.ndarray, squash: float, angle_rad: float) -> _Transform:
    """Maps a point in the ball's own unsquashed, unrotated circle space
    (local_x along the squash axis, local_y along the full-radius axis) to
    final pixel coordinates -- so every element of the ball (rim, spiral
    arms, motes) tilts together as one rigid disc."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    def transform(local_x: float, local_y: float) -> tuple[int, int]:
        sx = local_x * squash
        rx = sx * cos_a - local_y * sin_a
        ry = sx * sin_a + local_y * cos_a
        return int(center[0] + rx), int(center[1] + ry)

    return transform


def _draw_spiral_arm(
    img: np.ndarray, transform: _Transform, max_radius: float, turns: float, phase: float,
    color: tuple[int, int, int], thickness: int, samples: int = 20,
) -> None:
    """One Archimedean-spiral vortex arm from the center out to max_radius,
    in the ball's local circle space (transform handles the tilt)."""
    pts = []
    for i in range(samples + 1):
        t = i / samples
        angle = phase + turns * 2 * math.pi * t
        r = max_radius * t
        pts.append(transform(math.cos(angle) * r, math.sin(angle) * r))
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)


class RasenganEffect:
    """Renders a spinning chakra-ball anchored to each open-palm hand."""

    def __init__(self, cfg: RasenganConfig) -> None:
        self._cfg = cfg
        self.enabled = cfg.enabled_by_default
        self._states: dict[str, _RasenganState] = {}

    def render(self, frame: np.ndarray, hands: list[HandResult]) -> None:
        if not self.enabled:
            return

        active_labels = {h.handedness for h in hands}
        for label in list(self._states):
            if label not in active_labels:
                state = self._states[label]
                state.strength = max(0.0, state.strength - self._cfg.fade_step)
                if state.strength <= 0.0:
                    del self._states[label]

        if not hands and not self._states:
            return

        glow = np.zeros_like(frame)  # blurred ambient halo -- ball glow
        sharp = np.zeros_like(frame)  # crisp spiral/mote linework, drawn unblurred so detail reads
        any_visible = False

        for hand in hands:
            state = self._states.setdefault(hand.handedness, _RasenganState())
            # "Four" covers a flat hand held with the thumb tucked in rather
            # than splayed out -- geometrically identical palm otherwise.
            target = 1.0 if hand.gesture in ("Open Palm", "Four") else 0.0
            state.strength += (target - state.strength) * self._cfg.fade_step
            if state.strength < 0.01 and target == 0.0:
                continue

            any_visible = True
            center, radius, squash, angle = self._palm_orientation(hand.landmarks)
            state.phase += self._cfg.rotation_speed
            self._update_motes(state)
            self._draw_ball(glow, sharp, center, radius, squash, angle, state)

        if not any_visible:
            return

        glow = cv2.GaussianBlur(glow, (0, 0), self._cfg.glow_sigma)
        cv2.add(frame, glow, dst=frame)
        cv2.add(frame, sharp, dst=frame)

    def _palm_orientation(self, pts: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Palm centroid/radius plus a squash factor and tilt angle that make
        the ball read as a disc rotating in 3D as the palm turns.

        The wrist-index_mcp-pinky_mcp triangle's on-screen area (normalized
        by hand size) is largest when the palm faces the camera flat-on and
        shrinks toward zero as the hand rotates to show its edge -- exactly
        how a real disc's silhouette would flatten. `min_squash` keeps the
        ball from ever fully vanishing at that extreme.
        """
        center = pts[list(_PALM_LANDMARKS)].mean(axis=0)
        scale = float(np.linalg.norm(pts[_MIDDLE_MCP] - pts[_WRIST])) or 1.0
        radius = scale * self._cfg.radius_scale

        wrist = pts[_WRIST]
        v1 = pts[_INDEX_MCP] - wrist
        v2 = pts[_PINKY_MCP] - wrist
        facing = abs(float(v1[0] * v2[1] - v1[1] * v2[0])) / (scale * scale)
        squash = float(np.clip(facing / self._cfg.facing_reference, self._cfg.min_squash, 1.0))

        width_vec = pts[_PINKY_MCP] - pts[_INDEX_MCP]
        angle = math.atan2(float(width_vec[1]), float(width_vec[0]))

        return center, radius, squash, angle

    def _update_motes(self, state: _RasenganState) -> None:
        if not state.motes:
            for _ in range(self._cfg.num_motes):
                state.motes.append(
                    _Mote(
                        angle=random.uniform(0, 2 * math.pi),
                        radius_frac=random.uniform(0.8, 1.05),
                        speed=random.uniform(0.25, 0.5) * random.choice((-1, 1)),
                        size=random.uniform(1.5, 3.0),
                    )
                )
        for mote in state.motes:
            mote.angle += mote.speed

    def _draw_ball(
        self, glow: np.ndarray, sharp: np.ndarray, center: np.ndarray, radius: float,
        squash: float, angle: float, state: _RasenganState,
    ) -> None:
        s = state.strength
        cx, cy = int(center[0]), int(center[1])
        axes = (max(1, int(radius * squash)), max(1, int(radius)))
        angle_deg = math.degrees(angle)
        transform = _ellipse_transform(center, squash, angle)

        # Ambient halo -- a dim, translucent-looking base sphere; blurred, so
        # keep it to broad filled shapes rather than fine linework. Kept
        # noticeably dimmer than the spiral arms/core below so those read as
        # bright blades against it instead of washing out into one white disc.
        cv2.ellipse(glow, (cx, cy), axes, angle_deg, 0, 360, _scaled(_OUTER_COLOR, s * 0.5), -1, cv2.LINE_AA)
        inner_axes = (max(1, int(axes[0] * 0.45)), max(1, int(axes[1] * 0.45)))
        cv2.ellipse(glow, (cx, cy), inner_axes, angle_deg, 0, 360, _scaled(_INNER_COLOR, s * 0.35), -1, cv2.LINE_AA)

        # A soft highlight offset toward one side of the disc, like a light
        # source -- the main cue that reads as "sphere" rather than "flat
        # glowing disc" once the spin makes the spiral pattern hard to track.
        hx, hy = transform(-radius * 0.35, -radius * 0.35)
        cv2.circle(glow, (hx, hy), max(1, int(radius * 0.28 * squash)), _scaled(_HIGHLIGHT_COLOR, s * 0.4), -1, cv2.LINE_AA)

        # Crisp sphere rim + rotating vortex arms, drawn unblurred and bold
        # so the swirl pattern is legible against the dim base sphere.
        cv2.ellipse(sharp, (cx, cy), axes, angle_deg, 0, 360, _scaled(_INNER_COLOR, s * 0.7), max(1, int(2 * s)), cv2.LINE_AA)
        cfg = self._cfg
        for i in range(cfg.num_arms):
            arm_offset = i * (2 * math.pi / cfg.num_arms)
            phase = state.phase + arm_offset
            # Each arm is a duller blue backing stroke with a brighter,
            # thinner near-white line on top, like the anime's shaded blades.
            _draw_spiral_arm(sharp, transform, radius, cfg.arm_turns, phase, _scaled(_INNER_COLOR, s), max(2, int(5 * s)))
            _draw_spiral_arm(sharp, transform, radius, cfg.arm_turns, phase, _scaled(_CORE_COLOR, s), max(1, int(2 * s)))

        # Bright white-hot core -- kept small so it accents the center
        # rather than covering the spiral arms.
        pulse = 0.16 + 0.02 * math.sin(state.phase * 3)
        core_axes = (max(1, int(radius * pulse * squash * s)), max(1, int(radius * pulse * s)))
        cv2.ellipse(sharp, (cx, cy), core_axes, angle_deg, 0, 360, _scaled(_CORE_COLOR, s), -1, cv2.LINE_AA)

        # Chakra motes swirling near the surface, tilted with the disc.
        for mote in state.motes:
            mr = radius * mote.radius_frac
            mx, my = transform(math.cos(mote.angle) * mr, math.sin(mote.angle) * mr)
            size = max(1, int(mote.size * s))
            cv2.circle(sharp, (mx, my), size, _scaled(_CORE_COLOR, s), -1, cv2.LINE_AA)
