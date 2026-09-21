"""Virtual whiteboard driven by hand gestures.

Two extended fingers (index + middle, ring/pinky folded) draw a freehand
stroke that follows the midpoint between the two fingertips. A closed-fist
pinch (thumb and index tips touching, other fingers folded) grabs the most
recently completed stroke and stretches it: moving the pinch point away from
that stroke's centroid scales it up, moving it back in shrinks it down. A
fully closed fist (thumb tucked in too, not pinched) erases any stroke
within reach. Three more poses -- index only, index+middle+ring, and
index+pinky ("rock on") -- switch the pen color; each is a distinct finger
combination from the draw/pinch/erase poses so they never fight over the
same gesture.

State is tracked per hand (keyed by MediaPipe's "Left"/"Right" label) so two
hands can draw, pinch, erase, and pick colors independently in the same
frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import DrawingConfig
from hand_gestures import HandResult

# MediaPipe Hands landmark indices (see MediaPipe Hands documentation).
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP, _INDEX_PIP = 8, 6
_MIDDLE_TIP, _MIDDLE_PIP, _MIDDLE_MCP = 12, 10, 9
_RING_TIP, _RING_PIP = 16, 14
_PINKY_TIP, _PINKY_PIP = 20, 18


@dataclass
class Stroke:
    """One freehand polyline, in pixel space."""

    points: np.ndarray  # (N, 2) float32
    color: tuple[int, int, int]
    thickness: int


@dataclass
class _PinchState:
    centroid: np.ndarray
    initial_dist: float
    original_points: np.ndarray
    target: Stroke


@dataclass
class _HandState:
    active_stroke: Stroke | None = None
    pinch: _PinchState | None = None


def _extended(pts: np.ndarray, tip: int, pip: int) -> bool:
    """Same rule as hand_gestures._finger_extended: farther from the wrist
    than its PIP joint means the finger is extended."""
    wrist = pts[_WRIST]
    return float(np.linalg.norm(pts[tip] - wrist)) > float(np.linalg.norm(pts[pip] - wrist)) * 1.05


class DrawingCanvas:
    """Freehand drawing + pinch-to-stretch overlay, driven by hand landmarks."""

    def __init__(self, cfg: DrawingConfig) -> None:
        self._cfg = cfg
        self.enabled = cfg.enabled_by_default
        self.current_color = cfg.stroke_color
        self.strokes: list[Stroke] = []
        self._hand_states: dict[str, _HandState] = {}

    def clear(self) -> None:
        self.strokes.clear()
        self._hand_states.clear()

    def undo(self) -> None:
        """Remove the most recently drawn stroke, if any."""
        if self.strokes:
            self.strokes.pop()

    def update(self, hands: list[HandResult]) -> None:
        """Advance drawing/pinch state from this frame's detected hands."""
        if not self.enabled:
            return

        seen = set()
        for hand in hands:
            seen.add(hand.handedness)
            state = self._hand_states.setdefault(hand.handedness, _HandState())
            self._update_hand(state, hand.landmarks)

        # A hand that left the frame stops drawing/pinching cleanly rather
        # than leaving a dangling stroke or pinch anchor.
        for label in list(self._hand_states):
            if label not in seen:
                del self._hand_states[label]

    def _update_hand(self, state: _HandState, pts: np.ndarray) -> None:
        scale = float(np.linalg.norm(pts[_MIDDLE_MCP] - pts[_WRIST])) or 1.0

        index_up = _extended(pts, _INDEX_TIP, _INDEX_PIP)
        middle_up = _extended(pts, _MIDDLE_TIP, _MIDDLE_PIP)
        ring_up = _extended(pts, _RING_TIP, _RING_PIP)
        pinky_up = _extended(pts, _PINKY_TIP, _PINKY_PIP)
        pinch_dist = float(np.linalg.norm(pts[_THUMB_TIP] - pts[_INDEX_TIP]))

        drawing_pose = index_up and middle_up and not ring_up and not pinky_up
        # Deliberately excludes OK-Sign-like poses (middle/ring/pinky
        # extended) so pinch-to-stretch doesn't fire while making that sign.
        pinch_pose = (not middle_up and not ring_up and not pinky_up
                      and pinch_dist < self._cfg.pinch_threshold * scale)
        # A loosely closed fist (thumb *not* drawn in to the index tip, which
        # would instead read as the pinch-stretch pose above) erases.
        eraser_pose = not index_up and not middle_up and not ring_up and not pinky_up and not pinch_pose

        color_pose = None
        if index_up and not middle_up and not ring_up and not pinky_up:
            color_pose = 0
        elif index_up and middle_up and ring_up and not pinky_up:
            color_pose = 1
        elif index_up and not middle_up and not ring_up and pinky_up:
            color_pose = 2

        if drawing_pose:
            self._end_pinch(state)
            point = ((pts[_INDEX_TIP] + pts[_MIDDLE_TIP]) / 2.0).astype(np.float32)
            self._extend_stroke(state, point)
            return

        self._end_stroke(state)

        if pinch_pose:
            pinch_point = ((pts[_THUMB_TIP] + pts[_INDEX_TIP]) / 2.0).astype(np.float32)
            self._update_pinch(state, pinch_point)
            return
        self._end_pinch(state)

        if eraser_pose:
            self._erase_near(pts[_WRIST], scale * self._cfg.eraser_radius_scale)
        elif color_pose is not None and color_pose < len(self._cfg.palette):
            self.current_color = self._cfg.palette[color_pose]

    def _extend_stroke(self, state: _HandState, point: np.ndarray) -> None:
        if state.active_stroke is None:
            state.active_stroke = Stroke(
                points=point.reshape(1, 2),
                color=self.current_color,
                thickness=self._cfg.stroke_thickness,
            )
            self.strokes.append(state.active_stroke)
            return
        last = state.active_stroke.points[-1]
        if float(np.linalg.norm(point - last)) >= self._cfg.min_point_distance:
            state.active_stroke.points = np.vstack([state.active_stroke.points, point])

    def _end_stroke(self, state: _HandState) -> None:
        state.active_stroke = None

    def _update_pinch(self, state: _HandState, pinch_point: np.ndarray) -> None:
        if state.pinch is None:
            target = next((s for s in reversed(self.strokes) if len(s.points) >= 2), None)
            if target is None:
                return
            centroid = target.points.mean(axis=0)
            initial_dist = float(np.linalg.norm(pinch_point - centroid)) or 1.0
            state.pinch = _PinchState(
                centroid=centroid,
                initial_dist=initial_dist,
                original_points=target.points.copy(),
                target=target,
            )
            return

        pinch = state.pinch
        current_dist = float(np.linalg.norm(pinch_point - pinch.centroid))
        factor = np.clip(current_dist / pinch.initial_dist, self._cfg.min_stretch, self._cfg.max_stretch)
        pinch.target.points = pinch.centroid + (pinch.original_points - pinch.centroid) * factor

    def _end_pinch(self, state: _HandState) -> None:
        state.pinch = None

    def _erase_near(self, center: np.ndarray, radius: float) -> None:
        """Drop any stroke with a point within `radius` of `center`."""
        kept = []
        for stroke in self.strokes:
            dists = np.linalg.norm(stroke.points - center, axis=1)
            if not np.any(dists <= radius):
                kept.append(stroke)
        self.strokes = kept

    def render(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            cv2.polylines(
                frame, [stroke.points.astype(np.int32)], False, stroke.color, stroke.thickness, cv2.LINE_AA
            )
