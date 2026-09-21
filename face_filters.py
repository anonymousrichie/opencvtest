"""AR sunglasses anchored to MediaPipe Face Mesh eye landmarks.

Purely a screen-space overlay: given each face's 468-point mesh (from
FaceMeshDetector), a pair of dark lenses is drawn over the eyes, sized from
the inter-eye distance and rotated to match head tilt (roll), so the glasses
track the face as it moves, turns, and tilts.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from config import FaceFilterConfig

# MediaPipe Face Mesh landmark indices (fixed by the model's topology).
_RIGHT_EYE_OUTER, _RIGHT_EYE_INNER = 33, 133
_LEFT_EYE_INNER, _LEFT_EYE_OUTER = 362, 263

_HIGHLIGHT_COLOR = (230, 230, 230)


class FaceFilters:
    """Renders AR sunglasses over every detected face's mesh."""

    def __init__(self, cfg: FaceFilterConfig) -> None:
        self._cfg = cfg
        self.enabled = cfg.enabled_by_default

    def render(self, frame: np.ndarray, meshes: list[np.ndarray]) -> None:
        if not self.enabled:
            return
        for pts in meshes:
            self._draw_sunglasses(frame, pts)

    def _draw_sunglasses(self, frame: np.ndarray, pts: np.ndarray) -> None:
        right_center = (pts[_RIGHT_EYE_OUTER] + pts[_RIGHT_EYE_INNER]) / 2.0
        left_center = (pts[_LEFT_EYE_INNER] + pts[_LEFT_EYE_OUTER]) / 2.0

        eye_vec = left_center - right_center
        inter_eye = float(np.linalg.norm(eye_vec)) or 1.0
        angle_deg = math.degrees(math.atan2(float(eye_vec[1]), float(eye_vec[0])))

        lens_radius = inter_eye * self._cfg.lens_radius_scale
        axes = (int(lens_radius), int(lens_radius * 0.72))

        for eye_center in (right_center, left_center):
            center = (int(eye_center[0]), int(eye_center[1]))
            cv2.ellipse(frame, center, axes, angle_deg, 0, 360, self._cfg.lens_color, -1, cv2.LINE_AA)
            cv2.ellipse(frame, center, axes, angle_deg, 0, 360, (10, 10, 10), 2, cv2.LINE_AA)

            # A small glossy highlight, offset toward the upper side of the
            # lens along its own tilted axis so it stays put as the head turns.
            offset_angle = math.radians(angle_deg) - 0.6
            hx = eye_center[0] + math.cos(offset_angle) * lens_radius * 0.4
            hy = eye_center[1] + math.sin(offset_angle) * lens_radius * 0.4
            cv2.ellipse(
                frame, (int(hx), int(hy)), (max(1, int(lens_radius * 0.22)), max(1, int(lens_radius * 0.12))),
                angle_deg, 0, 360, _HIGHLIGHT_COLOR, -1, cv2.LINE_AA,
            )

        # Bridge connecting the two lenses.
        cv2.line(
            frame, (int(right_center[0]), int(right_center[1])), (int(left_center[0]), int(left_center[1])),
            (10, 10, 10), max(1, int(lens_radius * 0.12)), cv2.LINE_AA,
        )
