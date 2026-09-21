"""Optional dense face landmark overlay via MediaPipe Face Mesh.

This runs independently of the YuNet/SFace detection+recognition path, which
only needs YuNet's 5-point landmarks for alignment. MediaPipe's 468-point mesh
is purely a richer visual overlay, toggled off with 'l' if the extra CPU cost
(and visual clutter) isn't wanted.
"""
from __future__ import annotations

import mediapipe as mp
import numpy as np

from config import FaceMeshConfig


class FaceMeshDetector:
    """Wraps mediapipe.solutions.face_mesh for per-frame dense landmarks."""

    def __init__(self, cfg: FaceMeshConfig) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.max_num_faces,
            refine_landmarks=cfg.refine_landmarks,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def process(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Return one (468, 2) pixel-space landmark array per detected face."""
        h, w = frame_rgb.shape[:2]
        result = self._mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return []
        meshes = []
        for face_landmarks in result.multi_face_landmarks:
            pts = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)
            meshes.append(pts)
        return meshes

    def close(self) -> None:
        self._mesh.close()
