"""Face detection via OpenCV's YuNet (cv2.FaceDetectorYN).

YuNet is a small ONNX model distributed through the OpenCV Zoo that runs
comfortably in real time on CPU. It is used here instead of InsightFace's
detector (RetinaFace) to keep this base system's dependency footprint small
(OpenCV + MediaPipe only) and Windows-friendly -- the task spec explicitly
allows "OpenCV SFace + YuNet" as an alternative to InsightFace, and it pairs
directly with cv2.FaceRecognizerSF (SFace) for embeddings, sharing the same
5-point landmark output for alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import FaceDetectionConfig


class ModelLoadError(RuntimeError):
    """Raised when a required ONNX model file is missing or fails to load."""


@dataclass
class DetectedFace:
    """One face detection: box + 5-point landmarks + the raw YuNet output row.

    `raw` is kept because cv2.FaceRecognizerSF.alignCrop expects the detector's
    original 1x15 output row (box + landmarks + score), not a reconstructed one.
    """

    bbox: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray  # (5, 2): right eye, left eye, nose tip, mouth-right, mouth-left
    score: float
    raw: np.ndarray


class FaceDetector:
    """Wraps cv2.FaceDetectorYN for per-frame face detection."""

    def __init__(self, cfg: FaceDetectionConfig) -> None:
        self._cfg = cfg
        if not cfg.model_path.exists():
            raise ModelLoadError(
                f"Face detection model not found at {cfg.model_path}. "
                "Run `python download_models.py` first."
            )
        self._detector = cv2.FaceDetectorYN.create(
            str(cfg.model_path),
            "",
            (320, 320),
            score_threshold=cfg.score_threshold,
            nms_threshold=cfg.nms_threshold,
            top_k=cfg.top_k,
        )
        self._input_size: tuple[int, int] = (320, 320)

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in a BGR frame."""
        h, w = frame.shape[:2]
        if (w, h) != self._input_size:
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)

        _, faces = self._detector.detect(frame)
        if faces is None:
            return []

        results = []
        for row in faces:
            x, y, fw, fh = row[0:4].astype(int)
            landmarks = row[4:14].reshape(5, 2)
            score = float(row[14])
            results.append(DetectedFace(bbox=(int(x), int(y), int(fw), int(fh)), landmarks=landmarks, score=score, raw=row))
        return results
