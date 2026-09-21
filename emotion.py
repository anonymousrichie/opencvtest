"""Facial emotion classification via FER+ (an ONNX CNN run through cv2.dnn).

Unlike hand gestures, facial expressions don't reduce to a handful of clean
geometric rules -- the visual difference between, say, anger and disgust is
subtle and doesn't map to a simple landmark measurement. So this uses a small
pretrained CNN (FER+, ~34 MB, from the ONNX Model Zoo) instead, run the same
way YuNet/SFace are: via cv2.dnn, no extra dependency beyond
opencv-contrib-python already in requirements.txt.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import EmotionConfig
from face_detection import DetectedFace, ModelLoadError

# Output order fixed by the FER+ model's training labels.
_LABELS = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]


class EmotionRecognizer:
    """Wraps the FER+ ONNX CNN to classify one of 8 emotions from a face crop."""

    def __init__(self, cfg: EmotionConfig) -> None:
        self._cfg = cfg
        if not cfg.model_path.exists():
            raise ModelLoadError(
                f"Emotion model not found at {cfg.model_path}. "
                "Run `python download_models.py` first."
            )
        try:
            self._net = cv2.dnn.readNetFromONNX(str(cfg.model_path))
        except cv2.error as exc:
            # A truncated/corrupt file (e.g. an interrupted download) still
            # passes the exists() check above but fails here -- treat it the
            # same as "missing" rather than letting a raw cv2 error crash the
            # whole pipeline over an optional feature.
            raise ModelLoadError(
                f"Emotion model at {cfg.model_path} failed to load ({exc}). "
                "It may be corrupt or incomplete -- delete it and re-run "
                "`python download_models.py`."
            ) from exc

    def classify(self, frame: np.ndarray, face: DetectedFace) -> tuple[str, float]:
        """Return (label, confidence) for the dominant emotion in `face`'s crop.

        Falls back to ("Neutral", 0.0) if the bbox is clipped down to nothing
        by the frame edge.
        """
        x, y, w, h = face.bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return "Neutral", 0.0

        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # FER+ expects a single-channel 64x64 input (Nx1x64x64); blobFromImage
        # with a grayscale source produces that shape directly.
        blob = cv2.dnn.blobFromImage(gray, scalefactor=1.0, size=(64, 64))
        self._net.setInput(blob)
        logits = self._net.forward().flatten()

        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        return _LABELS[idx], float(probs[idx])


def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x))
    return exp / exp.sum()
