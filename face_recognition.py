"""Face embedding extraction (SFace) and matching against the enrollment database."""
from __future__ import annotations

import cv2
import numpy as np

from config import FaceRecognitionConfig
from database import EnrollmentDatabase, PersonInfo
from face_detection import DetectedFace, ModelLoadError


class FaceRecognizer:
    """Wraps cv2.FaceRecognizerSF to turn a DetectedFace into a 128-d embedding
    and match it against an EnrollmentDatabase."""

    def __init__(self, cfg: FaceRecognitionConfig, database: EnrollmentDatabase) -> None:
        self._cfg = cfg
        self._db = database
        if not cfg.model_path.exists():
            raise ModelLoadError(
                f"Face recognition model not found at {cfg.model_path}. "
                "Run `python download_models.py` first."
            )
        self._recognizer = cv2.FaceRecognizerSF.create(str(cfg.model_path), "")

    def embed(self, frame: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Align + crop the face and extract its embedding vector."""
        aligned = self._recognizer.alignCrop(frame, face.raw)
        feature = self._recognizer.feature(aligned)
        return feature.flatten()

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return (name, cosine_similarity). Below the configured threshold
        (or with an empty database) the name is reported as "Unknown" even
        though a raw best-score is still returned for display/debugging."""
        best_name, best_score = self._db.best_match(embedding)
        if best_name is None or best_score < self._cfg.match_threshold:
            return "Unknown", max(best_score, 0.0)
        return best_name, best_score

    def enroll(self, name: str, embedding: np.ndarray, info: PersonInfo | None = None) -> None:
        self._db.add(name, embedding, info)
