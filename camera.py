"""Webcam capture wrapper with clean lifecycle management and error handling."""
from __future__ import annotations

import platform

import cv2
import numpy as np

from config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the requested camera cannot be opened or read from."""


class Camera:
    """Thin wrapper around cv2.VideoCapture with sane defaults.

    Use as a context manager (`with Camera(cfg) as cam:`) or call
    open()/release() manually.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the configured camera index, raising CameraError on failure.

        On Windows, DirectShow (CAP_DSHOW) opens faster and more reliably than
        the default MSMF backend on most laptops, so it is tried first there;
        CAP_ANY is used as a fallback (and as the only option elsewhere).
        """
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(self._cfg.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
                cap.set(cv2.CAP_PROP_FPS, self._cfg.request_fps)
                self._cap = cap
                return
            cap.release()
        raise CameraError(f"Could not open camera index {self._cfg.index}.")

    def read(self) -> np.ndarray:
        """Read one BGR frame, raising CameraError if the device stops delivering."""
        if self._cap is None:
            raise CameraError("Camera is not open. Call open() first.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to read frame from camera (disconnected?).")
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
