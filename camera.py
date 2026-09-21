"""Webcam capture wrapper with clean lifecycle management and error handling."""
from __future__ import annotations

import platform

import cv2
import numpy as np

from config import CameraConfig

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraError(RuntimeError):
    """Raised when the requested camera cannot be opened or read from."""


class Camera:
    """Thin wrapper around cv2.VideoCapture with sane defaults.

    Use as a context manager (`with Camera(cfg) as cam:`) or call
    open()/release() manually. `open()` can be called again with a
    different `source` to switch cameras live -- the new source is opened
    and verified *before* the old one is released, so a failed switch
    (e.g. a virtual-camera app that isn't running) leaves the previous,
    working camera untouched instead of leaving the app with none at all.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self.active_source: int | str | None = None

    def open(self, source: int | str | None = None) -> None:
        """Open `source` (a device index, or a phone/network URL), raising
        CameraError on failure. Defaults to `CameraConfig.phone_url` if set,
        else `CameraConfig.index`.

        On Windows, DirectShow (CAP_DSHOW) opens faster and more reliably than
        the default MSMF backend on most laptops, so it is tried first there;
        CAP_ANY is used as a fallback (and as the only option elsewhere).
        """
        if source is None:
            source = self._cfg.phone_url or self._cfg.index

        if isinstance(source, str):
            # A network stream, not a local device -- no backend fallback or
            # width/height/fps hints apply (the phone's streaming app controls
            # its own resolution). Explicit open/read timeouts matter here in
            # a way they don't for a local device: without them, an
            # unreachable URL can hang VideoCapture for minutes with zero
            # feedback instead of failing fast.
            timeout = self._cfg.phone_timeout_ms
            new_cap = cv2.VideoCapture(
                source, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout],
            )
            if not new_cap.isOpened():
                new_cap.release()
                raise CameraError(
                    f"Could not connect to phone camera at {source}. "
                    "Check the phone and this PC are on the same network and the streaming app is running."
                )
            self.release()
            self._cap = new_cap
            self.active_source = source
            return

        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
        for backend in backends:
            new_cap = cv2.VideoCapture(source, backend)
            if new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
                new_cap.set(cv2.CAP_PROP_FPS, self._cfg.request_fps)
                self.release()
                self._cap = new_cap
                self.active_source = source
                return
            new_cap.release()
        raise CameraError(f"Could not open camera index {source}.")

    def read(self) -> np.ndarray:
        """Read one BGR frame, raising CameraError if the device stops delivering.

        A phone stream over Wi-Fi drops the occasional frame far more often
        than a wired USB webcam does, so a couple of quick retries happen
        here before giving up -- a local device (1 attempt) behaves exactly
        as before.
        """
        if self._cap is None:
            raise CameraError("Camera is not open. Call open() first.")
        is_network = isinstance(self.active_source, str)
        attempts = 3 if is_network else 1
        for _ in range(attempts):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                rotation = _ROTATIONS.get(self._cfg.phone_rotate) if is_network else None
                return cv2.rotate(frame, rotation) if rotation is not None else frame
        raise CameraError("Failed to read frame from camera (disconnected?).")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
