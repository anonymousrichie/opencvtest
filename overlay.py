"""Drawing helpers -- keeps main.py free of raw cv2 drawing calls."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from database import PersonInfo

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_KNOWN_COLOR = (60, 220, 60)
_UNKNOWN_COLOR = (60, 60, 220)
_TEXT_COLOR = (255, 255, 255)
_HAND_COLOR = (220, 180, 40)
_MESH_COLOR = (0, 200, 255)
_KEYPOINT_COLOR = (255, 255, 0)


def draw_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    keypoints: np.ndarray,
    track_id: int,
    name: str,
    score: float,
    emotion: str = "",
    emotion_score: float = 0.0,
    show_emotion: bool = False,
    info: PersonInfo | None = None,
) -> None:
    """Draw a face bounding box, its 5 detector keypoints, and an identity
    label, optionally appending the classified emotion and a profile-details
    line (nickname/department/ID/hall/age) below the box."""
    x, y, w, h = bbox
    color = _UNKNOWN_COLOR if name == "Unknown" else _KNOWN_COLOR
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    for kx, ky in keypoints.astype(int):
        cv2.circle(frame, (kx, ky), 2, _KEYPOINT_COLOR, -1)

    label = f"#{track_id} {name} ({score:.2f})"
    if show_emotion and emotion:
        label += f" - {emotion} ({emotion_score:.2f})"
    _draw_label_above(frame, label, x, y, color)

    if info is not None:
        details = _format_info(info)
        if details:
            _draw_label_below(frame, details, x, y + h, color)


def _format_info(info: PersonInfo) -> str:
    parts = []
    if info.nickname:
        parts.append(f'"{info.nickname}"')
    if info.department:
        parts.append(info.department)
    if info.person_id:
        parts.append(f"ID {info.person_id}")
    if info.hall:
        parts.append(f"{info.hall} Hall")
    if info.age:
        parts.append(f"Age {info.age}")
    return " | ".join(parts)


def _draw_label_above(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.55, 1)
    top = max(0, y - th - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + 8), color, -1)
    cv2.putText(frame, text, (x + 3, top + th + 2), _FONT, 0.55, _TEXT_COLOR, 1, cv2.LINE_AA)


def _draw_label_below(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    top = y + 4
    cv2.rectangle(frame, (x, top), (x + tw + 6, top + th + 8), color, -1)
    cv2.putText(frame, text, (x + 3, top + th + 2), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)


def draw_face_mesh(frame: np.ndarray, points: np.ndarray) -> None:
    for x, y in points.astype(int):
        cv2.circle(frame, (x, y), 1, _MESH_COLOR, -1)


def draw_hand(
    frame: np.ndarray,
    points: np.ndarray,
    connections: Iterable[tuple[int, int]],
    handedness: str,
    gesture: str,
    show_gesture: bool,
) -> None:
    for a, b in connections:
        pa, pb = points[a].astype(int), points[b].astype(int)
        cv2.line(frame, tuple(pa), tuple(pb), _HAND_COLOR, 2)
    for x, y in points.astype(int):
        cv2.circle(frame, (x, y), 3, _HAND_COLOR, -1)

    wrist = points[0].astype(int)
    label = f"{handedness}: {gesture}" if show_gesture else handedness
    cv2.putText(frame, label, (wrist[0] - 20, wrist[1] + 30), _FONT, 0.6, _TEXT_COLOR, 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), _FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def draw_status(frame: np.ndarray, lines: list[str]) -> None:
    h = frame.shape[0]
    for i, line in enumerate(reversed(lines)):
        y = h - 10 - i * 22
        cv2.putText(frame, line, (10, y), _FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
