"""MediaPipe Hands wrapper + geometric gesture classification.

Gestures are classified with simple distance/position rules over the 21 hand
landmarks rather than a trained classifier. That keeps the base system
dependency-free and fast, at the cost of being less robust to unusual hand
poses or camera angles than a learned model would be. `classify_gesture` is
the intended extension point -- swap it for a trained classifier later
without touching capture, detection, or display code.
"""
from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np

from config import HandConfig

_mp_hands = mp.solutions.hands

# MediaPipe Hands landmark indices (see MediaPipe Hands documentation).
_WRIST = 0
_THUMB_MCP, _THUMB_TIP = 2, 4
_FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
_FINGER_MCPS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class HandResult:
    """One detected hand: pixel-space landmarks, handedness, and its gesture."""

    landmarks: np.ndarray  # (21, 2) pixel coordinates
    handedness: str  # "Left" or "Right"
    gesture: str


class HandGestureRecognizer:
    """Wraps mediapipe.solutions.hands and classifies a gesture per hand."""

    def __init__(self, cfg: HandConfig) -> None:
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def process(self, frame_rgb: np.ndarray) -> list[HandResult]:
        h, w = frame_rgb.shape[:2]
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return []

        outputs = []
        for hand_landmarks, handedness_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            pts = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark], dtype=np.float32)
            # main.py mirrors the frame before it reaches here (so the feed
            # behaves like a mirror on screen), which flips MediaPipe's
            # handedness classification relative to the physical hand -- swap
            # it back so "Left"/"Right" still names the hand the user is
            # actually holding up.
            raw_label = handedness_info.classification[0].label
            label = "Left" if raw_label == "Right" else "Right"
            gesture = classify_gesture(pts)
            outputs.append(HandResult(landmarks=pts, handedness=label, gesture=gesture))
        return outputs

    def close(self) -> None:
        self._hands.close()


def _hand_scale(pts: np.ndarray) -> float:
    """A rough hand-size unit (wrist-to-middle-knuckle distance) used to
    normalize all other distances, so thresholds hold regardless of how close
    the hand is to the camera."""
    return float(np.linalg.norm(pts[_FINGER_MCPS["middle"]] - pts[_WRIST])) or 1.0


def _finger_extended(pts: np.ndarray, finger: str) -> bool:
    """A non-thumb finger counts as extended if its tip is meaningfully farther
    from the wrist than its PIP joint is. Comparing distances-from-wrist
    (rather than raw y-coordinates) keeps this correct regardless of hand
    rotation or tilt."""
    wrist = pts[_WRIST]
    tip = pts[_FINGER_TIPS[finger]]
    pip = pts[_FINGER_PIPS[finger]]
    return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist) * 1.05


def _thumb_extended(pts: np.ndarray) -> bool:
    """The thumb doesn't fold the same way as other fingers, so it is judged
    by how far its tip splays away from the index finger's base compared to
    the thumb's own knuckle."""
    tip = pts[_THUMB_TIP]
    mcp = pts[_THUMB_MCP]
    index_mcp = pts[_FINGER_MCPS["index"]]
    return np.linalg.norm(tip - index_mcp) > np.linalg.norm(mcp - index_mcp) * 1.1


def classify_gesture(pts: np.ndarray) -> str:
    """Classify one of a fixed gesture vocabulary from 21 hand landmarks.

    Order matters: the more specific pinch/orientation/thumb checks run
    before the general finger-counting fallbacks (open palm, fist, pointing,
    peace), since those would otherwise also satisfy a finger-count rule.
    """
    fingers_up = {f: _finger_extended(pts, f) for f in _FINGER_TIPS}
    thumb_up = _thumb_extended(pts)
    num_up = sum(fingers_up.values())
    scale = _hand_scale(pts)

    # OK sign: thumb and index tips pinched together, other three fingers extended.
    pinch_dist = float(np.linalg.norm(pts[_THUMB_TIP] - pts[_FINGER_TIPS["index"]]))
    if pinch_dist < 0.35 * scale and fingers_up["middle"] and fingers_up["ring"] and fingers_up["pinky"]:
        return "OK Sign"

    # Thumbs up/down: only the thumb extended; direction read from the thumb
    # tip's vertical position relative to the wrist (image y grows downward).
    if thumb_up and num_up == 0:
        wrist_y = pts[_WRIST][1]
        thumb_tip_y = pts[_THUMB_TIP][1]
        if thumb_tip_y < wrist_y - 0.4 * scale:
            return "Thumbs Up"
        if thumb_tip_y > wrist_y + 0.4 * scale:
            return "Thumbs Down"

    if num_up == 0 and not thumb_up:
        return "Fist"

    only = lambda *names: num_up == len(names) and all(fingers_up[n] for n in names)

    # ASL "I Love You": thumb, index, and pinky extended; middle/ring folded.
    if thumb_up and only("index", "pinky"):
        return "I Love You"

    # Rock on / devil horns: index and pinky extended, thumb folded over the
    # middle/ring fingers rather than out to the side.
    if not thumb_up and only("index", "pinky"):
        return "Rock On"

    # Shaka / "call me": only the pinky extended, plus the thumb out to the side.
    if thumb_up and only("pinky"):
        return "Call Me"

    # Gun / "L" shape: thumb and index extended out to the side, rest folded.
    if thumb_up and only("index"):
        return "L Shape"

    if num_up == 1 and fingers_up["index"]:
        return "Pointing"

    if only("index", "middle"):
        return "Peace / Victory"

    if only("index", "middle", "ring"):
        return "Three"

    # Four fingers without the thumb splayed out reads as counting "four"
    # rather than a fully open palm.
    if num_up == 4 and not thumb_up:
        return "Four"

    if num_up == 4:
        return "Open Palm"

    return "Unknown"
