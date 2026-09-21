"""On-screen HUD of virtual buttons for controlling the whole laptop.

Renders a fixed panel (category tabs + the active category's buttons) in
the top-right corner. A fingertip hovering a button highlights it; pinching
(thumb + index touching) while hovering fires it. Nothing outside the
panel's bounding rect is touched, so the drawing canvas's own pinch gesture
keeps working normally everywhere else on screen -- main.py is responsible
for excluding hands whose fingertip falls inside `claimed_handedness` from
DrawingCanvas.update() so the two features don't fight over the same pinch.

Mouse-cursor mode takes over the whole frame: while it's on, a hand's index
fingertip drives the real OS cursor and pinching performs a real click, so
the button panel stops dispatching pinches entirely. The toggle badge that
turns this mode on/off therefore can't rely on a pinch either (there would
be no way to pinch a *virtual* button once pinches are being sent to the OS)
-- it uses hover-and-hold instead, so it is always reachable in both modes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

import system_control as sc
from config import HudConfig
from hand_gestures import HandResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_MCP = 9

_PANEL_BG = (40, 40, 40)
_TAB_BG = (70, 70, 70)
_TAB_ACTIVE_BG = (0, 150, 200)
_BTN_BG = (90, 90, 90)
_BTN_HOVER_BG = (0, 180, 120)
_BADGE_OFF_BG = (70, 70, 70)
_BADGE_ON_BG = (0, 150, 200)
_TEXT_COLOR = (255, 255, 255)


@dataclass
class _Button:
    id: str
    label: str
    action: Callable[[], str]
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h in frame pixels


def _point_in_rect(point: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    x, y, w, h = rect
    return x <= point[0] <= x + w and y <= point[1] <= y + h


def _hand_scale(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[_MIDDLE_MCP] - pts[_WRIST])) or 1.0


class Hud:
    """Gesture-driven virtual button panel + hands-free mouse-cursor mode."""

    def __init__(self, cfg: HudConfig) -> None:
        self._cfg = cfg
        self.visible = cfg.visible_by_default
        self.mouse_mode = False
        self.active_category = "Media"
        self._categories: dict[str, list[_Button]] = self._build_categories()
        self._tab_bboxes: dict[str, tuple[int, int, int, int]] = {}
        self._panel_rect = (0, 0, 0, 0)
        self._toggle_rect = (0, 0, 0, 0)
        self._hovered_ids: set[str] = set()

        self._pinch_armed: dict[str, bool] = {}  # handedness -> was pinching last frame
        self._toggle_hover_start: float | None = None
        self._toggle_fired = False

    # ------------------------------------------------------------------ setup
    def _build_categories(self) -> dict[str, list[_Button]]:
        media = [
            _Button("play_pause", "Play/Pause", sc.play_pause),
            _Button("prev", "Prev Track", sc.prev_track),
            _Button("next", "Next Track", sc.next_track),
            _Button("vol_up", "Vol +", sc.volume_up),
            _Button("vol_down", "Vol -", sc.volume_down),
            _Button("mute", "Mute", sc.mute),
        ]
        keys = [
            _Button("alt_tab", "Alt+Tab", sc.alt_tab),
            _Button("copy", "Copy", sc.copy),
            _Button("paste", "Paste", sc.paste),
            _Button("screenshot", "Screenshot", sc.screenshot),
        ]
        apps = [
            _Button(f"app_{i}", label, lambda command=command: sc.launch(command))
            for i, (label, command) in enumerate(self._cfg.apps)
        ]
        return {"Media": media, "Apps": apps, "Keys": keys}

    # ----------------------------------------------------------------- layout
    def _layout(self, frame_shape: tuple[int, ...]) -> None:
        h, w = frame_shape[:2]
        cfg = self._cfg
        x0 = w - cfg.panel_width - cfg.margin
        y0 = cfg.margin

        names = list(self._categories)
        tab_w = cfg.panel_width // len(names)
        self._tab_bboxes = {
            name: (x0 + i * tab_w, y0, tab_w, cfg.tab_height) for i, name in enumerate(names)
        }

        y = y0 + cfg.tab_height + cfg.button_gap
        buttons = self._categories[self.active_category]
        for btn in buttons:
            btn.bbox = (x0, y, cfg.panel_width, cfg.button_height)
            y += cfg.button_height + cfg.button_gap

        self._panel_rect = (x0, y0, cfg.panel_width, y - y0)
        badge_w, badge_h = cfg.toggle_badge_size
        self._toggle_rect = (cfg.margin, cfg.margin, badge_w, badge_h)

    def _hit_test(self, point: np.ndarray) -> tuple[str, str] | None:
        for name, bbox in self._tab_bboxes.items():
            if _point_in_rect(point, bbox):
                return "tab", name
        for btn in self._categories[self.active_category]:
            if _point_in_rect(point, btn.bbox):
                return "button", btn.id
        return None

    # ----------------------------------------------------------------- update
    def update(self, hands: list[HandResult], frame_shape: tuple[int, ...]) -> tuple[str | None, set[str]]:
        """Advance HUD state for this frame.

        Returns (status_message_or_None, handedness_labels_claimed_by_the_hud)
        -- callers should keep claimed hands out of any other pinch-driven
        feature (e.g. the drawing canvas) for this frame.
        """
        if not self.visible:
            return None, set()

        self._layout(frame_shape)
        self._hovered_ids = set()
        status: str | None = None
        claimed: set[str] = set()

        toggle_hovered = any(_point_in_rect(h.landmarks[_INDEX_TIP], self._toggle_rect) for h in hands)
        toggle_status = self._update_toggle_dwell(toggle_hovered)
        if toggle_status:
            status = toggle_status
        if toggle_hovered:
            claimed.add("toggle")

        if self.mouse_mode:
            self._drive_cursor(hands, frame_shape)
            # The button panel is suspended while the OS cursor is live --
            # pinches go to whatever's under the real cursor, not our panel.
            return status, claimed

        for hand in hands:
            label = hand.handedness
            pts = hand.landmarks
            tip = pts[_INDEX_TIP]
            pinch_dist = float(np.linalg.norm(pts[_THUMB_TIP] - pts[_INDEX_TIP]))
            pinching = pinch_dist < self._cfg.pinch_threshold * _hand_scale(pts)

            hit = self._hit_test(tip)
            if hit is not None:
                claimed.add(label)
                self._hovered_ids.add(hit[1])

            was_pinching = self._pinch_armed.get(label, False)
            if hit is not None and pinching and not was_pinching:
                fired_status = self._fire(hit)
                if fired_status:
                    status = fired_status
            self._pinch_armed[label] = pinching

        return status, claimed

    def _fire(self, hit: tuple[str, str]) -> str:
        kind, value = hit
        if kind == "tab":
            self.active_category = value
            return f"{value} tab"
        btn = next(b for b in self._categories[self.active_category] if b.id == value)
        try:
            return btn.action()
        except Exception as exc:  # noqa: BLE001 -- surface any OS-control failure as a status line, never crash the loop
            return f"[ERROR] {btn.label}: {exc}"

    def _update_toggle_dwell(self, hovered: bool) -> str | None:
        if not hovered:
            self._toggle_hover_start = None
            self._toggle_fired = False
            return None
        now = time.monotonic()
        if self._toggle_hover_start is None:
            self._toggle_hover_start = now
        elif not self._toggle_fired and now - self._toggle_hover_start >= self._cfg.dwell_seconds:
            self._toggle_fired = True
            self.mouse_mode = not self.mouse_mode
            return f"Cursor mode {'ON' if self.mouse_mode else 'OFF'}"
        return None

    def _drive_cursor(self, hands: list[HandResult], frame_shape: tuple[int, ...]) -> None:
        if not hands:
            return
        hand = hands[0]
        pts = hand.landmarks
        h, w = frame_shape[:2]
        tip = pts[_INDEX_TIP]

        margin = 4
        nx = np.clip(tip[0] / w, 0.0, 1.0)
        ny = np.clip(tip[1] / h, 0.0, 1.0)
        sx = int(np.clip(nx * sc.SCREEN_WIDTH, margin, sc.SCREEN_WIDTH - margin))
        sy = int(np.clip(ny * sc.SCREEN_HEIGHT, margin, sc.SCREEN_HEIGHT - margin))

        pinch_dist = float(np.linalg.norm(pts[_THUMB_TIP] - pts[_INDEX_TIP]))
        pinching = pinch_dist < self._cfg.pinch_threshold * _hand_scale(pts)
        was_pinching = self._pinch_armed.get(hand.handedness, False)

        try:
            sc.move_cursor(sx, sy)
            if pinching and not was_pinching:
                sc.click()
        except Exception:  # noqa: BLE001 -- e.g. pyautogui's fail-safe corner abort; never crash the video loop
            pass
        self._pinch_armed[hand.handedness] = pinching

    # ----------------------------------------------------------------- render
    def render(self, frame: np.ndarray) -> None:
        if not self.visible:
            return
        self._render_toggle_badge(frame)
        if self.mouse_mode:
            return
        self._render_panel(frame)

    def _render_toggle_badge(self, frame: np.ndarray) -> None:
        x, y, w, h = self._toggle_rect
        bg = _BADGE_ON_BG if self.mouse_mode else _BADGE_OFF_BG
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), _TEXT_COLOR, 1)
        label = f"Cursor: {'ON' if self.mouse_mode else 'OFF'}"
        cv2.putText(frame, label, (x + 10, y + h // 2 + 6), _FONT, 0.55, _TEXT_COLOR, 1, cv2.LINE_AA)

        if self._toggle_hover_start is not None and not self._toggle_fired:
            frac = min(1.0, (time.monotonic() - self._toggle_hover_start) / self._cfg.dwell_seconds)
            cv2.rectangle(frame, (x, y + h - 4), (x + int(w * frac), y + h), (0, 255, 0), -1)

    def _render_panel(self, frame: np.ndarray) -> None:
        px, py, pw, ph = self._panel_rect
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), _PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        for name, (x, y, w, h) in self._tab_bboxes.items():
            active = name == self.active_category
            bg = _TAB_ACTIVE_BG if active else (_BTN_HOVER_BG if name in self._hovered_ids else _TAB_BG)
            cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), _TEXT_COLOR, 1)
            (tw, th), _ = cv2.getTextSize(name, _FONT, 0.5, 1)
            cv2.putText(
                frame, name, (x + (w - tw) // 2, y + (h + th) // 2), _FONT, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA
            )

        for btn in self._categories[self.active_category]:
            x, y, w, h = btn.bbox
            bg = _BTN_HOVER_BG if btn.id in self._hovered_ids else _BTN_BG
            cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), _TEXT_COLOR, 1)
            (tw, th), _ = cv2.getTextSize(btn.label, _FONT, 0.55, 1)
            cv2.putText(
                frame,
                btn.label,
                (x + (w - tw) // 2, y + (h + th) // 2),
                _FONT,
                0.55,
                _TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
