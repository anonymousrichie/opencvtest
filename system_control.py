"""Thin wrappers around pyautogui/os.startfile for controlling the host
laptop -- media/volume keys, clipboard and window shortcuts, launching apps,
and moving the real mouse cursor -- from gesture-driven HUD buttons.

Kept free of any hand-tracking/UI logic so hud.py only has to decide *when*
to call these, never *how* the underlying OS call works.
"""
from __future__ import annotations

import os

import pyautogui
from pycaw.pycaw import AudioUtilities

# A held-open video loop calls move_cursor() every frame, so the default
# ~0.1s PAUSE after every pyautogui call would tank the frame rate. FAILSAFE
# (abort on cursor hitting a screen corner) stays on as a safety net.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


def play_pause() -> str:
    pyautogui.press("playpause")
    return "Play/Pause"


def next_track() -> str:
    pyautogui.press("nexttrack")
    return "Next track"


def prev_track() -> str:
    pyautogui.press("prevtrack")
    return "Previous track"


def volume_up() -> str:
    pyautogui.press("volumeup")
    return "Volume up"


def volume_down() -> str:
    pyautogui.press("volumedown")
    return "Volume down"


def mute() -> str:
    pyautogui.press("volumemute")
    return "Mute toggled"


def set_volume(percent: float) -> str:
    """Set system output volume to an exact level (0-100), via the Windows
    Core Audio API (pycaw) rather than the relative volume-up/-down media
    keys, which can only nudge the level one step at a time."""
    percent = max(0, min(100, round(percent)))
    volume = AudioUtilities.GetSpeakers().EndpointVolume
    volume.SetMasterVolumeLevelScalar(percent / 100.0, None)
    return f"Volume set to {percent}%"


def alt_tab() -> str:
    pyautogui.hotkey("alt", "tab")
    return "Alt+Tab"


def copy() -> str:
    pyautogui.hotkey("ctrl", "c")
    return "Copied"


def paste() -> str:
    pyautogui.hotkey("ctrl", "v")
    return "Pasted"


def screenshot() -> str:
    # Win+PrintScreen saves straight to Pictures/Screenshots with no dialog.
    pyautogui.hotkey("win", "printscreen")
    return "Screenshot saved"


def launch(command: str) -> str:
    """Open an app, path, or URL the way double-clicking it would."""
    os.startfile(command)
    return f"Launched {command}"


def move_cursor(x: int, y: int) -> None:
    pyautogui.moveTo(x, y)


def click() -> str:
    pyautogui.click()
    return "Clicked"
