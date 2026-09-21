"""Thin wrappers around pyautogui/os.startfile for controlling the host
laptop -- media/volume keys, clipboard and window shortcuts, launching apps,
and moving the real mouse cursor -- from gesture-driven HUD buttons.

Kept free of any hand-tracking/UI logic so hud.py only has to decide *when*
to call these, never *how* the underlying OS call works.
"""
from __future__ import annotations

import difflib
import os
import urllib.parse

import psutil
import pyautogui
from pycaw.pycaw import AudioUtilities

# Never terminate these via close_app(), regardless of how the spoken name
# matches -- these are core OS processes where killing the wrong one can
# destabilize the whole session, not just close an app.
_PROTECTED_PROCESSES = {
    "system", "system idle process", "registry", "smss", "csrss", "wininit",
    "winlogon", "services", "lsass", "svchost", "dwm", "explorer", "fontdrvhost",
    "memory compression", "runtimebroker", "sihost", "taskhostw", "ctfmon",
}

# A held-open video loop calls move_cursor() every frame, so the default
# ~0.1s PAUSE after every pyautogui call would tank the frame rate. FAILSAFE
# (abort on cursor hitting a screen corner) stays on as a safety net.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

_START_MENU_DIRS = [
    os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
    os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"),
]


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


def _iter_start_menu_shortcuts() -> dict[str, str]:
    shortcuts: dict[str, str] = {}
    for base in _START_MENU_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fname in files:
                if fname.lower().endswith((".lnk", ".url")):
                    name = os.path.splitext(fname)[0]
                    shortcuts.setdefault(name, os.path.join(root, fname))
    return shortcuts


def open_app(name: str) -> str:
    """Open any installed app by (spoken) name, by fuzzy-matching it against
    the Start Menu's shortcuts -- unlike `launch`, this needs no hardcoded
    path for each app."""
    name = name.strip()
    if not name:
        return "No app name heard."

    shortcuts = _iter_start_menu_shortcuts()
    if not shortcuts:
        # No Start Menu shortcuts found (unusual) -- fall back to letting
        # Windows resolve it directly, e.g. if it's on PATH.
        try:
            os.startfile(name)
            return f"Launched {name}"
        except OSError as exc:
            return f"Couldn't find or launch '{name}': {exc}"

    lname = name.lower()
    exact = [n for n in shortcuts if n.lower() == lname]
    substring = [n for n in shortcuts if lname in n.lower() or n.lower() in lname]
    match = exact[0] if exact else (substring[0] if substring else None)
    if match is None:
        # A tight cutoff matters here more than for a typical fuzzy search --
        # this is a live launch action, not a suggestion, so a wrong match
        # opens the wrong app rather than just annoying with a bad guess.
        close = difflib.get_close_matches(name, list(shortcuts), n=1, cutoff=0.65)
        match = close[0] if close else None

    if match is None:
        return f"Couldn't find an app named '{name}'."
    os.startfile(shortcuts[match])
    return f"Opening {match}"


def close_app(name: str) -> str:
    """Terminate any running process matching (spoken) `name` -- the
    counterpart to open_app(). Deliberately stricter than open_app's
    matching (exact or prefix only, no loose "contains" or fuzzy match):
    a wrong match here kills a process instead of just opening the wrong
    window, and protected core OS processes are refused outright."""
    name = name.strip()
    if not name:
        return "No app name heard."
    lname = name.lower()
    if lname in _PROTECTED_PROCESSES:
        return f"Won't close '{name}' -- that's a core system process."

    matches = []
    for proc in psutil.process_iter(["name"]):
        pname = proc.info.get("name") or ""
        base = pname.rsplit(".", 1)[0].lower()
        if not base or base in _PROTECTED_PROCESSES:
            continue  # skip unnamed processes -- an empty base is a prefix of everything
        if base == lname or base.startswith(lname) or lname.startswith(base):
            matches.append(proc)

    if not matches:
        return f"No running app found matching '{name}'."

    closed = set()
    for proc in matches:
        try:
            proc.terminate()
            closed.add(proc.info.get("name"))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not closed:
        return f"Couldn't close '{name}' (access denied?)."
    return f"Closed {', '.join(sorted(closed))}"


def type_text(text: str) -> str:
    """Type literal text into whatever has focus -- dictation."""
    text = text.strip()
    if not text:
        return "Nothing to type."
    pyautogui.write(text, interval=0.02)
    return f"Typed: {text}"


def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        return "No search query heard."
    url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
    os.startfile(url)
    return f"Searching: {query}"


def select_all() -> str:
    pyautogui.hotkey("ctrl", "a")
    return "Select all"


def cut() -> str:
    pyautogui.hotkey("ctrl", "x")
    return "Cut"


def undo() -> str:
    pyautogui.hotkey("ctrl", "z")
    return "Undo"


def redo() -> str:
    pyautogui.hotkey("ctrl", "y")
    return "Redo"


def save() -> str:
    pyautogui.hotkey("ctrl", "s")
    return "Save"


def find() -> str:
    pyautogui.hotkey("ctrl", "f")
    return "Find"


def new_tab() -> str:
    pyautogui.hotkey("ctrl", "t")
    return "New tab"


def close_tab() -> str:
    pyautogui.hotkey("ctrl", "w")
    return "Close tab"


def refresh() -> str:
    pyautogui.press("f5")
    return "Refresh"


def minimize_window() -> str:
    pyautogui.hotkey("win", "down")
    return "Minimize window"


def maximize_window() -> str:
    pyautogui.hotkey("win", "up")
    return "Maximize window"


def show_desktop() -> str:
    pyautogui.hotkey("win", "d")
    return "Show desktop"


def close_window() -> str:
    pyautogui.hotkey("alt", "f4")
    return "Close window"


def lock_screen() -> str:
    pyautogui.hotkey("win", "l")
    return "Locked"


def snap_left() -> str:
    pyautogui.hotkey("win", "left")
    return "Snapped left"


def snap_right() -> str:
    pyautogui.hotkey("win", "right")
    return "Snapped right"


def task_view() -> str:
    pyautogui.hotkey("win", "tab")
    return "Task view"


def new_desktop() -> str:
    pyautogui.hotkey("ctrl", "win", "d")
    return "New virtual desktop"


def close_desktop() -> str:
    pyautogui.hotkey("ctrl", "win", "f4")
    return "Closed virtual desktop"


def next_desktop() -> str:
    pyautogui.hotkey("ctrl", "win", "right")
    return "Next virtual desktop"


def prev_desktop() -> str:
    pyautogui.hotkey("ctrl", "win", "left")
    return "Previous virtual desktop"


def open_task_manager() -> str:
    pyautogui.hotkey("ctrl", "shift", "esc")
    return "Task Manager"


def open_settings() -> str:
    pyautogui.hotkey("win", "i")
    return "Settings"


def zoom_in() -> str:
    pyautogui.hotkey("ctrl", "+")
    return "Zoomed in"


def zoom_out() -> str:
    pyautogui.hotkey("ctrl", "-")
    return "Zoomed out"


def zoom_reset() -> str:
    pyautogui.hotkey("ctrl", "0")
    return "Zoom reset"


def scroll_up() -> str:
    pyautogui.scroll(400)
    return "Scrolled up"


def scroll_down() -> str:
    pyautogui.scroll(-400)
    return "Scrolled down"


def right_click() -> str:
    pyautogui.rightClick()
    return "Right click"


def double_click() -> str:
    pyautogui.doubleClick()
    return "Double click"


def set_brightness(percent: float) -> str:
    """Set screen brightness to an exact level (0-100) via WMI. Only works
    on displays that expose DDC/CI brightness control this way -- typically
    laptop panels, not most external monitors -- so failures are reported
    rather than raised."""
    percent = max(0, min(100, round(percent)))
    try:
        import wmi

        c = wmi.WMI(namespace="wmi")
        methods = c.WmiMonitorBrightnessMethods()
        if not methods:
            return "No brightness-controllable display found."
        for method in methods:
            method.WmiSetBrightness(percent, 0)
        return f"Brightness set to {percent}%"
    except Exception as exc:  # noqa: BLE001 -- WMI errors vary by hardware; report, don't crash
        return f"Couldn't set brightness: {exc}"


def move_cursor(x: int, y: int) -> None:
    pyautogui.moveTo(x, y)


def click() -> str:
    pyautogui.click()
    return "Clicked"
