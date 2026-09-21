"""Spoken feedback for voice commands, via pyttsx3 (SAPI5 on Windows).

Runs on its own dedicated thread with a queue: `engine.runAndWait()` blocks
for as long as the utterance takes to speak, which the video loop can't
afford to stall on, and pyttsx3's SAPI5 engine isn't meant to be shared
across threads, so the engine is created and used only within this one
worker thread rather than from the main thread that calls `say()`.
"""
from __future__ import annotations

import queue
import threading


class VoiceFeedback:
    """Queues short strings to be spoken aloud, one at a time, in the background."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if self.enabled and text:
            self._queue.put(text)

    def _run(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        while True:
            text = self._queue.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:  # noqa: BLE001 -- a TTS glitch should never take down the app
                pass
