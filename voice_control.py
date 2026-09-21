"""Voice control: background speech recognition mapped to app/laptop commands.

Microphone capture goes through `sounddevice` rather than the more common
`PyAudio` -- `sounddevice` wraps the same PortAudio library but installs
cleanly from a prebuilt wheel on Windows, whereas PyAudio often needs a C
build toolchain. A simple energy-based endpointer (record while the signal
is louder than a calibrated ambient-noise threshold, stop once it's been
quiet for `pause_threshold` seconds) segments the stream into phrases, each
handed to Google's free Web Speech API via the `speech_recognition` package
for transcription.

All of that -- capture, endpointing, network transcription -- runs on a
background thread, since it takes real time the video loop can't block on.
The thread only resolves recognized text to a zero-arg callable (a fixed
phrase's handler, or a regex command's handler pre-bound with its captured
group) and queues it; the callable itself only ever runs on the main thread,
via `process_pending()`, so command handlers (which touch pyautogui, the
drawing canvas, the HUD, etc.) never run concurrently with the rest of the
app or each other.

Three tiers of matching, in priority order, let voice cover a much wider
surface than a fixed phrase list ever could:

1. `priority_commands` -- regex patterns anchored to the start of the
   utterance (e.g. "type ...", "search for ..."), checked first and
   exclusively of everything else. Anchoring means a command like "copy"
   can't accidentally fire just because the word appears inside whatever
   the user is dictating.
2. `commands` -- a fixed phrase -> zero-arg handler dict, matched by
   substring containment (longest phrase wins ties). This is the simple
   case for one-shot actions ("volume up", "copy", "lock screen", ...).
3. `fallback_commands` -- regex patterns checked last, for
   parametrized-but-generic actions ("open <anything>", "set the volume to
   N percent") that would otherwise be too broad to risk matching first.
"""
from __future__ import annotations

import queue
import re
import threading
from typing import Callable

import numpy as np
import sounddevice as sd
import speech_recognition as sr

from config import VoiceConfig

CommandHandlers = dict[str, Callable[[], str]]
ParamCommand = tuple[re.Pattern, Callable[[re.Match], str]]

_SAMPLE_RATE = 16000
_BLOCK_DURATION = 0.1  # seconds per audio block read from the mic


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


class VoiceController:
    """Listens for voice commands and queues them for the main thread."""

    def __init__(
        self,
        cfg: VoiceConfig,
        commands: CommandHandlers,
        priority_commands: list[ParamCommand] | None = None,
        fallback_commands: list[ParamCommand] | None = None,
    ) -> None:
        self._cfg = cfg
        self._commands = commands
        self._priority_commands = priority_commands or []
        self._fallback_commands = fallback_commands or []
        self._recognizer = sr.Recognizer()
        self._pending: queue.Queue[tuple[str, Callable[[], str]]] = queue.Queue()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.listening = False
        self.last_heard = ""

    def start(self) -> str:
        if self.listening:
            return "Voice control already on."
        try:
            threshold = self._calibrate()
            self._stream = sd.InputStream(
                samplerate=_SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=int(_SAMPLE_RATE * _BLOCK_DURATION), callback=self._on_block,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            return f"No microphone available: {exc}"

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._listen_loop, args=(threshold,), daemon=True)
        self._worker.start()
        self.listening = True
        return "Voice control ON -- try 'volume up', 'open spotify', 'type hello', 'search for cats', ..."

    def stop(self) -> str:
        if not self.listening:
            return "Voice control already off."
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.listening = False
        return "Voice control OFF."

    def _calibrate(self) -> float:
        """Record ~0.5s of ambient noise and return an energy threshold above it."""
        sample = sd.rec(int(_SAMPLE_RATE * 0.5), samplerate=_SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        return max(float(self._cfg.energy_threshold), _rms(sample) * 2.5)

    def _on_block(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        self._audio_queue.put(indata.copy())

    def _listen_loop(self, threshold: float) -> None:
        buffer: list[np.ndarray] = []
        recording = False
        silence_run = 0.0
        recorded_duration = 0.0

        while not self._stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            speaking = _rms(block) > threshold

            if not recording:
                if speaking:
                    recording, buffer, silence_run, recorded_duration = True, [block], 0.0, _BLOCK_DURATION
                continue

            buffer.append(block)
            recorded_duration += _BLOCK_DURATION
            silence_run = 0.0 if speaking else silence_run + _BLOCK_DURATION

            if silence_run >= self._cfg.pause_threshold or recorded_duration >= self._cfg.phrase_time_limit:
                self._process_phrase(buffer)
                recording, buffer = False, []

    def _process_phrase(self, buffer: list[np.ndarray]) -> None:
        audio_np = np.concatenate(buffer, axis=0)
        audio_data = sr.AudioData(audio_np.tobytes(), _SAMPLE_RATE, 2)
        try:
            raw_text = self._recognizer.recognize_google(audio_data, language=self._cfg.language)
        except sr.UnknownValueError:
            return
        except sr.RequestError as exc:
            self._pending.put(("(voice)", lambda exc=exc: f"Voice recognition service unreachable: {exc}"))
            return

        self.last_heard = raw_text
        run = self._resolve(raw_text)
        if run is not None:
            self._pending.put((raw_text, run))

    def _resolve(self, raw_text: str) -> Callable[[], str] | None:
        for pattern, handler in self._priority_commands:
            m = pattern.search(raw_text)
            if m:
                return lambda m=m, handler=handler: handler(m)

        text = raw_text.lower().strip()
        phrase = self._match_fixed(text)
        if phrase is not None:
            return self._commands[phrase]

        for pattern, handler in self._fallback_commands:
            m = pattern.search(raw_text)
            if m:
                return lambda m=m, handler=handler: handler(m)
        return None

    def _match_fixed(self, text: str) -> str | None:
        # Longest phrase first, so a more specific command wins over a
        # shorter one that happens to be a substring of what was said.
        for phrase in sorted(self._commands, key=len, reverse=True):
            if phrase in text:
                return phrase
        return None

    def process_pending(self) -> str | None:
        """Run any queued commands on the calling (main) thread. Returns the
        last status message produced this call, if any, for display."""
        status = None
        while True:
            try:
                heard, run = self._pending.get_nowait()
            except queue.Empty:
                break
            try:
                result = run()
            except Exception as exc:  # noqa: BLE001 -- never let a bad voice command crash the video loop
                result = f"[ERROR] {exc}"
            status = f"Heard '{heard}' -> {result}"
        return status
