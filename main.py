"""Real-time face recognition + hand gesture pipeline -- entry point.

Run with `python main.py`. See README.md for setup, controls, and the
enrollment workflow.
"""
from __future__ import annotations

import re
import sys

import cv2
import mediapipe as mp
import numpy as np

import overlay
from camera import Camera, CameraError
from config import CONFIG
from database import EnrollmentDatabase, PersonInfo
from drawing import DrawingCanvas
from emotion import EmotionRecognizer
from emotion_fx import EmotionEffects
from face_detection import DetectedFace, FaceDetector, ModelLoadError
from face_filters import FaceFilters
from face_mesh import FaceMeshDetector
from face_recognition import FaceRecognizer
from fps import FPSCounter
from hand_gestures import HandGestureRecognizer
from hud import Hud
from rasengan import RasenganEffect
from repulsor import RepulsorEffect
from shield import ShieldEffect
import system_control as sc
from tracking import IoUTracker
from voice_control import VoiceController

_HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def _enroll_face(frame: np.ndarray, faces: list[DetectedFace], recognizer: FaceRecognizer) -> str:
    """Enroll the largest currently-detected face under a name typed in the terminal.

    The video loop is (briefly) blocked on `input()` while the operator types
    the name -- acceptable for a base system per the spec, and simpler than
    building an on-screen text-entry widget.
    """
    if not faces:
        return "No face detected -- cannot enroll."
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    if len(faces) > 1:
        print(
            f"[NOTE] {len(faces)} faces detected -- enrolling the largest one "
            f"(bbox={face.bbox}, detector score={face.score:.2f}). If that isn't "
            "the intended face, cancel (empty name) and reframe so only one face is in view."
        )
    name = input("Enter name for enrollment: ").strip()
    if not name:
        return "Enrollment cancelled (empty name)."

    print("Optional profile details -- press Enter to skip (or keep the existing value).")
    info = PersonInfo(
        age=input("Age: ").strip(),
        department=input("Department: ").strip(),
        person_id=input("ID: ").strip(),
        nickname=input("Nickname: ").strip(),
        hall=input("Hall: ").strip(),
    )

    embedding = recognizer.embed(frame, face)
    recognizer.enroll(name, embedding, info)
    print(f"[NOTE] Enrolled from bbox={face.bbox}, detector score={face.score:.2f}.")
    return f"Enrolled '{name}'."


def main() -> int:
    try:
        detector = FaceDetector(CONFIG.face_detection)
        database = EnrollmentDatabase(CONFIG.face_recognition.db_path)
        recognizer = FaceRecognizer(CONFIG.face_recognition, database)
    except ModelLoadError as exc:
        print(f"[FATAL] {exc}")
        return 1

    face_mesh = FaceMeshDetector(CONFIG.face_mesh)
    hands = HandGestureRecognizer(CONFIG.hands)
    canvas = DrawingCanvas(CONFIG.drawing)
    # Open-palm "powers" -- press 's' to cycle which one (if any) is active.
    palm_powers: list[tuple[str, object | None]] = [
        ("Off", None),
        ("Fire Shield", ShieldEffect(CONFIG.shield)),
        ("Rasengan", RasenganEffect(CONFIG.rasengan)),
        ("Repulsor Blast", RepulsorEffect(CONFIG.repulsor)),
    ]
    palm_power_index = 1
    emotion_fx = EmotionEffects(CONFIG.emotion_fx)
    face_filters = FaceFilters(CONFIG.face_filter)
    hud = Hud(CONFIG.hud)
    tracker = IoUTracker(CONFIG.tracking)
    fps_counter = FPSCounter()

    def _select_power(name: str) -> str:
        nonlocal palm_power_index
        for i, (label, _effect) in enumerate(palm_powers):
            if label == name:
                palm_power_index = i
                return f"Open-palm power: {label}"
        return "Unknown power."

    def _clear_canvas() -> str:
        canvas.clear()
        return "Canvas cleared."

    def _toggle_drawing() -> str:
        canvas.enabled = not canvas.enabled
        return f"Drawing {'ON' if canvas.enabled else 'OFF'}"

    def _set_sunglasses(on: bool) -> str:
        face_filters.enabled = on
        return f"Sunglasses {'ON' if on else 'OFF'}"

    def _set_hud(visible: bool) -> str:
        hud.visible = visible
        return f"Laptop-control HUD {'ON' if visible else 'OFF'}"

    # Spoken phrases -> handler. Matched by substring containment against
    # what Google's Web Speech API transcribed (see VoiceController._match_fixed),
    # so phrases are picked to be distinctive and unlikely to appear inside
    # each other or ordinary speech.
    voice_commands = {
        "volume up": sc.volume_up,
        "volume down": sc.volume_down,
        "mute": sc.mute,
        "play music": sc.play_pause,
        "pause music": sc.play_pause,
        "next track": sc.next_track,
        "skip song": sc.next_track,
        "previous track": sc.prev_track,
        "copy": sc.copy,
        "paste": sc.paste,
        "cut": sc.cut,
        "undo": sc.undo,
        "redo": sc.redo,
        "select all": sc.select_all,
        "save": sc.save,
        "find": sc.find,
        "new tab": sc.new_tab,
        "close tab": sc.close_tab,
        "refresh": sc.refresh,
        "take a screenshot": sc.screenshot,
        "switch window": sc.alt_tab,
        "minimize window": sc.minimize_window,
        "maximize window": sc.maximize_window,
        "show desktop": sc.show_desktop,
        "close window": sc.close_window,
        "lock screen": sc.lock_screen,
        "lock computer": sc.lock_screen,
        "open browser": lambda: sc.launch("https://www.google.com"),
        "open notepad": lambda: sc.launch("notepad.exe"),
        "open calculator": lambda: sc.launch("calc.exe"),
        "open explorer": lambda: sc.launch("explorer.exe"),
        "clear canvas": _clear_canvas,
        "toggle drawing": _toggle_drawing,
        "fire shield": lambda: _select_power("Fire Shield"),
        "rasengan": lambda: _select_power("Rasengan"),
        "repulsor blast": lambda: _select_power("Repulsor Blast"),
        "powers off": lambda: _select_power("Off"),
        "show hud": lambda: _set_hud(True),
        "hide hud": lambda: _set_hud(False),
        "sunglasses on": lambda: _set_sunglasses(True),
        "sunglasses off": lambda: _set_sunglasses(False),
    }

    # Free-text commands -- anchored to the *start* of the utterance and
    # checked before anything else, so a word like "copy" inside whatever
    # you're dictating can't accidentally trigger the "copy" command above.
    voice_priority_commands = [
        (re.compile(r"^type\s+(.+)$", re.IGNORECASE), lambda m: sc.type_text(m.group(1))),
        (re.compile(r"^(?:search for|google)\s+(.+)$", re.IGNORECASE), lambda m: sc.web_search(m.group(1))),
    ]

    # Generic parametrized commands -- checked only after the fixed phrases
    # above, so e.g. "open notepad" still gets the reliable hardcoded path
    # rather than a fuzzy Start Menu search, but "open spotify" (or anything
    # else not special-cased) still works.
    _PERCENT_RE = r"{word}.{{0,20}}?(\d{{1,3}})\s*(?:percent|%)|(\d{{1,3}})\s*(?:percent|%).{{0,20}}?{word}"
    voice_fallback_commands = [
        (
            re.compile(_PERCENT_RE.format(word="volume"), re.IGNORECASE),
            lambda m: sc.set_volume(int(m.group(1) or m.group(2))),
        ),
        (
            re.compile(_PERCENT_RE.format(word="brightness"), re.IGNORECASE),
            lambda m: sc.set_brightness(int(m.group(1) or m.group(2))),
        ),
        (re.compile(r"^open\s+(.+)$", re.IGNORECASE), lambda m: sc.open_app(m.group(1))),
    ]

    voice = VoiceController(CONFIG.voice, voice_commands, voice_priority_commands, voice_fallback_commands)

    try:
        emotion_recognizer: EmotionRecognizer | None = EmotionRecognizer(CONFIG.emotion)
    except ModelLoadError as exc:
        print(f"[WARN] Emotion recognition disabled: {exc}")
        emotion_recognizer = None

    show_landmarks = CONFIG.display.show_landmarks
    show_gestures = CONFIG.display.show_gestures
    show_emotion = CONFIG.display.show_emotion and emotion_recognizer is not None
    status_message = "Ready. Press 'e' to enroll, 'q' to quit."

    camera = Camera(CONFIG.camera)
    try:
        camera.open()
    except CameraError as exc:
        print(f"[FATAL] {exc}")
        face_mesh.close()
        hands.close()
        return 1

    cv2.namedWindow(CONFIG.display.window_name, cv2.WINDOW_NORMAL)

    print(
        "Cam IntelliSense running. Controls: q=quit  e=enroll  l=landmarks  g=gestures  m=emotion  "
        "d=drawing  c=clear canvas  z=undo  v=laptop-control HUD  f=sunglasses  r=voice control  "
        "s=cycle open-palm power (off/fire shield/Rasengan/repulsor)"
    )
    print(
        "Drawing: 2 fingers=draw  pinch-fist=stretch last stroke  closed fist=erase  "
        "1 finger/3 fingers/index+pinky=pick color"
    )
    print(
        "Voice (after 'r'): 'volume/brightness up/down/to N percent', 'open <any app>', "
        "'type <text>', 'search for <query>', 'lock screen', 'save', 'undo', ... -- see README.md"
    )

    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as exc:
                print(f"[ERROR] {exc}")
                break

            # Mirror the feed so it behaves like a mirror (move your hand
            # right, it goes right on screen) instead of the raw camera view,
            # which feels backwards. Everything downstream -- detection,
            # overlays, the drawing canvas, the HUD's cursor mapping --
            # operates on this flipped frame, so it all stays consistent.
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            voice_status = voice.process_pending()
            if voice_status:
                status_message = voice_status

            faces = detector.detect(frame)
            tracks = tracker.update([f.bbox for f in faces])

            for face, track in zip(faces, tracks):
                track.frames_since_recognition += 1
                # Only re-run SFace embedding/matching periodically per track;
                # see TrackingConfig.recognition_interval for why.
                if track.frames_since_recognition >= CONFIG.tracking.recognition_interval:
                    embedding = recognizer.embed(frame, face)
                    track.identity, track.confidence = recognizer.match(embedding)
                    track.frames_since_recognition = 0

                if emotion_recognizer is not None:
                    track.frames_since_emotion += 1
                    if track.frames_since_emotion >= CONFIG.emotion.emotion_interval:
                        track.emotion, track.emotion_score = emotion_recognizer.classify(frame, face)
                        track.frames_since_emotion = 0

                info = database.get_info(track.identity) if track.identity != "Unknown" else None
                overlay.draw_face(
                    frame,
                    track.bbox,
                    face.landmarks,
                    track.track_id,
                    track.identity,
                    track.confidence,
                    track.emotion,
                    track.emotion_score,
                    show_emotion,
                    info,
                )

            if show_emotion:
                emotion_fx.update(tracks)

            # Face Mesh is only worth the CPU cost when something actually
            # reads its landmarks -- the raw dot overlay ('l') or the AR
            # sunglasses, either of which may be on independently of the other.
            mesh_points_list = []
            if show_landmarks or face_filters.enabled:
                mesh_points_list = face_mesh.process(frame_rgb)
                if show_landmarks:
                    for mesh_points in mesh_points_list:
                        overlay.draw_face_mesh(frame, mesh_points)
            face_filters.render(frame, mesh_points_list)

            hand_results = hands.process(frame_rgb)

            hud_status, hud_claimed = hud.update(hand_results, frame.shape)
            if hud_status:
                status_message = hud_status

            # Mouse-cursor mode commandeers the fingertip for the real OS
            # cursor; otherwise, hands the HUD is using (hovering/pinching a
            # button) are kept out of the drawing canvas so one pinch can't
            # both click a button and stretch a stroke.
            drawing_hands = [] if hud.mouse_mode else [h for h in hand_results if h.handedness not in hud_claimed]
            canvas.update(drawing_hands)

            for hand in hand_results:
                overlay.draw_hand(frame, hand.landmarks, _HAND_CONNECTIONS, hand.handedness, hand.gesture, show_gestures)
            canvas.render(frame)
            active_power = palm_powers[palm_power_index][1]
            if active_power is not None:
                active_power.render(frame, hand_results)
            hud.render(frame)
            if show_emotion:
                emotion_fx.render(frame)

            fps = fps_counter.tick()
            if CONFIG.display.show_fps:
                overlay.draw_fps(frame, fps)
            overlay.draw_status(
                frame,
                [
                    status_message,
                    f"Enrolled identities: {', '.join(database.names()) or 'none'}",
                    "q=quit  e=enroll  l=landmarks  g=gestures  m=emotion  d=drawing  c=clear  z=undo  v=HUD  "
                    "f=sunglasses  r=voice  s=palm power",
                    f"Open-palm power: {palm_powers[palm_power_index][0]}",
                ],
            )

            cv2.imshow(CONFIG.display.window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("e"):
                status_message = _enroll_face(frame, faces, recognizer)
                print(status_message)
            elif key == ord("l"):
                show_landmarks = not show_landmarks
                status_message = f"Landmarks {'ON' if show_landmarks else 'OFF'}"
            elif key == ord("g"):
                show_gestures = not show_gestures
                status_message = f"Gesture labels {'ON' if show_gestures else 'OFF'}"
            elif key == ord("m"):
                if emotion_recognizer is None:
                    status_message = "Emotion recognition unavailable -- run download_models.py"
                else:
                    show_emotion = not show_emotion
                    status_message = f"Emotion labels {'ON' if show_emotion else 'OFF'}"
            elif key == ord("d"):
                canvas.enabled = not canvas.enabled
                status_message = f"Drawing {'ON' if canvas.enabled else 'OFF'}"
            elif key == ord("c"):
                canvas.clear()
                status_message = "Canvas cleared."
            elif key == ord("z"):
                canvas.undo()
                status_message = "Undo."
            elif key == ord("v"):
                hud.visible = not hud.visible
                status_message = f"Laptop-control HUD {'ON' if hud.visible else 'OFF'}"
            elif key == ord("f"):
                face_filters.enabled = not face_filters.enabled
                status_message = f"Sunglasses {'ON' if face_filters.enabled else 'OFF'}"
            elif key == ord("r"):
                status_message = voice.stop() if voice.listening else voice.start()
            elif key == ord("s"):
                palm_power_index = (palm_power_index + 1) % len(palm_powers)
                status_message = f"Open-palm power: {palm_powers[palm_power_index][0]}"
    finally:
        voice.stop()
        camera.release()
        face_mesh.close()
        hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
