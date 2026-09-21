# Cam IntelliSense

A single-camera, real-time computer vision base system: face detection +
tracking + recognition, hand gesture classification, and a growing set of
gesture/voice-driven extras (a drawing canvas, movie-style "powers" anchored
to an open palm, AR sunglasses, emotion-reactive effects, a virtual button
panel that controls the laptop, and voice commands) -- all built on clean
OpenCV overlays. Built as a foundation to extend later, not a finished
product.

## Architecture

| File | Responsibility |
|---|---|
| `config.py` | All tunables (thresholds, model paths, feature flags) in one place |
| `camera.py` | Webcam open/read/release, backend fallback, error handling |
| `face_detection.py` | YuNet face detection (`cv2.FaceDetectorYN`) |
| `tracking.py` | Greedy IoU tracker assigning stable per-face track IDs |
| `face_recognition.py` | SFace embeddings (`cv2.FaceRecognizerSF`) + matching |
| `database.py` | Local pickle-backed store of enrolled face embeddings + profile info (age, department, ID, nickname, hall) |
| `emotion.py` | FER+ emotion classification (ONNX CNN via `cv2.dnn`) |
| `emotion_fx.py` | Confetti on a smile, a red vignette flash while angry -- driven by `emotion.py`'s output |
| `face_mesh.py` | Dense 468-point face landmark mesh (MediaPipe), used for the overlay dots and to anchor the sunglasses |
| `face_filters.py` | AR sunglasses anchored to face-mesh eye landmarks |
| `hand_gestures.py` | MediaPipe Hands + geometric gesture classification |
| `drawing.py` | Gesture-driven whiteboard: draw, pinch-to-stretch, erase, pick colors |
| `shield.py` | Doctor-Strange-style fire shield, anchored to an open palm |
| `rasengan.py` | Naruto-style spinning chakra ball, anchored to an open palm |
| `repulsor.py` | Iron-Man-style repulsor blast, anchored to an open palm |
| `hud.py` | On-screen virtual buttons (media/apps/shortcuts) + a hands-free mouse-cursor mode |
| `system_control.py` | `pyautogui`/`os.startfile` wrappers the HUD and voice commands call into |
| `voice_control.py` | Background speech recognition (mic capture via `sounddevice`, transcription via Google's Web Speech API) mapped to app/laptop commands |
| `voice_feedback.py` | Spoken feedback for voice commands (`pyttsx3`/SAPI5), on its own thread so speaking doesn't stall the video loop |
| `overlay.py` | All `cv2.putText`/`cv2.rectangle` drawing, kept out of `main.py` |
| `fps.py` | Rolling-window FPS counter |
| `main.py` | Wires everything into the capture -> process -> display loop |

## Key design decisions

- **Detection + recognition backend: OpenCV YuNet + SFace, not InsightFace.**
  The spec allows either. YuNet/SFace are small ONNX models that ship through
  the OpenCV Zoo and run via `opencv-contrib-python` (already a MediaPipe
  dependency), so the whole system needs only two pip packages and two small
  model files -- no extra native build step, which matters most on Windows.
  They also share YuNet's 5-point landmark output directly for face alignment.
- **Tracking: greedy IoU, not SORT/DeepSORT/ByteTrack.** For a handful of
  faces in frame, matching each new detection to the closest previous box by
  intersection-over-union is enough to keep IDs stable and is dependency-free.
  See `tracking.py` for the extension point if fast motion / occlusion
  handling is needed later.
- **Recognition runs on a duty cycle, not every frame.** SFace inference
  (`FaceRecognizer.embed` + `match`) is the most expensive step per face.
  Each track only re-embeds/re-matches once every `recognition_interval`
  frames (default 10) and reuses its cached identity otherwise -- this is
  what keeps the pipeline at the 20-25 FPS target on a laptop CPU. Tune it in
  `config.py` (`TrackingConfig.recogniti on_interval`).
- **Matching threshold: cosine similarity >= 0.36.** This is the SFace
  authors' published operating point (~99.6% precision on LFW). Configurable
  via `FaceRecognitionConfig.match_threshold`; raise it to cut false accepts,
  lower it if real enrollments are being missed in poor lighting.
- **Gestures: geometric rules over MediaPipe's 21 hand landmarks, not a
  trained classifier.** Finger-extended state is judged by tip-to-wrist vs.
  pip-to-wrist distance (robust to hand rotation), the thumb by its splay
  from the index-finger base, and pinch gestures (OK sign) by tip-to-tip
  distance -- all normalized by a per-hand size unit so thresholds hold at
  any distance from the camera. See `hand_gestures.py:classify_gesture` for
  the exact rule order and rationale; it's the natural place to swap in a
  trained model later.
- **Emotion: a small pretrained CNN (FER+), not geometric rules.** Unlike
  hand gestures, facial expressions don't reduce to clean landmark
  measurements -- anger vs. disgust vs. fear are subtle and overlapping. FER+
  is a ~34 MB ONNX model (from the ONNX Model Zoo) run through `cv2.dnn`, the
  same way YuNet/SFace run, so it adds no new dependency. It classifies one of
  8 emotions (neutral, happy, surprise, sad, angry, disgust, fear, contempt)
  from a 64x64 grayscale crop of each tracked face, on the same
  duty-cycle pattern as recognition (`EmotionConfig.emotion_interval`, default
  every 5 frames -- cheaper than SFace so it can run more often). If the model
  file isn't present, the feature disables itself with a warning instead of
  blocking startup.

## Install

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
```

`download_models.py` fetches the three ONNX models (YuNet detector, SFace
recognizer, FER+ emotion classifier) into `models/`. Run it once; if your
network blocks GitHub, the script prints the URLs to fetch manually.

## Run

```powershell
python main.py
```

A window opens showing your webcam feed with live overlays. Press **q** to quit.

## Enrolling faces

1. Look at the camera so exactly the face you want to enroll is clearly
   detected (green/red box visible). If more than one face is in frame, the
   terminal warns you which one (the largest) is about to be enrolled.
2. Press **e**. The video window will freeze on that frame while the
   terminal prompts:
   ```
   Enter name for enrollment:
   ```
3. Type a name and press Enter, then fill in the optional profile prompts
   (age, department, ID, nickname, hall) -- press Enter on any of them to
   skip. The largest detected face in that frame is embedded and stored,
   along with the profile, in `enrollment_data/faces.pkl`.
4. From then on, that person is recognized in real time and labeled with
   their name and a confidence score, plus a second line showing whichever
   profile fields were filled in. Unmatched faces are labeled `Unknown`.

Re-enrolling the same name (e.g. from a different angle) adds another
embedding without needing to retype the whole profile -- leaving a profile
prompt blank keeps that field's previously stored value rather than erasing it.

Enroll the same person multiple times (different angles/lighting) for more
robust matching -- each enrollment adds an embedding rather than replacing
the previous one.

## Keyboard controls

| Key | Action |
|---|---|
| `q` | Quit |
| `e` | Enroll the current largest detected face (prompts for a name in the terminal) |
| `l` | Toggle the dense face-mesh landmark overlay |
| `g` | Toggle hand gesture text labels |
| `m` | Toggle emotion labels + emotion-reactive effects (confetti/anger vignette) |
| `d` | Toggle the drawing canvas |
| `c` | Clear the drawing canvas |
| `z` | Undo the last stroke |
| `s` | Cycle the open-palm "power": Off -> Fire Shield -> Rasengan -> Repulsor Blast |
| `f` | Toggle AR sunglasses |
| `v` | Toggle the laptop-control HUD (virtual buttons + hands-free mouse mode) |
| `r` | Toggle voice control on/off |
| `t` | Toggle spoken voice feedback on/off |

## Supported hand gestures

Open Palm, Fist, Thumbs Up, Thumbs Down, Pointing, Peace / Victory, OK Sign,
Three, Four, L Shape, Call Me, Rock On, I Love You -- detected for both hands
independently, labeled `Left`/`Right` from the viewer's perspective. See
`hand_gestures.py:classify_gesture` for the exact rules.

## Drawing canvas

Press `d` to turn it on. Poses are geometrically distinct so none of them
fight over the same gesture:

| Pose | Action |
|---|---|
| Index + middle finger extended | Draw, following the midpoint between the two fingertips |
| Thumb + index touching, other fingers folded | Pinch-grab the most recent stroke and stretch it as you move your hand |
| Fully closed fist | Erase any stroke within reach |
| Index finger only / index+middle+ring / index+pinky ("rock on") | Pick red / green / blue |

## Open-palm powers

Press `s` to cycle which effect (if any) appears when you hold your palm
open toward the camera -- a Doctor-Strange-style fire shield, a Naruto-style
Rasengan, or an Iron-Man-style repulsor blast. Each fades in/out smoothly
and tracks each hand independently.

## Laptop-control HUD

Press `v` to show a virtual button panel (top-right) for media/volume keys,
launching apps, and keyboard shortcuts (Alt+Tab, copy/paste, screenshot) --
hover a fingertip over a button and pinch to fire it. A "Cursor: ON/OFF"
badge (top-left, hover-and-hold to toggle) hands your fingertip control of
the real OS mouse cursor, with a pinch as a click.

## Voice control

Press `r` to start listening, `t` to toggle spoken feedback (on by default --
each command's result is read back aloud via `pyttsx3`/SAPI5, on its own
background thread so it doesn't stall the video loop). Speech is captured
via `sounddevice` and transcribed with Google's free Web Speech API
(requires internet + a working microphone). Matching happens in three
tiers (see `VoiceController._resolve` in `voice_control.py`):

1. **Free-text commands**, anchored to the start of what you said, so a word
   inside whatever you're dictating can't accidentally trigger another
   command:
   - "type `<anything>`" -- types it into whatever has focus (dictation)
   - "search for `<anything>`" / "google `<anything>`" -- opens a Google
     search for it in your browser
2. **Fixed phrases** (matched anywhere in the sentence):
   - "volume up" / "volume down" / "mute"
   - "play music" / "next track" / "previous track"
   - "copy" / "paste" / "cut" / "undo" / "redo" / "select all" / "save" / "find"
   - "new tab" / "close tab" / "refresh" / "take a screenshot" / "switch window"
   - "minimize window" / "maximize window" / "show desktop" / "close window"
   - "snap window left" / "snap window right" / "task view"
   - "new desktop" / "close desktop" / "next desktop" / "previous desktop"
   - "task manager" / "open settings"
   - "zoom in" / "zoom out" / "reset zoom" / "scroll up" / "scroll down"
   - "left click" / "right click" / "double click"
   - "lock screen"
   - "open browser" / "open notepad" / "open calculator" / "open explorer"
   - "open downloads" / "open documents" / "open desktop folder" / "open pictures"
   - "clear canvas" / "toggle drawing"
   - "fire shield" / "rasengan" / "repulsor blast" / "powers off"
   - "show hud" / "hide hud" / "sunglasses on" / "sunglasses off"
3. **Generic parametrized commands** (checked last, as a catch-all):
   - "set the volume to `N` percent" / "set the brightness to `N` percent" --
     an exact level, not just nudging one step at a time (brightness only
     works on displays that expose it over WMI, typically laptop panels)
   - "open `<any app>`" -- fuzzy-matches the name against your Start Menu
     shortcuts, so it's not limited to the apps hardcoded above
   - "close `<any app>`" -- the counterpart to "open": finds the running
     process by name and terminates it. Matching is deliberately stricter
     here (exact/prefix only, no fuzzy guessing) and a fixed list of core OS
     processes (explorer, svchost, winlogon, ...) is refused outright,
     since a wrong match here kills a process rather than just opening the
     wrong window

See `voice_commands`/`voice_priority_commands`/`voice_fallback_commands` in
`main.py` to add more. Deliberately **not** included: shutdown, restart, or
sign-out -- those are one misheard phrase away from losing unsaved work, so
they're left out rather than gated behind anything voice could bypass.

## Changing the camera

Edit `CameraConfig.index` in `config.py` (0 is the default webcam; try 1, 2,
... for external cameras).

## Performance tuning

If FPS is below target on your machine, in `config.py`:
- Lower `CameraConfig.width`/`height` (e.g. 960x540).
- Increase `TrackingConfig.recognition_interval` (recognize less often).
- Increase `EmotionConfig.emotion_interval` (classify emotion less often).
- Reduce `HandConfig.max_num_hands` to 1 if only one hand is needed.
- Toggle off face mesh (`l`), gestures (`g`), and/or emotion (`m`) at runtime.
