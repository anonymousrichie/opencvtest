"""Central configuration for the real-time CV pipeline.

Every tunable knob (model paths, thresholds, feature flags) lives here so the
rest of the codebase never hardcodes a magic number. Import `CONFIG` and read
from its nested dataclasses rather than passing loose arguments around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
ENROLLMENT_DB_PATH = PROJECT_ROOT / "enrollment_data" / "faces.pkl"


@dataclass
class CameraConfig:
    """Webcam capture settings."""

    index: int = 0  # change to select a different camera (1, 2, ...)
    width: int = 1280
    height: int = 720
    request_fps: int = 30


@dataclass
class FaceDetectionConfig:
    """YuNet face detector settings (cv2.FaceDetectorYN)."""

    model_path: Path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    score_threshold: float = 0.7
    nms_threshold: float = 0.3
    top_k: int = 20


@dataclass
class FaceRecognitionConfig:
    """SFace embedding + matching settings (cv2.FaceRecognizerSF)."""

    model_path: Path = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    # Cosine similarity cutoff above which a match counts as "known". 0.36 is
    # the SFace authors' published operating point (~99.6% precision on LFW);
    # raise it to reduce false accepts, lower it if real enrollments are being
    # missed under poor lighting.
    match_threshold: float = 0.36
    db_path: Path = ENROLLMENT_DB_PATH


@dataclass
class EmotionConfig:
    """FER+ emotion classification settings (ONNX CNN run via cv2.dnn)."""

    model_path: Path = MODELS_DIR / "emotion_ferplus.onnx"
    # Re-running the CNN on every face every frame is unnecessary -- like face
    # recognition, each track only re-classifies once per this many frames and
    # otherwise reuses its cached label. The model is cheap (64x64 grayscale
    # input) so this can be shorter than TrackingConfig.recognition_interval.
    emotion_interval: int = 5


@dataclass
class FaceMeshConfig:
    """MediaPipe Face Mesh settings, used only for the optional dense overlay."""

    max_num_faces: int = 3
    refine_landmarks: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class HandConfig:
    """MediaPipe Hands settings and gesture-geometry thresholds."""

    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass
class TrackingConfig:
    """Greedy IoU tracker settings."""

    iou_match_threshold: float = 0.3
    max_missed_frames: int = 15  # frames a track survives without a matching detection
    # Re-running SFace on every face every frame is the single most expensive
    # step in the pipeline. Recognition results change slowly for a tracked
    # face, so each track only re-embeds/matches once per this many frames and
    # otherwise reuses its cached identity -- this is what keeps the pipeline
    # at the 20-25 FPS target on a laptop CPU.
    recognition_interval: int = 10


@dataclass
class DrawingConfig:
    """Hand-gesture drawing overlay: two extended fingers (index + middle)
    draws a freehand stroke; a closed-fist pinch (thumb + index touching)
    grabs the most recently drawn stroke and stretches it as the pinch point
    moves away from (or toward) that stroke's centroid."""

    enabled_by_default: bool = True
    stroke_color: tuple[int, int, int] = (0, 220, 255)
    stroke_thickness: int = 4
    min_point_distance: float = 4.0  # px; drop points closer than this to keep strokes smooth
    pinch_threshold: float = 0.35  # thumb-index distance below this (x hand scale) counts as a pinch
    min_stretch: float = 0.25  # clamp how far a pinch can shrink a stroke
    max_stretch: float = 4.0  # clamp how far a pinch can enlarge a stroke
    # Colors for the three color-picker poses, in order: index-finger-only,
    # index+middle+ring, and index+pinky ("rock on"). Deliberately distinct
    # from the draw (index+middle only), pinch-stretch (thumb+index touching)
    # and eraser (closed fist) poses so color-picking never fights with them.
    palette: tuple[tuple[int, int, int], ...] = (
        (60, 60, 230),  # red
        (60, 200, 60),  # green
        (230, 160, 60),  # blue
    )
    eraser_radius_scale: float = 1.6  # eraser radius as a multiple of hand scale, centered on the fist


@dataclass
class ShieldConfig:
    """Doctor-Strange-style rotating fire shield, drawn anchored to an open
    palm (HandResult.gesture == "Open Palm")."""

    enabled_by_default: bool = True
    radius_scale: float = 2.3  # shield radius as a multiple of the hand's wrist-to-middle-mcp scale
    rotation_speed: float = 0.05  # radians added to the rotation phase each frame
    fade_step: float = 0.15  # how fast strength eases toward its target each frame (0..1)
    glow_sigma: float = 5.0  # Gaussian blur sigma for the additive glow pass
    max_embers: int = 40
    ember_spawn_rate: int = 2  # new embers spawned per frame while fully active
    ember_outflow: float = 0.01  # radial drift per frame, as a fraction of shield radius
    ember_decay: float = 0.02  # life lost per frame (life goes 1 -> 0)


@dataclass
class RasenganConfig:
    """Naruto-style spinning chakra-ball ("Rasengan"), drawn anchored to an
    open palm (HandResult.gesture == "Open Palm")."""

    enabled_by_default: bool = True
    radius_scale: float = 1.6  # ball radius as a multiple of the hand's wrist-to-middle-mcp scale
    rotation_speed: float = 0.35  # radians added to the rotation phase each frame -- fast spin
    fade_step: float = 0.15  # how fast strength eases toward its target each frame (0..1)
    glow_sigma: float = 6.0  # Gaussian blur sigma for the additive glow pass
    num_arms: int = 3  # spiral vortex arms
    arm_turns: float = 1.4  # how many full turns each spiral arm makes from center to edge
    num_motes: int = 10  # bright chakra motes swirling near the surface
    # A flat, front-facing palm's index-MCP/wrist/pinky-MCP triangle area
    # (normalized by hand-scale^2) that counts as "fully facing the camera".
    # Tune down if the ball looks too squashed while facing the camera
    # dead-on, up if it doesn't squash enough when the palm turns edge-on.
    facing_reference: float = 1.3
    min_squash: float = 0.35  # never squash thinner than this, so the ball stays visible edge-on


@dataclass
class RepulsorConfig:
    """Iron-Man-style repulsor blast, drawn anchored to an open palm
    (HandResult.gesture in ("Open Palm", "Four"))."""

    enabled_by_default: bool = True
    radius_scale: float = 1.1  # core radius as a multiple of the hand's wrist-to-middle-mcp scale
    rotation_speed: float = 0.06  # radians added to the spoke rotation phase each frame
    fade_step: float = 0.15  # how fast strength eases toward its target each frame (0..1)
    glow_sigma: float = 6.0  # Gaussian blur sigma for the additive glow pass
    pulse_interval: int = 18  # frames between new shockwave pulses while fully active
    pulse_speed: float = 0.05  # shockwave radius growth per frame, as a fraction of core radius
    pulse_max_radius: float = 3.2  # shockwave despawns after growing past this multiple of core radius
    num_spokes: int = 8


@dataclass
class EmotionFxConfig:
    """Screen-space payoff for the emotion recognizer: confetti on a smile,
    a red vignette flash while angry."""

    confetti_spawn_rate: int = 3  # new confetti pieces per frame per smiling face
    max_confetti: int = 220
    happy_threshold: float = 0.6  # FER+ confidence required to count as "smiling"
    angry_threshold: float = 0.6
    anger_fade_step: float = 0.12  # how fast the vignette eases toward its target each frame (0..1)
    anger_max_alpha: float = 0.55  # peak strength of the red vignette tint


@dataclass
class FaceFilterConfig:
    """AR sunglasses anchored to MediaPipe Face Mesh eye landmarks."""

    enabled_by_default: bool = True
    lens_color: tuple[int, int, int] = (25, 20, 15)
    lens_radius_scale: float = 0.62  # lens radius as a multiple of the inter-eye distance


@dataclass
class VoiceConfig:
    """Background speech recognition (captured via `sounddevice`, transcribed
    with the free Google Web Speech API through the `speech_recognition`
    package) mapped to app/laptop voice commands."""

    enabled_by_default: bool = False  # requires mic access + internet; opt in with 'r'
    language: str = "en-US"
    energy_threshold: int = 300  # minimum mic RMS level that counts as speech, before ambient calibration
    pause_threshold: float = 0.7  # seconds of silence that ends a phrase
    phrase_time_limit: float = 6.0  # max seconds captured per phrase
    speak_feedback: bool = True  # speak each command's result aloud (toggle with 't')


@dataclass
class HudConfig:
    """Virtual on-screen buttons that control the whole laptop: media/volume
    keys, keyboard shortcuts, launching apps, and a hands-free mouse-cursor
    mode. A fingertip hovering a button highlights it; pinching (thumb +
    index touching) while hovering fires it."""

    visible_by_default: bool = True
    panel_width: int = 220
    margin: int = 12
    tab_height: int = 36
    button_height: int = 42
    button_gap: int = 8
    pinch_threshold: float = 0.35  # thumb-index distance below this (x hand scale) counts as a pinch/click
    dwell_seconds: float = 0.6  # how long to hover the cursor-mode badge to toggle it
    toggle_badge_size: tuple[int, int] = (170, 44)
    # (label, command) launched with os.startfile when its Apps button fires.
    apps: tuple[tuple[str, str], ...] = (
        ("Browser", "https://www.google.com"),
        ("Notepad", "notepad.exe"),
        ("Calculator", "calc.exe"),
        ("Explorer", "explorer.exe"),
    )


@dataclass
class DisplayConfig:
    """Window and overlay feature flags."""

    window_name: str = "Cam IntelliSense"
    show_landmarks: bool = True
    show_gestures: bool = True
    show_emotion: bool = True
    show_fps: bool = True


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    face_mesh: FaceMeshConfig = field(default_factory=FaceMeshConfig)
    hands: HandConfig = field(default_factory=HandConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    rasengan: RasenganConfig = field(default_factory=RasenganConfig)
    repulsor: RepulsorConfig = field(default_factory=RepulsorConfig)
    emotion_fx: EmotionFxConfig = field(default_factory=EmotionFxConfig)
    face_filter: FaceFilterConfig = field(default_factory=FaceFilterConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


CONFIG = AppConfig()
