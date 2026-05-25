from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# ── Paths ─────────────────────────────────────────────────────────────────────
USERS_DIR      = BASE_DIR / _cfg["paths"]["users_dir"]
EMBEDDINGS_DIR = BASE_DIR / _cfg["paths"]["embeddings_dir"]
METADATA_PATH  = EMBEDDINGS_DIR / _cfg["paths"]["metadata_file"]
MEDIAPIPE_MODEL = str(BASE_DIR / _cfg["paths"]["mediapipe_model"])

# ── Modelo FaceNet ────────────────────────────────────────────────────────────
FACENET_MODEL      = _cfg["model"]["facenet"]
DETECTOR_BACKEND   = _cfg["model"]["detector_backend"]
DISTANCE_THRESHOLD = _cfg["model"]["distance_threshold"]
CONFIDENCE_THRESHOLD = _cfg["model"]["confidence_threshold"]

# ── Câmera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = _cfg["camera"]["index"]
FRAME_WIDTH  = _cfg["camera"]["width"]
FRAME_HEIGHT = _cfg["camera"]["height"]

# ── Arduino / Porta ───────────────────────────────────────────────────────────
SERIAL_PORT       = _cfg["door"]["serial_port"]
SERIAL_BAUDRATE   = _cfg["door"]["baudrate"]
DOOR_OPEN_SECONDS = _cfg["door"]["open_seconds"]

# ── Liveness detection ────────────────────────────────────────────────────────
EYE_BLINK_THRESHOLD = _cfg["liveness"]["eye_blink_threshold"]
BLINK_REQUIRED      = _cfg["liveness"]["blinks_required"]
