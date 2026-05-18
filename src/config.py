from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Dados ────────────────────────────────────────────────────────────────────
USERS_DIR = BASE_DIR / "data_base" / "users"
EMBEDDINGS_DIR = BASE_DIR / "data_base" / "embeddings"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

# ── Modelo FaceNet ────────────────────────────────────────────────────────────
FACENET_MODEL = "Facenet512"
DETECTOR_BACKEND = "opencv"

# Distância cosseno máxima para que um embedding individual seja contado como "voto a favor".
# 0.0 = idêntico, 1.0 = completamente diferente. Mais alto = mais tolerante.
# Câmera de corredor justifica ser mais permissivo (0.40).
DISTANCE_THRESHOLD = 0.40

# Percentual mínimo de embeddings cadastrados que devem votar a favor para conceder acesso.
# Ex.: Eduardo tem 45 fotos → pelo menos 80% (≥36) devem ter distância ≤ DISTANCE_THRESHOLD.
CONFIDENCE_THRESHOLD = 60.0  # soma mínima 180 / 300 pontos

# ── Câmera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ── Arduino / Porta ───────────────────────────────────────────────────────────
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 9600
DOOR_OPEN_SECONDS = 5      # tempo que a porta fica aberta

# ── Liveness detection ────────────────────────────────────────────────────────
MEDIAPIPE_MODEL = str(BASE_DIR / "models" / "face_landmarker.task")
EYE_BLINK_THRESHOLD = 0.20  # EAR abaixo desse valor = olho fechado
BLINK_REQUIRED = 2           # piscadas mínimas para passar no anti-spoofing
