import logging
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import MEDIAPIPE_MODEL, EYE_BLINK_THRESHOLD, BLINK_REQUIRED

log = logging.getLogger(__name__)

# MediaPipe 478-point face mesh — índices dos olhos
# Ordem: canto-externo, topo-externo, topo-interno, canto-interno, base-interna, base-externa
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]


def _ear(landmarks, indices: list[int], w: int, h: int) -> float:
    """Eye Aspect Ratio: razão entre abertura vertical e horizontal do olho."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hz = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * hz)) if hz > 0 else 0.0


class LivenessDetector:
    def __init__(self) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._blink_count = 0
        self._eyes_closed = False
        log.info("LivenessDetector inicializado (MediaPipe EAR).")

    def reset(self) -> None:
        self._blink_count = 0
        self._eyes_closed = False

    @property
    def blink_count(self) -> int:
        return self._blink_count

    @property
    def passed(self) -> bool:
        return self._blink_count >= BLINK_REQUIRED

    def check(self, frame: np.ndarray) -> float:
        """
        Processa frame BGR, atualiza contagem de piscadas e retorna EAR médio.
        Retorna 1.0 (olhos abertos) se nenhum rosto for detectado.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        if not result.face_landmarks:
            return 1.0

        lm = result.face_landmarks[0]
        ear = (_ear(lm, _RIGHT_EYE, w, h) + _ear(lm, _LEFT_EYE, w, h)) / 2.0

        if ear < EYE_BLINK_THRESHOLD:
            self._eyes_closed = True
        elif self._eyes_closed:
            self._blink_count += 1
            self._eyes_closed = False
            log.info(f"Piscada detectada ({self._blink_count}/{BLINK_REQUIRED})")

        return ear
