"""
Detecção de rostos em frames BGR usando Haar Cascade do OpenCV.

Responsabilidade: localizar e recortar regiões de rosto no frame bruto.
O recognizer.py usa o DeepFace internamente para alinhar o rosto antes
do embedding — aqui só precisamos saber *onde* há rostos para exibir
bounding boxes e decidir quando acionar o reconhecimento.
"""
import cv2
import logging
import numpy as np
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Margem adicionada ao redor do bounding box antes de recortar
_CROP_MARGIN = 0.20


@dataclass(frozen=True)
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    crop: np.ndarray  # ROI BGR pronto para passar ao recognizer


class FaceDetector:
    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Não foi possível carregar o Haar Cascade: {cascade_path}")
        log.info("FaceDetector inicializado (Haar Cascade).")

    def detect(self, frame: np.ndarray) -> list[FaceRegion]:
        """
        Recebe frame BGR e retorna lista de FaceRegion detectadas.
        Retorna lista vazia se nenhum rosto for encontrado.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        raw = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(raw) == 0:
            return []

        h_frame, w_frame = frame.shape[:2]
        regions = []

        for x, y, w, h in raw:
            # Adiciona margem para incluir testa e queixo no crop
            margin_x = int(w * _CROP_MARGIN)
            margin_y = int(h * _CROP_MARGIN)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_frame, x + w + margin_x)
            y2 = min(h_frame, y + h + margin_y)

            crop = frame[y1:y2, x1:x2].copy()
            regions.append(FaceRegion(x=x, y=y, w=w, h=h, crop=crop))

        return regions

    @staticmethod
    def draw(frame: np.ndarray, regions: list[FaceRegion], label: str = "") -> np.ndarray:
        """Desenha bounding boxes e label no frame (in-place). Retorna o frame."""
        color = (0, 255, 0)
        for r in regions:
            cv2.rectangle(frame, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 2)
            if label:
                cv2.putText(
                    frame,
                    label,
                    (r.x, r.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
        return frame
