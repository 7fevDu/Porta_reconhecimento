import logging

import cv2

from src.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)


class Camera:
    """
    Gerencia a captura de frames via webcam usando OpenCV.

    Suporta uso como context manager:

        with Camera() as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        device_index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Abre a câmera e configura a resolução."""
        logger.info(f"Abrindo câmera no índice {self.device_index} ({self.width}x{self.height})")
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera no índice {self.device_index}. "
                "Verifique se o dispositivo está conectado e não está em uso."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info("Câmera aberta com sucesso.")

    def read(self):
        """
        Captura e retorna um frame da câmera.

        Retorna
        -------
        np.ndarray
            Frame BGR capturado.

        Raises
        ------
        RuntimeError
            Se a câmera não foi inicializada ou a leitura falhar.
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Câmera não inicializada. Chame open() antes de ler frames.")

        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError("Erro ao capturar frame. A câmera pode ter sido desconectada.")

        return frame

    def release(self) -> None:
        """Libera o recurso da câmera."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Câmera liberada.")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False