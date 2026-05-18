"""
Reconhecimento facial baseado em embeddings FaceNet512.

Fluxo:
  1. Na inicialização, carrega todos os embeddings de data_base/embeddings/.
  2. Para cada frame recebido, computa o embedding do rosto detectado.
  3. Para cada usuário, conta quantos embeddings têm distância cosseno ≤ DISTANCE_THRESHOLD
     — isso é o "voto" de cada embedding armazenado.
  4. Confiança = (votos / total_embeddings) * 100.
  5. Acesso concedido apenas se confiança ≥ CONFIDENCE_THRESHOLD (padrão: 80%).
"""
import json
import logging
import numpy as np
from pathlib import Path
from deepface import DeepFace

from src.config import (
    EMBEDDINGS_DIR,
    METADATA_PATH,
    FACENET_MODEL,
    DETECTOR_BACKEND,
    DISTANCE_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)

log = logging.getLogger(__name__)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distância cosseno entre dois vetores 1-D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


class FaceRecognizer:
    def __init__(self) -> None:
        # dict: nome → array (N, 512)
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_embeddings()

    # ── Inicialização ─────────────────────────────────────────────────────────

    def _load_embeddings(self) -> None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"metadata.json não encontrado em {METADATA_PATH}. "
                "Execute scripts/modelo_treino.py primeiro."
            )

        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

        for user in metadata["users"]:
            npy_path = EMBEDDINGS_DIR / user["embeddings_file"]
            if not npy_path.exists():
                log.warning(f"Arquivo de embeddings não encontrado: {npy_path}")
                continue
            self._embeddings[user["name"]] = np.load(str(npy_path))
            log.info(
                f"  Carregado: {user['name']} "
                f"({user['num_embeddings']} embeddings)"
            )

        if not self._embeddings:
            raise RuntimeError("Nenhum embedding carregado. Verifique data_base/embeddings/.")

        log.info(f"FaceRecognizer pronto. Usuários: {list(self._embeddings.keys())}")

    # ── API pública ───────────────────────────────────────────────────────────

    def identify(self, image: np.ndarray) -> tuple[str | None, float]:
        """
        Recebe um frame BGR (numpy array) com um rosto.
        Retorna (nome, confiança%) se confiança ≥ CONFIDENCE_THRESHOLD, ou (None, confiança%).
        """
        embedding = self._compute_embedding(image)
        if embedding is None:
            return None, 0.0

        best_name, best_confidence = self._vote(embedding)

        if best_confidence >= CONFIDENCE_THRESHOLD:
            log.debug(f"Reconhecido: {best_name} ({best_confidence:.1f}%)")
            return best_name, best_confidence

        log.debug(f"Não reconhecido. Melhor candidato: {best_name} ({best_confidence:.1f}%)")
        return None, best_confidence

    @property
    def authorized_users(self) -> list[str]:
        return list(self._embeddings.keys())

    # ── Internos ──────────────────────────────────────────────────────────────

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray | None:
        try:
            result = DeepFace.represent(
                img_path=image,
                model_name=FACENET_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
            return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception as exc:
            log.debug(f"Embedding falhou: {exc}")
            return None

    def _vote(self, query: np.ndarray) -> tuple[str, float]:
        """
        Sistema de votação: para cada usuário, calcula a porcentagem de embeddings
        cadastrados com distância cosseno ≤ DISTANCE_THRESHOLD.
        Retorna (nome_do_melhor_candidato, confiança_em_%).
        """
        best_name = ""
        best_confidence = 0.0

        for name, stored in self._embeddings.items():
            distances = np.array([_cosine_distance(query, ref) for ref in stored])
            votes = int(np.sum(distances <= DISTANCE_THRESHOLD))
            confidence = (votes / len(stored)) * 100.0
            log.debug(f"  {name}: {votes}/{len(stored)} votos → {confidence:.1f}%")
            if confidence > best_confidence:
                best_confidence = confidence
                best_name = name

        return best_name, best_confidence
