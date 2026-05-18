"""
Treinamento: computa embeddings FaceNet512 para cada usuário autorizado
e os salva em data_base/embeddings/.

Uso:
    python scripts/modelo_treino.py
"""
import json
import logging
import numpy as np
from pathlib import Path
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent.parent
USERS_DIR = BASE_DIR / "data_base" / "users"
EMBEDDINGS_DIR = BASE_DIR / "data_base" / "embeddings"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_user_embeddings(user_dir: Path) -> np.ndarray:
    """Retorna array (N, 512) com embeddings de todas as fotos válidas do usuário."""
    image_files = [f for f in user_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        log.warning(f"Nenhuma imagem encontrada em: {user_dir}")
        return np.empty((0, 512), dtype=np.float32)

    embeddings = []
    for img_path in sorted(image_files):
        try:
            result = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
            # result é lista; usa o primeiro rosto detectado na imagem
            vec = np.array(result[0]["embedding"], dtype=np.float32)
            embeddings.append(vec)
            log.info(f"    OK  {img_path.name}")
        except Exception as exc:
            log.warning(f"    SKIP {img_path.name}: {exc}")

    if not embeddings:
        return np.empty((0, 512), dtype=np.float32)

    return np.stack(embeddings)


def main() -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    user_dirs = sorted(d for d in USERS_DIR.iterdir() if d.is_dir())
    if not user_dirs:
        log.error(f"Nenhuma pasta de usuário encontrada em: {USERS_DIR}")
        return

    log.info(f"Modelo: {MODEL_NAME}  |  Detector: {DETECTOR_BACKEND}")
    log.info(f"Usuários encontrados: {[d.name for d in user_dirs]}\n")

    metadata: dict = {"model": MODEL_NAME, "detector": DETECTOR_BACKEND, "users": []}
    total_embeddings = 0

    for user_dir in user_dirs:
        # stem remove a extensão caso a pasta tenha nome como "Eduardo.jpg"
        user_name = user_dir.stem
        log.info(f"[{user_name}] processando {user_dir.name}...")

        embeddings = compute_user_embeddings(user_dir)

        if len(embeddings) == 0:
            log.warning(f"[{user_name}] nenhum embedding gerado — pulando.\n")
            continue

        out_path = EMBEDDINGS_DIR / f"{user_name}.npy"
        np.save(str(out_path), embeddings)

        count = len(embeddings)
        total_embeddings += count
        metadata["users"].append(
            {
                "name": user_name,
                "folder": user_dir.name,
                "embeddings_file": out_path.name,
                "num_embeddings": count,
            }
        )
        log.info(f"[{user_name}] {count} embeddings salvos → {out_path.name}\n")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info("=" * 50)
    log.info(f"Treinamento concluído.")
    log.info(f"Usuários processados : {len(metadata['users'])}")
    log.info(f"Total de embeddings  : {total_embeddings}")
    log.info(f"Metadados            : {METADATA_PATH}")


if __name__ == "__main__":
    main()
