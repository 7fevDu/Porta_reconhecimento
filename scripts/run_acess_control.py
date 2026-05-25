"""
Loop principal do sistema de controle de acesso por reconhecimento facial.

Fluxo por frame:
  1. Captura frame da câmera
  2. Detecta rostos (Haar Cascade)
  3. Ao detectar rosto, coleta 3 amostras rápidas de confiança (FaceNet512)
  4. Calcula a média das 3 amostras
  5. Se média ≥ CONFIDENCE_THRESHOLD → abre porta
  6. Exibe overlay com progresso das verificações e percentual em tempo real

Uso:
    python scripts/run_acess_control.py
"""
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer
from src.door import DoorController
from src.config import CONFIDENCE_THRESHOLD, BLINK_REQUIRED
from src.liveness import LivenessDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SAMPLES_REQUIRED  = 3     # número de verificações por tentativa
_SAMPLE_INTERVAL   = 0.3   # segundos mínimos entre submissões ao worker
_STABLE_DELAY      = 0.5   # aguarda este tempo após detecção para face estabilizar antes de V1
_RESULT_HOLD_TIME  = 2.0   # segundos exibindo o resultado antes de resetar
_DOOR_COOLDOWN     = 5.0   # segundos mínimos entre acionamentos da porta
_LIVENESS_TIMEOUT  = 8.0   # segundos máximos aguardando piscadas após as amostras


# ── Helpers de cor e overlay ──────────────────────────────────────────────────

def _color_for(confidence: float) -> tuple:
    if confidence >= CONFIDENCE_THRESHOLD:
        return (0, 220, 0)      # verde  — aprovado
    if confidence >= 50:
        return (0, 200, 255)    # amarelo — analisando
    return (0, 60, 220)         # vermelho — baixo


def _draw_overlay(
    frame,
    state: str,
    samples: list[float],
    display_confidence: float,
    final_user: str | None,
    blink_count: int,
) -> None:
    """Painel no topo do frame com as 3 verificações e barra de progresso."""
    _, w = frame.shape[:2]
    panel_h = 82

    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(bg, 0.65, frame, 0.35, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_w = w // _SAMPLES_REQUIRED

    # ── Linha 1: V1 / V2 / V3 ────────────────────────────────────────────────
    for i in range(_SAMPLES_REQUIRED):
        if i < len(samples):                          # amostra já coletada
            color = _color_for(samples[i])
            label = f"V{i+1}: {samples[i]:.0f}%"
        elif i == len(samples) and state == "collecting":  # amostra atual
            color = (0, 200, 255)
            label = f"V{i+1}: ..."
        else:                                          # ainda não chegou
            color = (90, 90, 90)
            label = f"V{i+1}: --"

        cv2.putText(frame, label, (col_w * i + 12, 26), font, 0.70, color, 2)

    # ── Linha 2: status / resultado ───────────────────────────────────────────
    max_sum = _SAMPLES_REQUIRED * 100  # 300 pontos possíveis
    threshold_sum = CONFIDENCE_THRESHOLD * _SAMPLES_REQUIRED  # 270 para 90%

    if state == "idle":
        status_text = "Aguardando rosto..."
        status_color = (160, 160, 160)
    elif state == "collecting":
        status_text = f"Coletando... ({len(samples)}/{_SAMPLES_REQUIRED})  |  Piscadas: {blink_count}/{BLINK_REQUIRED}"
        status_color = (0, 200, 255)
    elif state == "waiting_liveness":
        needed = BLINK_REQUIRED - blink_count
        status_text = f"Pisque {needed} vez(es)  |  Piscadas: {blink_count}/{BLINK_REQUIRED}"
        status_color = (0, 200, 255)
    else:
        name = final_user or "Desconhecido"
        verdict = "ACESSO CONCEDIDO" if state == "granted" else "ACESSO NEGADO"
        status_text = (
            f"{name}  |  Soma: {display_confidence:.0f} / {threshold_sum:.0f}"
            f"  |  Piscadas: {blink_count}/{BLINK_REQUIRED}  |  {verdict}"
        )
        status_color = _color_for(display_confidence / _SAMPLES_REQUIRED)

    cv2.putText(frame, status_text, (12, 54), font, 0.62, status_color, 2)

    # ── Barra de progresso + marcador de threshold ────────────────────────────
    bx1, by1, bx2, by2 = 10, 66, w - 10, 76
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (70, 70, 70), -1)
    if display_confidence > 0:
        fill = bx1 + int((bx2 - bx1) * min(display_confidence, max_sum) / max_sum)
        cv2.rectangle(
            frame, (bx1, by1), (fill, by2),
            _color_for(display_confidence / _SAMPLES_REQUIRED), -1
        )
    # Linha branca marcando onde fica o threshold (90% da barra)
    tx = bx1 + int((bx2 - bx1) * CONFIDENCE_THRESHOLD / 100)
    cv2.line(frame, (tx, by1 - 3), (tx, by2 + 3), (255, 255, 255), 2)


# ── Conexão Arduino ───────────────────────────────────────────────────────────

def _try_connect_door() -> DoorController | None:
    door = DoorController()
    try:
        door.connect()
        return door
    except RuntimeError as exc:
        log.warning(f"Arduino não disponível: {exc}")
        log.warning("Rodando sem controle de porta (modo simulação).")
        return None


# ── Loop principal ────────────────────────────────────────────────────────────

def run() -> None:
    log.info("Inicializando sistema de controle de acesso...")

    detector   = FaceDetector()
    recognizer = FaceRecognizer()
    liveness   = LivenessDetector()
    executor   = ThreadPoolExecutor(max_workers=1)
    door       = _try_connect_door()

    log.info("Aquecendo modelo de reconhecimento em background...")
    executor.submit(recognizer.warmup)

    log.info(f"Usuários autorizados: {recognizer.authorized_users}")
    log.info(f"Verificações por tentativa: {_SAMPLES_REQUIRED} | Threshold: {CONFIDENCE_THRESHOLD}%")
    log.info("Pressione Q para encerrar.\n")

    # ── Estado da máquina ─────────────────────────────────────────────────────
    state: str                  = "idle"     # idle | collecting | waiting_liveness | granted | denied
    samples:      list[float]   = []
    sample_users: list[str | None] = []
    last_sample_time      = 0.0
    last_door_time        = 0.0
    result_time           = 0.0
    liveness_start_time   = 0.0
    collecting_start_time = 0.0
    final_confidence    = 0.0
    final_user:     str | None    = None
    pending_future: "Future | None" = None

    try:
        with Camera() as cam:
            while True:
                frame   = cam.read()
                regions = detector.detect(frame)
                now     = time.monotonic()

                # ── Transições de estado ──────────────────────────────────────
                if state in ("granted", "denied"):
                    if (now - result_time) >= _RESULT_HOLD_TIME:
                        state = "idle"
                        samples.clear()
                        sample_users.clear()
                        final_confidence = 0.0
                        final_user = None

                elif state == "idle":
                    if regions:
                        log.info("Rosto detectado — iniciando verificações.")
                        state = "collecting"
                        samples.clear()
                        sample_users.clear()
                        last_sample_time      = now   # V1 aguarda _STABLE_DELAY antes de disparar
                        collecting_start_time = now
                        liveness.reset()

                elif state == "collecting":
                    if not regions:
                        log.info("Rosto perdido — reiniciando.")
                        state = "idle"
                        samples.clear()
                        sample_users.clear()
                        if pending_future is not None:
                            pending_future.cancel()
                            pending_future = None
                    else:
                        liveness.check(frame)

                        # coleta resultado pronto do worker
                        if pending_future is not None and pending_future.done():
                            try:
                                user, conf = pending_future.result()
                            except Exception as exc:
                                log.debug(f"Reconhecimento falhou: {exc}")
                                user, conf = None, 0.0
                            pending_future = None
                            samples.append(conf)
                            sample_users.append(user)
                            log.info(
                                f"  [{len(samples)}/{_SAMPLES_REQUIRED}] "
                                f"{user or 'desconhecido'}: {conf:.1f}%"
                            )
                            if len(samples) == _SAMPLES_REQUIRED:
                                state = "waiting_liveness"
                                liveness_start_time = now
                                log.info("Amostras coletadas — aguardando piscadas...")

                        # submete próximo crop se o worker estiver livre e face estabilizou
                        if (
                            pending_future is None
                            and len(samples) < _SAMPLES_REQUIRED
                            and (now - collecting_start_time) >= _STABLE_DELAY
                            and (now - last_sample_time) >= _SAMPLE_INTERVAL
                        ):
                            face = max(regions, key=lambda r: r.w * r.h)
                            pending_future = executor.submit(recognizer.identify, face.crop.copy())
                            last_sample_time = now

                elif state == "waiting_liveness":
                    if not regions:
                        log.info("Rosto perdido — reiniciando.")
                        state = "idle"
                        samples.clear()
                        sample_users.clear()
                        liveness.reset()
                    else:
                        liveness.check(frame)
                        liveness_timeout = (now - liveness_start_time) >= _LIVENESS_TIMEOUT
                        if liveness.passed or liveness_timeout:
                            final_confidence = sum(samples)
                            threshold_sum = CONFIDENCE_THRESHOLD * _SAMPLES_REQUIRED
                            valid_users = [u for u in sample_users if u is not None]
                            final_user = (
                                max(set(valid_users), key=valid_users.count)
                                if valid_users else None
                            )
                            result_time = now
                            door_ready = (now - last_door_time) >= _DOOR_COOLDOWN
                            if (
                                final_user
                                and final_confidence >= threshold_sum
                                and liveness.passed
                                and door_ready
                            ):
                                state = "granted"
                                log.info(
                                    f"ACESSO CONCEDIDO — {final_user} "
                                    f"| amostras={[f'{s:.1f}' for s in samples]} "
                                    f"| soma={final_confidence:.1f} / {threshold_sum:.0f} "
                                    f"| piscadas={liveness.blink_count}/{BLINK_REQUIRED}"
                                )
                                if door:
                                    door.open_door(final_user)
                                last_door_time = now
                            else:
                                reason = "anti-spoofing" if not liveness.passed else "confiança"
                                state = "denied"
                                log.info(
                                    f"ACESSO NEGADO ({reason}) — {final_user or 'desconhecido'} "
                                    f"| amostras={[f'{s:.1f}' for s in samples]} "
                                    f"| soma={final_confidence:.1f} / {threshold_sum:.0f} "
                                    f"| piscadas={liveness.blink_count}/{BLINK_REQUIRED}"
                                )

                # ── Renderização ──────────────────────────────────────────────
                # display_conf é sempre a soma acumulada (0–300)
                display_conf = sum(samples) if samples else final_confidence
                _draw_overlay(frame, state, samples, display_conf, final_user, liveness.blink_count)

                face_label = ""
                if state in ("granted", "denied") and final_user:
                    face_label = f"{final_user} {final_confidence:.0f}pts"
                detector.draw(frame, regions, face_label)

                cv2.imshow("Controle de Acesso", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Encerrando por comando do usuário.")
                    break

    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        cv2.destroyAllWindows()
        if door:
            door.disconnect()
        log.info("Sistema encerrado.")


if __name__ == "__main__":
    run()
