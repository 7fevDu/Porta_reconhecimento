"""
Controle da porta física via comunicação serial com Arduino.

Protocolo simples:
  - Envia b'1' → Arduino abre a fechadura pelo tempo configurado
  - Envia b'0' → Arduino fecha a fechadura imediatamente (fallback)

O Arduino deve fechar automaticamente após DOOR_OPEN_SECONDS; o comando b'0'
existe apenas como segurança adicional no shutdown do sistema.
"""
import logging
import time
import threading

import serial
from serial import SerialException

from src.config import SERIAL_PORT, SERIAL_BAUDRATE, DOOR_OPEN_SECONDS

log = logging.getLogger(__name__)


class DoorController:
    def __init__(
        self,
        port: str = SERIAL_PORT,
        baudrate: int = SERIAL_BAUDRATE,
        open_seconds: float = DOOR_OPEN_SECONDS,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._open_seconds = open_seconds
        self._serial: serial.Serial | None = None
        self._lock = threading.Lock()

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Abre a conexão serial com o Arduino."""
        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=1,
            )
            # Arduino reinicia ao abrir serial; aguarda estabilização
            time.sleep(2)
            log.info(f"Arduino conectado em {self._port} @ {self._baudrate} baud.")
        except SerialException as exc:
            raise RuntimeError(
                f"Não foi possível conectar ao Arduino em {self._port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Fecha a conexão serial."""
        with self._lock:
            if self._serial and self._serial.is_open:
                self._send(b"0")
                self._serial.close()
                log.info("Conexão serial encerrada.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ── API pública ───────────────────────────────────────────────────────────

    def open_door(self, user: str = "") -> None:
        """
        Envia sinal de abertura ao Arduino e agenda o fechamento automático
        em uma thread separada para não bloquear o loop principal.
        """
        label = f" ({user})" if user else ""
        log.info(f"Acesso concedido{label} — abrindo porta por {self._open_seconds}s.")
        with self._lock:
            self._send(b"1")
        threading.Thread(target=self._auto_close, daemon=True).start()

    # ── Internos ──────────────────────────────────────────────────────────────

    def _send(self, command: bytes) -> None:
        if self._serial is None or not self._serial.is_open:
            log.warning("Tentativa de envio sem conexão serial ativa.")
            return
        try:
            self._serial.write(command)
            self._serial.flush()
        except SerialException as exc:
            log.error(f"Erro ao enviar comando serial: {exc}")

    def _auto_close(self) -> None:
        time.sleep(self._open_seconds)
        with self._lock:
            self._send(b"0")
        log.info("Porta fechada automaticamente.")
