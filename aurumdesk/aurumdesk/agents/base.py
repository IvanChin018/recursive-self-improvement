"""Base class for every desk agent: lifecycle, heartbeat, guarded tick."""
from __future__ import annotations

import threading
import time
from typing import Optional

from ..bus import Bus
from ..utils.logging import get_logger


class Agent:
    """Agents can be either:
      - thread-driven (run_forever loop calling tick() every interval), or
      - scheduler-driven (desk calls tick() in the main loop).
    The desk uses the scheduler style for determinism; threads stay idle.
    """

    name = "Agent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        self.bus = bus
        self.cfg = cfg
        self.log = get_logger(self.name.lower(), log_dir)
        self._stop = threading.Event()
        self.last_tick_t: float = 0.0
        self.ticks: int = 0
        self._born_t: float = time.time()
        self.errors: int = 0
        self.last_error: str = ""

    # ---------------- lifecycle ---------------- #
    def setup(self) -> None:      # override
        pass

    def tick(self) -> None:       # override — must be exception-safe-ish
        raise NotImplementedError

    def safe_tick(self) -> bool:
        try:
            self.tick()
            self.last_tick_t = time.time()
            self.ticks += 1
            return True
        except Exception as exc:
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.log.exception("tick failed: %s", exc)
            self.bus.publish("alert", {"level": "error", "agent": self.name,
                                       "msg": self.last_error})
            return False

    def stop(self) -> None:
        self._stop.set()

    def start_thread(self, interval_sec: float) -> Optional[threading.Thread]:
        if interval_sec <= 0:
            return None

        def _loop() -> None:
            while not self._stop.is_set():
                self.safe_tick()
                self._stop.wait(interval_sec)

        th = threading.Thread(target=_loop, name=self.name, daemon=True)
        th.start()
        return th

    # ---------------- helpers ---------------- #
    def heartbeat_ok(self, timeout_sec: float) -> bool:
        return (time.time() - self.last_tick_t) < max(timeout_sec, 1.0)
