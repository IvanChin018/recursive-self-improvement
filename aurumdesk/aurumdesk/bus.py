"""
Bus: tiny thread-safe pub/sub + shared blackboard.

Agents never call each other directly; they publish typed messages and read
the blackboard. That keeps the desk decoupled, replayable and debuggable.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List


class Bus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: Dict[str, List[Callable[[str, Any], None]]] = defaultdict(list)
        self._board: Dict[str, Any] = {}
        self._history: Deque = deque(maxlen=5000)
        self._stale_ms: Dict[str, float] = {}

    # ---------------- pub/sub ---------------- #
    def subscribe(self, topic: str, cb: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._subs[topic].append(cb)

    def publish(self, topic: str, payload: Any) -> None:
        with self._lock:
            self._history.append((time.time(), topic, payload))
            cbs = list(self._subs.get(topic, []))
        for cb in cbs:
            try:
                cb(topic, payload)
            except Exception as exc:  # a subscriber bug must never kill the desk
                import logging
                logging.getLogger("aurumdesk.bus").exception(
                    "subscriber error on %s: %s", topic, exc)

    # ---------------- blackboard ---------------- #
    def set(self, key: str, value: Any, max_age_sec: float | None = None) -> None:
        with self._lock:
            self._board[key] = value
            if max_age_sec is not None:
                self._stale_ms[key] = time.time() + max_age_sec
            else:
                self._stale_ms.pop(key, None)

    def get(self, key: str, default: Any = None, allow_stale: bool = False) -> Any:
        with self._lock:
            exp = self._stale_ms.get(key)
            if not allow_stale and exp is not None and time.time() > exp:
                return default
            return self._board.get(key, default)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._board.keys())

    def recent(self, n: int = 50) -> List:
        with self._lock:
            return list(self._history)[-n:]
