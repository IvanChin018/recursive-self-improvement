"""
Market store: in-memory OHLCV per (symbol, timeframe) with strict guards.

Guards against the classic silent killers:
  * duplicate/overlapping bars      -> deduped by candle open time
  * out-of-order history            -> sorted insert
  * NaN / non-positive prices       -> rejected
  * stale data                      -> age check for consumers
  * unbounded memory                -> ring-buffer via candles_keep
"""
from __future__ import annotations

import bisect
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models import Candle, Timeframe


class Series:
    __slots__ = ("t", "o", "h", "l", "c", "v")

    def __init__(self):
        self.t: List[int] = []
        self.o: List[float] = []
        self.h: List[float] = []
        self.l: List[float] = []
        self.c: List[float] = []
        self.v: List[float] = []

    def arrays(self) -> Dict[str, np.ndarray]:
        return dict(t=np.asarray(self.t, float), o=np.asarray(self.o, float),
                    h=np.asarray(self.h, float), l=np.asarray(self.l, float),
                    c=np.asarray(self.c, float), v=np.asarray(self.v, float))

    def last_closed(self) -> Optional[Candle]:
        if len(self.t) < 2:
            return None
        i = -2
        return Candle(symbol="", timeframe=Timeframe.M1, t=self.t[i], o=self.o[i],
                      h=self.h[i], l=self.l[i], c=self.c[i], volume=self.v[i])


class MarketStore:
    def __init__(self, keep: Dict[str, int]):
        self._lock = threading.RLock()
        self._data: Dict[Tuple[str, Timeframe], Series] = {}
        self._keep = keep
        self.last_update: Dict[Tuple[str, Timeframe], float] = {}

    def _series(self, symbol: str, tf: Timeframe) -> Series:
        key = (symbol, tf)
        if key not in self._data:
            self._data[key] = Series()
        return self._data[key]

    # ---------------- ingest ---------------- #
    def update_history(self, symbol: str, tf: Timeframe, candles: List[Candle]) -> int:
        """Bulk set from broker copy_rates; returns accepted count."""
        accepted = 0
        with self._lock:
            s = self._series(symbol, tf)
            existing = set(s.t)
            for cd in candles:
                if not self._valid(cd):
                    continue
                if cd.t in existing:
                    self._overwrite(s, cd)
                    continue
                idx = bisect.bisect_left(s.t, cd.t)
                s.t.insert(idx, cd.t); s.o.insert(idx, cd.o); s.h.insert(idx, cd.h)
                s.l.insert(idx, cd.l); s.c.insert(idx, cd.c); s.v.insert(idx, cd.volume)
                existing.add(cd.t)
                accepted += 1
            self._trim(s, tf)
            self.last_update[(symbol, tf)] = time.time()
        return accepted

    def update_live_bar(self, cd: Candle) -> bool:
        """Upsert the currently-forming bar (last row)."""
        if not self._valid(cd):
            return False
        with self._lock:
            s = self._series(cd.symbol, cd.timeframe)
            if s.t and s.t[-1] == cd.t:
                s.o[-1], s.h[-1], s.l[-1], s.c[-1], s.v[-1] = \
                    cd.o, max(s.h[-1], cd.h), min(s.l[-1], cd.l), cd.c, cd.volume
            else:
                if s.t and cd.t < s.t[-1]:
                    return False          # late tick for older bar — ignore
                s.t.append(cd.t); s.o.append(cd.o); s.h.append(cd.h)
                s.l.append(cd.l); s.c.append(cd.c); s.v.append(cd.volume)
            self._trim(s, cd.timeframe)
            self.last_update[(cd.symbol, cd.timeframe)] = time.time()
        return True

    @staticmethod
    def _valid(cd: Candle) -> bool:
        try:
            vals = [cd.o, cd.h, cd.l, cd.c]
        except Exception:
            return False
        if any(v is None or not np.isfinite(v) or v <= 0 for v in vals):
            return False
        if cd.h < cd.l or cd.h < cd.o - 1e-9 or cd.h < cd.c - 1e-9 \
           or cd.l > cd.o + 1e-9 or cd.l > cd.c + 1e-9:
            return False
        return True

    def _overwrite(self, s: Series, cd: Candle) -> None:
        i = s.t.index(cd.t)
        s.o[i], s.h[i], s.l[i], s.c[i], s.v[i] = cd.o, cd.h, cd.l, cd.c, cd.volume

    def _trim(self, s: Series, tf: Timeframe) -> None:
        keep = int(self._keep.get(tf.value, 500))
        for arr in (s.t, s.o, s.h, s.l, s.c, s.v):
            del arr[:-keep]

    # ---------------- reads ---------------- #
    def get(self, symbol: str, tf: Timeframe) -> Optional[Series]:
        with self._lock:
            return self._data.get((symbol, tf))

    def arrays(self, symbol: str, tf: Timeframe) -> Optional[Dict[str, np.ndarray]]:
        with self._lock:
            s = self._data.get((symbol, tf))
            return s.arrays() if s and len(s.t) >= 30 else None

    def n_bars(self, symbol: str, tf: Timeframe) -> int:
        with self._lock:
            s = self._data.get((symbol, tf))
            return len(s.t) if s else 0

    def age_sec(self, symbol: str, tf: Timeframe) -> float:
        with self._lock:
            lu = self.last_update.get((symbol, tf))
            return (time.time() - lu) if lu else 1e9

    def fresh(self, symbol: str, tf: Timeframe, stale_after: float) -> bool:
        return self.age_sec(symbol, tf) < stale_after
