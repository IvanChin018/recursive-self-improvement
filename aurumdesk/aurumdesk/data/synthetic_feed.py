"""
Synthetic market simulator — realistic gold/silver microstructure.

Purpose: (1) let the whole desk be tested end-to-end anywhere (no broker),
(2) act as a failover feed if MT5 drops, so agents never starve for data.
It is clearly labelled in every log line; it is NOT a substitute for real
market data when trading real money.

Model per symbol:
  * cross-rate consistency: XAUUSD = XAUAUD / AUDUSD etc. (arbitrage-free-ish)
  * GBM with stochastic vol (vol clustering), occasional news shocks
  * intraday session volatility profile (London/NY overlap hottest)
  * spread widening around simulated red-folder events
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

from ..models import Candle, Quote, Timeframe, TIMEFRAME_SECONDS

BASE_PARAMS = {
    # start price, annualized vol, avg spread (in price units)
    "XAUUSD": dict(p0=4180.0, vol=0.16, spread=0.35),
    "XAGUSD": dict(p0=52.40, vol=0.28, spread=0.020),
    "AUDUSD": dict(p0=0.6550, vol=0.10, spread=0.00012),
    "EURUSD": dict(p0=1.0780, vol=0.09, spread=0.00010),
}
CROSS_DEFS = {
    "XAUAUD": ("XAUUSD", "AUDUSD"), "XAUEUR": ("XAUUSD", "EURUSD"),
    "XAGAUD": ("XAGUSD", "AUDUSD"), "XAGEUR": ("XAGUSD", "EURUSD"),
}

SESSION_VOL_MULT = {   # hour-of-day UTC -> vol multiplier (gold's day)
    **{h: 0.55 for h in range(0, 6)},
    **{h: 0.9 for h in range(6, 8)},
    **{h: 1.25 for h in range(8, 12)},
    **{h: 1.7 for h in range(12, 16)},
    **{h: 1.35 for h in range(16, 21)},
    **{h: 0.7 for h in range(21, 24)},
}


class SyntheticFeed:
    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self.t0 = int(time.time())
        self.params = {k: dict(v) for k, v in BASE_PARAMS.items()}
        for s in BASE_PARAMS:
            self.params[s]["p"] = self.params[s]["p0"]
            self.params[s]["v"] = self.params[s]["vol"] / math.sqrt(365 * 24 * 60)
            self.params[s]["vo"] = 1.0
        self.events: List[dict] = []      # simulated red-folder events
        self._next_event_gap = self.rng.uniform(20, 90) * 60
        self._last_event_check = self.t0

    # ---------------- helpers ---------------- #
    def _session_mult(self, t: float) -> float:
        return SESSION_VOL_MULT.get(int((t // 3600) % 24), 1.0)

    def _step_base(self, sym: str, dt_sec: float, shock: float) -> None:
        pr = self.params[sym]
        # OU mean-reverting vol of vol (clustering)
        pr["vo"] += 0.02 * (1.0 - pr["vo"]) + 0.06 * self.rng.gauss(0, 1)
        pr["vo"] = max(0.35, min(pr["vo"], 3.2))
        sigma = pr["v"] * pr["vo"] * self._session_mult(pr.get("t", self.t0))
        g = self.rng.gauss(0, 1) + shock
        r = -0.5 * sigma ** 2 + sigma * g
        pr["p"] *= math.exp(r * math.sqrt(max(dt_sec, 1)))
        pr["p"] = max(pr["p"], 1e-3)
        pr["t"] = self.t0

    def price(self, sym: str) -> float:
        if sym in CROSS_DEFS:
            num, den = CROSS_DEFS[sym]
            return self.params[num]["p"] / self.params[den]["p"]
        return self.params[sym]["p"]

    # ---------------- public api ---------------- #
    def advance_to(self, now: float) -> None:
        """Move the simulation from last update to `now` (epoch seconds)."""
        for sym in ("XAUUSD", "XAGUSD", "AUDUSD", "EURUSD"):
            self._step_base(sym, 60.0, 0.0)
        self.t0 = int(now)
        for sym in self.params:
            self.params[sym]["t"] = self.t0

    def maybe_event(self, now: float) -> Optional[dict]:
        if now - self._last_event_check > self._next_event_gap:
            self._last_event_check = now
            self._next_event_gap = self.rng.uniform(20, 90) * 60
            ev = {"t": now + 600, "impact": self.rng.choice([-1, 1]) *
                  self.rng.uniform(0.5, 2.5), "kind": "sim-red-event"}
            self.events.append(ev)
            return ev
        return None

    def apply_shocks(self, now: float) -> float:
        shock = 0.0
        for ev in self.events:
            if abs(now - ev["t"]) < 120:
                shock += ev["impact"] * 0.0009
        return shock

    def quote(self, sym: str) -> Optional[Quote]:
        if sym not in self.params and sym not in CROSS_DEFS:
            return None
        mid = self.price(sym)
        spr = self.params.get(CROSS_DEFS.get(sym, (sym,))[0], {}).get("spread", 0.001)
        return Quote(symbol=sym, bid=mid - spr / 2, ask=mid + spr / 2, t=time.time())

    def candles(self, sym: str, tf: Timeframe, count: int) -> List[Candle]:
        """Generate plausible OHLC history consistent with current price."""
        if sym not in self.params and sym not in CROSS_DEFS:
            return []
        out: List[Candle] = []
        step = TIMEFRAME_SECONDS[tf]
        p_now = self.price(sym)
        base = self.params[CROSS_DEFS.get(sym, (sym, ""))[0]] if sym in CROSS_DEFS else self.params[sym]
        sig = base["vol"] / math.sqrt(365 * 24 * 3600 / step)
        # walk backwards from now
        path = [p_now]
        for i in range(count):
            m = self._session_mult(self.t0 - i * step)
            path.append(path[-1] / math.exp(self.rng.gauss(0, sig * m)))
        path.reverse()
        for i in range(len(path) - 1):
            o, c = path[i], path[i + 1]
            wick = abs(self.rng.gauss(0, sig * o)) * 0.7
            h = max(o, c) + wick * self.rng.random()
            l = min(o, c) - wick * self.rng.random()
            t_open = int((self.t0 - (len(path) - 2 - i) * step) // step * step)
            out.append(Candle(symbol=sym, timeframe=tf, t=t_open,
                              o=o, h=h, l=max(l, 1e-3), c=c,
                              volume=self.rng.uniform(500, 2500)))
        return out
