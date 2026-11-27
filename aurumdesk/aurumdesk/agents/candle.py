"""
CandlePatternAgent — candlestick & bar-microstructure reading.

The user asked: "the chart is always moving 24/7, trade consistently
depending on the candlesticks." This agent is the dedicated candle reader.

Patterns detected on closed candles (M5/M15/H1), each with position context
(a pattern in the middle of nowhere is noise; at a level it's information):
  * Pin bars / hammers / shooting stars   (wick rejection)
  * Engulfing (bull/bear)                 (momentum transfer)
  * Inside bar breakout setup             (compression)
  * Doji at extremes                      (indecision + level)
  * Morning/evening star (3-bar)
  * Three white soldiers / black crows
  * Outside bars / fakey-liquidity sweeps (stop-hunt reclaim)
Confidence scales with: pattern size vs ATR, location vs S/R or EMA,
session quality, and agreement across timeframes.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import Signal, SignalDirection, Timeframe
from .base import Agent


def _anatomy(o, h, l, c):
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    up = h - max(o, c)
    lo = min(o, c) - l
    return rng, body, up, lo


class CandlePatternAgent(Agent):
    name = "CandlePatternAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self._seen: Dict[str, int] = {}

    def tick(self) -> None:
        self.store = self.bus.get("market.store")
        if self.store is None:
            return
        now = time.time()
        signals: List[Signal] = []
        for sym in self.cfg["symbols"]:
            for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
                s = self.store.get(sym, tf)
                if s is None or len(s.t) < 60:
                    continue
                last_closed = s.t[-2] if len(s.t) > 1 else s.t[-1]
                key = f"{sym}:{tf.value}"
                if self._seen.get(key) == last_closed:
                    continue
                self._seen[key] = last_closed
                sig = self._evaluate(sym, tf, last_closed, now)
                if sig:
                    signals.append(sig)
        if signals:
            self.bus.publish("signals.candle", signals)

    # ---------------- evaluation ---------------- #
    def _evaluate(self, sym: str, tf: Timeframe, t_closed: int, now: float) -> Signal | None:
        a = self.store.arrays(sym, tf)
        if a is None:
            return None
        o, h, l, c, v = a["o"], a["h"], a["l"], a["c"], a["v"]
        atr = ind.atr(h, l, c, 14)
        if np.isnan(atr[-1]) or atr[-1] <= 0:
            return None
        i = len(c) - 2                       # last CLOSED bar
        score, reasons = 0.0, []

        # ---- context: where are we? ----
        e50 = ind.ema(c, 50)
        hh, ll = ind.donchian(h, l, 20)
        near_high = not np.isnan(hh[i]) and c[i] > hh[i] - 0.5 * atr[i]
        near_low = not np.isnan(ll[i]) and c[i] < ll[i] + 0.5 * atr[i]
        above_ema = c[i] > e50[i] if not np.isnan(e50[i]) else None

        patterns = [
            self._pin_bar(o, h, l, c, i, atr),
            self._engulfing(o, c, i, atr),
            self._inside_bar(o, h, l, c, i, atr),
            self._morning_evening(o, c, i, atr),
            self._three_soldiers_crows(c, i, atr),
            self._liquidity_sweep(h, l, c, i, atr),
            self._doji_extreme(o, h, l, c, i, atr, near_high, near_low),
        ]
        for p_score, p_name in patterns:
            if abs(p_score) >= 0.05:
                score += p_score
                reasons.append(f"{tf.value}:{p_name}")

        # position context boosts
        if score != 0.0:
            if score > 0 and near_high:
                score *= 1.15; reasons.append("at range high (continuation)")
            if score < 0 and near_low:
                score *= 1.15; reasons.append("at range low (continuation)")
            if score > 0 and near_low:
                score *= 1.25; reasons.append("rejection at lows (reversal)")
            if score < 0 and near_high:
                score *= 1.25; reasons.append("rejection at highs (reversal)")
            # volume confirmation (tick volume proxy)
            vm = np.mean(v[max(i - 20, 0):i]) if i > 5 else 0
            if vm > 0 and v[i] > 1.3 * vm:
                score *= 1.10; reasons.append("volume surge confirm")

        if abs(score) < 0.18:
            return None
        d = SignalDirection.LONG if score > 0 else SignalDirection.SHORT
        sig = Signal(agent=self.name, symbol=sym, direction=d,
                     score=float(np.clip(score, -1, 1)), timeframe=tf, t=now,
                     reasons=reasons)
        self.bus.publish("signal", {"agent": self.name, "symbol": sym,
                                    "direction": d.value, "score": sig.score,
                                    "tf": tf.value, "reasons": reasons})
        return sig

    # ---------------- pattern detectors ---------------- #
    @staticmethod
    def _pin_bar(o, h, l, c, i, atr) -> Tuple[float, str]:
        rng, body, up, lo = _anatomy(o[i], h[i], l[i], c[i])
        if rng < 0.7 * atr[i]:
            return 0.0, ""
        if lo > 2.0 * body and lo > 0.55 * rng and up < 0.3 * rng:
            return 0.45, "bull pin/hammer"
        if up > 2.0 * body and up > 0.55 * rng and lo < 0.3 * rng:
            return -0.45, "bear pin/shooting star"
        return 0.0, ""

    @staticmethod
    def _engulfing(o, c, i, atr) -> Tuple[float, str]:
        if i < 1:
            return 0.0, ""
        b0, b1 = c[i - 1] - o[i - 1], c[i] - o[i]
        if b0 < 0 and b1 > 0 and c[i] >= o[i - 1] and o[i] <= c[i - 1] \
           and abs(b1) > 1.1 * abs(b0):
            return 0.40, "bullish engulfing"
        if b0 > 0 and b1 < 0 and c[i] <= o[i - 1] and o[i] >= c[i - 1] \
           and abs(b1) > 1.1 * abs(b0):
            return -0.40, "bearish engulfing"
        return 0.0, ""

    @staticmethod
    def _inside_bar(o, h, l, c, i, atr) -> Tuple[float, str]:
        if i < 1:
            return 0.0, ""
        if h[i] < h[i - 1] and l[i] > l[i - 1] and (h[i - 1] - l[i - 1]) > atr[i - 1]:
            # inside bar after trend bar -> directional bias from mother bar
            mom = c[i - 1] - o[i - 1]
            return (0.20 if mom > 0 else -0.20), "inside bar coil"
        return 0.0, ""

    @staticmethod
    def _morning_evening(o, c, i, atr) -> Tuple[float, str]:
        if i < 2:
            return 0.0, ""
        b = [abs(c[j] - o[j]) for j in (i - 2, i - 1, i)]
        if b[1] < 0.35 * atr[i - 1]:
            if c[i - 2] < o[i - 2] and b[0] > 0.8 * atr[i - 2] and c[i] > o[i] \
               and b[2] > 0.8 * atr[i] and c[i] > (o[i - 2] + c[i - 2]) / 2:
                return 0.50, "morning star"
            if c[i - 2] > o[i - 2] and b[0] > 0.8 * atr[i - 2] and c[i] < o[i] \
               and b[2] > 0.8 * atr[i] and c[i] < (o[i - 2] + c[i - 2]) / 2:
                return -0.50, "evening star"
        return 0.0, ""

    @staticmethod
    def _three_soldiers_crows(c, i, atr) -> Tuple[float, str]:
        if i < 3:
            return 0.0, ""
        inc = all(c[i - k] > c[i - k - 1] and (c[i - k] - c[i - k - 1]) > 0.3 * atr[i]
                  for k in range(3))
        dec = all(c[i - k] < c[i - k - 1] and (c[i - k - 1] - c[i - k]) > 0.3 * atr[i]
                  for k in range(3))
        if inc:
            return 0.35, "three white soldiers"
        if dec:
            return -0.35, "three black crows"
        return 0.0, ""

    @staticmethod
    def _liquidity_sweep(h, l, c, i, atr) -> Tuple[float, str]:
        """Close reclaims prior N-bar extreme after wick beyond it = stop hunt."""
        if i < 10:
            return 0.0, ""
        ph = max(h[i - 10:i]); pl = min(l[i - 10:i])
        if h[i] > ph and c[i] < ph and (h[i] - c[i]) > 0.4 * atr[i]:
            return -0.45, "high sweep failed (bear trap)"
        if l[i] < pl and c[i] > pl and (c[i] - l[i]) > 0.4 * atr[i]:
            return 0.45, "low sweep reclaimed (bull trap flip)"
        return 0.0, ""

    @staticmethod
    def _doji_extreme(o, h, l, c, i, atr, near_high, near_low) -> Tuple[float, str]:
        rng, body, up, lo = _anatomy(o[i], h[i], l[i], c[i])
        if body > 0.1 * rng or rng < 0.5 * atr[i]:
            return 0.0, ""
        if near_low and lo > 0.5 * rng:
            return 0.25, "dragonfly doji at lows"
        if near_high and up > 0.5 * rng:
            return -0.25, "gravestone doji at highs"
        return 0.0, ""
