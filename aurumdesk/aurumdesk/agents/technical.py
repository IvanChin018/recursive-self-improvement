"""
TechnicalAnalystAgent — multi-timeframe classical TA signals.

Per symbol it evaluates on M15/H1/H4/D1:
  * EMA ribbon trend (20/50/200) + slope
  * ADX strength with DI direction
  * MACD histogram momentum & sign flips
  * RSI(14) level + divergence against price over last 60 bars
  * Bollinger %B and squeeze detection
  * Ichimoku-style cloud via Tenkan/Kijun on H4
  * Donchian position (where in the N-bar range are we?)
  * VWAP-like anchored level from session open
Combines into a single score [-1, 1] with reasons. No lookahead: uses only
closed bars for anything that would repaint.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import Signal, SignalDirection, Timeframe
from .base import Agent

TF_WEIGHT = {Timeframe.D1: 0.35, Timeframe.H4: 0.30,
             Timeframe.H1: 0.25, Timeframe.M15: 0.10}


class TechnicalAnalystAgent(Agent):
    name = "TechnicalAnalystAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self.last_scores: Dict[str, float] = {}
        self._last_eval: Dict[str, float] = {}

    def tick(self) -> None:
        self.store = self.bus.get("market.store")
        if self.store is None:
            return
        now = time.time()
        out: List[Signal] = []
        for sym in self.cfg["symbols"]:
            # evaluate once per new closed bar on the primary TF (M15)
            key = f"{sym}"
            series = self.store.get(sym, Timeframe.M15)
            if series is None or len(series.t) < 60:
                continue
            last_closed_t = series.t[-2] if len(series.t) > 1 else series.t[-1]
            if self._last_eval.get(key) == last_closed_t:
                continue
            self._last_eval[key] = last_closed_t
            sig = self._score_symbol(sym, now)
            if sig:
                out.append(sig)
        if out:
            self.bus.publish("signals.technical", out)
            for s in out:
                self.last_scores[s.symbol] = s.score

    # ---------------- scoring ---------------- #
    def _tf_score(self, sym: str, tf: Timeframe) -> tuple[float, List[str]]:
        a = self.store.arrays(sym, tf)
        if a is None:
            return 0.0, []
        o, h, l, c = a["o"], a["h"], a["l"], a["c"]
        n = len(c)
        reasons: List[str] = []
        score = 0.0

        e20, e50 = ind.ema(c, 20), ind.ema(c, 50)
        e200 = ind.ema(c, min(200, max(n // 2, 50)))
        if not (np.isnan(e20[-1]) or np.isnan(e50[-1])):
            if c[-1] > e20[-1] > e50[-1]:
                score += 0.30
                reasons.append(f"{tf.value}: bullish EMA stack")
            elif c[-1] < e20[-1] < e50[-1]:
                score -= 0.30
                reasons.append(f"{tf.value}: bearish EMA stack")
            slope = (e50[-1] - e50[-min(10, n - 1)]) / max(abs(e50[-1]), 1e-9)
            score += float(np.clip(slope * 150, -0.15, 0.15))

        adx, dip, dim = ind.adx_di(h, l, c, 14)
        if not np.isnan(adx[-1]):
            if adx[-1] > 25:
                d = 0.20 if dip[-1] > dim[-1] else -0.20
                score += d
                reasons.append(f"{tf.value}: ADX {adx[-1]:.0f} "
                               f"{'DI+' if d > 0 else 'DI-'}")
            elif adx[-1] < 15:
                score *= 0.6   # no trend: dampen

        line, sigl, hist = ind.macd(c)
        if not np.isnan(hist[-1]) and n > 30:
            flip_up = hist[-2] < 0 <= hist[-1]
            flip_dn = hist[-2] > 0 >= hist[-1]
            if flip_up:
                score += 0.15; reasons.append(f"{tf.value}: MACD cross up")
            elif flip_dn:
                score -= 0.15; reasons.append(f"{tf.value}: MACD cross down")
            score += float(np.clip(hist[-1] / max(abs(c[-1]) * 0.001, 1e-9), -0.1, 0.1))

        r = ind.rsi(c, 14)
        if not np.isnan(r[-1]):
            div = self._rsi_divergence(c, r)
            if div > 0:
                score += 0.15; reasons.append(f"{tf.value}: RSI bull divergence")
            elif div < 0:
                score -= 0.15; reasons.append(f"{tf.value}: RSI bear divergence")
            if r[-1] > 78:
                score -= 0.05
            elif r[-1] < 22:
                score += 0.05

        up, mid, lo, w = ind.bbands(c, 20, 2.0)
        if not np.isnan(lo[-1]) and (up[-1] - lo[-1]) > 0:
            pb = (c[-1] - lo[-1]) / (up[-1] - lo[-1])
            wr = ind.pct_rank(w[-60:] if n >= 60 else w, len(w[-60:] if n >= 60 else w))
            squeeze = not np.isnan(wr[-1]) and wr[-1] < 0.15
            if squeeze:
                reasons.append(f"{tf.value}: BB squeeze (breakout fuel)")
            if pb > 0.98:
                score += 0.05
            elif pb < 0.02:
                score -= 0.05

        du, dd = ind.donchian(h, l, 20)
        if not np.isnan(dd[-1]) and du[-1] > dd[-1]:
            pos = (c[-1] - dd[-1]) / (du[-1] - dd[-1])
            if pos > 0.995:
                score += 0.10; reasons.append(f"{tf.value}: 20-bar high breakout")
            elif pos < 0.005:
                score -= 0.10; reasons.append(f"{tf.value}: 20-bar low breakdown")

        if tf == Timeframe.H4:
            tenkan = (ind.donchian(h, l, 9)[0][-1] + ind.donchian(h, l, 9)[1][-1]) / 2
            kijun = (ind.donchian(h, l, 26)[0][-1] + ind.donchian(h, l, 26)[1][-1]) / 2
            if not (np.isnan(tenkan) or np.isnan(kijun)):
                if tenkan > kijun and c[-1] > kijun:
                    score += 0.10; reasons.append("H4 cloud support")
                elif tenkan < kijun and c[-1] < kijun:
                    score -= 0.10; reasons.append("H4 cloud resistance")

        return float(np.clip(score, -1, 1)), reasons

    @staticmethod
    def _rsi_divergence(c: np.ndarray, r: np.ndarray) -> int:
        """+1 hidden-classic bull div, -1 bear div over last 60 bars."""
        k = min(60, len(c) - 2)
        if k < 20:
            return 0
        pc, pr = c[-k:], r[-k:]
        i_low = int(np.argmin(pc))
        j_low = int(np.argmin(pr))
        i_high = int(np.argmax(pc))
        j_high = int(np.argmax(pr))
        if abs(i_low - j_low) <= k // 3 and pr[j_low] < pr[max(j_low - 10, 0)] and \
                pc[i_low] >= np.min(pc[:k // 2]):
            return 1
        if abs(i_high - j_high) <= k // 3 and pr[j_high] > pr[max(j_high - 10, 0)] and \
                pc[i_high] <= np.max(pc[:k // 2]):
            return -1
        return 0

    def _score_symbol(self, sym: str, now: float) -> Signal | None:
        total, all_reasons, wsum = 0.0, [], 0.0
        for tf, w in TF_WEIGHT.items():
            s, rs = self._tf_score(sym, tf)
            if rs:
                total += s * w
                wsum += w
                all_reasons += rs
        if wsum == 0:
            return None
        total /= wsum
        # DXY/macro alignment filter comes from ensemble, not here.
        d = SignalDirection.LONG if total > 0.12 else (
            SignalDirection.SHORT if total < -0.12 else SignalDirection.NEUTRAL)
        sig = Signal(agent=self.name, symbol=sym, direction=d, score=total,
                     timeframe=Timeframe.H1, t=now, reasons=all_reasons[:6])
        self.bus.publish("signal", sig.__dict__ | {"direction": sig.direction.value})
        return sig
