"""
SentimentRegimeAgent — tells the desk WHAT game is being played.

Regime classification per symbol (H1/H4):
  trend_up / trend_down : directional strategies get full weight
  chop                  : mean-reversion weighted, breakout needs confirmation
  volatile_breakout     : vol expansion -> momentum/breakout + wider stops

Tools: Hurst exponent, ADX, realized-vol percentile, BB width percentile,
range compression (NR7), session-liquidity map, and cross-market sanity
(XAU/XAG ratio shift = risk appetite tell).
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import RegimeInfo, Timeframe
from .base import Agent
from ..utils.timeutils import session_of, SESSION_LIQUIDITY


class SentimentRegimeAgent(Agent):
    name = "SentimentRegimeAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self.regimes: Dict[str, RegimeInfo] = {}
        self._last_eval: Dict[str, int] = {}

    def tick(self) -> None:
        self.store = self.bus.get("market.store")
        if self.store is None:
            return
        now = time.time()
        for sym in self.cfg["symbols"]:
            s = self.store.get(sym, Timeframe.H1)
            if s is None or len(s.t) < 120:
                continue
            closed = s.t[-2]
            if self._last_eval.get(sym) == closed:
                continue
            self._last_eval[sym] = closed
            self.regimes[sym] = self._classify(sym, now)
        # market-wide sentiment read via gold/silver ratio
        self._ratio_sentiment(now)
        self.bus.set("regime.map", self.regimes, max_age_sec=3600)

    # ---------------- classify ---------------- #
    def _classify(self, sym: str, now: float) -> RegimeInfo:
        a = self.store.arrays(sym, Timeframe.H1)
        if a is None:
            return RegimeInfo()
        o, h, l, c = a["o"], a["h"], a["l"], a["c"]
        atr_fast = ind.atr(h, l, c, 7)
        atr_slow = ind.atr(h, l, c, 60)
        vol_mult = float(atr_fast[-1] / atr_slow[-1]) if atr_slow[-1] > 0 else 1.0
        adx, _, _ = ind.adx_di(h, l, c, 14)
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        hurst = ind.hurst(c[-300:])
        _, _, _, bbw = ind.bbands(c, 20, 2.0)
        bbw_rank = float(np.nanmean(bbw[-20:] < bbw[-1])) if not np.isnan(bbw[-1]) else 0.5

        trend_dir = 0.0
        if not (np.isnan(e50[-1]) or np.isnan(e200[-1])):
            gap = (e50[-1] - e200[-1]) / max(e200[-1], 1e-9)
            trend_dir = float(np.clip(gap * 120, -1, 1))
        strength = float(np.clip((adx[-1] - 15) / 25, 0, 1)) if not np.isnan(adx[-1]) else 0.0

        if vol_mult > 1.6 and strength > 0.5:
            regime, conf = "volatile_breakout", min(0.9, vol_mult / 2.2)
        elif strength > 0.45 and abs(trend_dir) > 0.25 and hurst > 0.52:
            regime, conf = ("trend_up" if trend_dir > 0 else "trend_down"), \
                min(0.95, 0.5 + strength / 2)
        elif hurst < 0.45 or (strength < 0.25 and bbw_rank < 0.4):
            regime, conf = "chop", 0.6 + 0.3 * (0.5 - hurst)
        else:
            regime, conf = "chop", 0.45

        info = RegimeInfo(regime=regime, vol_mult=vol_mult,
                          confidence=float(np.clip(conf, 0, 1)), t=now)
        prev = self.regimes.get(sym)
        if prev and prev.regime != regime:
            self.log.info("%s regime: %s -> %s (vol x%.2f)", sym,
                          prev.regime, regime, vol_mult)
            self.bus.publish("regime.changed", {"symbol": sym, "new": regime})
        return info

    # ---------------- XAU/XAG ratio ---------------- #
    def _ratio_sentiment(self, now: float) -> None:
        ga = self.store.arrays("XAGUSD", Timeframe.H1) if self.store else None
        au = self.store.arrays("XAUUSD", Timeframe.H1) if self.store else None
        if ga is None or au is None:
            return
        ratio = au["c"] / np.maximum(ga["c"], 1e-9)
        if len(ratio) < 60:
            return
        rank = float(np.mean(ratio[-60:] < ratio[-1]))
        z = (ratio[-1] - np.mean(ratio[-60:])) / max(np.std(ratio[-60:]), 1e-9)
        risk_appetite = -z          # falling ratio = silver outperforming = risk-on
        self.bus.set("sentiment.risk_appetite",
                     float(np.clip(risk_appetite, -3, 3)), max_age_sec=3600)
        self.bus.set("sentiment.gold_silver_ratio", float(ratio[-1]),
                     max_age_sec=3600)

    def liquidity_score(self, now: float) -> float:
        return SESSION_LIQUIDITY.get(session_of(now), 0.5)
