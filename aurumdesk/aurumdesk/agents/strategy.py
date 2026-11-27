"""
StrategyEnsembleAgent — the head strategist / desk CIO.

Takes every analyst output + strategy votes, applies regime-dependent
weighting and adaptive performance weights (each strategy's live track
record), then produces TradeProposals with stop/TP placement logic.

Confidence model:
    raw = Σ (vote_direction * vote_conviction * base_weight * adapt_weight)
          / Σ possible_weights
    conf = |raw| * agreement_bonus * freshness * macro_alignment

Exit plan per proposal (handed to risk/exec):
    SL  = max(atr_mult*ATR, structure level +/- buffer)
    TP  = rr * risk, plus partial take-profit & trailing rules
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import (Side, Signal, SignalDirection, StrategyVote, Timeframe,
                      TradeProposal)
from ..strategies.library import (Breakout, CandleEdge, CarryMacro,
                                  MeanReversion, PairsRelative, SessionMomentum,
                                  TrendFollowing)
from .base import Agent

STRATEGIES = [TrendFollowing(), Breakout(), MeanReversion(), CandleEdge(),
              CarryMacro(), PairsRelative(), SessionMomentum()]

REGIME_BASE_WEIGHT = {
    "trend":        {"trend_up": 1.4, "trend_down": 1.4, "chop": 0.3, "volatile_breakout": 1.0},
    "breakout":     {"trend_up": 1.0, "trend_down": 1.0, "chop": 0.8, "volatile_breakout": 1.6},
    "meanrev":      {"trend_up": 0.2, "trend_down": 0.2, "chop": 1.5, "volatile_breakout": 0.1},
    "candlestick":  {"trend_up": 0.9, "trend_down": 0.9, "chop": 1.2, "volatile_breakout": 1.1},
    "carry_macro":  {"trend_up": 1.0, "trend_down": 1.0, "chop": 0.7, "volatile_breakout": 0.8},
    "pairs_relative": {"trend_up": 0.7, "trend_down": 0.7, "chop": 1.1, "volatile_breakout": 0.6},
    "session_momentum": {"trend_up": 1.0, "trend_down": 1.0, "chop": 0.5, "volatile_breakout": 1.3},
    "ml_forecast":  {"trend_up": 1.0, "trend_down": 1.0, "chop": 0.9, "volatile_breakout": 1.0},
}


class _Ctx:
    """Read-only bundle handed to strategies."""
    def __init__(self, now, macro, candle_signals, fx_momentum):
        self.now = now
        self.macro = macro
        self.candle_signals = candle_signals
        self.fx_momentum = fx_momentum


class StrategyEnsembleAgent(Agent):
    name = "StrategyEnsembleAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self.weights: Dict[str, float] = {s.name: cfg["strategies"].get("weights_init", 1.0)
                                         for s in STRATEGIES}
        self.weights["ml_forecast"] = cfg["strategies"].get("weights_init", 1.0)
        self.perf: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=120))
        self._candle_sigs: Dict[str, List[Signal]] = defaultdict(list)
        self._last_bar: Dict[str, int] = {}
        self._open_refs: Dict[str, dict] = {}
        self.state_file = os.path.join(cfg.get("overseer", {}).get("state_dir", "state"),
                                       "ensemble_state.json")
        self._load()
        bus.subscribe("signals.candle", self._on_candle)

    def _on_candle(self, topic, payload) -> None:
        now = time.time()
        for sig in payload:
            lst = self._candle_sigs[sig.symbol]
            lst[:] = [s for s in lst if now - s.t < 4 * 900]
            lst.append(sig)

    def _load(self) -> None:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                self.weights.update(d.get("weights", {}))
                for k, v in d.get("perf", {}).items():
                    self.perf[k].extend(v)
        except Exception:
            pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as fh:
                json.dump(dict(weights=self.weights,
                               perf={k: list(v)[-60:] for k, v in self.perf.items()}), fh)
        except Exception:
            pass

    # ---------------- feedback loop ---------------- #
    def record_outcome(self, strategy: str, r_multiple: float) -> None:
        """Called by Journal when a tagged trade closes."""
        self.perf[strategy].append(float(np.clip(r_multiple, -3, 3)))
        self._recompute_weight(strategy)

    def _recompute_weight(self, name: str) -> None:
        hist = list(self.perf[name])[-40:]
        if len(hist) < 8:
            return
        ewma_r = _ewma(hist, halflife=12)
        wr = sum(h > 0 for h in hist) / len(hist)
        w = 0.5 + 1.8 * ewma_r + 0.8 * (wr - 0.5)
        self.weights[name] = float(np.clip(w, 0.15, 2.2))

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        self.store = self.bus.get("market.store")
        if self.store is None:
            return
        now = time.time()
        macro = self.bus.get("macro.context")
        regimes = self.bus.get("regime.map", {})
        ml_votes = self.bus.get("ml.votes", {})
        quotes = self.bus.get("market.quotes", {})
        stale = set(self.bus.get("market.stale_symbols", []))
        halted = self.bus.get("risk.halted", False)
        if halted:
            return
        fx_mom = {}
        for q in ("AUD", "EUR"):
            a = self.store.arrays(f"{q}USD", Timeframe.H1)
            if a is not None and len(a["c"]) > 2:
                fx_mom[q] = float(a["c"][-1] / a["c"][-2] - 1.0)
        ctx = _Ctx(now, macro, self._candle_sigs, fx_mom)

        proposals: List[TradeProposal] = []
        for sym in self.cfg["symbols"]:
            if sym in stale:
                continue
            series = self.store.get(sym, Timeframe.M15)
            if series is None or len(series.t) < 60:
                continue
            closed = series.t[-2]
            if self._last_bar.get(sym) == closed:
                continue
            self._last_bar[sym] = closed

            regime = getattr(regimes.get(sym), "regime", "chop")
            votes: List[StrategyVote] = []
            for strat in STRATEGIES:
                if not self.cfg["strategies"]["components"].get(
                        strat.name, {}).get("enabled", True):
                    continue
                if regime not in strat.regimes:
                    continue
                try:
                    v = strat.vote(self.store, sym, ctx)
                except Exception as exc:
                    self.log.warning("strategy %s/%s error: %s", strat.name, sym, exc)
                    continue
                if v:
                    v.weight = REGIME_BASE_WEIGHT.get(strat.name, {}).get(regime, 1.0) \
                        * self.weights.get(strat.name, 1.0)
                    votes.append(v)
            mv = ml_votes.get(sym)
            if mv is not None:
                mv.weight = REGIME_BASE_WEIGHT["ml_forecast"].get(regime, 1.0) \
                    * self.weights.get("ml_forecast", 1.0)
                votes.append(mv)

            prop = self._combine(sym, votes, regime, quotes.get(sym), now)
            if prop:
                proposals.append(prop)

        if proposals:
            self.bus.publish("proposals", proposals)
            for p in proposals:
                self.log.info("PROPOSAL %s %s conf=%.2f rr=%.2f sl=%.2f tp=%.2f <- %s",
                              p.symbol, p.side.value, p.confidence, p.rr,
                              p.stop_loss, p.take_profit, ",".join(p.strategies))

    # ---------------- combination ---------------- #
    def _combine(self, sym, votes: List[StrategyVote], regime, quote, now):
        if not votes or quote is None:
            return None
        a = self.store.arrays(sym, Timeframe.M15)
        atr_ = float(ind.atr(a["h"], a["l"], a["c"], 14)[-1])
        if np.isnan(atr_) or atr_ <= 0:
            return None
        long_w = sum(v.weight * v.conviction for v in votes
                     if v.direction == SignalDirection.LONG)
        short_w = sum(v.weight * v.conviction for v in votes
                      if v.direction == SignalDirection.SHORT)
        total_w = sum(v.weight for v in votes)
        if total_w <= 0:
            return None
        net = long_w - short_w
        side = Side.BUY if net > 0 else Side.SELL if net < 0 else Side.FLAT
        if side == Side.FLAT:
            return None
        aligned = [v for v in votes
                   if (v.direction == SignalDirection.LONG) == (side == Side.BUY)]
        n_aligned = len(aligned)
        if self.cfg["strategies"].get("require_two_agreeing", True) and n_aligned < 2:
            return None
        raw = abs(net) / total_w
        agree_bonus = min(0.25, 0.08 * (n_aligned - 1))
        # macro alignment multiplier
        macro = self.bus.get("macro.context")
        m_align = 1.0
        if macro is not None:
            mb = macro.macro_bias * (1 if side == Side.BUY else -1)
            m_align = 1.0 + 0.3 * float(np.clip(mb, -1, 1))
        confidence = float(np.clip(raw * 1.6 + agree_bonus, 0, 1) * m_align)
        if confidence < self.cfg["strategies"]["min_confidence"]:
            return None

        entry = quote.ask if side == Side.BUY else quote.bid
        stop_cfg = self.cfg["risk"]["stop"]
        mult = float(np.clip(stop_cfg["atr_mult"] *
                             getattr(self.bus.get("regime.map", {}).get(sym),
                                     "vol_mult", 1.0) ** 0.35,
                             stop_cfg["min_atr"], stop_cfg["max_atr"]))
        risk_dist = mult * atr_
        # structure-aware stop: use recent swing if tighter-but-safe
        hi, lo = ind.donchian(a["h"][:-1], a["l"][:-1], 12)
        if side == Side.BUY and not np.isnan(lo[-1]):
            sw = entry - (lo[-1] - stop_cfg["structure_buffer_atr"] * atr_)
            risk_dist = max(min(sw, risk_dist * 1.4), risk_dist * 0.6)
        elif side == Side.SELL and not np.isnan(hi[-1]):
            sw = (hi[-1] + stop_cfg["structure_buffer_atr"] * atr_) - entry
            risk_dist = max(min(sw, risk_dist * 1.4), risk_dist * 0.6)

        rr_target = float(stop_cfg.get("tp_rr", 2.0))
        if regime == "chop":
            rr_target *= 0.7
        elif regime == "volatile_breakout":
            rr_target *= 1.2
        sl = entry - risk_dist if side == Side.BUY else entry + risk_dist
        tp = entry + rr_target * risk_dist if side == Side.BUY else entry - rr_target * risk_dist

        prop = TradeProposal(
            symbol=sym, side=side, entry_ref=entry, stop_loss=sl, take_profit=tp,
            confidence=min(confidence, 0.99), rr=rr_target,
            horizon_tf=aligned[0].horizon_tf if aligned else Timeframe.M15,
            strategies=[v.strategy for v in aligned],
            reasons=sum((v.reasons for v in aligned[:3]), [])[:8],
            votes=[dict(strategy=v.strategy, direction=v.direction.value,
                        weight=round(v.weight, 3), conviction=round(v.conviction, 3),
                        reasons=v.reasons[:3]) for v in votes])
        return prop
