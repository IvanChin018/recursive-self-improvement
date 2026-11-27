"""
Strategy modules. Each exposes:

    name: str
    regimes: set of regime names where it should be active
    vote(store, sym, ctx) -> StrategyVote | None

`ctx` carries macro context, other agents' signals, current price/ATR and
the regime info. Strategies must return None when they have no edge —
silence is a valid, important answer (prevents forced trades).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np

from ..indicators import library as ind
from ..models import SignalDirection, StrategyVote, Timeframe
from ..utils.timeutils import session_of, in_momentum_window


def _dir(score: float, thr: float = 0.1) -> SignalDirection:
    if score > thr:
        return SignalDirection.LONG
    if score < -thr:
        return SignalDirection.SHORT
    return SignalDirection.NEUTRAL


# ===================================================================== #
# 1. Trend following (multi-TF confluence)
# ===================================================================== #
class TrendFollowing:
    name = "trend"
    regimes = {"trend_up", "trend_down", "volatile_breakout"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        a15, a1, a4 = (store.arrays(sym, tf) for tf in
                       (Timeframe.M15, Timeframe.H1, Timeframe.H4))
        if a1 is None or a4 is None:
            return None
        c1, c4 = a1["c"], a4["c"]
        e20_1, e50_1 = ind.ema(c1, 20), ind.ema(c1, 50)
        e50_4, e200_4 = ind.ema(c4, 50), ind.ema(c4, 200)
        adx4, dip4, dim4 = ind.adx_di(a4["h"], a4["l"], c4, 14)
        if any(np.isnan(x[-1]) for x in (e20_1, e50_1, e50_4, e200_4)):
            return None
        bull = (c1[-1] > e20_1[-1] > e50_1[-1]) and (e50_4[-1] > e200_4[-1])
        bear = (c1[-1] < e20_1[-1] < e50_1[-1]) and (e50_4[-1] < e200_4[-1])
        strength = 0.0
        if bull:
            strength = 0.5 + 0.3 * min(max((adx4[-1] - 20) / 25, 0), 1)
        elif bear:
            strength = -(0.5 + 0.3 * min(max((adx4[-1] - 20) / 25, 0), 1))
        else:
            return None
        reasons = [f"H4 ADX {adx4[-1]:.0f}", "M15/H4 trend aligned"]
        if not in_momentum_window(ctx.now):
            strength *= 0.7
            reasons.append("off-peak session haircut")
        return StrategyVote(self.name, sym, _dir(strength), weight=1.0,
                            conviction=float(min(abs(strength), 1.0)),
                            horizon_tf=Timeframe.H1, reasons=reasons)


# ===================================================================== #
# 2. Breakout (vol expansion from compression)
# ===================================================================== #
class Breakout:
    name = "breakout"
    regimes = {"chop", "volatile_breakout", "trend_up", "trend_down"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        a = store.arrays(sym, Timeframe.M15)
        if a is None:
            return None
        o, h, l, c = a["o"], a["h"], a["l"], a["c"]
        du, dd = ind.donchian(h[:-1], l[:-1], 20)     # exclude forming bar
        atr_ = ind.atr(h, l, c, 14)
        _, _, _, bbw = ind.bbands(c, 20, 2.0)
        if np.isnan(du[-1]) or np.isnan(atr_[-1]):
            return None
        squeeze_rank = float(np.nanmean(bbw[-80:] < bbw[-2])) if len(bbw) > 80 else 0.5
        up_move = (h[-1] - du[-1]) / max(atr_[-1], 1e-9)
        dn_move = (dd[-1] - l[-1]) / max(atr_[-1], 1e-9)
        score = 0.0
        reasons = []
        if up_move > 0.15 and c[-1] > du[-1]:
            score = 0.55 + 0.25 * (1 - squeeze_rank)
            reasons.append(f"cleared 20-bar high by {up_move:.2f} ATR")
        elif dn_move > 0.15 and c[-1] < dd[-1]:
            score = -(0.55 + 0.25 * (1 - squeeze_rank))
            reasons.append(f"broke 20-bar low by {dn_move:.2f} ATR")
        else:
            return None
        if squeeze_rank < 0.2:
            score *= 1.2
            reasons.append("post-squeeze expansion (high quality)")
        if session_of(ctx.now) in ("asia", "sydney"):
            score *= 0.6
            reasons.append("thin-session breakout discount")
        return StrategyVote(self.name, sym, _dir(score), weight=1.0,
                            conviction=min(abs(score), 1.0),
                            horizon_tf=Timeframe.M15, reasons=reasons)


# ===================================================================== #
# 3. Mean reversion (only in chop)
# ===================================================================== #
class MeanReversion:
    name = "meanrev"
    regimes = {"chop"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        a = store.arrays(sym, Timeframe.M5)
        ah = store.arrays(sym, Timeframe.H1)
        if a is None or ah is None:
            return None
        c = a["c"]; h, l = a["h"], a["l"]
        up, mid, lo, w = ind.bbands(c, 20, 2.0)
        r = ind.rsi(c, 14)
        z = ind.zscore(c, 60)
        adx_h, _, _ = ind.adx_di(ah["h"], ah["l"], ah["c"], 14)
        if np.isnan(up[-1]) or up[-1] == lo[-1]:
            return None
        pb = (c[-1] - lo[-1]) / (up[-1] - lo[-1])
        score = 0.0; reasons = []
        if pb < 0.03 and r[-1] < 28 and (np.isnan(adx_h[-1]) or adx_h[-1] < 28):
            score = 0.45 + 0.2 * (0.03 - pb) / 0.03
            reasons.append(f"M5 lower-band tag, RSI {r[-1]:.0f}")
        elif pb > 0.97 and r[-1] > 72 and (np.isnan(adx_h[-1]) or adx_h[-1] < 28):
            score = -(0.45 + 0.2 * (pb - 0.97) / 0.03)
            reasons.append(f"M5 upper-band tag, RSI {r[-1]:.0f}")
        if abs(z[-1]) > 2.4 and score == 0:
            score = -0.35 * math.copysign(1, z[-1])
            reasons.append(f"z-score stretch {z[-1]:+.1f}")
        if score == 0:
            return None
        return StrategyVote(self.name, sym, _dir(score), weight=1.0,
                            conviction=min(abs(score), 1.0),
                            horizon_tf=Timeframe.M5, reasons=reasons)


# ===================================================================== #
# 4. Candlestick event-driven (wraps CandlePatternAgent output)
# ===================================================================== #
class CandleEdge:
    name = "candlestick"
    regimes = {"chop", "trend_up", "trend_down", "volatile_breakout"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        best = None
        for sig in ctx.candle_signals.get(sym, []):
            if time.time() - sig.t > 3 * 900:      # fresh within 3 M15 bars
                continue
            if best is None or abs(sig.score) > abs(best.score):
                best = sig
        if best is None or abs(best.score) < 0.25:
            return None
        return StrategyVote(self.name, sym, best.direction, weight=1.0,
                            conviction=min(abs(best.score), 1.0),
                            horizon_tf=best.timeframe, reasons=list(best.reasons))


# ===================================================================== #
# 5. Carry & Macro (real-rate / USD / risk sentiment structural bias)
# ===================================================================== #
class CarryMacro:
    name = "carry_macro"
    regimes = {"trend_up", "trend_down", "chop", "volatile_breakout"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        macro = ctx.macro
        if macro is None or abs(macro.macro_bias) < 0.12:
            return None
        usd_ccy = sym[3:]
        # gold priced in AUD/EUR gets extra tail when that ccy is weak vs USD
        adj = 1.0
        fx = ctx.fx_momentum.get(usd_ccy, 0.0)   # e.g. EURUSD 1h momentum
        d = SignalDirection.LONG if macro.macro_bias > 0 else SignalDirection.SHORT
        conv = min(abs(macro.macro_bias) + 0.4 * abs(fx), 1.0)
        return StrategyVote(self.name, sym, d, weight=1.0, conviction=conv,
                            horizon_tf=Timeframe.D1,
                            reasons=list(macro.macro_reasons[:3]))


# ===================================================================== #
# 6. Pairs / relative value (metal cross consistency + metal ratio)
# ===================================================================== #
class PairsRelative:
    name = "pairs_relative"
    regimes = {"chop", "trend_up", "trend_down", "volatile_breakout"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        base, qccy = sym[:3], sym[3:]
        usd_sym = f"{base}USD"
        au = store.arrays(usd_sym, Timeframe.M15)
        fx = store.arrays(f"{qccy}USD", Timeframe.M15) if qccy != "USD" else None
        mine = store.arrays(sym, Timeframe.M15)
        if au is None or mine is None:
            return None
        if qccy == "USD":
            # XAU vs XAG relative value via ratio z-score
            ag = store.arrays("XAGUSD", Timeframe.M15)
            if ag is None or base != "XAU":
                return None
            ratio = au["c"] / np.maximum(ag["c"], 1e-9)
            if len(ratio) < 60:
                return None
            z = (ratio[-1] - np.mean(ratio[-60:])) / max(np.std(ratio[-60:]), 1e-9)
            if abs(z) < 1.6:
                return None
            d = SignalDirection.SHORT if z > 0 else SignalDirection.LONG
            return StrategyVote(self.name, sym, d, weight=1.0,
                                conviction=min(abs(z) / 3, 1.0),
                                horizon_tf=Timeframe.H1,
                                reasons=[f"gold/silver ratio z={z:+.2f} reversion"])
        if fx is None:
            return None
        implied = au["c"] / np.maximum(fx["c"], 1e-9)
        dev = (mine["c"][-1] - implied[-1]) / max(implied[-1], 1e-9)
        atr_ = ind.atr(mine["h"], mine["l"], mine["c"], 14)
        dev_atr = dev * mine["c"][-1] / max(atr_[-1], 1e-9)
        if abs(dev_atr) < 0.5:      # cross mispricing beyond half an ATR
            return None
        d = SignalDirection.SHORT if dev_atr > 0 else SignalDirection.LONG
        return StrategyVote(self.name, sym, d, weight=1.0,
                            conviction=min(abs(dev_atr) / 1.5, 1.0),
                            horizon_tf=Timeframe.M15,
                            reasons=[f"cross vs implied off by {dev_atr:+.2f} ATR"])


# ===================================================================== #
# 7. Session momentum (London open drift / NY overlap continuation)
# ===================================================================== #
class SessionMomentum:
    name = "session_momentum"
    regimes = {"trend_up", "trend_down", "volatile_breakout", "chop"}

    def vote(self, store, sym, ctx) -> Optional[StrategyVote]:
        a = store.arrays(sym, Timeframe.M15)
        if a is None:
            return None
        t = a["t"]; c = a["c"]
        now_h = int((ctx.now // 3600) % 24)
        sess = session_of(ctx.now)
        if sess not in ("london", "london_ny_overlap", "newyork"):
            return None
        # return over the last 6 hours
        n_back = max(int(len(t) * 0.02), 24)
        if len(c) <= n_back:
            return None
        ret = c[-1] / c[-1 - n_back] - 1.0
        atr_ = ind.atr(a["h"], a["l"], c, 14)
        r_atr = ret * c[-1] / max(atr_[-1], 1e-9)
        if abs(r_atr) < 1.2:
            return None
        boost = 1.25 if sess == "london_ny_overlap" else 1.0
        score = float(np.clip(r_atr / 4, -1, 1)) * boost
        return StrategyVote(self.name, sym, _dir(score), weight=1.0,
                            conviction=min(abs(score), 1.0),
                            horizon_tf=Timeframe.M15,
                            reasons=[f"{sess} drift {ret:+.2%} ({r_atr:.1f} ATR)"])
