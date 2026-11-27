"""
RiskManagerAgent — the desk's CRO. Nothing trades without passing here.

Checklist executed for EVERY proposal (each one independently logged):
  1. kill switches: daily loss, weekly drawdown, max DD reducer
  2. news blackout window
  3. market freshness / spread cost / session liquidity
  4. position sizing: fixed-fractional RISK per trade x confidence multiplier,
     capped by Kelly fraction (<=25% of full Kelly) and broker limits
  5. portfolio: open position count, correlation-adjusted exposure caps,
     per-symbol & group caps, margin headroom
  6. duplicate/flip protection: no adding to a losing same-direction book,
     no instant reversal churn
  7. sanity: SL/TP distance vs ATR, RR floor, stop direction correctness

It can APPROVE, REDUCE (smaller size), VETO or DEFER. Its word is law;
Execution cannot override it (only Compliance can *tighten*, never loosen).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import (RiskDecision, RiskVerdict, Side, SignalDirection,
                      Timeframe, TradeProposal)
from ..risk.portfolio import (kelly_fraction, position_size_by_risk, round_step,
                              max_drawdown)
from ..utils.timeutils import market_open, session_of, SESSION_LIQUIDITY, utc_day, utc_week_key
from .base import Agent

GOLD_SYMS = {"XAUUSD", "XAUAUD", "XAUEUR"}
SILVER_SYMS = {"XAGUSD", "XAGAUD", "XAGEUR"}


class RiskManagerAgent(Agent):
    name = "RiskManagerAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self.gateway = None
        self.equity_peak: float = 0.0
        self.day_key: str = ""
        self.week_key: str = ""
        self.day_start_equity: float = 0.0
        self.week_start_equity: float = 0.0
        self.symbol_day_pnl: Dict[str, float] = {}
        self.recent_trades: Dict[str, List[float]] = {s: [] for s in cfg["symbols"]}
        self.halted_reason: str = ""
        self.last_stats: Dict[str, dict] = {}   # per-strategy live stats from journal
        bus.subscribe("trade.closed", self._on_close)

    def _on_close(self, topic, payload) -> None:
        sym = payload.get("symbol", "")
        pnl = float(payload.get("pnl", 0.0))
        if utc_day(time.time()) == self.day_key:
            self.symbol_day_pnl[sym] = self.symbol_day_pnl.get(sym, 0.0) + pnl

    # ---------------- helpers ---------------- #
    def _value_per_unit(self, sym: str) -> float:
        spec = self.gateway.spec(sym) if self.gateway else None
        if spec is None:
            return 100.0
        fx = 1.0
        if spec.needs_fx_conversion and self.store is not None:
            a = self.store.arrays(spec.fx_symbol, Timeframe.M15)
            if a is not None:
                fx = float(a["c"][-1])
        return spec.contract_size * fx

    def _book_r_multiple_stats(self) -> tuple[float, float]:
        wins = [v for d in self.last_stats.values() for v in [d.get("avg_win_r", 0)] if v > 0]
        losses = [v for d in self.last_stats.values() for v in [d.get("avg_loss_r", 0)] if v > 0]
        wr = sum(d.get("win_rate", 0) for d in self.last_stats.values()) / max(len(self.last_stats), 1)
        aw = np.mean(wins) if wins else 1.5
        al = np.mean(losses) if losses else 1.0
        return float(aw), float(al) if wr else 1.0

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        self.store = self.bus.get("market.store")
        snap = self.bus.get("account.snapshot")
        now = time.time()
        if snap is None:
            return
        eq = snap.equity
        # day/week anchors & kill switches
        if utc_day(now) != self.day_key:
            self.day_key = utc_day(now)
            self.day_start_equity = eq
            self.symbol_day_pnl.clear()
            self.bus.set("risk.halted", False)
            self.halted_reason = ""
        if utc_week_key(now) != self.week_key:
            self.week_key = utc_week_key(now)
            self.week_start_equity = eq
        self.equity_peak = max(self.equity_peak, eq)

        rc = self.cfg["risk"]
        day_dd_pct = (eq - self.day_start_equity) / max(self.day_start_equity, 1e-9) * 100
        week_dd_pct = (eq - self.week_start_equity) / max(self.week_start_equity, 1e-9) * 100
        peak_dd_pct = (eq - self.equity_peak) / max(self.equity_peak, 1e-9) * 100
        halt_new = False
        reason = ""
        if day_dd_pct <= -rc["daily_loss_limit_pct"]:
            halt_new, reason = True, f"daily loss limit ({day_dd_pct:.2f}%)"
        elif week_dd_pct <= -rc["weekly_drawdown_halt_pct"]:
            halt_new, reason = True, f"weekly drawdown halt ({week_dd_pct:.2f}%)"
        if halt_new and not self.bus.get("risk.halted", False):
            self.log.error("KILL SWITCH: %s — no new entries until reset", reason)
            self.bus.publish("alert", {"level": "critical", "agent": self.name,
                                       "msg": f"kill switch: {reason}"})
        self.bus.set("risk.halted", halt_new)
        self.bus.set("risk.size_mult",
                     0.5 if peak_dd_pct <= -rc["max_drawdown_reducer_pct"] else 1.0)
        self.bus.set("risk.kill_reason", reason)

        # consume proposals synchronously each tick via publish subscription
        proposals = self.bus.get("pending.proposals", [])
        if proposals:
            self.bus.set("pending.proposals", [])
            decisions = []
            for prop in proposals:
                decisions.append(self.evaluate(prop, snap, now))
            self.bus.publish("risk.decisions", decisions)

    # ---------------- core evaluation ---------------- #
    def evaluate(self, prop: TradeProposal, snap, now: float) -> RiskDecision:
        rc = self.cfg["risk"]
        checks: Dict[str, bool] = {}
        notes: List[str] = []
        veto = lambda note, **kw: RiskDecision(prop.id, RiskVerdict.VETOED, 0.0,
                                               0.0, notes=[note], checks=checks)

        # 1. kill switches
        if self.bus.get("risk.halted", False):
            return veto(f"halted: {self.bus.get('risk.kill_reason', '')}")
        if not market_open(now):
            return veto("market closed")
        # 2. news blackout
        macro = self.bus.get("macro.context")
        if macro and now < macro.news_blackout_until:
            return veto("news blackout window")
        # 3. data quality / spread / liquidity
        stale = set(self.bus.get("market.stale_symbols", []))
        if prop.symbol in stale:
            return veto("stale data")
        quotes = self.bus.get("market.quotes", {})
        q = quotes.get(prop.symbol)
        if q is None or q.bid <= 0:
            return veto("no quote")
        a = self.store.arrays(prop.symbol, Timeframe.M15) if self.store else None
        if a is None:
            return veto("no candles")
        atr_ = float(ind.atr(a["h"], a["l"], a["c"], 14)[-1])
        if np.isnan(atr_) or atr_ <= 0:
            return veto("bad ATR")
        if q.spread > rc["spread_max_atr_frac"] * atr_:
            return veto(f"spread too wide ({q.spread:.3f} > "
                        f"{rc['spread_max_atr_frac']:.2f} ATR)")
        liq = SESSION_LIQUIDITY.get(session_of(now), 0.5)
        if liq < 0.45:
            return veto(f"illiquid session ({session_of(now)})")
        # 4. structural sanity of levels
        sign = prop.side.sign
        risk_dist = abs(prop.entry_ref - prop.stop_loss)
        reward_dist = abs(prop.take_profit - prop.entry_ref)
        checks["levels_sane"] = (sign * (prop.entry_ref - prop.stop_loss) > 0 and
                                 sign * (prop.take_profit - prop.entry_ref) > 0)
        if not checks["levels_sane"]:
            return veto("SL/TP on wrong side of entry")
        if risk_dist < 0.4 * atr_:
            return veto("stop tighter than 0.4 ATR (noise zone)")
        if reward_dist / max(risk_dist, 1e-9) < 0.8:
            return veto("RR below 0.8 floor")
        # 5. per-symbol daily loss limiter
        sym_pl = self.symbol_day_pnl.get(prop.symbol, 0.0)
        if sym_pl < -(rc["per_symbol_daily_loss_pct"] / 100) * snap.equity:
            return veto(f"{prop.symbol} daily loss lockout ({sym_pl:.2f})")
        # 6. churn protection: min spacing between entries same symbol
        rt = self.recent_trades[prop.symbol]
        rt[:] = [t for t in rt if now - t < 3600]
        if len(rt) >= 2:
            return veto("too many trades this hour (churn guard)")
        # 7. sizing
        conf_mult = np.interp(prop.confidence, [0.5, 1.0], rc["conf_scale"])
        base_risk = rc["risk_per_trade_pct"] * conf_mult * liq
        # Kelly cap using realized book stats
        aw, al = self._book_r_multiple_stats()
        wr = float(np.clip(0.5 + (prop.confidence - 0.5) * 0.6, 0, 1))
        kf = kelly_fraction(wr, aw, al)
        kelly_risk_pct = kf * 25.0          # quarter-Kelly ceiling
        risk_pct = min(base_risk, kelly_risk_pct) if kf > 0 else base_risk
        risk_pct *= float(self.bus.get("risk.size_mult", 1.0))
        vpu = self._value_per_unit(prop.symbol)
        lots = position_size_by_risk(snap.equity, risk_pct, risk_dist, vpu)
        spec = self.gateway.spec(prop.symbol) if self.gateway else None
        verdict = RiskVerdict.APPROVED
        if spec:
            if lots > rc["max_position_lots"]:
                lots = rc["max_position_lots"]
                verdict = RiskVerdict.REDUCED
                notes.append("capped by max_position_lots")
            lots = round_step(lots, spec.lot_step, "down")
            if lots < spec.min_lot:
                return veto("size below broker minimum")
        # 8. portfolio constraints
        poss = snap.open_positions
        same_dir = [p for p in poss if p.symbol == prop.symbol and p.side == prop.side]
        losing_same = [p for p in same_dir if p.profit < -0.3 * risk_dist * vpu * p.lots]
        if losing_same:
            return veto("would average down into a loser")
        opposite = [p for p in poss if p.symbol == prop.symbol and p.side != prop.side]
        if opposite and prop.confidence < 0.8:
            return veto("reversal requires confidence >= 0.8")
        if len(poss) >= rc["max_open_positions"] and not prop.reduce_only:
            return veto("max open positions reached")
        notional = lots * spec.contract_size * prop.entry_ref * (vpu / max(spec.contract_size, 1e-9)) \
            if spec else lots * vpu * prop.entry_ref
        proj_exposure_pct = (snap.positions_value + notional) / max(snap.equity, 1e-9) * 100
        if proj_exposure_pct > rc["max_exposure_pct_equity"]:
            allowed = max(rc["max_exposure_pct_equity"] / 100 * snap.equity
                          - snap.positions_value, 0)
            if allowed <= 0:
                return veto("exposure cap reached")
            lots = round_step(allowed / max(prop.entry_ref * vpu, 1e-9),
                              spec.lot_step if spec else 0.01, "down")
            verdict = RiskVerdict.REDUCED
            notes.append("reduced by exposure cap")
            if lots < (spec.min_lot if spec else 0.01):
                return veto("exposure cap leaves no room")
        # correlation-group caps (gold/silver books are near-duplicates)
        grp_syms = GOLD_SYMS if prop.symbol in GOLD_SYMS else SILVER_SYMS
        grp_val = sum(p.lots * self._value_per_unit(p.symbol) * p.open_price
                      for p in poss if p.symbol in grp_syms) + lots * vpu * prop.entry_ref
        grp_cap = rc["correlation_cap"]["gold_group" if prop.symbol in GOLD_SYMS
                                          else "silver_group"] * snap.equity / 100 * 1000
        if grp_val > grp_cap:
            return veto(f"{prop.symbol[:3]} group exposure cap hit")
        # margin headroom
        margin_need = lots * spec.contract_size * prop.entry_ref * (
            vpu / max(spec.contract_size, 1e-9)) / 100.0 if spec else 0
        if snap.margin_used + margin_need > rc["max_margin_used_pct"] / 100 * snap.equity:
            return veto("margin headroom insufficient")
        max_loss = lots * risk_dist * vpu
        self.recent_trades[prop.symbol].append(now)
        notes.insert(0, f"risk={risk_pct:.2f}% of equity, conf={prop.confidence:.2f}, "
                        f"liq={liq:.2f}")
        return RiskDecision(proposal_id=prop.id, verdict=verdict, lots=lots,
                            max_loss_amount=max_loss, notes=notes, checks=checks)
