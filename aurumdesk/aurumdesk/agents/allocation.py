"""
PortfolioAllocAgent — capital allocator / position manager.

Responsibilities:
  * Maintain EWMA correlation matrix across the metals book.
  * Compute correlation-adjusted effective exposure vs equity.
  * Propose rebalancing trims when a group is overheated.
  * Manage open positions' lifecycle policies that are NOT broker-side:
      - breakeven move after +1 ATR
      - trailing stop at N x ATR
      - partial take-profit at 1R (close X%)
      - time-stop for stale trades
      - pre-news risk reduction (trim X% before red events)
  * Publishes AllocationActions consumed by ExecutionAgent.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import Side, Timeframe
from ..instruments import get_spec
from ..risk.portfolio import ewma_corr_matrix
from .base import Agent


class PortfolioAllocAgent(Agent):
    name = "PortfolioAllocAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self._returns: Dict[str, List[float]] = {s: [] for s in cfg["symbols"]}
        self._last_t: Dict[str, int] = {}
        self.corr: np.ndarray | None = None
        self._done_partial: set[int] = set()
        self._done_be: set[int] = set()

    # ---------------- correlation ---------------- #
    def _usd_notional(self, pos, quote) -> float:
        """USD notional of a position (contract_size * price * fx_to_usd)."""
        spec = self.gateway.spec(pos.symbol) if getattr(self, "gateway", None) else \
            get_spec(pos.symbol)
        px = (quote.mid if quote is not None else pos.open_price)
        fx = 1.0
        if spec.needs_fx_conversion and self.store is not None:
            a = self.store.arrays(spec.fx_symbol, Timeframe.M15)
            if a is not None:
                fx = float(a["c"][-1])
        return pos.lots * spec.contract_size * px * fx

    def _update_corr(self, now: float) -> None:
        self.store = self.bus.get("market.store")
        if self.store is None:
            return
        keys = self.cfg["symbols"]
        rets = []
        for s in keys:
            ser = self.store.get(s, Timeframe.M15)
            if ser is None or len(ser.t) < 30:
                rets.append([0.0] * 60)
                continue
            t = ser.t[-61:]
            c = ser.c[-61:]
            r = list(np.diff(np.log(np.maximum(c, 1e-9))))
            rets.append((r + [0.0] * 60)[:60])
        self.corr = ewma_corr_matrix(rets, halflife=48)
        self.bus.set("portfolio.corr", dict(zip(keys, [list(map(float, row))
                                                       for row in self.corr])),
                     max_age_sec=3600)

    # ---------------- management tick ---------------- #
    def tick(self) -> None:
        now = time.time()
        self._update_corr(now)
        snap = self.bus.get("account.snapshot")
        quotes = self.bus.get("market.quotes", {})
        if snap is None:
            return
        actions: List[dict] = []
        rc = self.cfg["risk"]
        stop_cfg = rc["stop"]
        macro = self.bus.get("macro.context")

        for pos in snap.open_positions:
            q = quotes.get(pos.symbol)
            if q is None:
                continue
            a = self.store.arrays(pos.symbol, Timeframe.M15) if self.store else None
            if a is None:
                continue
            atr_ = float(ind.atr(a["h"], a["l"], a["c"], 14)[-1])
            if np.isnan(atr_) or atr_ <= 0:
                continue
            px = q.bid if pos.side == Side.BUY else q.ask
            sign = pos.side.sign
            gain = (px - pos.open_price) * sign
            gain_atr = gain / atr_

            # 1. breakeven after +1 ATR
            be_at = stop_cfg.get("breakeven_atr", 1.0)
            if gain_atr >= be_at and pos.ticket not in self._done_be:
                new_sl = pos.open_price + sign * 0.1 * atr_
                if (pos.sl == 0) or (sign > 0 and new_sl > pos.sl) or (sign < 0 and (pos.sl == 0 or new_sl < pos.sl)):
                    actions.append(dict(kind="modify", ticket=pos.ticket, sl=new_sl, tp=pos.tp,
                                        why=f"breakeven @+{gain_atr:.1f} ATR"))
                    self._done_be.add(pos.ticket)

            # 2. trail stop behind 1.5 ATR once beyond 1R
            trail = stop_cfg.get("trail_atr", 1.5)
            if gain_atr >= trail:
                cand = px - sign * trail * atr_
                if (sign > 0 and (pos.sl == 0 or cand > pos.sl)) or \
                   (sign < 0 and (pos.sl == 0 or cand < pos.sl)):
                    actions.append(dict(kind="modify", ticket=pos.ticket, sl=cand, tp=pos.tp,
                                        why=f"trail to {cand:.2f}"))
                    if pos.sl != 0:
                        pos.sl = cand  # optimistic; exec confirms

            # 3. partial TP at 1R
            ptp = stop_cfg.get("partial_tp", {})
            risk_dist = abs(pos.open_price - pos.sl) if pos.sl else atr_
            if ptp.get("enabled") and risk_dist > 0 and \
               gain / max(risk_dist, 1e-9) >= ptp.get("at_rr", 1.0) and \
               pos.ticket not in self._done_partial:
                cut = round(pos.lots * ptp.get("close_pct", 50) / 100.0, 2)
                if cut >= 0.01 and pos.lots - cut >= 0.0:
                    actions.append(dict(kind="close_partial", ticket=pos.ticket, lots=cut,
                                        why=f"partial TP at {ptp['at_rr']}R ({ptp['close_pct']}%)"))
                    self._done_partial.add(pos.ticket)

            # 4. time stop
            age_h = (now - pos.opened_t) / 3600
            if age_h > rc.get("max_trade_age_hours", 36) and gain < 0:
                actions.append(dict(kind="close_full", ticket=pos.ticket,
                                    why=f"time-stop after {age_h:.0f}h underwater"))

        # 5. pre-news risk reduction
        if macro and macro.upcoming_events:
            lo, hi = rc.get("news_blackout_min", [-15, 5])
            for e in macro.upcoming_events:
                lead = (e["t"] - now) / 60
                if e.get("red") and lo <= lead <= 0:
                    trim_pct = rc.get("news_close_reduce_pct", 50) / 100.0
                    for pos in snap.open_positions:
                        cut = round(pos.lots * trim_pct, 2)
                        if cut >= 0.01:
                            actions.append(dict(kind="close_partial", ticket=pos.ticket,
                                                lots=cut, why=f"pre-news trim '{e['name']}'"))
                    break

        # 6. correlation-adjusted total exposure check (signed $-notional per symbol)
        if self.corr is not None and snap.open_positions:
            keys = self.cfg["symbols"]
            v = {s: sum(p.side.sign * p.lots * self._usd_notional(p, quotes.get(p.symbol))
                        for p in snap.open_positions if p.symbol == s) for s in keys}
            eff = float(np.sqrt(max(sum(v[a] * v[b] * self.corr[i][j]
                                        for i, a in enumerate(keys)
                                        for j, b in enumerate(keys)), 0.0)))
            budget = rc["max_exposure_pct_equity"] / 100 * snap.equity
            if eff > budget * 1.15:
                worst = min(snap.open_positions, key=lambda p: p.profit)
                cut = round(worst.lots * 0.3, 2)
                if cut >= 0.01:
                    actions.append(dict(kind="close_partial", ticket=worst.ticket,
                                        lots=cut, why="correlation-exposure trim"))

        # prune done sets of closed tickets
        live = {p.ticket for p in snap.open_positions}
        self._done_partial &= live
        self._done_be &= live

        if actions:
            self.bus.publish("alloc.actions", actions)
            self.log.info("ALLOC: %d actions (%s)", len(actions),
                          "; ".join(a["why"] for a in actions[:3]))
