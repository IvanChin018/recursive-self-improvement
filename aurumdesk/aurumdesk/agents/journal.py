"""
JournalAgent — the desk's historian and performance accountant.

Every fill, close, decision and equity print is written to a durable journal
(logs/trades.csv + logs/events.jsonl via TradeJournal). On trade close it:

  * attributes PnL and R-multiple back to the strategy tags on the position
    comment -> feeds StrategyEnsemble's adaptive weighting loop
  * publishes trade.closed (consumed by RiskManager per-symbol lockouts)
  * maintains rolling stats: win rate, avg win/loss R, expectancy, Sharpe,
    max drawdown, per-symbol and per-strategy breakdowns
This closes the learning loop: strategies that stop working get down-weighted
automatically; nothing is ever traded on stale conviction.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..bus import Bus
from ..risk.portfolio import max_drawdown, sharpe
from ..utils.logging import TradeJournal
from ..utils.timeutils import utc_day
from .base import Agent


class JournalAgent(Agent):
    name = "JournalAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.tj = TradeJournal(log_dir)
        self.gateway = None
        self.ensemble = None                 # wired by desk for feedback loop
        self._known_tickets: set[int] = set()
        self._ticket_meta: Dict[int, dict] = {}
        self._equity_curve: List[float] = []
        self._rets: List[float] = []
        self._last_eq: float = 0.0
        self.trades: List[dict] = []
        self.stats_path = os.path.join(cfg.get("overseer", {}).get("state_dir", "state"),
                                       "journal_stats.json")
        self._load()
        bus.subscribe("order.filled", self._on_fill)

    def _load(self) -> None:
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                self.trades = d.get("trades", [])[-3000:]
                self._ticket_meta = {int(k): v for k, v in d.get("meta", {}).items()}
                self._known_tickets = set(self._ticket_meta)
        except Exception:
            pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            with open(self.stats_path, "w", encoding="utf-8") as fh:
                json.dump(dict(trades=self.trades[-3000:],
                               meta={str(k): v for k, v in
                                     list(self._ticket_meta.items())[-500:]}), fh)
        except Exception as exc:
            self.log.warning("journal save failed: %s", exc)

    # ---------------- intake ---------------- #
    def _on_fill(self, topic, payload) -> None:
        o = payload.get("order")
        pid = payload.get("proposal_id", "")
        if o is None:
            return
        tag = (o.comment or "").split("|")[-1]
        self.tj.write("fill", symbol=o.symbol, side=o.side.value, lots=o.filled_lots,
                      price=o.avg_fill_price, sl=o.stop_loss, tp=o.take_profit,
                      strategies=tag, order_id=o.id, proposal_id=pid,
                      notes="entry" if not o.reduce_only else "reduce")
        # remember tickets opened this cycle (paper backend records history too)
        if not o.reduce_only and self.gateway is not None:
            try:
                for p in self.gateway.positions():
                    if p.ticket not in self._known_tickets and p.symbol == o.symbol:
                        self._known_tickets.add(p.ticket)
                        self._ticket_meta[p.ticket] = dict(
                            symbol=p.symbol, lots=p.lots,
                            strategies=[s.strip() for s in tag.split(",")
                                        if s.strip()],
                            risk_dist=abs(o.avg_fill_price - (p.sl or o.stop_loss)),
                            opened_t=time.time(), entry=o.avg_fill_price)
            except Exception:
                pass

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        if self.gateway is None:
            self.gateway = self.bus.get("broker.gateway")
        snap = self.bus.get("account.snapshot")
        if snap is None:
            return
        now = time.time()

        # equity curve sampling (~1/min effective)
        if self._last_eq and abs(snap.equity - self._last_eq) > 1e-9:
            self._rets.append(snap.equity / self._last_eq - 1.0)
            if len(self._rets) > 5000:
                self._rets = self._rets[-5000:]
        self._last_eq = snap.equity
        self._equity_curve.append(snap.equity)
        if len(self._equity_curve) > 20000:
            self._equity_curve = self._equity_curve[-10000:]

        # detect closed tickets by diffing broker truth
        if self.gateway is not None:
            try:
                live = {p.ticket for p in self.gateway.positions()}
            except Exception:
                live = set(self._ticket_meta)
            gone = [t for t in self._known_tickets if t not in live]
            hist = []
            paper = getattr(self.gateway, "paper", None)
            if paper is not None:
                hist = paper.history
            elif hasattr(self.gateway, "mt5") and getattr(self.gateway, "using_mt5", False):
                hist = self.gateway.mt5.deal_history() if hasattr(
                    self.gateway.mt5, "deal_history") else []
            for t in gone:
                meta = self._ticket_meta.pop(t, {})
                rec = next((h for h in reversed(hist)
                            if h.get("ticket") == t and h.get("kind") in
                            ("close", "sl", "tp")), None)
                pnl = float(rec.get("pnl", 0.0)) if rec else 0.0
                r_mult = 0.0
                rd = meta.get("risk_dist", 0.0)
                vpu = 100.0
                if paper is not None and hasattr(paper, "value_per_unit"):
                    from ..instruments import get_spec as _gs
                    vpu = paper.value_per_unit(_gs(meta.get("symbol", "")))
                if rd and vpu > 0:
                    r_mult = pnl / max(abs(rd * vpu * meta.get("lots", 1.0)), 1e-9)
                strats = meta.get("strategies", ["unattributed"]) or ["unattributed"]
                tr = dict(ticket=t, symbol=meta.get("symbol", ""), closed_t=now,
                          pnl=pnl, r=r_mult, strategies=strats,
                          day=utc_day(now))
                self.trades.append(tr)
                if len(self.trades) > 5000:
                    self.trades = self.trades[-3000:]
                self.tj.write("close", symbol=tr["symbol"], pnl=round(pnl, 2),
                              strategies=",".join(strats), confidence="",
                              notes=f"R={r_mult:.2f} ticket={t}",
                              balance=snap.balance, equity=snap.equity)
                self.bus.publish("trade.closed", tr)
                if self.ensemble is not None:
                    for s in strats:
                        self.ensemble.record_outcome(s, r_mult if r_mult else
                                                     (0.3 if pnl > 0 else -0.7))
                self._known_tickets.discard(t)
            self.save()

        stats = self.compute_stats()
        self.bus.set("journal.stats", stats, max_age_sec=None)
        self.bus.publish("journal.tick", dict(equity=snap.equity,
                                              realized_today=snap.realized_today,
                                              n_trades=len(self.trades)))

    # ---------------- analytics ---------------- #
    def compute_stats(self) -> dict:
        rs = [t["r"] for t in self.trades if t.get("r")]
        pnls = [t["pnl"] for t in self.trades]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]
        wr = len(wins) / len(rs) if rs else 0.0
        aw = float(np.mean(wins)) if wins else 0.0
        al = float(np.mean([abs(x) for x in losses])) if losses else 0.0
        expectancy = wr * aw - (1 - wr) * al if rs else 0.0
        per_strategy: Dict[str, dict] = {}
        agg: Dict[str, List[dict]] = defaultdict(list)
        for t in self.trades:
            for s in t.get("strategies", []):
                agg[s].append(t)
        for s, ts in agg.items():
            srs = [x["r"] for x in ts if x.get("r")]
            sw = [x for x in srs if x > 0]
            sl_ = [abs(x) for x in srs if x <= 0]
            per_strategy[s] = dict(
                n=len(ts),
                win_rate=(len(sw) / len(srs)) if srs else 0.0,
                avg_win_r=float(np.mean(sw)) if sw else 0.0,
                avg_loss_r=float(np.mean(sl_)) if sl_ else 0.0,
                total_pnl=sum(x["pnl"] for x in ts),
            )
        per_symbol: Dict[str, float] = defaultdict(float)
        for t in self.trades:
            per_symbol[t["symbol"]] += t["pnl"]
        return dict(
            n_trades=len(self.trades),
            win_rate=wr, avg_win_r=aw, avg_loss_r=al, expectancy_r=expectancy,
            total_pnl=sum(pnls),
            today_pnl=sum(t["pnl"] for t in self.trades if t.get("day") == utc_day(time.time())),
            sharpe=sharpe(self._rets[-500:]) if len(self._rets) > 30 else 0.0,
            max_dd_pct=max_drawdown(self._equity_curve[-5000:]) * 100,
            per_strategy=per_strategy, per_symbol=dict(per_symbol),
        )
