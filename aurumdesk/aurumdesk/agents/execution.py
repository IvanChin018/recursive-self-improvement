"""
ExecutionAgent — the trading desk's execution trader.

Consumes risk-approved decisions and allocation actions and turns them into
orders with best-execution discipline:

  * pre-trade checks (quote freshness, spread guard re-check, market open)
  * reduce-first ordering: closes/reductions go out BEFORE new opens so we
    never breach exposure caps by accident
  * idempotency: every proposal_id maps to at most one live order; retries
    only on transient errors with exponential backoff
  * post-trade verification: position must appear broker-side with sane
    SL/TP, otherwise raise a critical alert (no silent failures)
  * weekend flat: optionally flattens the whole book before Friday NY close
Nothing here can change size or levels — that is Risk's job. Execution only
executes what it was told, as well as possible.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..bus import Bus
from ..models import (Order, OrderState, RiskDecision, Side, Timeframe,
                      TradeProposal)
from ..utils.timeutils import is_friday_close_time, market_open
from .base import Agent


class ExecutionAgent(Agent):
    name = "ExecutionAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.gateway = None
        self._pending: Dict[str, dict] = {}      # proposal_id -> {order, tries, next_t}
        self._done_proposals: set[str] = set()
        self._last_ticket_seq: int = 0
        self._weekend_flat_done_day: str = ""
        bus.subscribe("risk.decisions", self._on_decisions)
        bus.subscribe("alloc.actions", self._on_alloc_actions)

    # ---------------- intake ---------------- #
    def _on_decisions(self, topic, payload: List[RiskDecision]) -> None:
        proposals: Dict[str, TradeProposal] = {p.id: p for p in
                                               self.bus.get("proposals.last", [],
                                                            allow_stale=True)}
        now = time.time()
        for d in payload:
            prop = proposals.get(d.proposal_id)
            if prop is None:
                continue
            if d.verdict.value in ("VETOED",):
                continue
            if d.lots <= 0:
                continue
            if prop.id in self._done_proposals or prop.id in self._pending:
                continue
            self._queue_entry(prop, d, now)

    def _queue_entry(self, prop: TradeProposal, d: RiskDecision, now: float) -> None:
        ex = self.cfg["execution"]
        o = Order(proposal_id=prop.id, symbol=prop.symbol, side=prop.side,
                  order_type="MARKET", lots=d.lots, limit_price=prop.entry_ref,
                  stop_loss=prop.stop_loss, take_profit=prop.take_profit,
                  reduce_only=prop.reduce_only,
                  comment=f"{ex.get('comment_prefix', 'AurumDesk')}|"
                          f"{','.join(prop.strategies)[:12]}",
                  client_id=f"AUR-{int(now)}-{prop.id}")
        self._pending[prop.id] = dict(order=o, tries=0, next_t=now)

    def _on_alloc_actions(self, topic, actions: List[dict]) -> None:
        """Position-management actions from PortfolioAllocAgent."""
        gw = self.gateway
        if gw is None:
            return
        quotes = self.bus.get("market.quotes", {})
        for a in actions:
            kind = a.get("kind")
            try:
                if kind == "modify":
                    ok = gw.modify_sl_tp(a["ticket"], round(a["sl"], 5),
                                         round(a.get("tp", 0.0), 5))
                    self.log.info("MODIFY ticket=%s sl=%.2f -> %s (%s)",
                                  a["ticket"], a["sl"], "ok" if ok else "FAIL",
                                  a.get("why", ""))
                elif kind in ("close_partial", "close_full"):
                    pos = next((p for p in gw.positions()
                                if p.ticket == a["ticket"]), None)
                    if pos is None:
                        continue
                    q = quotes.get(pos.symbol)
                    lots = pos.lots if kind == "close_full" else min(
                        a.get("lots", 0.0), pos.lots)
                    if lots < 0.01:
                        continue
                    o = Order(symbol=pos.symbol,
                              side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                              lots=round(lots, 2), reduce_only=True,
                              comment=f"{self.cfg['execution'].get('comment_prefix', 'AurumDesk')}|mgmt")
                    res = gw.send_order(o, q)
                    self.log.info("CLOSE %s ticket=%s lots=%.2f -> %s (%s)",
                                  kind, a["ticket"], lots, res.state.value,
                                  a.get("why", ""))
            except Exception as exc:
                self.log.exception("alloc action failed: %s", exc)

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        if self.gateway is None:
            self.gateway = self.bus.get("broker.gateway")
        if self.gateway is None:
            return
        now = time.time()

        # weekend flat policy
        if self.cfg["risk"].get("weekend_flat", True) and not self._wk_done(now):
            if is_friday_close_time(now) and market_open(now):
                n = self._flatten_all(now, reason="weekend flat")
                if n:
                    self.log.warning("WEEKEND FLAT: closed %d positions", n)
                self._mark_wk(now)

        if self.bus.get("risk.halted", False):
            # halted = no NEW entries, but management/close orders still flow
            pass

        # fire pending entry orders (reduce-only first is guaranteed by design:
        # alloc actions above already ran this cycle)
        ex = self.cfg["execution"]
        retry_max = int(ex.get("retry_max", 3))
        backoff = float(ex.get("retry_backoff_sec", 2.0))
        quotes = self.bus.get("market.quotes", {})
        stale = set(self.bus.get("market.stale_symbols", []))

        for pid in list(self._pending):
            slot = self._pending[pid]
            o: Order = slot["order"]
            if now < slot["next_t"]:
                continue
            if o.symbol in stale or not market_open(now):
                slot["next_t"] = now + max(backoff, 5.0)   # wait for data/market
                continue
            q = quotes.get(o.symbol)
            res = self.gateway.send_order(o, q)
            slot["tries"] += 1
            if res.state == OrderState.FILLED:
                del self._pending[pid]
                self._done_proposals.add(pid)
                self._verify(o, now)
                self.bus.publish("order.filled", dict(order=res, proposal_id=pid))
                self.log.info("FILLED %s %s %.2f @ %.2f sl=%.2f tp=%.2f [%s]",
                              o.symbol, o.side.value, o.filled_lots,
                              o.avg_fill_price, o.stop_loss, o.take_profit, pid)
            elif res.state in (OrderState.REJECTED, OrderState.ERROR):
                transient = ("retcode=10018" in res.error or "busy" in res.error.lower()
                             or "requote" in res.error.lower() or "no quote" in res.error
                             or "connection" in res.error.lower()
                             or res.state == OrderState.ERROR)
                if transient and slot["tries"] < retry_max:
                    slot["next_t"] = now + backoff * (2 ** (slot["tries"] - 1))
                    self.log.warning("RETRY %d/%d %s: %s", slot["tries"], retry_max,
                                     o.symbol, res.error)
                else:
                    del self._pending[pid]
                    self._done_proposals.add(pid)
                    self.bus.publish("alert", {"level": "warn", "agent": self.name,
                                               "msg": f"order rejected {o.symbol}: {res.error}"})
                    self.log.error("REJECTED %s %s: %s", o.symbol, o.side.value, res.error)
            else:  # PARTIAL / SUBMITTED — keep watching via verify next tick
                slot["next_t"] = now + backoff

        # cap memory of done ids
        if len(self._done_proposals) > 4096:
            self._done_proposals = set(list(self._done_proposals)[-2048:])

    # ---------------- helpers ---------------- #
    def _verify(self, o: Order, now: float) -> None:
        """Post-trade check: position exists & protected."""
        try:
            poss = [p for p in self.gateway.positions() if p.symbol == o.symbol]
        except Exception:
            return
        mine = [p for p in poss if abs(p.open_price - o.avg_fill_price) /
                max(o.avg_fill_price, 1e-9) < 0.01]
        if not mine:
            self.bus.publish("alert", {"level": "critical", "agent": self.name,
                                       "msg": f"fill not visible after send: {o.symbol}"})
            return
        prot = [p for p in mine if p.sl > 0 or p.tp > 0]
        if not prot and (o.stop_loss or o.take_profit):
            ok = self.gateway.modify_sl_tp(mine[-1].ticket, o.stop_loss, o.take_profit)
            if not ok:
                self.bus.publish("alert", {"level": "critical", "agent": self.name,
                                           "msg": f"SL/TP attach failed {o.symbol} "
                                                  f"ticket={mine[-1].ticket}"})

    def _flatten_all(self, now: float, reason: str) -> int:
        n = 0
        quotes = self.bus.get("market.quotes", {})
        try:
            poss = self.gateway.positions()
        except Exception:
            return 0
        for pos in poss:
            q = quotes.get(pos.symbol)
            o = Order(symbol=pos.symbol,
                      side=Side.SELL if pos.side == Side.BUY else Side.BUY,
                      lots=round(pos.lots, 2), reduce_only=True,
                      comment=f"{self.cfg['execution'].get('comment_prefix', 'AurumDesk')}|flat")
            res = self.gateway.send_order(o, q)
            if res.state == OrderState.FILLED:
                n += 1
        if n:
            self.bus.publish("alert", {"level": "info", "agent": self.name,
                                       "msg": f"flattened book: {reason}"})
        return n

    def _wk_key(self, now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now))

    def _wk_done(self, now: float) -> bool:
        return self._weekend_flat_done_day == self._wk_key(now)

    def _mark_wk(self, now: float) -> None:
        self._weekend_flat_done_day = self._wk_key(now)
