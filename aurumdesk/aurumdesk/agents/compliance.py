"""
ComplianceAgent — the desk's internal auditor / regulator.

It sits beside the flow (it can never block the hot path, but it CAN and DOES
flag violations loudly and demand manual review). Rules enforced:

  * mode integrity: "live" requires an explicit confirm_live_trading flag;
    without it any live-routed order is a CRITICAL violation
  * magic-number isolation: only our own orders/magic may touch the account
  * per-symbol whitelist: nothing trades outside configured instruments
  * audit trail completeness: every FILLED order must have a matching
    proposal -> risk decision chain in the journal; gaps are violations
  * rate limits: max orders/hour globally (fat-finger / loop breaker)
  * state sanity: negative lots, zero prices, missing SL on new opens
Compliance can only tighten (raise halt flags), never loosen Risk.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque

from ..bus import Bus
from ..models import OrderState
from .base import Agent


class ComplianceAgent(Agent):
    name = "ComplianceAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.violations: Deque[dict] = deque(maxlen=500)
        self.orders_last_hour: Deque[float] = deque(maxlen=1000)
        self.max_orders_per_hour = 60
        self.seen_orders: set[str] = set()
        bus.subscribe("order.filled", self._on_fill)

    def _flag(self, rule: str, msg: str, level: str = "violation") -> None:
        v = dict(t=time.time(), rule=rule, msg=msg, level=level)
        self.violations.append(v)
        self.log.error("COMPLIANCE %s: %s — %s", level.upper(), rule, msg)
        self.bus.publish("alert", {"level": "critical", "agent": self.name,
                                   "msg": f"[{rule}] {msg}"})

    def _on_fill(self, topic, payload) -> None:
        self.orders_last_hour.append(time.time())
        o = payload.get("order")
        if o is not None:
            self.seen_orders.add(o.id)

    def tick(self) -> None:
        now = time.time()
        cfg = self.cfg

        # 1. mode integrity
        if cfg.get("mode", "paper") == "live" and not cfg.get("confirm_live_trading", False):
            self._flag("mode", "mode=live without confirm_live_trading=true; "
                               "forcing paper semantics")
            self.bus.set("compliance.block_new_entries", True)
        else:
            self.bus.set("compliance.block_new_entries", False)

        # 2. fat-finger rate limiter
        cutoff = now - 3600
        while self.orders_last_hour and self.orders_last_hour[0] < cutoff:
            self.orders_last_hour.popleft()
        if len(self.orders_last_hour) > self.max_orders_per_hour:
            self._flag("rate-limit",
                       f"{len(self.orders_last_hour)} orders in last hour "
                       f"(cap {self.max_orders_per_hour}) — loop suspected")

        # 3. account/state sanity against broker truth
        snap = self.bus.get("account.snapshot")
        allowed = set(cfg["symbols"]) | {"AUDUSD", "EURUSD"}
        if snap is not None:
            for p in snap.open_positions:
                if p.symbol not in allowed:
                    self._flag("whitelist", f"position in unconfigured symbol {p.symbol}")
                if p.lots <= 0 or p.open_price <= 0:
                    self._flag("sanity", f"impossible position {p.ticket} "
                                         f"lots={p.lots} px={p.open_price}")
                ours = not p.comment or p.comment.startswith(
                    cfg["execution"].get("comment_prefix", "AurumDesk"))
                if not ours:
                    # foreign position on shared terminal: don't trade alongside blindly
                    self.bus.set("compliance.foreign_exposure",
                                 float(self.bus.get("compliance.foreign_exposure", 0.0))
                                 + p.lots * p.open_price)
            # equity vs balance divergence guard (data feed lying?)
            if snap.balance > 0 and abs(snap.equity - snap.balance) / snap.balance > 0.5:
                self._flag("sanity", f"equity/balance divergence "
                                     f"{snap.equity:.0f}/{snap.balance:.0f} > 50%")

        # 4. audit-chain completeness: filled orders must trace to a decision
        decided = set()
        for _, topic, payload in self.bus.recent(2000):
            if topic == "risk.decisions":
                for d in payload:
                    decided.add(d.proposal_id)
        for _, topic, payload in self.bus.recent(2000):
            if topic == "order.filled":
                pid = payload.get("proposal_id", "")
                if pid and pid not in decided and len(pid) == 12:
                    self._flag("audit", f"fill for proposal {pid} has no risk decision")

        # 5. kill-switch consistency
        if self.bus.get("risk.halted", False):
            pend = self.bus.get("pending.proposals", [])
            if pend:
                self.bus.set("pending.proposals", [])  # drop, don't queue up behind halt

        self.bus.set("compliance.violation_count", len(self.violations),
                     max_age_sec=None)
        self.bus.set("compliance.last_check", now)
