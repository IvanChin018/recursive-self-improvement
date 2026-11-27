"""
BrokerGateway: one interface, two brains (MT5 live/demo OR PaperBroker).

The rest of the desk never knows which backend is active. `mode` decides:
  paper -> PaperBroker unless MT5 attached in demo (then MT5 passthrough)
  live  -> requires MT5 attached AND explicit confirm flag in config
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ..instruments import ContractSpec, get_spec
from ..models import AccountSnapshot, Order, Position, Quote, Side
from .paper_broker import PaperBroker


class BrokerGateway:
    def __init__(self, cfg: dict, mt5_feed=None, log=None):
        self.cfg = cfg
        self.mt5 = mt5_feed
        self.log = log
        self.mode = cfg.get("mode", "paper")
        self.paper = PaperBroker(cfg, state_path="state/paper_account.json", log=log) \
            if self.mode == "paper" or mt5_feed is None or not getattr(mt5_feed, "connected", False) else None
        self.backend = "mt5" if (mt5_feed and getattr(mt5_feed, "connected", False)
                                 and self.paper is None) else ("paper" if self.paper else "none")
        # In paper mode with MT5 connected to a DEMO terminal we still use MT5
        # directly — that IS paper trading on the user's API-linked terminal.
        if self.mode == "paper" and mt5_feed and getattr(mt5_feed, "connected", False):
            self.paper = None
            self.backend = "mt5-demo"

    # ---------------- info ---------------- #
    @property
    def using_mt5(self) -> bool:
        return self.backend.startswith("mt5") and self.mt5 is not None

    def spec(self, symbol: str) -> ContractSpec:
        if self.using_mt5:
            return self.mt5.spec(symbol)
        return get_spec(symbol, self.cfg.get("account_currency", "USD"))

    # ---------------- market data ---------------- #
    def quote(self, symbol: str) -> Optional[Quote]:
        if self.using_mt5:
            return self.mt5.quote(symbol)
        return None

    def positions(self) -> List[Position]:
        if self.using_mt5:
            return self.mt5.positions()
        if self.paper:
            return list(self.paper.positions.values())
        return []

    def account(self, quotes: Optional[Dict[str, Quote]] = None) -> Optional[AccountSnapshot]:
        if self.using_mt5:
            return self.mt5.account()
        if self.paper:
            return self.paper.snapshot(quotes or {})
        return None

    # ---------------- orders ---------------- #
    def send_order(self, order: Order, quote: Optional[Quote] = None) -> Order:
        if self.using_mt5:
            return self.mt5.send_market(order)
        if self.paper:
            return self.paper.send(order, quote)
        order.error = "no broker backend"
        from ..models import OrderState
        order.state = OrderState.ERROR
        return order

    def close_position(self, ticket: int, lots: Optional[float] = None,
                       quote: Optional[Quote] = None) -> bool:
        if self.using_mt5:
            return self.mt5.close_position(ticket, lots)
        if self.paper:
            return self.paper.close_ticket(ticket, quote, lots)
        return False

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        if self.using_mt5:
            return self.mt5.modify_position_sl_tp(ticket, sl, tp)
        if self.paper:
            return self.paper.modify_sl_tp(ticket, sl, tp)
        return False

    def tick_exits(self, quotes: Dict[str, Quote]) -> List[dict]:
        """SL/TP simulation events (paper backend only; MT5 does it server-side)."""
        if self.paper:
            return self.paper.check_exits(quotes)
        return []

    def roll_day(self) -> None:
        if self.paper:
            self.paper.roll_day()

    def save_state(self) -> None:
        if self.paper:
            self.paper.save()
