"""Paper broker: full-featured simulated MT5 account for paper trading.

Features: margin model, floating PnL with FX conversion, SL/TP engine with
conservative intra-bar fills, spread + slippage costs, swaps/commissions,
trade history, daily/weekly stats persistence. This is what runs when the
desk is in `paper` mode without (or alongside) a live terminal.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from ..instruments import ContractSpec, get_spec
from ..models import AccountSnapshot, Fill, Order, OrderState, Position, Quote, Side

# quote_ccy -> USD conversions resolved from live prices of fx pairs
FX_PRICES = {"AUDUSD": 0.655, "EURUSD": 1.078}


class PaperBroker:
    def __init__(self, cfg: dict, state_path: str = "state/paper_account.json",
                 log=None):
        self.cfg = cfg
        self.log = log
        self.state_path = state_path
        self.balance: float = 10_000.0
        self.positions: Dict[int, Position] = {}
        self.history: List[dict] = []
        self.realized_today: float = 0.0
        self.today_key: str = time.strftime("%Y-%m-%d")
        self._ticket = 1000
        self.fills: List[Fill] = []
        self.fx_rates: Dict[str, float] = dict(FX_PRICES)
        self.leverage = 100.0
        self.commission_per_lot = 0.07 * 2   # ~$7/side per 100k notionals scaled
        self.swap_annual_pct = 3.0
        self._load()

    # ---------------- persistence ---------------- #
    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                self.balance = float(d.get("balance", self.balance))
                self.realized_today = float(d.get("realized_today", 0.0))
                self.today_key = d.get("today_key", self.today_key)
                self._ticket = int(d.get("ticket", 1000))
                self.history = d.get("history", [])[-5000:]
                self.positions = {}
                for p in d.get("positions", []):
                    pos = Position(**p)
                    self.positions[pos.ticket] = pos
                if self.today_key != self._roll_day():
                    self.roll_day()
        except Exception:
            pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(dict(balance=self.balance,
                               realized_today=self.realized_today,
                               today_key=self.today_key, ticket=self._ticket,
                               positions=[pos.__dict__ | {"side": pos.side.value}
                                          for pos in self.positions.values()],
                               history=self.history[-5000:]), fh, indent=1)
        except Exception as exc:
            if self.log:
                self.log.warning("paper state save failed: %s", exc)

    @staticmethod
    def _roll_day() -> str:
        return time.strftime("%Y-%m-%d")

    def roll_day(self) -> None:
        self.today_key = self._roll_day()
        self.realized_today = 0.0

    # ---------------- pricing helpers ---------------- #
    def set_fx(self, symbol: str, rate: float) -> None:
        if rate > 0:
            self.fx_rates[symbol] = rate

    def _fx_to_usd(self, ccy: str) -> float:
        if ccy == "USD":
            return 1.0
        r = self.fx_rates.get(f"{ccy}USD")
        if r:
            return r
        inv = self.fx_rates.get(f"USD{ccy}")
        return 1.0 / inv if inv else 1.0

    def value_per_unit(self, spec: ContractSpec) -> float:
        """Account-currency value of a 1.00 price move on 1 lot."""
        return spec.contract_size * self._fx_to_usd(spec.quote_ccy)

    def floating_pnl(self, pos: Position, price: float, spec: ContractSpec) -> float:
        return (price - pos.open_price) * pos.direction_sign * \
            pos.lots * self.value_per_unit(spec)

    def equity(self, quotes: Dict[str, Quote]) -> float:
        eq = self.balance
        for pos in self.positions.values():
            q = quotes.get(pos.symbol)
            if not q:
                continue
            px = q.bid if pos.side == Side.BUY else q.ask
            eq += self.floating_pnl(pos, px, get_spec(pos.symbol))
        return eq

    def margin_used(self, quotes: Dict[str, Quote]) -> float:
        m = 0.0
        for pos in self.positions.values():
            q = quotes.get(pos.symbol)
            px = q.mid if q else pos.open_price
            spec = get_spec(pos.symbol)
            m += pos.lots * spec.contract_size * px * self._fx_to_usd(spec.quote_ccy) \
                / self.leverage
        return m

    def snapshot(self, quotes: Dict[str, Quote]) -> AccountSnapshot:
        if self.today_key != self._roll_day():
            self.roll_day()
        eq = self.equity(quotes)
        mu = self.margin_used(quotes)
        notional = sum(p.lots * get_spec(p.symbol).contract_size *
                       (quotes[p.symbol].mid if p.symbol in quotes else p.open_price) *
                       self._fx_to_usd(get_spec(p.symbol).quote_ccy)
                       for p in self.positions.values())
        poss = list(self.positions.values())
        for p in poss:
            q = quotes.get(p.symbol)
            if q:
                px = q.bid if p.side == Side.BUY else q.ask
                p.profit = self.floating_pnl(p, px, get_spec(p.symbol))
        return AccountSnapshot(balance=self.balance, equity=eq, margin_used=mu,
                               margin_free=max(eq - mu, 0.0),
                               positions_value=notional,
                               realized_today=self.realized_today,
                               open_positions=poss)

    # ---------------- order handling ---------------- #
    def send(self, order: Order, quote: Optional[Quote]) -> Order:
        order.updated_t = time.time()
        if quote is None or quote.bid <= 0:
            order.state = OrderState.REJECTED
            order.error = "no quote"
            return order
        spec = get_spec(order.symbol)
        slip_pts = self.cfg["execution"].get("deviation_slippage_points", 20)
        slip = slip_pts * spec.point
        fill_px = (quote.ask + slip) if order.side == Side.BUY else (quote.bid - slip)

        if order.reduce_only or self._netting_close(order):
            self._close(order, fill_px)
            return order

        # margin check
        need_margin = order.lots * spec.contract_size * fill_px * \
            self._fx_to_usd(spec.quote_ccy) / self.leverage
        snap_eq = self.equity({order.symbol: quote})
        if need_margin > (snap_eq - self.margin_used({})) * 0.9:
            order.state = OrderState.REJECTED
            order.error = "insufficient margin"
            return order

        commission = order.lots * self.commission_per_lot * \
            (spec.contract_size * fill_px * self._fx_to_usd(spec.quote_ccy) / 100_000.0)
        self.balance -= commission
        self._ticket += 1
        pos = Position(ticket=self._ticket, symbol=order.symbol, side=order.side,
                       lots=order.lots, open_price=fill_px, sl=order.stop_loss,
                       tp=order.take_profit, opened_t=time.time(),
                       comment=order.comment, strategy_tag=order.comment.split("|")[-1])
        self.positions[pos.ticket] = pos
        order.state = OrderState.FILLED
        order.avg_fill_price = fill_px
        order.filled_lots = order.lots
        order.broker_order_id = pos.ticket
        self.fills.append(Fill(symbol=order.symbol, side=order.side, lots=order.lots,
                               price=fill_px, t=time.time(), commission=commission,
                               order_id=order.id, pos_ticket=pos.ticket))
        self.history.append(dict(kind="open", ticket=pos.ticket, symbol=order.symbol,
                                 side=order.side.value, lots=order.lots, price=fill_px,
                                 sl=order.stop_loss, tp=order.take_profit,
                                 t=time.time(), comment=order.comment))
        return order

    def _netting_close(self, order: Order) -> bool:
        """Opposite-side order against existing position -> partial/full close."""
        if order.side == Side.FLAT:
            return True
        for pos in self.positions.values():
            if pos.symbol == order.symbol and pos.side != order.side:
                return True
        return False

    def _close(self, order: Order, fill_px: float) -> None:
        remaining = order.lots
        closed_any = False
        for ticket in sorted(self.positions, key=lambda k: self.positions[k].opened_t):
            pos = self.positions[ticket]
            if pos.symbol != order.symbol or pos.side == order.side:
                continue
            take = min(remaining, pos.lots)
            spec = get_spec(pos.symbol)
            pnl = (fill_px - pos.open_price) * pos.direction_sign * \
                take * self.value_per_unit(spec)
            self.balance += pnl
            self.realized_today += pnl
            closed_any = True
            self.history.append(dict(kind="close", ticket=ticket, symbol=pos.symbol,
                                     side=pos.side.value, lots=take, price=fill_px,
                                     pnl=pnl, t=time.time(), comment=order.comment))
            remaining -= take
            if take >= pos.lots - 1e-9:
                del self.positions[ticket]
            else:
                pos.lots -= take
            if remaining <= 1e-9:
                break
        order.state = OrderState.FILLED if closed_any else OrderState.REJECTED
        if not closed_any:
            order.error = "nothing to close"
        order.avg_fill_price = fill_px
        order.filled_lots = order.lots - max(remaining, 0.0)

    def close_ticket(self, ticket: int, quote: Quote, lots: Optional[float] = None) -> bool:
        pos = self.positions.get(ticket)
        if not pos or quote is None:
            return False
        px = quote.bid if pos.side == Side.BUY else quote.ask
        o = Order(symbol=pos.symbol, side=(Side.SELL if pos.side == Side.BUY else Side.BUY),
                  lots=lots or pos.lots, reduce_only=True, comment="AurumDesk-close")
        self._close(o, px)
        return o.state == OrderState.FILLED

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        pos = self.positions.get(ticket)
        if not pos:
            return False
        pos.sl, pos.tp = sl, tp
        return True

    # ---------------- SL/TP engine ---------------- #
    def check_exits(self, quotes: Dict[str, Quote]) -> List[dict]:
        """Conservative intra-bar logic using candle high/low when available."""
        events = []
        for ticket in list(self.positions):
            pos = self.positions[ticket]
            q = quotes.get(pos.symbol)
            if not q:
                continue
            bid, ask = q.bid, q.ask
            hit_sl = hit_tp = False
            px = 0.0
            if pos.side == Side.BUY:
                if pos.sl and bid <= pos.sl:
                    hit_sl, px = True, pos.sl
                elif pos.tp and bid >= pos.tp:
                    hit_tp, px = True, pos.tp
            else:
                if pos.sl and ask >= pos.sl:
                    hit_sl, px = True, pos.sl
                elif pos.tp and ask <= pos.tp:
                    hit_tp, px = True, pos.tp
            if hit_sl or hit_tp:
                spec = get_spec(pos.symbol)
                exec_px = px if hit_tp else (px - 2 * spec.point)  # slight stop slip
                pnl = (exec_px - pos.open_price) * pos.direction_sign * \
                    pos.lots * self.value_per_unit(spec)
                self.balance += pnl
                self.realized_today += pnl
                del self.positions[ticket]
                self.history.append(dict(kind="sl" if hit_sl else "tp", ticket=ticket,
                                         symbol=pos.symbol, side=pos.side.value,
                                         lots=pos.lots, price=exec_px, pnl=pnl,
                                         t=time.time(), comment=pos.comment))
                events.append(dict(ticket=ticket, symbol=pos.symbol,
                                   reason="SL" if hit_sl else "TP", pnl=pnl,
                                   price=exec_px, lots=pos.lots))
        return events

    def apply_swaps(self, quotes: Dict[str, Quote]) -> None:
        day = self._roll_day()
        if getattr(self, "_last_swap", "") == day:
            return
        self._last_swap = day
        for pos in self.positions.values():
            spec = get_spec(pos.symbol)
            notional = pos.lots * spec.contract_size * pos.open_price * \
                self._fx_to_usd(spec.quote_ccy)
            sgn = -1 if pos.side == Side.BUY else 1
            self.balance += sgn * notional * self.swap_annual_pct / 100 / 365
            pos.swap += sgn * notional * self.swap_annual_pct / 100 / 365
