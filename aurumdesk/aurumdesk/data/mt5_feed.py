"""
MTFeed: MetaTrader5 bridge.

Works when running on the user's Windows machine with a logged-in MT5
terminal (paper or live). On any other platform it degrades gracefully and
reports unavailable so the desk can fall back to the synthetic feed.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Dict, List, Optional

try:
    import MetaTrader5 as mt5
    _HAVE_MT5 = True
except ImportError:          # non-Windows / not installed
    mt5 = None
    _HAVE_MT5 = False

from ..instruments import ContractSpec, get_spec
from ..models import AccountSnapshot, Candle, Order, Position, Quote, Side, Timeframe

TF_MAP_NAME = {
    Timeframe.M1: "M1", Timeframe.M5: "M5", Timeframe.M15: "M15",
    Timeframe.H1: "H1", Timeframe.H4: "H4", Timeframe.D1: "D1",
}


class MTFeed:
    def __init__(self, cfg: dict, log=None):
        self.cfg = cfg
        self.log = log
        self.connected = False
        self.mode = cfg.get("mode", "paper")
        self.symbol_map: Dict[str, str] = cfg["mt5"].get("symbol_map", {})
        self._specs: Dict[str, ContractSpec] = {}
        self._tf_const: Dict[Timeframe, int] = {}
        self.last_error = ""

    # ---------------- connection ---------------- #
    def connect(self) -> bool:
        if not _HAVE_MT5 or not self.cfg["mt5"].get("enabled", True):
            self.last_error = "MT5 module unavailable"
            return False
        try:
            path = self.cfg["mt5"].get("terminal_path") or None
            kw = {}
            if self.cfg["mt5"].get("login"):
                kw = dict(login=int(self.cfg["mt5"]["login"]),
                          password=self.cfg["mt5"].get("password", ""),
                          server=self.cfg["mt5"].get("server", "") or None)
            ok = mt5.initialize(path=path, **kw) if kw else mt5.initialize(path=path)
            if not ok:
                self.last_error = f"initialize failed: {mt5.last_error()}"
                return False
            info = mt5.terminal_info()
            acct = mt5.account_info()
            if info is None or acct is None:
                self.last_error = "terminal/account not ready (log in to MT5 first)"
                return False
            self.connected = True
            for tf in Timeframe:
                self._tf_const[tf] = getattr(mt5, f"TIMEFRAME_{TF_MAP_NAME[tf]}", None)
            if self.log:
                self.log.info("MT5 attached: %s | login=%s balance=%.2f margin_mode=%s",
                              info.name, acct.login, acct.balance,
                              "demo/paper" if getattr(acct, "trade_mode", 1) == 1 else "REAL")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def shutdown(self) -> None:
        if _HAVE_MT5 and self.connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self.connected = False

    def ping(self) -> bool:
        if not (_HAVE_MT5 and self.connected):
            return False
        try:
            return mt5.terminal_info() is not None
        except Exception:
            self.connected = False
            return False

    # ---------------- symbols ---------------- #
    def ensure_symbol(self, symbol: str) -> bool:
        broker_sym = self.symbol_map.get(symbol, symbol)
        if not _HAVE_MT5 or not self.connected:
            return False
        try:
            mt5.symbol_select(broker_sym, True)
            info = mt5.symbol_info(broker_sym)
            if info is None:
                return False
            spec = get_spec(symbol, self.cfg.get("account_currency", "USD"))
            spec.contract_size = float(info.trade_contract_size or spec.contract_size)
            spec.lot_step = float(info.volume_step or 0.01)
            spec.min_lot = float(info.volume_min or 0.01)
            spec.max_lot = float(info.volume_max or 100.0)
            spec.digits = int(info.digits)
            spec.point = float(info.point)
            self._specs[symbol] = spec
            return True
        except Exception:
            return False

    def spec(self, symbol: str) -> ContractSpec:
        if symbol not in self._specs:
            self.ensure_symbol(symbol)
        return self._specs.get(symbol, get_spec(symbol))

    # ---------------- market data ---------------- #
    def candles(self, symbol: str, tf: Timeframe, count: int) -> List[Candle]:
        broker_sym = self.symbol_map.get(symbol, symbol)
        if not (_HAVE_MT5 and self.connected):
            return []
        tconst = self._tf_const.get(tf) or getattr(mt5, f"TIMEFRAME_{TF_MAP_NAME[tf]}")
        rates = mt5.copy_rates_from_pos(broker_sym, tconst, 0, count)
        out: List[Candle] = []
        if rates is None or len(rates) == 0:
            return out
        for r in rates:
            out.append(Candle(symbol=symbol, timeframe=tf, t=int(r["time"]),
                              o=float(r["open"]), h=float(r["high"]),
                              l=float(r["low"]), c=float(r["close"]),
                              volume=float(r.get("tick_volume", 0))))
        return out

    def quote(self, symbol: str) -> Optional[Quote]:
        broker_sym = self.symbol_map.get(symbol, symbol)
        if not (_HAVE_MT5 and self.connected):
            return None
        tick = mt5.symbol_info_tick(broker_sym)
        if tick is None or tick.bid <= 0:
            return None
        return Quote(symbol=symbol, bid=float(tick.bid), ask=float(tick.ask),
                     t=float(tick.time))

    # ---------------- account / positions ---------------- #
    def account(self) -> Optional[AccountSnapshot]:
        if not (_HAVE_MT5 and self.connected):
            return None
        a = mt5.account_info()
        if a is None:
            return None
        poss = self.positions()
        notional = sum(abs(p.lots) * self.spec(p.symbol).contract_size * p.open_price
                       for p in poss)
        return AccountSnapshot(balance=float(a.balance), equity=float(a.equity),
                               margin_used=float(a.margin), margin_free=float(a.margin_free),
                               positions_value=notional, realized_today=0.0,
                               open_positions=poss)

    def positions(self) -> List[Position]:
        if not (_HAVE_MT5 and self.connected):
            return []
        rows = mt5.positions_get() or []
        out = []
        for p in rows:
            side = Side.BUY if p.type == mt5.POSITION_TYPE_BUY else Side.SELL
            out.append(Position(ticket=int(p.ticket), symbol=self._ours(p.symbol),
                                side=side, lots=float(p.volume),
                                open_price=float(p.price_open), sl=float(p.sl),
                                tp=float(p.tp), opened_t=float(p.time),
                                profit=float(p.profit), swap=float(p.swap),
                                comment=p.comment or "",
                                strategy_tag=(p.comment or "").split("|")[-1]))
        return out

    def _ours(self, broker_sym: str) -> str:
        for k, v in self.symbol_map.items():
            if v == broker_sym:
                return k
        return broker_sym

    # ---------------- orders ---------------- #
    def send_market(self, order: Order) -> Order:
        """Send a MARKET order; mutates & returns the same Order object."""
        from ..models import OrderState
        if not (_HAVE_MT5 and self.connected):
            order.state = OrderState.ERROR
            order.error = "MT5 not connected"
            return order
        broker_sym = self.symbol_map.get(order.symbol, order.symbol)
        tick = mt5.symbol_info_tick(broker_sym)
        if tick is None:
            order.state = OrderState.REJECTED
            order.error = "no tick"
            return order
        mtype = mt5.ORDER_TYPE_BUY if order.side == Side.BUY else mt5.ORDER_TYPE_SELL
        price = tick.ask if order.side == Side.BUY else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": round(order.lots, 2),
            "type": mtype,
            "price": price,
            "sl": order.stop_loss if order.stop_loss > 0 else 0.0,
            "tp": order.take_profit if order.take_profit > 0 else 0.0,
            "deviation": int(self.cfg["execution"].get("deviation_slippage_points", 20)),
            "magic": int(self.cfg["execution"].get("magic_base", 0)),
            "comment": order.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        order.updated_t = time.time()
        if res is None:
            order.state = OrderState.ERROR
            order.error = f"order_send None: {mt5.last_error()}"
            return order
        rc = int(getattr(res, "retcode", -1))
        if rc in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL,
                  mt5.TRADE_RETCODE_PLACED):
            order.state = (OrderState.FILLED if rc != mt5.TRADE_RETCODE_DONE_PARTIAL
                           else OrderState.PARTIAL)
            order.broker_order_id = int(res.order)
            order.avg_fill_price = float(res.price or price)
            order.filled_lots = float(res.volume or order.lots)
        else:
            order.state = OrderState.REJECTED
            order.error = f"retcode={rc} {getattr(res, 'comment', '')}"
        return order

    def modify_position_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        if not (_HAVE_MT5 and self.connected):
            return False
        pos = [p for p in (mt5.positions_get() or []) if p.ticket == ticket]
        if not pos:
            return False
        p = pos[0]
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket,
               "symbol": p.symbol, "sl": float(sl), "tp": float(tp)}
        res = mt5.order_send(req)
        return res is not None and int(getattr(res, "retcode", -1)) == mt5.TRADE_RETCODE_DONE

    def close_position(self, ticket: int, lots: Optional[float] = None) -> bool:
        if not (_HAVE_MT5 and self.connected):
            return False
        pos = [p for p in (mt5.positions_get() or []) if p.ticket == ticket]
        if not pos:
            return False
        p = pos[0]
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            return False
        closing_buy = p.type == mt5.POSITION_TYPE_SELL   # close short by buying
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": p.symbol,
            "volume": round(float(lots if lots else p.volume), 2),
            "type": mt5.ORDER_TYPE_BUY if closing_buy else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if closing_buy else tick.bid,
            "deviation": int(self.cfg["execution"].get("deviation_slippage_points", 20)),
            "magic": int(self.cfg["execution"].get("magic_base", 0)),
            "comment": "AurumDesk-close",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        return res is not None and int(getattr(res, "retcode", -1)) in (
            mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL)
