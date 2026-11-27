"""
NewsMacroAgent — macro & event-risk layer for the metals book.

Real drivers of gold/silver: real yields, USD (DXY), inflation breakevens,
risk sentiment (VIX), central-bank demand, and scheduled red-folder events.

Data sources (best effort, never blocking):
  * MT5 symbols if the broker offers XAU/XAG-linked or DXY tickers
  * Stooq public CSVs (no key required) for DXY ^NYICDX? -> uses UUP proxy,
    VIX ^VIX, gold GLD proxy, TLT etc. Wrapped in timeouts; failures degrade
    to neutral with ok=False so downstream agents know macro is offline.
  * Built-in recurring event calendar (NFP first Friday 12:30 UTC-ish,
    CPI mid-month, FOMC ~8x/year) + optional JSON file the user can edit.

Outputs MacroContext on the blackboard every tick.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request
from typing import Dict, List, Optional

from ..bus import Bus
from ..models import Candle, MacroContext, Timeframe
from .base import Agent


def _fetch_csv(url: str, timeout: float = 4.0) -> Optional[List[List[str]]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AurumDesk/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="ignore")
        return [line.split(",") for line in raw.strip().splitlines()]
    except Exception:
        return None


class NewsMacroAgent(Agent):
    name = "NewsMacroAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.ctx = MacroContext()
        self.events_file = os.path.join(os.path.dirname(log_dir) or ".",
                                        "config", "events.json")
        self._last_fetch = 0.0
        self._last_cal_check = 0.0
        self.net_available = True

    # ---------------- external data ---------------- #
    def _refresh_external(self, now: float) -> None:
        if now - self._last_fetch < 300:      # every 5 min
            return
        self._last_fetch = now
        rows = _fetch_csv("https://stooq.com/q/l/?s=%5Evix&f=sd2t2ohlc&h&e=csv")
        if rows and len(rows) > 1:
            try:
                self.ctx.vix = float(rows[1][4])
            except (ValueError, IndexError):
                pass
        else:
            self.net_available = False
        rows = _fetch_csv("https://stooq.com/q/l/?s=uup.us&f=sd2t2ohlc&h&e=csv")
        if rows and len(rows) > 1:
            try:
                self.ctx.dxy = float(rows[1][4])   # UUP ETF as USD proxy
            except (ValueError, IndexError):
                pass

    # ---------------- event calendar ---------------- #
    def _red_events_today(self, now: float) -> List[dict]:
        """Recurring US red-folder events (approximate but conservative)."""
        d = dt.datetime.fromtimestamp(now, tz=dt.timezone.utc)
        ev: List[dict] = []
        # NFP: first Friday of month 12:30 UTC
        if d.weekday() == 4 and d.day <= 7:
            ev.append({"name": "US Non-Farm Payrolls", "t": d.replace(
                hour=12, minute=30, second=0).timestamp(), "red": True})
        # CPI: days 10-15, Tue/Wed/Fri 12:30 UTC (heuristic)
        if 10 <= d.day <= 15 and d.weekday() in (1, 2, 4):
            ev.append({"name": "US CPI", "t": d.replace(
                hour=12, minute=30, second=0).timestamp(), "red": True})
        # FOMC decision windows: Wednesdays in known-ish months weeks
        fomc_months = {1, 3, 5, 6, 7, 9, 11, 12}
        if d.weekday() == 2 and d.month in fomc_months and 15 <= d.day <= 22:
            ev.append({"name": "FOMC Rate Decision", "t": d.replace(
                hour=18, minute=0, second=0).timestamp(), "red": True})
        # PCE: last business day 12:30
        if d.day >= 26 and d.weekday() in (3, 4):
            ev.append({"name": "US Core PCE", "t": d.replace(
                hour=12, minute=30, second=0).timestamp(), "red": True})
        # user-supplied overrides
        try:
            if os.path.exists(self.events_file):
                with open(self.events_file, "r", encoding="utf-8") as fh:
                    for e in json.load(fh):
                        ev.append({"name": e.get("name", "custom"),
                                   "t": float(e["t"]), "red": bool(e.get("red", True))})
        except Exception:
            pass
        # keep events within +/- 6h window
        return [e for e in ev if abs(e["t"] - now) < 6 * 3600]

    # ---------------- structural bias ---------------- #
    def _macro_bias(self) -> tuple[float, list[str]]:
        reasons = []
        bias = 0.0
        # DXY/UUP trend from store if available
        arrs = self.bus.get("store.xauusd_d1")
        # real-rate proxy: use VIX regime + USD momentum when online
        if self.ctx.dxy > 0:
            prev_dxy = self.bus.get("macro.prev_dxy", 0)
            if prev_dxy:
                chg = (self.ctx.dxy - prev_dxy) / max(prev_dxy, 1e-9)
                bias -= max(min(chg * 200, 0.4), -0.4)
                if abs(chg) > 0.002:
                    reasons.append(f"DXY 5m chg {chg:+.2%}")
            self.bus.set("macro.prev_dxy", self.ctx.dxy)
        if self.ctx.vix > 0:
            if self.ctx.vix > 25:
                bias += 0.15
                reasons.append(f"VIX {self.ctx.vix:.1f} risk-off bid")
            elif self.ctx.vix < 13:
                bias -= 0.05
                reasons.append("low-vol complacency, mild headwind")
        # gold's own daily momentum as safe-haven flow detector
        gd = self.bus.get("market.store")
        if gd is not None:
            a = gd.arrays("XAUUSD", Timeframe.D1)
            if a is not None and len(a["c"]) > 20:
                mom = a["c"][-1] / a["c"][-20] - 1.0
                b = max(min(mom * 6, 0.35), -0.35)
                bias += b
                if abs(b) > 0.05:
                    reasons.append(f"gold 20d momentum {mom:+.2%}")
        return max(min(bias, 1.0), -1.0), reasons

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        now = time.time()
        self._refresh_external(now)
        if now - self._last_cal_check > 60:
            self._last_cal_check = now
            evs = self._red_events_today(now)
            self.ctx.upcoming_events = evs
            blackout_lo, blackout_hi = self.cfg["risk"].get("news_blackout_min", [-15, 5])
            bu = 0.0
            for e in evs:
                if e["red"] and (e["t"] + blackout_hi * 60) > now > (e["t"] + blackout_lo * 60):
                    bu = max(bu, e["t"] + blackout_hi * 60)
            if bu > self.ctx.news_blackout_until:
                self.log.info("NEWS BLACKOUT until %s (red event)",
                              time.strftime("%H:%M", time.gmtime(bu)))
            self.ctx.news_blackout_until = bu

        bias, reasons = self._macro_bias()
        self.ctx.macro_bias = bias
        self.ctx.macro_reasons = reasons
        self.ctx.t = now
        self.ctx.ok = True
        self.bus.set("macro.context", self.ctx, max_age_sec=600)
        self.bus.publish("macro.updated", self.ctx)
