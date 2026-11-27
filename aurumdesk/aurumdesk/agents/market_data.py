"""
MarketDataAgent — the desk's single source of truth for price.

* Attaches to MT5 when available (user's local API-linked terminal).
* Falls back / failovers to a synthetic feed so downstream agents never
  receive gaps, NaNs or stale candles without an explicit alert.
* Maintains multi-timeframe OHLCV in MarketStore + live quotes.
* Publishes: data.updated, quote, alert(data_stale), candle_close events.
"""
from __future__ import annotations

import time
from typing import Dict

from ..bus import Bus
from ..data.mt5_feed import MTFeed
from ..data.store import MarketStore
from ..data.synthetic_feed import SyntheticFeed
from ..models import Timeframe
from .base import Agent


class MarketDataAgent(Agent):
    name = "MarketDataAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.symbols: list[str] = cfg["symbols"]
        self.tfs = [Timeframe(t) for t in cfg["timeframes"]]
        self.store = MarketStore(cfg.get("candles_keep", {}))
        self.mt5 = MTFeed(cfg, self.log)
        self.synth = SyntheticFeed()
        self.quotes: Dict[str, object] = {}
        self.source = "none"
        self.stale_after = float(cfg["data"].get("stale_after_sec", 180))
        self._last_hist_fetch: Dict[str, float] = {}
        self._last_candle_t: Dict[str, Dict[Timeframe, int]] = {
            s: {} for s in self.symbols}
        self._synth_bar_bucket = 0

    def setup(self) -> None:
        if self.mt5.connect():
            ok = all(self.mt5.ensure_symbol(s) for s in self.symbols)
            self.source = "mt5" if ok else "mt5-partial"
            self.log.info("MT5 feed active (%s)", self.source)
        else:
            self.log.warning("MT5 unavailable (%s) -> SYNTHETIC paper feed. "
                             "On your PC with MT5 running this auto-switches to real data.",
                             self.mt5.last_error)
            self.source = "synthetic"
        # prime history
        for s in self.symbols:
            for tf in self.tfs:
                self._fetch_history(s, tf, force=True)
        self.bus.set("market.source", self.source)

    # ---------------- fetching ---------------- #
    def _fetch_history(self, symbol: str, tf: Timeframe, force: bool = False) -> None:
        key = f"{symbol}:{tf.value}"
        now = time.time()
        interval = {Timeframe.M1: 20, Timeframe.M5: 45, Timeframe.M15: 90,
                    Timeframe.H1: 180, Timeframe.H4: 600, Timeframe.D1: 1800}[tf]
        if not force and now - self._last_hist_fetch.get(key, 0) < interval:
            return
        count = int(self.cfg.get("candles_keep", {}).get(tf.value, 300))
        candles = []
        if self.source.startswith("mt5"):
            candles = self.mt5.candles(symbol, tf, count)
        if not candles and self.source != "mt5":
            candles = self.synth.candles(symbol, tf, min(count, 400))
        if candles:
            self.store.update_history(symbol, tf, candles)
            self._last_hist_fetch[key] = now

    def _live_bar_from_mt5(self, symbol: str, tf: Timeframe) -> None:
        rows = self.mt5.candles(symbol, tf, 2)
        for cd in rows[-1:]:
            closed_before = self._last_candle_t[symbol].get(tf)
            self.store.update_live_bar(cd)
            self._last_candle_t[symbol][tf] = cd.t
            if closed_before is not None and cd.t > closed_before:
                self.bus.publish("candle_close", {"symbol": symbol, "tf": tf.value})

    def _live_bar_synth(self, symbol: str, tf: Timeframe, now: float) -> None:
        step = {Timeframe.M1: 60, Timeframe.M5: 300, Timeframe.M15: 900,
                Timeframe.H1: 3600, Timeframe.H4: 14400, Timeframe.D1: 86400}[tf]
        q = self.synth.quote(symbol)
        if q is None:
            return
        self.quotes[symbol] = q
        bucket = int(now // step) * step
        prev = self._last_candle_t[symbol].get(tf)
        from ..models import Candle
        bar = Candle(symbol=symbol, timeframe=tf, t=bucket,
                     o=q.mid, h=q.mid, l=q.mid, c=q.mid, volume=100.0)
        if prev is not None and bucket > prev:
            self.bus.publish("candle_close", {"symbol": symbol, "tf": tf.value})
        self.store.update_live_bar(bar)
        self._last_candle_t[symbol][tf] = bucket

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        now = time.time()
        # connection health & failover
        if self.source.startswith("mt5") and not self.mt5.ping():
            self.log.error("MT5 link lost -> switching to synthetic failover feed")
            self.bus.publish("alert", {"level": "critical", "agent": self.name,
                                       "msg": "MT5 disconnected"})
            self.source = "synthetic-failover"
        elif self.source == "synthetic-failover" and self.mt5.connect():
            self.log.info("MT5 recovered — resuming real feed")
            self.source = "mt5"

        if self.source == "synthetic" or self.source == "synthetic-failover":
            shock = self.synth.apply_shocks(now)
            self.synth.advance_to(now)
            if self.source == "synthetic-failover":
                for s in self.symbols:
                    for tf in self.tfs:
                        self._fetch_history(s, tf)
            for s in self.symbols:
                for tf in self.tfs:
                    self._live_bar_synth(s, tf, now)
        else:
            for s in self.symbols:
                q = self.mt5.quote(s)
                if q:
                    self.quotes[s] = q
                for tf in self.tfs:
                    self._fetch_history(s, tf)
                    self._live_bar_from_mt5(s, tf)

        # staleness guard
        stale = []
        for s in self.symbols:
            if self.store.age_sec(s, Timeframe.M5) > self.stale_after:
                stale.append(s)
        self.bus.set("market.quotes", self.quotes, max_age_sec=self.stale_after)
        self.bus.set("market.stale_symbols", stale)
        if stale:
            self.bus.publish("alert", {"level": "warn", "agent": self.name,
                                       "msg": f"stale data: {stale}"})
        self.bus.publish("data.updated", {"t": now, "source": self.source})


class FeedAdapter:
    """Convenience view used by non-agent code (backtests/tests)."""

    def __init__(self, agent: MarketDataAgent):
        self.agent = agent

    def arrays(self, symbol: str, tf: Timeframe):
        return self.agent.store.arrays(symbol, tf)
