"""
Desk — wires every agent together and runs the deterministic tick loop.

Pipeline order each cycle (this ordering IS the risk architecture):

  1. MarketData      refresh candles/quotes, flag stale symbols
  2. NewsMacro       macro context, event blackout windows
  3. Technical       multi-TF technical reads
  4. CandlePattern   candlestick signals on closed bars
  5. SentimentRegime regime classification per symbol
  6. QuantForecast   ML directional votes
  7. Execution(poll) drain pending orders/retries   [management closes first]
  8. Account snapshot -> blackboard (broker truth)
  9. RiskManager     kill switches + evaluate proposals
 10. StrategyEnsemble produce new proposals (skipped while halted)
 11. PortfolioAlloc  position management actions (BE/trail/partial/news trim)
 12. Execution       send approved entries & mgmt orders
 13. Journal         bookkeeping + strategy feedback loop
 14. Compliance      audit everything that just happened
 15. Overseer        health check + dashboard
"""
from __future__ import annotations

import signal
import threading
import time
from typing import List, Optional

from .agents.allocation import PortfolioAllocAgent
from .agents.base import Agent
from .agents.candle import CandlePatternAgent
from .agents.compliance import ComplianceAgent
from .agents.execution import ExecutionAgent
from .agents.journal import JournalAgent
from .agents.market_data import MarketDataAgent
from .agents.news_macro import NewsMacroAgent
from .agents.overseer import OverseerAgent
from .agents.quant_forecast import QuantForecastAgent
from .agents.risk import RiskManagerAgent
from .agents.sentiment import SentimentRegimeAgent
from .agents.strategy import StrategyEnsembleAgent
from .agents.technical import TechnicalAnalystAgent
from .bus import Bus
from .config import load_config
from .execution.gateway import BrokerGateway
from .utils.logging import get_logger


class Desk:
    def __init__(self, cfg: Optional[dict] = None, config_path: Optional[str] = None):
        self.cfg = cfg or load_config(config_path)
        log_dir = self.cfg["overseer"].get("log_dir", "logs")
        self.log = get_logger("desk", log_dir)
        self.bus = Bus()
        self.stop_evt = threading.Event()

        # ---- market data & broker ---- #
        self.market = MarketDataAgent(self.bus, self.cfg, log_dir)
        self.bus.set("market.store", self.market.store)
        self.gateway = BrokerGateway(self.cfg, mt5_feed=self.market.mt5, log=self.log)
        self.bus.set("broker.gateway", self.gateway)
        self.bus.set("broker.spec_fn", self.gateway.spec)
        self.bus.set("broker.backend", self.gateway.backend)

        # ---- analysts ---- #
        self.news = NewsMacroAgent(self.bus, self.cfg, log_dir)
        self.tech = TechnicalAnalystAgent(self.bus, self.cfg, log_dir)
        self.candle = CandlePatternAgent(self.bus, self.cfg, log_dir)
        self.regime = SentimentRegimeAgent(self.bus, self.cfg, log_dir)
        self.ml = QuantForecastAgent(self.bus, self.cfg, log_dir)

        # ---- decision layer ---- #
        self.strategy = StrategyEnsembleAgent(self.bus, self.cfg, log_dir)
        self.risk = RiskManagerAgent(self.bus, self.cfg, log_dir)
        self.risk.gateway = self.gateway
        self.alloc = PortfolioAllocAgent(self.bus, self.cfg, log_dir)
        self.alloc.gateway = self.gateway
        self.exec_ = ExecutionAgent(self.bus, self.cfg, log_dir)
        self.exec_.gateway = self.gateway

        # ---- control layer ---- #
        self.journal = JournalAgent(self.bus, self.cfg, log_dir)
        self.journal.gateway = self.gateway
        self.journal.ensemble = self.strategy
        self.compliance = ComplianceAgent(self.bus, self.cfg, log_dir)
        self.overseer = OverseerAgent(self.bus, self.cfg, log_dir)

        # proposals queue: strategy -> risk consumes via blackboard
        def _queue_proposals(topic, payload):
            cur = self.bus.get("pending.proposals", [], allow_stale=True)
            self.bus.set("pending.proposals", cur + list(payload))
        self.bus.subscribe("proposals", _queue_proposals)

        self.pipeline: List[Agent] = [
            self.market, self.news, self.tech, self.candle, self.regime, self.ml,
            self.risk, self.strategy, self.alloc, self.exec_,
            self.journal, self.compliance, self.overseer,
        ]
        self.overseer.register(self.pipeline)
        for a in self.pipeline:
            a._born_t = time.time()

    # ---------------- lifecycle ---------------- #
    def setup(self) -> None:
        self.log.info("AurumDesk starting | mode=%s backend=%s symbols=%s",
                      self.cfg.get("mode"), self.gateway.backend, self.cfg["symbols"])
        for a in self.pipeline:
            try:
                a.setup()
            except Exception as exc:
                self.log.exception("setup failed for %s: %s", a.name, exc)
                self.bus.publish("alert", {"level": "critical", "agent": a.name,
                                           "msg": f"setup failed: {exc}"})

    def cycle(self) -> None:
        # 1-6 analysis
        for a in (self.market, self.news, self.tech, self.candle,
                  self.regime, self.ml):
            a.safe_tick()
        # 7-8 broker truth: fx feeds, exits engine + account snapshot
        quotes = self.bus.get("market.quotes", {})
        try:
            for f, r in (self.bus.get("market.fx_rates", {}) or {}).items():
                if self.gateway.paper is not None:
                    self.gateway.paper.set_fx(f, r)
            for ev in self.gateway.tick_exits(quotes):
                self.bus.publish("broker.exit", ev)
                self.log.info("BROKER EXIT %s %s pnl=%.2f @ %.2f",
                              ev["symbol"], ev["reason"], ev["pnl"], ev["price"])
            if self.gateway.paper is not None:
                self.gateway.paper.apply_swaps(quotes)
            snap = self.gateway.account(quotes)
            if snap is not None:
                self.bus.set("account.snapshot", snap, max_age_sec=120)
        except Exception as exc:
            self.log.exception("broker poll failed: %s", exc)
        # 9 risk kill-switches evaluate queued proposals
        self.risk.safe_tick()
        # 10 strategy produces proposals (no-op while halted)
        self.strategy.safe_tick()
        # 11 allocation / position management
        self.alloc.safe_tick()
        # 12 execution (entries + mgmt)
        self.exec_.safe_tick()
        # 13-15 control layer
        self.journal.safe_tick()
        self.compliance.safe_tick()
        self.overseer.safe_tick()
        self.gateway.save_state()
        self.strategy.save()

    def run_forever(self) -> None:
        self.setup()
        poll = float(self.cfg.get("poll_seconds", 20))

        def _sigint(*_):
            self.log.warning("SIGINT received -> shutting down")
            self.stop_evt.set()
        try:
            signal.signal(signal.SIGINT, _sigint)
            signal.signal(signal.SIGTERM, _sigint)
        except ValueError:
            pass  # not main thread

        n = 0
        while not self.stop_evt.is_set():
            t0 = time.time()
            try:
                self.cycle()
            except Exception as exc:
                self.log.exception("cycle crashed: %s", exc)
                self.bus.publish("alert", {"level": "critical", "agent": "Desk",
                                           "msg": f"cycle crash: {exc}"})
            n += 1
            if n % 300 == 0:
                self.market.prune_check() if hasattr(self.market, "prune_check") else None
            elapsed = time.time() - t0
            self.stop_evt.wait(max(1.0, poll - elapsed))
        self.shutdown()

    def shutdown(self) -> None:
        self.overseer.shutdown()
        try:
            self.market.mt5.shutdown()
        except Exception:
            pass
        self.log.info("AurumDesk stopped.")
