"""
OverseerAgent — the desk supervisor / chief of staff.

Watches everything else:
  * heartbeats: any agent whose tick is stale -> alert + it gets skipped,
    never allowed to inject blind decisions
  * error rates per agent (spikes are alerted)
  * data source health (mt5 vs synthetic), feed failover suggestions
  * aggregates a live dashboard dict on the blackboard ("desk.dashboard")
    and periodically prints a one-screen status board to console/log
  * escalation ladder: warn -> critical -> FULL STOP (sets risk.halted and
    blocks execution) when multiple subsystems are unhealthy
  * graceful shutdown: persists all state (broker, ensemble, journal)
It has no opinion on trades — only on the health of the machine that makes them.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from ..bus import Bus
from .base import Agent


class OverseerAgent(Agent):
    name = "OverseerAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.agents: List[Agent] = []
        self._last_print = 0.0
        self._prev_errors: Dict[str, int] = {}
        self.critical_streak = 0
        self.state_path = os.path.join(cfg.get("overseer", {}).get("state_dir", "state"),
                                       "overseer.json")
        bus.subscribe("alert", self._on_alert)

    def register(self, agents: List[Agent]) -> None:
        self.agents = list(agents)

    def _on_alert(self, topic, payload) -> None:
        lvl = payload.get("level", "info")
        msg = f"{payload.get('agent', '?')}: {payload.get('msg', '')}"
        if lvl == "critical":
            self.critical_streak += 1
            self.log.error("CRITICAL ALERT (%d recent) %s", self.critical_streak, msg)
        elif lvl == "error":
            self.log.error("ERROR ALERT %s", msg)
        else:
            self.log.warning("ALERT [%s] %s", lvl, msg)

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        now = time.time()
        hb_timeout = self.cfg["overseer"].get("heartbeat_timeout_sec", 120)
        unhealthy: List[str] = []
        for a in self.agents:
            if a.ticks == 0 and now - getattr(a, "_born_t", now) > 3 * hb_timeout:
                unhealthy.append(f"{a.name}:never-ticked")
                continue
            if a.last_tick_t and not a.heartbeat_ok(hb_timeout):
                unhealthy.append(f"{a.name}:stale")
            prev = self._prev_errors.get(a.name, 0)
            if a.errors - prev >= 3:
                unhealthy.append(f"{a.name}:{a.errors - prev} errors")
                self.bus.publish("alert", {"level": "error", "agent": a.name,
                                           "msg": f"error spike: {a.last_error}"})
            self._prev_errors[a.name] = a.errors

        source = self.bus.get("market.source", "?")
        snap = self.bus.get("account.snapshot")
        stats = self.bus.get("journal.stats", {}) or {}
        dash = dict(
            t=now, source=source, backend=self.bus.get("broker.backend", "?"),
            halted=bool(self.bus.get("risk.halted", False)),
            kill_reason=self.bus.get("risk.kill_reason", ""),
            n_positions=len(snap.open_positions) if snap else 0,
            equity=round(snap.equity, 2) if snap else 0.0,
            balance=round(snap.balance, 2) if snap else 0.0,
            realized_today=round(snap.realized_today, 2) if snap else 0.0,
            win_rate=round(stats.get("win_rate", 0), 3),
            expectancy_r=round(stats.get("expectancy_r", 0), 3),
            n_trades=stats.get("n_trades", 0),
            violations=self.bus.get("compliance.violation_count", 0),
            unhealthy=unhealthy,
            alerts_critical=self.critical_streak,
        )
        self.bus.set("desk.dashboard", dash, max_age_sec=None)

        # escalation: several unhealthy subsystems -> full stop new entries
        if len(unhealthy) >= 3:
            if not self.bus.get("risk.halted", False):
                self.bus.set("risk.halted", True)
                self.bus.set("risk.kill_reason", "overseer: system degradation")
                self.bus.publish("alert", {"level": "critical", "agent": self.name,
                                           "msg": f"FULL STOP — unhealthy: {unhealthy}"})
        elif unhealthy:
            self.log.warning("health issues: %s", unhealthy)

        # periodic console board
        if now - self._last_print > 60:
            self._last_print = now
            self._print_board(dash)
            self._save(dash)

    def _print_board(self, d: dict) -> None:
        line = (f"[DESK] src={d['source']} be={d['backend']} "
                f"eq={d['equity']:.2f} bal={d['balance']:.2f} "
                f"pos={d['n_positions']} today={d['realized_today']:+.2f} "
                f"WR={d['win_rate']:.0%} E[R]={d['expectancy_r']:+.2f} "
                f"N={d['n_trades']} {'HALTED:' + d['kill_reason'] if d['halted'] else 'ARMED'}")
        self.log.info(line)
        print(line, flush=True)

    def _save(self, d: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=1, default=str)
        except Exception:
            pass

    def shutdown(self) -> None:
        self.log.info("overseer: persisting final state")
        gw = self.bus.get("broker.gateway")
        if gw is not None:
            gw.save_state()
        for a in self.agents:
            for meth in ("save",):
                fn = getattr(a, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
