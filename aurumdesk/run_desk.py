#!/usr/bin/env python3
"""
AurumDesk entry point.

Usage:
  python run_desk.py                 # paper trading (default)
  python run_desk.py --cycles 50     # run N cycles then exit (smoke test)
  python run_desk.py --config my.yaml
  python run_desk.py --status        # print last dashboard snapshot & exit
  python run_desk.py --report        # print journal performance report & exit

On a Windows PC with an MT5 (demo) terminal running and the MetaTrader5
package installed, the desk automatically attaches to it and paper-trades on
real broker data. Everywhere else it falls back to a realistic synthetic
metals feed so you can validate the whole pipeline first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description="AurumDesk AI metals trading desk")
    ap.add_argument("--config", default=None, help="path to settings.yaml")
    ap.add_argument("--cycles", type=int, default=0,
                    help="run N cycles and exit (0 = forever)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)                      # state/logs/ relative paths are stable
    cfg_path = args.config or os.path.join(root, "config", "settings.yaml")

    if args.status:
        p = os.path.join(root, "state", "overseer.json")
        print(json.dumps(json.load(open(p)) if os.path.exists(p) else {}, indent=1))
        return 0

    if args.report:
        from aurumdesk.agents.journal import JournalAgent
        from aurumdesk.bus import Bus
        from aurumdesk.config import load_config
        cfg = load_config(cfg_path)
        ja = JournalAgent(Bus(), cfg, cfg["overseer"].get("log_dir", "logs"))
        st = ja.compute_stats()
        print(json.dumps(st, indent=1))
        return 0

    from aurumdesk.desk import Desk
    from aurumdesk.config import load_config
    cfg = load_config(cfg_path)
    desk = Desk(cfg)

    if args.cycles > 0:
        desk.setup()
        for i in range(args.cycles):
            desk.cycle()
            if i % 10 == 0:
                d = desk.bus.get("desk.dashboard", {})
                print(f"cycle {i+1}/{args.cycles}: "
                      f"eq={d.get('equity', 0):.2f} pos={d.get('n_positions', 0)} "
                      f"N={d.get('n_trades', 0)} src={d.get('source')}", flush=True)
        desk.shutdown()
        d = desk.bus.get("desk.dashboard", {})
        print("FINAL:", json.dumps(d, indent=1, default=str))
        return 0

    print("AurumDesk live loop — Ctrl+C to stop gracefully.")
    desk.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
