"""Desk-wide logging: console + rotating files (trades.csv journal, desk.log)."""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str = "desk", log_dir: str = "logs") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(f"aurumdesk.{name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        try:
            os.makedirs(log_dir, exist_ok=True)
            fh = RotatingFileHandler(os.path.join(log_dir, "desk.log"),
                                     maxBytes=5_000_000, backupCount=5)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass
        logger.propagate = False
    _LOGGERS[name] = logger
    return logger


class TradeJournal:
    """Append-only CSV + JSONL journal. Every decision leaves a fingerprint."""

    FIELDS = ["ts", "event", "symbol", "side", "lots", "price", "sl", "tp",
              "strategies", "confidence", "verdict", "order_id", "proposal_id",
              "pnl", "balance", "equity", "notes"]

    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "trades.csv")
        self.jsonl_path = os.path.join(log_dir, "events.jsonl")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.FIELDS)

    def write(self, event: str, **row: Any) -> None:
        rec = {k: row.get(k, "") for k in self.FIELDS}
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["event"] = event
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([rec[k] for k in self.FIELDS])
        payload = dict(row)
        payload["_ts"] = rec["ts"]
        payload["_event"] = event
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            pass
