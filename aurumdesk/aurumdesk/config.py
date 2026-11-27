"""
Config loader for AurumDesk.  YAML file -> nested dicts with safe defaults.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "mode": "paper",                      # paper | live
    "timezone": "UTC",
    "symbols": ["XAUUSD", "XAUAUD", "XAUEUR", "XAGUSD", "XAGAUD", "XAGEUR"],
    "account_currency": "USD",
    "timeframes": ["M5", "M15", "H1", "H4", "D1"],
    "poll_seconds": 20,                   # main loop cadence
    "candles_keep": {"M5": 600, "M15": 600, "H1": 800, "H4": 500, "D1": 600},

    "mt5": {
        "enabled": True,                  # try to attach to local MT5 terminal
        "terminal_path": "",              # optional: path to terminal64.exe
        "login": 0,
        "password": "",
        "server": "",
        "symbol_map": {                   # our names -> broker symbol names
            "XAUUSD": "XAUUSD", "XAUAUD": "XAUAUD", "XAUEUR": "XAUEUR",
            "XAGUSD": "XAGUSD", "XAGAUD": "XAGAUD", "XAGEUR": "XAGEUR",
        },
    },

    "data": {
        "fallback": "synthetic",          # synthetic GBM+vol-cluster feed if no MT5
        "stale_after_sec": 180,           # candle age beyond which data is stale
        "cache_dir": "data",
    },

    "strategies": {
        "min_confidence": 0.55,
        "require_two_agreeing": True,     # at least 2 weighted votes aligned
        "weights_init": 1.0,
        "weight_halflife_days": 14,
        "components": {
            "trend": {"enabled": True},
            "breakout": {"enabled": True},
            "meanrev": {"enabled": True},
            "candlestick": {"enabled": True},
            "carry_macro": {"enabled": True},
            "pairs_relative": {"enabled": True},
            "session_momentum": {"enabled": True},
            "ml_forecast": {"enabled": True, "retrain_every_bars": 96},
        },
    },

    "risk": {
        "risk_per_trade_pct": 0.75,       # % equity risked per trade (SL distance)
        "conf_scale": [0.5, 1.25],        # size multiplier range by confidence
        "max_position_lots": 2.0,
        "max_open_positions": 8,
        "max_exposure_pct_equity": 30.0,  # notional / equity
        "max_margin_used_pct": 25.0,
        "per_symbol_daily_loss_pct": 1.5, # kill that symbol for the day
        "daily_loss_limit_pct": 3.0,      # halt new entries for the day
        "weekly_drawdown_halt_pct": 6.0,  # halt desk until next week
        "max_drawdown_reducer_pct": 12.0, # halve all sizing below this equity peak DD
        "news_blackout_min": [-15, 5],    # minutes around red events: no new trades
        "news_close_reduce_pct": 50,      # % of exposure trimmed before red events
        "correlation_cap": {              # combined weight cap across correlated books
            "gold_group": 2.0, "silver_group": 2.0, "usd_short_group": 2.5,
        },
        "stop": {
            "atr_mult": 1.8, "min_atr": 0.5, "max_atr": 3.5,
            "structure_buffer_atr": 0.25, "breakeven_atr": 1.0,
            "trail_atr": 1.5, "tp_rr": 2.0, "partial_tp": {
                "enabled": True, "at_rr": 1.0, "close_pct": 50,
            },
        },
        "spread_max_atr_frac": 0.12,      # skip if spread > 12% of ATR
        "slippage_max_points": 30,
        "weekend_flat": True,             # close everything Friday NY close
        "max_trade_age_hours": 36,        # time-stop default
    },

    "execution": {
        "deviation_slippage_points": 20,
        "retry_max": 3,
        "retry_backoff_sec": 2.0,
        "magic_base": 20261128,
        "comment_prefix": "AurumDesk",
        "trailing_check_sec": 15,
    },

    "overseer": {
        "heartbeat_timeout_sec": 120,
        "alert_on_consecutive_losses": 5,
        "log_dir": "logs",
        "state_dir": "state",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "config", "settings.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user)
    cfg["_config_path"] = path
    return cfg
