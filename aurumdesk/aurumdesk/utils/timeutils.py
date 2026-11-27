"""Session / trading-hours helpers. All times UTC epoch seconds."""
from __future__ import annotations

import datetime as dt
from typing import Tuple

# FX/metals spot trading week (broker server time ~= UTC for our purposes):
# opens ~ Sunday 21:00 UTC, closes ~ Friday 21:00/22:00 UTC (NY 5pm = 21/22 UTC).


def to_dt(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)


def market_open(t: float) -> bool:
    d = to_dt(t)
    wd = d.weekday()          # Mon=0 .. Sun=6
    h = d.hour + d.minute / 60.0
    if wd == 5:               # Saturday
        return False
    if wd == 6:               # Sunday: open from 21:00 UTC
        return h >= 21.0
    if wd == 4:               # Friday: closed after 22:00 UTC
        return h < 22.0
    return True


def session_of(t: float) -> str:
    """Sydney / Asia / London / New York / off-hours (UTC)."""
    d = to_dt(t)
    wd = d.weekday()
    if wd == 5 or (wd == 6 and d.hour < 21):
        return "closed"
    h = d.hour
    if 21 <= h or h < 1:
        return "sydney"
    if 1 <= h < 7:
        return "asia"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "london_ny_overlap"      # the gold liquidity/vol sweet spot
    if 16 <= h < 21:
        return "newyork"
    return "ny_close"                   # 21-24 weekdays


SESSION_LIQUIDITY = {
    "sydney": 0.5, "asia": 0.6, "london": 0.85,
    "london_ny_overlap": 1.0, "newyork": 0.8,
    "ny_close": 0.4, "closed": 0.0,
}

# High-probability momentum window for gold: London/NY overlap + NY AM.
MOMENTUM_HOURS = set(list(range(12, 17)))


def in_momentum_window(t: float) -> bool:
    return to_dt(t).hour in MOMENTUM_HOURS


def utc_day(t: float) -> str:
    return to_dt(t).strftime("%Y-%m-%d")


def utc_week_key(t: float) -> str:
    iso = to_dt(t).isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def is_friday_close_time(t: float) -> bool:
    d = to_dt(t)
    return d.weekday() == 4 and d.hour >= 21
