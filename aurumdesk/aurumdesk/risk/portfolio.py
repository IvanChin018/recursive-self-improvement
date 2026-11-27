"""Portfolio math for the Allocation & Risk agents."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def kelly_fraction(win_rate: float, avg_win_r: float, avg_loss_r: float) -> float:
    """Kelly sizing in R-multiples; caller applies fractional cap (<=25%)."""
    if avg_loss_r <= 0 or avg_win_r <= 0:
        return 0.0
    q = 1.0 - win_rate
    k = win_rate / avg_loss_r - q / avg_win_r
    return max(0.0, min(k, 1.0))


def ewma_corr_matrix(returns: Sequence[Sequence[float]], halflife: int = 60) -> np.ndarray:
    T = len(next(iter(returns), []))
    if T < 10 or len(returns) < 2:
        n = len(returns)
        return np.eye(n)
    w = np.exp(np.log(0.5) * (T - np.arange(T)) / halflife)
    w /= w.sum()
    X = np.asarray(returns, float)
    X = X - np.average(X, axis=1, weights=w).reshape(-1, 1)
    cov = (X * w) @ X.T
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-9
    C = cov / np.outer(d, d)
    return np.clip(C, -1, 1)


def effective_exposure(weights_lots_value: Dict[str, float], corr: np.ndarray,
                       keys: List[str]) -> float:
    """sqrt(w' C w) — correlated books count more than once."""
    idx = {k: i for i, k in enumerate(keys)}
    v = np.array([weights_lots_value.get(k, 0.0) for k in keys])
    sub = corr[np.ix_([idx[k] for k in keys], [idx[k] for k in keys])] \
        if all(k in idx for k in keys) else np.eye(len(keys))
    return float(np.sqrt(max(v @ sub @ v, 0.0)))


def atr_stop(price: float, atr: float, side_sign: int, mult: float,
             buffer: float = 0.0) -> float:
    dist = max(atr * mult, 1e-9) + buffer
    return price - side_sign * dist


def position_size_by_risk(equity: float, risk_pct: float, stop_dist: float,
                          value_per_unit: float) -> float:
    """lots = cash_risk / (stop_dist * acct_ccy_value_of_1_price_move_per_lot)."""
    if stop_dist <= 0 or value_per_unit <= 0:
        return 0.0
    cash = equity * risk_pct / 100.0
    return cash / (stop_dist * value_per_unit)


def round_step(x: float, step: float, mode: str = "down") -> float:
    if step <= 0:
        return x
    if mode == "down":
        return math.floor(x / step + 1e-9) * step
    return math.ceil(x / step - 1e-9) * step


def _ewma(xs: Sequence[float], halflife: int = 12) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    a = 1 - 0.5 ** (1 / halflife)
    m = xs[0]
    for x in xs[1:]:
        m = m + a * (x - m)
    return m


def sharpe(returns: Sequence[float]) -> float:
    r = np.asarray(returns, float)
    if len(r) < 5 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(252))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    e = np.asarray(equity_curve, float)
    if len(e) < 2:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / np.maximum(peak, 1e-9)
    return float(abs(dd.min()))
