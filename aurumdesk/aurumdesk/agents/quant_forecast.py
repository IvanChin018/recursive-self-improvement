"""
QuantForecastAgent — lightweight ML edge with strict anti-overfit discipline.

Model: gradient-boosted trees (sklearn) predicting the sign/probability of a
positive 8-bar-forward return on M15, per symbol.

Anti-leak rules baked in:
  * features use only closed bars (shift >= 1)
  * target = future return over horizon (computed on later bars only)
  * walk-forward: train on first N, validate on next block; retrain every
    `retrain_every_bars` using expanding window capped at ~4000 bars
  * model only VOTES; it never trades directly, and its vote is discarded
    if validation hit-rate < 52% or AUC < 0.52
  * deterministic seed for reproducibility
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from ..bus import Bus
from ..indicators import library as ind
from ..models import SignalDirection, StrategyVote, Timeframe
from .base import Agent

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

HORIZON = 8          # M15 bars -> 2 hours ahead


class QuantForecastAgent(Agent):
    name = "QuantForecastAgent"

    def __init__(self, bus: Bus, cfg: dict, log_dir: str = "logs"):
        super().__init__(bus, cfg, log_dir)
        self.store = None
        self.models: Dict[str, object] = {}
        self.stats: Dict[str, dict] = {}
        self._bars_since_train: Dict[str, int] = {}
        self.enabled = cfg["strategies"]["components"].get(
            "ml_forecast", {}).get("enabled", True) and _HAS_SK
        self.retrain_every = cfg["strategies"]["components"].get(
            "ml_forecast", {}).get("retrain_every_bars", 96)
        self.last_votes: Dict[str, StrategyVote] = {}

    # ---------------- features ---------------- #
    @staticmethod
    def _features(o, h, l, c, v) -> tuple[np.ndarray, np.ndarray]:
        n = len(c)
        if n < 300:
            return np.empty((0, 0)), np.empty(0)
        f = []
        r1 = np.diff(np.log(c), prepend=np.log(c[0]))
        f.append(np.clip(r1 * 100, -5, 5))
        f.append(np.clip(pd_rolling_sum(r1, 4) * 100, -10, 10))
        f.append(np.clip(pd_rolling_sum(r1, 16) * 100, -20, 20))
        f.append(ind.rsi(c, 14) - 50)
        macd_line, _, hist = ind.macd(c)
        atr_ = ind.atr(h, l, c, 14)
        f.append(np.nan_to_num(hist / np.maximum(atr_, 1e-9)))
        adx, dip, dim = ind.adx_di(h, l, c, 14)
        f.append(np.nan_to_num(adx - 25))
        f.append(np.nan_to_num(dip - dim))
        up, mid, lo, w = ind.bbands(c, 20, 2.0)
        pb = np.nan_to_num((c - lo) / np.maximum(up - lo, 1e-9) - 0.5)
        f.append(pb * 4)
        rng = np.maximum(h - l, 1e-9)
        f.append(np.nan_to_num((h - np.maximum(o, c)) / rng))
        f.append(np.nan_to_num((np.minimum(o, c) - l) / rng))
        body = np.abs(c - o)
        f.append(np.nan_to_num(body / np.maximum(pd_rolling_mean(body, 20), 1e-9) - 1))
        vol_z = np.nan_to_num((v - pd_rolling_mean(v, 20)) /
                              np.maximum(pd_rolling_std(v, 20), 1e-9))
        f.append(np.clip(vol_z, -4, 4))
        X = np.column_stack(f)
        y_ret = np.full(n, np.nan)
        y_ret[:-HORIZON] = np.log(c[HORIZON:] / c[:-HORIZON])
        return X, y_ret

    # ---------------- training ---------------- #
    def _train(self, sym: str) -> None:
        a = self.store.arrays(sym, Timeframe.M15)
        if a is None:
            return
        X, y = self._features(a["o"], a["h"], a["l"], a["c"], a["v"])
        if X.size == 0:
            return
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[valid], y[valid]
        if len(X) < 400:
            return
        lab = (y > 0).astype(int)
        split = int(len(X) * 0.8)
        Xtr, ytr, Xva, yva = X[:split], lab[:split], X[split:], lab[split:]
        try:
            m = HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.06, max_depth=4,
                min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15,
                random_state=42)
            m.fit(Xtr, ytr)
            proba = m.predict_proba(Xva)[:, 1]
            hit = float(np.mean((proba > 0.5) == (yva == 1)))
            auc = _auc(yva, proba)
            self.models[sym] = m
            self.stats[sym] = dict(hit=hit, auc=auc, n=len(X))
            ok = hit >= 0.52 and auc >= 0.52
            self.bus.set(f"ml.ok.{sym}", ok, max_age_sec=86400)
            self.log.info("ML %s trained: hit=%.3f auc=%.3f n=%d usable=%s",
                          sym, hit, auc, len(X), ok)
        except Exception as exc:
            self.log.warning("ML train failed %s: %s", sym, exc)
            self.models.pop(sym, None)

    # ---------------- tick ---------------- #
    def tick(self) -> None:
        if not self.enabled:
            return
        self.store = self.bus.get("market.store")
        if self.store is None:
            return
        votes = {}
        for sym in self.cfg["symbols"]:
            s = self.store.get(sym, Timeframe.M15)
            if s is None or len(s.t) < 400:
                continue
            nb = self._bars_since_train.get(sym, 0) + 1
            self._bars_since_train[sym] = nb
            if sym not in self.models or nb >= self.retrain_every:
                self._bars_since_train[sym] = 0
                self._train(sym)
            m = self.models.get(sym)
            if m is None or not self.bus.get(f"ml.ok.{sym}", False):
                continue
            a = self.store.arrays(sym, Timeframe.M15)
            X, y = self._features(a["o"], a["h"], a["l"], a["c"], a["v"])
            if X.size == 0:
                continue
            row = X[-1][np.isfinite(X[-1])]
            if len(row) != X.shape[1]:
                continue
            try:
                p = float(m.predict_proba(X[-1:].reshape(1, -1))[0, 1])
            except Exception:
                continue
            if abs(p - 0.5) < 0.06:
                continue
            d = SignalDirection.LONG if p > 0.5 else SignalDirection.SHORT
            votes[sym] = StrategyVote("ml_forecast", sym, d, weight=1.0,
                                      conviction=min(abs(p - 0.5) * 3, 1.0),
                                      horizon_tf=Timeframe.M15,
                                      reasons=[f"ML P(up {HORIZON}xM15)={p:.2f}"])
        self.last_votes = votes
        self.bus.set("ml.votes", votes, max_age_sec=1800)


# ---------------- tiny helpers (avoid pandas import noise) ---------------- #
def pd_rolling_sum(x, n):
    cs = np.cumsum(np.nan_to_num(x))
    out = np.empty_like(x, dtype=float)
    out[n - 1:] = cs[n - 1:] - np.concatenate(([0], cs[:-n]))
    out[:n - 1] = cs[:n - 1]
    return out


def pd_rolling_mean(x, n):
    return pd_rolling_sum(x, n) / n


def pd_rolling_std(x, n):
    m = pd_rolling_mean(x, n)
    m2 = pd_rolling_mean(np.square(np.nan_to_num(x)), n)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def _auc(y, p) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos = max(y.sum(), 1e-9)
    nneg = max(len(y) - y.sum(), 1e-9)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
