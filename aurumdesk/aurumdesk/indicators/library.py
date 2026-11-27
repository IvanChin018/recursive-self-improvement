"""
Vectorized technical indicators (numpy/pandas only — no TA-Lib dependency).

All functions accept/return numpy arrays and align to the input length,
using only past data (no lookahead): value at index i uses bars 0..i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
#  Moving averages
# ------------------------------------------------------------------ #
def sma(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(n, min_periods=n).mean().to_numpy()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    return s.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def wma(x: np.ndarray, n: int) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=float)
    s = pd.Series(x)
    out = s.rolling(n).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)
    return out.to_numpy()


def hma(x: np.ndarray, n: int) -> np.ndarray:
    half = max(int(n / 2), 1)
    return wma(2.0 * wma(x, half) - wma(x, n), max(int(np.sqrt(n)), 1))


def kama(x: np.ndarray, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    x = np.asarray(x, float)
    ch = np.abs(np.diff(x, prepend=x[0]))
    vol = pd.Series(ch).rolling(n).sum().to_numpy()
    er = np.where(vol > 0, np.abs(np.diff(x, prepend=x[0]) * 0 + (x - np.roll(x, n))) / np.maximum(vol, 1e-12), 0.0)
    er[:n] = 0.0
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + sc[i] * (x[i] - out[i - 1])
    return out


# ------------------------------------------------------------------ #
#  Volatility
# ------------------------------------------------------------------ #
def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    tr = true_range(h, l, c)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def rvi_ema_std(x: np.ndarray, n: int = 10) -> np.ndarray:
    return pd.Series(x).rolling(n).std(ddof=0).to_numpy()


def bbands(x: np.ndarray, n: int = 20, k: float = 2.0):
    mid = sma(x, n)
    sd = pd.Series(x).rolling(n).std(ddof=0).to_numpy()
    up, lo = mid + k * sd, mid - k * sd
    width = np.where(mid > 0, (up - lo) / mid, np.nan)
    rng = np.nanmax(x[-n:], axis=-1) - np.nanmin(x[-n:], axis=-1) if False else None
    return up, mid, lo, width


def donchian(h: np.ndarray, l: np.ndarray, n: int = 20):
    up = pd.Series(h).rolling(n, min_periods=n).max().to_numpy()
    lo = pd.Series(l).rolling(n, min_periods=n).min().to_numpy()
    return up, lo


def parkinson(h: np.ndarray, l: np.ndarray, n: int = 20) -> np.ndarray:
    pl = np.log(np.maximum(h, 1e-12) / np.maximum(l, 1e-12)) ** 2
    return np.sqrt(pd.Series(pl).rolling(n).mean().to_numpy() / (4 * np.log(2)))


def garman_klass(o, h, l, c, n: int = 20) -> np.ndarray:
    o, h, l, c = map(lambda a: np.maximum(np.asarray(a, float), 1e-12), (o, h, l, c))
    var = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(np.maximum(pd.Series(var).rolling(n).mean().to_numpy(), 0.0))


# ------------------------------------------------------------------ #
#  Oscillators
# ------------------------------------------------------------------ #
def rsi(x: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(x, prepend=x[0])
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    rs = (pd.Series(up).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() /
          pd.Series(dn).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().replace(0, np.nan))
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def stoch_k(h, l, c, n: int = 14) -> np.ndarray:
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(l).rolling(n).min().to_numpy()
    denom = np.maximum(hh - ll, 1e-12)
    return 100 * (np.asarray(c) - ll) / denom


def stoch_kdf(h, l, c, n: int = 14, d: int = 3):
    k = stoch_k(h, l, c, n)
    kk = sma(k, d)
    dd = sma(kk, d)
    return k, kk, dd


def macd(x: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(x, fast) - ema(x, slow)
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist


def cci(h, l, c, n: int = 20) -> np.ndarray:
    tp = (np.asarray(h) + np.asarray(l) + np.asarray(c)) / 3.0
    m = sma(tp, n)
    md = pd.Series(np.abs(tp - m)).rolling(n).mean().to_numpy()
    return (tp - m) / np.maximum(0.015 * md, 1e-12)


def williams_r(h, l, c, n: int = 14) -> np.ndarray:
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(l).rolling(n).min().to_numpy()
    return -100 * (hh - np.asarray(c)) / np.maximum(hh - ll, 1e-12)


def roc(x: np.ndarray, n: int = 12) -> np.ndarray:
    prev = np.roll(x, n)
    prev[:n] = x[:n]
    return 100 * (x - prev) / np.maximum(np.abs(prev), 1e-12)


def momentum(x: np.ndarray, n: int = 10) -> np.ndarray:
    prev = np.roll(x, n)
    prev[:n] = x[:n]
    return x - prev


def adx_di(h, l, c, n: int = 14):
    h, l, c = map(np.asarray, (h, l, c))
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_s = pd.Series(true_range(h, l, c)).ewm(alpha=1 / n, adjust=False).mean()
    plus = pd.Series(plus_dm).ewm(alpha=1 / n, adjust=False).mean()
    minus = pd.Series(minus_dm).ewm(alpha=1 / n, adjust=False).mean()
    di_p = 100 * (plus / tr_s.replace(0, np.nan)).fillna(0)
    di_m = 100 * (minus / tr_s.replace(0, np.nan)).fillna(0)
    dx = 100 * ((di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx.to_numpy(), di_p.to_numpy(), di_m.to_numpy()


def mfi(h, l, c, v, n: int = 14) -> np.ndarray:
    tp = (np.asarray(h) + np.asarray(l) + np.asarray(c)) / 3.0
    d = np.diff(tp, prepend=tp[0])
    pos = np.where(d > 0, tp * np.asarray(v), 0.0)
    neg = np.where(d < 0, tp * np.asarray(v), 0.0)
    ps = pd.Series(pos).rolling(n).sum()
    ns = pd.Series(neg).rolling(n).sum().replace(0, np.nan)
    return (100 - 100 / (1 + ps / ns)).fillna(50).to_numpy()


def obv(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    c, v = np.asarray(c), np.asarray(v)
    sign = np.sign(np.diff(c, prepend=c[0]))
    return np.cumsum(sign * v)


def cmf(h, l, c, v, n: int = 20) -> np.ndarray:
    h, l, c, v = map(np.asarray, (h, l, c, v))
    mult = ((c - l) - (h - c)) / np.maximum(h - l, 1e-12)
    vf = mult * v
    return (pd.Series(vf).rolling(n).sum() /
            pd.Series(v).rolling(n).sum().replace(0, np.nan)).fillna(0).to_numpy()


# ------------------------------------------------------------------ #
#  Structure helpers
# ------------------------------------------------------------------ #
def swing_points(arr: np.ndarray, lookback: int = 3):
    """Return boolean masks for local highs/lows using centered window."""
    arr = np.asarray(arr, float)
    n = len(arr)
    hi = np.zeros(n, bool)
    lo = np.zeros(n, bool)
    for i in range(lookback, n - lookback):
        w = arr[i - lookback:i + lookback + 1]
        if arr[i] == w.max() and (w.argmax() == lookback):
            hi[i] = True
        if arr[i] == w.min() and (w.argmin() == lookback):
            lo[i] = True
    return hi, lo


def pct_rank(x: np.ndarray, n: int = 100) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(n, min_periods=max(5, n // 4)).rank(pct=True).to_numpy()


def zscore(x: np.ndarray, n: int = 50) -> np.ndarray:
    m = sma(x, n)
    sd = pd.Series(x).rolling(n).std(ddof=0).to_numpy()
    return (np.asarray(x) - m) / np.maximum(sd, 1e-12)


def realized_vol(c: np.ndarray, n: int = 20) -> np.ndarray:
    r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(max(c[0], 1e-12)))
    return pd.Series(r).rolling(n).std(ddof=0).to_numpy() * np.sqrt(365 * 24 * 4)


def hurst(x: np.ndarray, max_lag: int = 40) -> float:
    """Cheap Hurst exponent estimate via variance scaling."""
    x = np.asarray(x, float)
    x = x[-400:]
    if len(x) < 80:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        d = x[lag:] - x[:-lag]
        tau.append(np.sqrt(np.mean(d * d) + 1e-18))
    try:
        p = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return float(np.clip(p, 0.0, 1.0))
    except Exception:
        return 0.5


# ------------------------------------------------------------------ #
#  Candle anatomy (used by the Candle Pattern Agent)
# ------------------------------------------------------------------ #
def candle_anatomy(o, h, l, c):
    o, h, l, c = map(lambda a: np.asarray(a, float), (o, h, l, c))
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = c > o
    return dict(rng=rng, body=body, upper=upper, lower=lower,
                bull=bull, body_pct=body / rng, up_pct=upper / rng,
                low_pct=lower / rng)
