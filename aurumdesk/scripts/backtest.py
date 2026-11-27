#!/usr/bin/env python3
"""AurumDesk fast backtester: replays synthetic (or exported) metals history
through the real Strategy->Risk sizing math to sanity-check edge & risk knobs.

Not a full tick simulator — treat results as directional validation only.
Usage: python scripts/backtest.py [--days 60] [--seed 11]
"""
from __future__ import annotations
import argparse, os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aurumdesk.config import load_config
from aurumdesk.indicators import library as ind
from aurumdesk.models import Side, SignalDirection, Timeframe, TradeProposal
from aurumdesk.risk.portfolio import position_size_by_risk, round_step
from aurumdesk.instruments import get_spec
from aurumdesk.strategies.library import (TrendFollowing, Breakout, MeanReversion,
    CandleEdge, CarryMacro, PairsRelative, SessionMomentum)
from aurumdesk.data.synthetic_feed import SyntheticFeed
from aurumdesk.models import Candle

class _Store:
    """minimal store interface over numpy arrays"""
    def __init__(self, data): self.data = data   # sym -> dict(t,o,h,l,c,v)
    def arrays(self, sym, tf): return self.data.get(sym)
    def get(self, sym, tf):
        a = self.data.get(sym)
        if a is None: return None
        class S: t=a["t"]; o=a["o"]; h=a["h"]; l=a["l"]; c=a["c"]; v=a["v"]
        return S()
class _Ctx:
    now=0.0; macro=None; candle_signals={}; fx_momentum={}
class _VoteCtx(_Ctx):
    def __init__(self, now): self.now=now; self.macro=None; self.candle_signals={}; self.fx_momentum={}

def gen_history(synth, sym, n, step=900, end_t=None):
    """Build an OHLCV history consistent with the live SyntheticFeed dynamics.

    Base metals follow a mean-reverting OU log-price (drift + vol from the
    feed params).  Crosses (XAUAUD etc.) are derived as metalUSD / XXXUSD so
    their candles are internally consistent with the legs — exactly like the
    live feed does it.
    """
    from aurumdesk.data.synthetic_feed import CROSS_DEFS, SESSION_VOL_MULT
    if sym in CROSS_DEFS:
        num, den = CROSS_DEFS[sym]
        pn = gen_history(synth, num, n, step, end_t)
        pd_ = gen_history(synth, den, n, step, end_t)
        ratio = pn["o"] / np.maximum(pd_["o"], 1e-9)
        rc = pn["c"] / np.maximum(pd_["c"], 1e-9)
        rh = pn["h"] / np.maximum(pd_["l"], 1e-9)
        rl = pn["l"] / np.maximum(pd_["h"], 1e-9)
        return dict(t=pn["t"], o=ratio, h=rh, l=rl, c=rc,
                    v=np.full(n, 1000.0))
    base = synth.params[sym]
    rng = random.Random(abs(hash(sym)) % 100000 + (getattr(synth, "seed", None) or 0))
    p = base["p0"]
    dt = step / 86400.0
    drift = float(base.get("drift", 0.05))     # mild positive metals drift/yr
    vol = float(base["vol"])                   # annualized vol
    sig = vol * math.sqrt(dt)
    t0 = int(end_t // step * step) - n * step if end_t else 0
    ts, os_, hs, ls, cs, vs = [], [], [], [], [], []
    for i in range(n):
        o = p
        # stochastic vol (OU around 1.0) + session profile — mirrors live feed
        vo = max(0.4, min(2.5, (ts and 1.0) or 1.0))
        hour = ((t0 + i * step) / 3600.0) % 24
        smult = SESSION_VOL_MULT.get(int(hour), 1.0)
        c = o * math.exp((drift / 365.0 - 0.5 * vol * vol) * dt
                         + vol * smult * math.sqrt(dt) * rng.gauss(0, 1))
        wick = abs(rng.gauss(0, sig * smult)) * o * 0.7
        h = max(o, c) + wick * rng.random()
        l = min(o, c) - wick * rng.random()
        ts.append(t0 + i * step); os_.append(o); hs.append(h)
        ls.append(l); cs.append(c); vs.append(rng.uniform(500, 2500))
        p = c
    a = np.array
    return dict(t=a(ts), o=a(os_), h=a(hs), l=a(ls), c=a(cs), v=a(vs))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--days",type=int,default=45)
    ap.add_argument("--seed",type=int,default=11); a=ap.parse_args()
    cfg=load_config(None); random.seed(a.seed); np.random.seed(a.seed)
    synth=SyntheticFeed(seed=a.seed)
    syms=cfg["symbols"]; n=a.days*96
    now=synth.t0
    data={s:gen_history(synth,s,n,900,now) for s in syms}
    store=_Store(data)
    full={s:dict(d) for s,d in data.items()}   # immutable master copies
    strats=[TrendFollowing(),Breakout(),MeanReversion(),CandleEdge(),CarryMacro(),PairsRelative(),SessionMomentum()]
    equity=10_000.0; peak=equity; curve=[]; trades=[]; warm=200
    stop_cfg=cfg["risk"]["stop"]
    max_pos=int(cfg["risk"].get("max_open_positions",8))
    open_trades=[]          # list of dicts: symbol, side, entry, sl, tp, lots, vpu, j index
    def vote_at(i):
        votes_all={}
        for s in syms:
            d={k:(full[s][k][:i+1]) for k,v in data[s].items()}
            store.data[s]=d
            ctx=_VoteCtx(float(d["t"][-1]))
            vs=[]
            for st in strats:
                try: v=st.vote(store,s,ctx)
                except Exception: v=None
                if v: vs.append(v)
            votes_all[s]=vs
        return votes_all
    i=warm
    while i<n-1:
        # 1) manage existing trades bar-by-bar (conservative: SL checked first)
        still=[]
        for tr in open_trades:
            hi=full[tr["symbol"]]["h"][tr["j"]]; lo=full[tr["symbol"]]["l"][tr["j"]]
            out=None; reason="time"
            if tr["side"]==Side.BUY:
                if lo<=tr["sl"]: out=tr["sl"]-tr["entry"]; reason="sl"
                elif hi>=tr["tp"]: out=tr["tp"]-tr["entry"]; reason="tp"
            else:
                if hi>=tr["sl"]: out=tr["entry"]-tr["sl"]; reason="sl"
                elif lo<=tr["tp"]: out=tr["entry"]-tr["tp"]; reason="tp"
            if out is None and tr["j"]>=min(n-1,tr["i0"]+96):   # 24h time-stop
                out=(full[tr["symbol"]]["c"][tr["j"]]-tr["entry"])*(1 if tr["side"]==Side.BUY else -1)
            if out is not None:
                pnl=out*tr["lots"]*tr["vpu"]
                pnl-=tr["lots"]*tr["vpu"]*tr["entry"]*(0.0003 if "USD"==tr["symbol"][3:] else 0.0005)
                equity+=pnl; trades.append(pnl)
            else:
                tr["j"]+=1; still.append(tr)
        open_trades=still
        # 2) new entries at close of bar i (fills next bar's open — no lookahead)
        held={tr["symbol"] for tr in open_trades}
        if len(open_trades)<max_pos:
            votes_all=vote_at(i)
            for s,votes in votes_all.items():
                if not votes or s in held or len(open_trades)>=max_pos: continue
                d=store.data[s]; atr_=float(ind.atr(d["h"],d["l"],d["c"],14)[-1])
                if np.isnan(atr_) or atr_<=0: continue
                lw=sum(v.weight*v.conviction for v in votes if v.direction==SignalDirection.LONG)
                sw=sum(v.weight*v.conviction for v in votes if v.direction==SignalDirection.SHORT)
                tw=sum(v.weight for v in votes)
                net=lw-sw; side=Side.BUY if net>0 else Side.SELL if net<0 else Side.FLAT
                if side==Side.FLAT: continue
                al=[v for v in votes if (v.direction==SignalDirection.LONG)==(side==Side.BUY)]
                if len(al)<2: continue
                conf=min(abs(net)/tw*1.6+min(0.25,0.08*(len(al)-1)),0.99)
                if conf<cfg["strategies"]["min_confidence"]: continue
                entry=float(full[s]["o"][i+1]) if i+1<n else float(d["c"][-1])
                risk=stop_cfg["atr_mult"]*atr_
                sl=entry-risk if side==Side.BUY else entry+risk
                tp=entry+stop_cfg["tp_rr"]*risk if side==Side.BUY else entry-stop_cfg["tp_rr"]*risk
                spec=get_spec(s); vpu=spec.contract_size
                lots=round_step(position_size_by_risk(equity,cfg["risk"]["risk_per_trade_pct"]*np.interp(conf,[0.5,1],[0.5,1.25]),risk,vpu),0.01,"down")
                if lots<spec.min_lot: continue
                open_trades.append(dict(symbol=s,side=side,entry=entry,sl=sl,tp=tp,
                                        lots=lots,vpu=vpu,i0=i+1,j=i+1))
                held.add(s)
        curve.append(equity); peak=max(peak,equity)
        i+=1
    wins=[t for t in trades if t>0]; losses=[t for t in trades if t<=0]
    run_peak = 10_000.0; dd = 0.0
    for e in curve:
        run_peak = max(run_peak, e)
        dd = max(dd, (run_peak - e) / max(run_peak, 1e-9))
    print(f"bars traded: {len(trades)} | final equity {equity:,.2f} ({(equity/10000-1)*100:+.1f}%)")
    print(f"win rate {len(wins)/max(len(trades),1):.0%} | avg win {np.mean(wins) if wins else 0:.2f} | avg loss {np.mean(losses) if losses else 0:.2f}")
    print(f"expectancy/trade {np.mean(trades) if trades else 0:+.2f} | max DD {dd*100:.1f}% | profit factor {sum(wins)/max(-sum(losses),1e-9):.2f}")
if __name__=="__main__": main()
