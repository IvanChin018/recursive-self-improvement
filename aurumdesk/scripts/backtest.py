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
    # walk forward from a start price using synth vol params
    from aurumdesk.data.synthetic_feed import CROSS_DEFS
    if sym in CROSS_DEFS:
        num, den = CROSS_DEFS[sym]
        rngx = random.Random((hash(num)+hash(den)) & 0xffff)
        pn = _path(synth.params[num], n, step, rngx)
        pd_ = _path(synth.params[den], n, step, rngx)
        arr = lambda x: np.array(x)
        ratio = np.array(pn["o"]) / np.array(pd_["o"])
        return dict(t=pn["t"], o=ratio, c=ratio,
                    h=ratio*1.0007, l=ratio*0.9993,
                    v=np.full(n, 1000.0))
    base = synth.params[sym]
    r = _path(base, n, step, random.Random(hash(sym) & 0xffff))
    a=np.array
    return dict(t=a(r["t"]),o=a(r["o"]),h=a(r["h"]),l=a(r["l"]),c=a(r["c"]),v=a(r["v"]))

def _path(base, n, step, rng):
    p = base["p0"]; sig = base["vol"]/math.sqrt(365*24*3600/step)
    ts,os_,hs,ls,cs,vs=[],[],[],[],[],[]
    for i in range(n):
        o=p; ret=rng.gauss(0,sig); c=o*math.exp(ret)
        wick=abs(rng.gauss(0,sig))*o*0.7
        h=max(o,c)+wick*rng.random(); l=min(o,c)-wick*rng.random()
        ts.append(i*step); os_.append(o); hs.append(h); ls.append(l); cs.append(c); vs.append(rng.uniform(500,2500))
        p=c
    return dict(t=ts,o=os_,h=hs,l=ls,c=cs,v=vs,p=[p])
    rng = random.Random(hash(sym) & 0xffff)
    t0 = int(end_t//step*step) - n*step
    ts,os_,hs,ls,cs,vs = [],[],[],[],[],[]
    for i in range(n):
        o=p; ret=rng.gauss(0,sig); c=o*math.exp(ret)
        wick=abs(rng.gauss(0,sig))*o*0.7
        h=max(o,c)+wick*rng.random(); l=min(o,c)-wick*rng.random()
        ts.append(t0+i*step); os_.append(o); hs.append(h); ls.append(l); cs.append(c); vs.append(rng.uniform(500,2500))
        p=c
    return dict(t=np.array(ts),o=np.array(os_),h=np.array(hs),l=np.array(ls),c=np.array(cs),v=np.array(vs))

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
    for i in range(warm,n-1):
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
        for s,votes in votes_all.items():
            if not votes: continue
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
            entry=d["c"][-1]; risk=stop_cfg["atr_mult"]*atr_
            sl=entry-risk if side==Side.BUY else entry+risk
            tp=entry+stop_cfg["tp_rr"]*risk if side==Side.BUY else entry-stop_cfg["tp_rr"]*risk
            spec=get_spec(s); vpu=spec.contract_size
            lots=round_step(position_size_by_risk(equity,cfg["risk"]["risk_per_trade_pct"]*np.interp(conf,[0.5,1],[0.5,1.25]),risk,vpu),0.01,"down")
            if lots<spec.min_lot: continue
            # simulate next bars until SL/TP or 96 bars (24h)
            j=i+1; end=min(n,i+96); out=None
            hi=full[s]["h"][j:end]; lo=full[s]["l"][j:end]
            for k in range(len(hi)):
                px_b=hi[k]; px_a=lo[k]
                if side==Side.BUY:
                    if px_a<=sl: out=-risk; break
                    if px_b>=tp: out=stop_cfg["tp_rr"]*risk; break
                else:
                    if px_b>=sl: out=-risk; break
                    if px_a<=tp: out=stop_cfg["tp_rr"]*risk; break
            if out is None:
                last=(full[s]["c"][end-1]-entry)*(1 if side==Side.BUY else -1); out=last
            pnl=out*lots*vpu
            # costs: spread ~ half of typical spread
            pnl-=lots*spec.contract_size*entry*(0.0003 if "USD" in s else 0.0005)
            equity+=pnl; trades.append(pnl); curve.append(equity)
            peak=max(peak,equity)
    wins=[t for t in trades if t>0]; losses=[t for t in trades if t<=0]
    run_peak = 10_000.0; dd = 0.0
    for e in curve:
        run_peak = max(run_peak, e)
        dd = max(dd, (run_peak - e) / max(run_peak, 1e-9))
    print(f"bars traded: {len(trades)} | final equity {equity:,.2f} ({(equity/10000-1)*100:+.1f}%)")
    print(f"win rate {len(wins)/max(len(trades),1):.0%} | avg win {np.mean(wins) if wins else 0:.2f} | avg loss {np.mean(losses) if losses else 0:.2f}")
    print(f"expectancy/trade {np.mean(trades) if trades else 0:+.2f} | max DD {dd*100:.1f}% | profit factor {sum(wins)/max(-sum(losses),1e-9):.2f}")
if __name__=="__main__": main()
