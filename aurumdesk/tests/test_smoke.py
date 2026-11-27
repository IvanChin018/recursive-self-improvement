"""Fast smoke tests: import surface, config, bus semantics, paper broker math."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    import aurumdesk.desk as d
    from aurumdesk.agents import (allocation, base, candle, compliance, execution,
                                  journal, market_data, news_macro, overseer,
                                  quant_forecast, risk, sentiment, strategy, technical)
    assert hasattr(d, "Desk")

def test_config_defaults():
    from aurumdesk.config import load_config
    cfg = load_config(None)
    assert cfg["mode"] in ("paper", "live")
    assert "XAUUSD" in cfg["symbols"]
    assert cfg["risk"]["risk_per_trade_pct"] <= 1.5

def test_bus_blackboard_ttl():
    from aurumdesk.bus import Bus
    b = Bus()
    b.set("k", 42, max_age_sec=0.05)
    assert b.get("k") == 42
    time.sleep(0.06)
    assert b.get("k", "gone") == "gone"
    assert b.get("k", "x", allow_stale=True) == 42
    seen = []
    b.subscribe("topic", lambda t, p: seen.append(p))
    b.publish("topic", 1)
    assert seen == [1]

def test_paper_broker_roundtrip():
    from aurumdesk.execution.paper_broker import PaperBroker
    from aurumdesk.models import Order, Side, Quote
    cfg = {"execution": {"deviation_slippage_points": 0}}
    state_path = os.path.join(os.environ.get("TMP", "/tmp"), "ad_test_paper.json")
    if os.path.exists(state_path):
        os.remove(state_path)
    pb = PaperBroker(cfg, state_path=state_path)
    pb.balance = 10_000.0
    q = Quote(symbol="XAUUSD", bid=4000.0, ask=4000.5, t=time.time())
    o = Order(symbol="XAUUSD", side=Side.BUY, lots=0.05, stop_loss=3990.0,
              take_profit=4020.0, comment="test|trend")
    r = pb.send(o, q)
    assert r.state.value == "FILLED", r.error
    poss = list(pb.positions.values())
    assert len(poss) == 1 and abs(poss[0].lots - 0.05) < 1e-9
    # price hits TP
    q2 = Quote(symbol="XAUUSD", bid=4021.0, ask=4021.5, t=time.time()+5)
    events = pb.check_exits({"XAUUSD": q2})
    assert events and events[0]["reason"] in ("tp", "TP", "take_profit"), events
    assert not pb.positions
    assert pb.balance > 10_000.0
    if os.path.exists(state_path):
        os.remove(state_path)

def test_sizing_math():
    from aurumdesk.risk.portfolio import position_size_by_risk, kelly_fraction, round_step
    # XAUUSD: contract 100 oz -> $100 per 1.00 price move per lot
    lots = position_size_by_risk(10_000, 0.75, 10.0, 100.0)   # $75 risk / $1000 loss
    assert abs(lots - 0.075) < 1e-9
    assert position_size_by_risk(10_000, 0.75, 0.0, 100.0) == 0.0
    assert round_step(0.037, 0.01, "down") == 0.03
    assert kelly_fraction(0.55, 1.8, 1.0) > 0
    assert kelly_fraction(0.3, 1.0, 1.0) <= 0

def test_indicators_finite():
    import numpy as np
    from aurumdesk.indicators import library as ind
    x = np.cumsum(np.random.RandomState(1).randn(300)) + 4000
    for fn in (lambda: ind.sma(x, 20), lambda: ind.ema(x, 20), lambda: ind.rsi(x, 14),
               lambda: ind.atr(x, x*1.01, x, 14), lambda: ind.macd(x)[2]):
        arr = fn()
        assert np.all(np.isfinite(arr[-50:])), fn

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} smoke tests passed")
