"""
AurumDesk core data models.

Every message that flows between agents is one of these typed objects.
Keeping them frozen-ish (dataclasses) means no agent can silently mutate
another agent's decision after the fact — a key "no leaks" design rule.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
#  Enums
# --------------------------------------------------------------------------- #
class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else (-1 if self is Side.SELL else 0)


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


TIMEFRAME_SECONDS: Dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.H1: 3600,
    Timeframe.H4: 14400,
    Timeframe.D1: 86400,
}


class OrderState(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class RiskVerdict(str, Enum):
    APPROVED = "APPROVED"
    REDUCED = "REDUCED"       # approved but with smaller size
    VETOED = "VETOED"
    DEFERRED = "DEFERRED"     # e.g. news blackout — retry later


class AgentName(str, Enum):
    MARKET_DATA = "MarketDataAgent"
    NEWS_MACRO = "NewsMacroAgent"
    TECHNICAL = "TechnicalAnalystAgent"
    CANDLE = "CandlePatternAgent"
    SENTIMENT = "SentimentRegimeAgent"
    STRATEGY = "StrategyEnsembleAgent"
    FORECAST = "QuantForecastAgent"
    RISK = "RiskManagerAgent"
    ALLOCATION = "PortfolioAllocAgent"
    EXECUTION = "ExecutionAgent"
    COMPLIANCE = "ComplianceAgent"
    JOURNAL = "JournalAgent"
    OVERSEER = "OverseerAgent"


# --------------------------------------------------------------------------- #
#  Market structures
# --------------------------------------------------------------------------- #
@dataclass
class Candle:
    symbol: str
    timeframe: Timeframe
    t: int                 # candle open epoch seconds (UTC)
    o: float
    h: float
    l: float
    c: float
    volume: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    t: float = field(default_factory=time.time)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.spread / m * 100.0) if m > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Signals & decisions
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    """Output of any analyst agent for one symbol."""
    agent: str
    symbol: str
    direction: SignalDirection
    score: float                      # normalized conviction in [-1, 1]
    timeframe: Timeframe
    t: float = field(default_factory=time.time)
    reasons: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


@dataclass
class StrategyVote:
    strategy: str
    symbol: str
    direction: SignalDirection
    weight: float                     # adaptive weight from performance
    conviction: float                 # 0..1 how strong this instance is
    horizon_tf: Timeframe
    reasons: List[str] = field(default_factory=list)


@dataclass
class TradeProposal:
    """What the Strategy Desk hands to the Risk Desk."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str = ""
    side: Side = Side.FLAT
    entry_ref: float = 0.0            # price used for SL/TP math
    stop_loss: float = 0.0
    take_profit: float = 0.0
    target_lots: float = 0.0          # requested size (pre-risk checks)
    confidence: float = 0.0           # 0..1 ensemble confidence
    rr: float = 0.0                   # reward:risk
    horizon_tf: Timeframe = Timeframe.M15
    strategies: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    reduce_only: bool = False         # closing/reducing an existing position
    created_t: float = field(default_factory=time.time)
    votes: List[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Risk & execution
# --------------------------------------------------------------------------- #
@dataclass
class RiskDecision:
    proposal_id: str
    verdict: RiskVerdict
    lots: float                       # final approved size (0 if vetoed)
    max_loss_amount: float            # cash at risk if SL hit
    notes: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Order:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    proposal_id: str = ""
    client_id: str = ""
    symbol: str = ""
    side: Side = Side.FLAT
    order_type: str = "MARKET"        # MARKET | LIMIT | STOP
    lots: float = 0.0
    filled_lots: float = 0.0
    limit_price: float = 0.0          # requested ref price
    avg_fill_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    state: OrderState = OrderState.PENDING
    broker_order_id: Optional[int] = None
    comment: str = ""
    created_t: float = field(default_factory=time.time)
    updated_t: float = field(default_factory=time.time)
    error: str = ""
    reduce_only: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state in (OrderState.FILLED, OrderState.REJECTED,
                              OrderState.CANCELLED, OrderState.ERROR)


@dataclass
class Position:
    ticket: int
    symbol: str
    side: Side
    lots: float
    open_price: float
    sl: float
    tp: float
    opened_t: float
    profit: float = 0.0               # floating pnl in account currency
    swap: float = 0.0
    comment: str = ""
    strategy_tag: str = ""

    @property
    def direction_sign(self) -> int:
        return self.side.sign


@dataclass
class Fill:
    symbol: str
    side: Side
    lots: float
    price: float
    t: float
    commission: float = 0.0
    order_id: str = ""
    pos_ticket: int = 0


# --------------------------------------------------------------------------- #
#  Contexts broadcast by upstream agents
# --------------------------------------------------------------------------- #
@dataclass
class MacroContext:
    """NewsMacroAgent output — the desk's 'is it safe / who's driving' layer."""
    t: float = field(default_factory=time.time)
    news_blackout_until: float = 0.0
    upcoming_events: List[dict] = field(default_factory=list)
    macro_bias: float = 0.0              # -1..1 structural gold bias
    macro_reasons: List[str] = field(default_factory=list)
    dxy: float = 0.0
    us10y: float = 0.0
    us10y_real: float = 0.0
    vix: float = 0.0
    silver_bars: Optional[Candle] = None  # latest XAU/XAG ratio candle
    xauusd_daily: Optional[Candle] = None
    ok: bool = True


@dataclass
class RegimeInfo:
    regime: str = "chop"      # trend_up | trend_down | chop | volatile_breakout
    vol_mult: float = 1.0     # current ATR vs long-run ATR
    confidence: float = 0.5
    t: float = field(default_factory=time.time)


@dataclass
class AccountSnapshot:
    balance: float
    equity: float
    margin_used: float
    margin_free: float
    positions_value: float
    realized_today: float
    open_positions: List[Position]
    t: float = field(default_factory=time.time)

    @property
    def leverage_used(self) -> float:
        return (self.positions_value / self.balance) if self.balance > 0 else 0.0


@dataclass
class EventLogEntry:
    t: float
    kind: str            # signal | proposal | risk | order | fill | pnl | alert ...
    agent: str
    symbol: str
    payload: dict = field(default_factory=dict)
