"""
Contract specifications for metals symbols.

If MT5 is attached, specs are pulled live from the broker and override these.
These defaults match typical retail CFD/spot conventions so that paper mode
and lot math remain correct even before a broker connection exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ContractSpec:
    symbol: str
    quote_ccy: str          # currency of the price (XAUUSD -> USD)
    contract_size: float    # troy oz per 1.0 lot
    lot_step: float
    min_lot: float
    max_lot: float
    digits: int
    point: float            # 10^-digits typically
    account_ccy: str = "USD"
    # fx crosses need conversion: value_per_unit_in_quote / FX rate to account ccy
    needs_fx_conversion: bool = False
    fx_symbol: str = ""     # e.g. AUDUSD to convert AUD -> USD


def _spec(symbol: str, quote: str, size: float, digits: int,
          fx: str = "") -> ContractSpec:
    return ContractSpec(
        symbol=symbol, quote_ccy=quote, contract_size=size,
        lot_step=0.01, min_lot=0.01, max_lot=100.0,
        digits=digits, point=10.0 ** -digits,
        needs_fx_conversion=bool(fx), fx_symbol=fx,
    )


DEFAULT_SPECS: Dict[str, ContractSpec] = {
    # Gold = 100 oz/lot on most retail brokers; Silver = 5000 oz/lot.
    "XAUUSD": _spec("XAUUSD", "USD", 100.0, 2),
    "XAGUSD": _spec("XAGUSD", "USD", 5000.0, 3),
    "XAUAUD": _spec("XAUAUD", "AUD", 100.0, 2, "AUDUSD"),
    "XAUEUR": _spec("XAUEUR", "EUR", 100.0, 2, "EURUSD"),
    "XAGAUD": _spec("XAGAUD", "AUD", 5000.0, 3, "AUDUSD"),
    "XAGEUR": _spec("XAGEUR", "EUR", 5000.0, 3, "EURUSD"),
}


def pip_value_per_lot(price: float, spec: ContractSpec,
                      fx_rate: float = 1.0) -> float:
    """Account-currency value of a 1.0 move in price for 1 lot."""
    return spec.contract_size * fx_rate


def get_spec(symbol: str, account_ccy: str = "USD") -> ContractSpec:
    s = DEFAULT_SPECS.get(symbol)
    if s is None:
        # generic fallback: treat like gold usd
        base = symbol[:3].upper()
        quote = symbol[3:].upper() if len(symbol) > 3 else "USD"
        size = 5000.0 if base == "XAG" else 100.0
        digits = 3 if base == "XAG" else 2
        s = _spec(symbol, quote, size, digits,
                  f"{quote}{base}" if quote != account_ccy else "")
    s.account_ccy = account_ccy
    return s
