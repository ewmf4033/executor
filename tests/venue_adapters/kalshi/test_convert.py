"""
Unit tests for Kalshi <-> canonical conversions.

Covers:
  - cents -> Decimal probability (1c = 0.01, 99c = 0.99)
  - Decimal -> cents with rejection of boundary/invalid values
  - YES / NO outcome canonicalization
  - Orderbook parse: bids descending, asks derived from opposite-side bids,
    empty sides, zero-size levels skipped
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from executor.venue_adapters.kalshi.convert import (
    canonicalize_outcome,
    cents_to_prob,
    parse_orderbook,
    prob_to_cents,
    side_to_action,
    to_native_outcome,
)
from executor.core.types import Side


# ---------------------------------------------------------------------------
# Cents <-> probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cents, prob",
    [
        (1, Decimal("0.01")),
        (50, Decimal("0.50")),
        (62, Decimal("0.62")),
        (99, Decimal("0.99")),
        (0, Decimal("0.00")),
        (100, Decimal("1.00")),
    ],
)
def test_cents_to_prob_exact(cents, prob):
    assert cents_to_prob(cents) == prob


def test_cents_to_prob_out_of_range():
    with pytest.raises(ValueError):
        cents_to_prob(-1)
    with pytest.raises(ValueError):
        cents_to_prob(101)


@pytest.mark.parametrize(
    "prob, cents",
    [
        (Decimal("0.01"), 1),
        (Decimal("0.50"), 50),
        (Decimal("0.62"), 62),
        (Decimal("0.99"), 99),
        (0.55, 55),
    ],
)
def test_prob_to_cents_round_trip(prob, cents):
    assert prob_to_cents(prob) == cents


@pytest.mark.parametrize("bad", [Decimal("0"), Decimal("1"), Decimal("-0.1"), Decimal("1.01")])
def test_prob_to_cents_rejects_boundary_and_oob(bad):
    with pytest.raises(ValueError):
        prob_to_cents(bad)


def test_prob_to_cents_banker_rounding():
    # 0.555 -> 56 (round half to even? 55.5 -> 56), 0.545 -> 54 (banker's)
    assert prob_to_cents(Decimal("0.555")) == 56
    assert prob_to_cents(Decimal("0.545")) == 54


# ---------------------------------------------------------------------------
# Outcome canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["yes", "YES", "Yes", " yes ", "yEs"])
def test_canonicalize_yes(raw):
    assert canonicalize_outcome(raw) == "YES"


@pytest.mark.parametrize("raw", ["no", "NO", "No"])
def test_canonicalize_no(raw):
    assert canonicalize_outcome(raw) == "NO"


def test_canonicalize_rejects_unknown():
    with pytest.raises(ValueError):
        canonicalize_outcome("maybe")
    with pytest.raises(ValueError):
        canonicalize_outcome(None)  # type: ignore[arg-type]


def test_to_native_outcome():
    assert to_native_outcome("YES") == "yes"
    assert to_native_outcome("NO") == "no"


def test_side_to_action():
    assert side_to_action(Side.BUY) == "buy"
    assert side_to_action(Side.SELL) == "sell"


# ---------------------------------------------------------------------------
# Orderbook parse
# ---------------------------------------------------------------------------


def _sample_book():
    # Kalshi REST shape: yes/no are bid-only arrays of [cents, size].
    # Best YES bid = highest yes cents (62). Best NO bid = highest no cents (35).
    # Implied best YES ask = 100 - 35 = 65.
    return {
        "orderbook": {
            "yes": [[60, 20], [62, 15]],
            "no": [[30, 10], [35, 25]],
        }
    }


def test_parse_orderbook_yes_side():
    ob = parse_orderbook("TICKER-1", _sample_book(), outcome="YES")
    assert ob.market_id == "TICKER-1"
    assert ob.outcome_id == "YES"
    # Bids descending.
    assert [lvl.price_prob for lvl in ob.bids] == [Decimal("0.62"), Decimal("0.60")]
    assert [lvl.size for lvl in ob.bids] == [Decimal("15"), Decimal("20")]
    # Asks ascending, derived from NO bids (best NO bid = 35 -> YES ask 0.65).
    assert [lvl.price_prob for lvl in ob.asks] == [Decimal("0.65"), Decimal("0.70")]
    assert [lvl.size for lvl in ob.asks] == [Decimal("25"), Decimal("10")]


def test_parse_orderbook_no_side():
    ob = parse_orderbook("TICKER-1", _sample_book(), outcome="NO")
    assert ob.outcome_id == "NO"
    assert [lvl.price_prob for lvl in ob.bids] == [Decimal("0.35"), Decimal("0.30")]
    # NO asks derived from YES bids: 100 - 62 = 38, 100 - 60 = 40.
    assert [lvl.price_prob for lvl in ob.asks] == [Decimal("0.38"), Decimal("0.40")]


def test_parse_orderbook_empty_sides():
    ob = parse_orderbook("X", {"orderbook": {"yes": [], "no": []}}, outcome="YES")
    assert ob.bids == ()
    assert ob.asks == ()


def test_parse_orderbook_skips_zero_size():
    raw = {"orderbook": {"yes": [[50, 0], [51, 3]], "no": [[40, 0], [42, 7]]}}
    ob = parse_orderbook("X", raw, outcome="YES")
    assert [lvl.price_prob for lvl in ob.bids] == [Decimal("0.51")]
    assert [lvl.price_prob for lvl in ob.asks] == [Decimal("0.58")]
