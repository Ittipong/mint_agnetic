"""Unit tests for the Smart CodeAct subgraph — pure functions only.

LLM-driven paths (codeact loop, llm_rerank) live in the LangSmith eval
harness; here we lock down the deterministic helpers (heuristic scoring,
time parsing) so a regression fails CI without spending tokens.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.graph.compute_subgraph.codeact.exceptions import (
    AmbiguousMatchError,
    ClarificationNeeded,
)
from src.graph.compute_subgraph.codeact.resolvers import (
    _score_pair,
    clarify,
    parse_period,
)


# ── parse_period — Thai/English time phrases ────────────────────────────────


TODAY = date(2026, 5, 5)


def test_parse_period_none_returns_all_time():
    s, e = parse_period(None, TODAY)
    assert s == date(1900, 1, 1)
    assert e == TODAY


def test_parse_period_empty_returns_all_time():
    s, e = parse_period("", TODAY)
    assert s == date(1900, 1, 1)
    assert e == TODAY


def test_parse_period_all_time_marker():
    for phrase in ["all time", "ทั้งหมด", "ตั้งแต่เริ่ม"]:
        s, e = parse_period(phrase, TODAY)
        assert s == date(1900, 1, 1)
        assert e == TODAY


def test_parse_period_this_month():
    s, e = parse_period("เดือนนี้", TODAY)
    assert s == date(2026, 5, 1)
    assert e == date(2026, 5, 31)


def test_parse_period_last_month():
    s, e = parse_period("เดือนที่แล้ว", TODAY)
    assert s == date(2026, 4, 1)
    assert e == date(2026, 4, 30)


def test_parse_period_last_n_months():
    s, e = parse_period("3 เดือนที่แล้ว", TODAY)
    assert s == date(2026, 2, 1)
    assert e == TODAY  # window ending today


def test_parse_period_named_month_thai():
    s, e = parse_period("กุมภาพันธ์ 2026", TODAY)
    assert s == date(2026, 2, 1)
    assert e == date(2026, 2, 28)


def test_parse_period_named_month_buddhist_year():
    """Buddhist Era input — 2569 BE = 2026 AD."""
    s, e = parse_period("กุมภาพันธ์ 2569", TODAY)
    assert s == date(2026, 2, 1)
    assert e == date(2026, 2, 28)


def test_parse_period_named_month_english():
    s, e = parse_period("February 2026", TODAY)
    assert s == date(2026, 2, 1)
    assert e == date(2026, 2, 28)


def test_parse_period_today():
    s, e = parse_period("วันนี้", TODAY)
    assert s == TODAY and e == TODAY


def test_parse_period_yesterday():
    s, e = parse_period("เมื่อวาน", TODAY)
    assert s == date(2026, 5, 4) and e == date(2026, 5, 4)


def test_parse_period_unrecognized_raises():
    with pytest.raises(ValueError, match="unrecognized period"):
        parse_period("xyzzy nonsense phrase", TODAY)


# ── _score_pair — heuristic name matching ───────────────────────────────────


def test_score_exact_match():
    assert _score_pair("TrueMonney", "TrueMonney") == 1.0


def test_score_case_insensitive_exact():
    assert _score_pair("truemonney", "TrueMonney") == 1.0


def test_score_substring():
    """'kbank' is a substring of 'KBank Savings' after normalization."""
    s = _score_pair("kbank", "KBank Savings")
    assert s >= 0.7  # contains rule


def test_score_no_match():
    assert _score_pair("xxx", "TrueMonney") < 0.3


def test_score_empty_query():
    assert _score_pair("", "TrueMonney") == 0.0


# ── clarify() raises ClarificationNeeded ────────────────────────────────────


def test_clarify_raises():
    with pytest.raises(ClarificationNeeded) as exc:
        clarify("which wallet?", options=["A", "B"])
    assert exc.value.question == "which wallet?"
    assert exc.value.options == ["A", "B"]


def test_clarify_no_options():
    with pytest.raises(ClarificationNeeded) as exc:
        clarify("when?")
    assert exc.value.options == []


# ── AmbiguousMatchError carries candidate list ──────────────────────────────


def test_ambiguous_error_includes_candidates():
    err = AmbiguousMatchError(query="savings", kind="wallet", candidates=["A", "B"])
    assert "savings" in str(err)
    assert "A" in str(err) and "B" in str(err)
    assert err.candidates == ["A", "B"]
