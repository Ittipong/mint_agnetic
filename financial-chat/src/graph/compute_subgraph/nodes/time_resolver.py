"""time_resolver — Thai/EN time phrase → TimeRange.

Strategy: cheap first, LLM last.
1. Empty / no phrase → default to "this month".
2. Quick regex on common Thai relative forms ("เดือนนี้", "เดือนที่แล้ว",
   "<n> เดือนที่แล้ว", "ปีนี้", "ปีที่แล้ว") — deterministic, no LLM cost.
3. dateparser with Thai locale — covers "1 เดือนตุลาคม 2005" style.
4. LLM Pydantic fallback — only when 1-3 fail.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from src.graph.compute_subgraph.schemas import TimeRange
from src.graph.compute_subgraph.state import ComputeSubState


# ── Regex shortcuts (Thai relative phrases) ──────────────────────────────────


_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_NUM_RE = re.compile(r"(\d+)")

# Sentinel start for "all-time" queries — earlier than any plausible
# transaction. Postgres compares timestamptz against the literal date at
# session timezone, so this stays correct under TZ shifts.
_ALL_TIME_START = date(1900, 1, 1)


def _month_range(today: date, offset: int) -> tuple[date, date]:
    """Month containing today shifted by `offset` whole months."""
    year = today.year
    month = today.month + offset
    while month <= 0:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


def _year_range(today: date, offset: int) -> tuple[date, date]:
    y = today.year + offset
    return date(y, 1, 1), date(y, 12, 31)


def _quick_thai(phrase: str, today: date) -> tuple[date, date, str] | None:
    """Return (start, end, granularity) on a hit, else None."""
    p = phrase.translate(_THAI_DIGITS).strip().lower()

    # All-time: explicit user intent OR planner echoing default phrasing.
    if p in {
        "all", "all time", "all-time", "alltime",
        "ทั้งหมด", "ตั้งแต่เริ่ม", "ตั้งแต่เริ่มต้น",
    }:
        return _ALL_TIME_START, today, "all"

    if p in {"เดือนนี้", "this month", "current month", "เดือน"}:
        s, e = _month_range(today, 0)
        return s, e, "month"
    if p in {"เดือนที่แล้ว", "เดือนก่อน", "last month", "previous month"}:
        s, e = _month_range(today, -1)
        return s, e, "month"
    if p in {"ปีนี้", "this year", "current year"}:
        s, e = _year_range(today, 0)
        return s, e, "year"
    if p in {"ปีที่แล้ว", "ปีก่อน", "last year", "previous year"}:
        s, e = _year_range(today, -1)
        return s, e, "year"
    if p in {"วันนี้", "today"}:
        return today, today, "day"
    if p in {"เมื่อวาน", "yesterday"}:
        d = today - timedelta(days=1)
        return d, d, "day"

    # "<n> เดือนที่แล้ว" / "ที่ผ่านมา" → window of n months ending today
    m = re.match(r"^(\d+)\s*เดือน(ที่แล้ว|ก่อน|ที่ผ่านมา)?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            s, _ = _month_range(today, -n)
            return s, today, "month"

    # "<n> วัน" same idea
    m = re.match(r"^(\d+)\s*วัน(ที่แล้ว|ก่อน|ที่ผ่านมา)?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            return today - timedelta(days=n), today, "day"

    # English / Thai month names with optional year — "February 2026",
    # "feb 2026", "กุมภาพันธ์ 2026". Year defaults to today's year.
    parsed = _parse_named_month(p, today)
    if parsed is not None:
        return parsed

    return None


_EN_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_TH_MONTHS = {
    "มกราคม": 1, "ม.ค.": 1, "มค": 1,
    "กุมภาพันธ์": 2, "ก.พ.": 2, "กพ": 2,
    "มีนาคม": 3, "มี.ค.": 3, "มีค": 3,
    "เมษายน": 4, "เม.ย.": 4, "เมย": 4,
    "พฤษภาคม": 5, "พ.ค.": 5, "พค": 5,
    "มิถุนายน": 6, "มิ.ย.": 6, "มิย": 6,
    "กรกฎาคม": 7, "ก.ค.": 7, "กค": 7,
    "สิงหาคม": 8, "ส.ค.": 8, "สค": 8,
    "กันยายน": 9, "ก.ย.": 9, "กย": 9,
    "ตุลาคม": 10, "ต.ค.": 10, "ตค": 10,
    "พฤศจิกายน": 11, "พ.ย.": 11, "พย": 11,
    "ธันวาคม": 12, "ธ.ค.": 12, "ธค": 12,
}


def _parse_named_month(phrase: str, today: date) -> tuple[date, date, str] | None:
    """Match 'february', 'feb 2026', 'กุมภาพันธ์ 2026' → full month range."""
    tokens = phrase.replace(",", " ").split()
    if not tokens:
        return None
    month: int | None = None
    year: int | None = None
    for tok in tokens:
        key = tok.strip(".").lower()
        if key in _EN_MONTHS and month is None:
            month = _EN_MONTHS[key]
            continue
        if tok in _TH_MONTHS and month is None:
            month = _TH_MONTHS[tok]
            continue
        if tok.isdigit():
            n = int(tok)
            # Buddhist-era → Gregorian (rough heuristic — anyone writing
            # "2569" probably means 2026 BE).
            if n > 2400:
                n -= 543
            if 1900 < n < 2100:
                year = n
    if month is None:
        return None
    if year is None:
        year = today.year
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last), "month"


# ── LLM fallback ─────────────────────────────────────────────────────────────


class _LLMRange(BaseModel):
    """Structured output for the rare cases regex/dateparser miss."""

    start_date: str = Field(description="ISO date YYYY-MM-DD")
    end_date: str = Field(description="ISO date YYYY-MM-DD, inclusive")
    granularity: Literal["day", "week", "month", "quarter", "year"] = "month"


async def _llm_resolve(phrase: str, today: date) -> tuple[date, date, str, float] | None:
    try:
        from src.llm import llm
    except Exception:
        return None
    prompt = (
        f"Today is {today.isoformat()}. Convert this Thai/English time phrase "
        f"to an absolute date range (inclusive). Phrase: {phrase!r}. "
        f"If unclear, choose the most common interpretation."
    )
    try:
        structured = llm.with_structured_output(_LLMRange)
        out: _LLMRange = await structured.ainvoke(prompt)
        return (
            date.fromisoformat(out.start_date),
            date.fromisoformat(out.end_date),
            out.granularity,
            0.7,
        )
    except Exception:
        return None


# ── Node ─────────────────────────────────────────────────────────────────────


async def time_resolve_node(state: ComputeSubState) -> dict:
    today = date.fromisoformat(state["today"])
    plan = state.get("plan")
    phrase = (plan.time_phrase if plan else None) or ""

    # 1. No phrase → all time. Users who don't specify a window almost always
    #    want every record they have, not just the calendar month they happen
    #    to be in. Explicit windows (e.g. "เดือนนี้") still work via the
    #    quick_thai branch below.
    if not phrase.strip():
        return {
            "time_range": TimeRange(
                start=_ALL_TIME_START,
                end=today,
                granularity="all",
                confidence=0.9,
                raw_phrase=None,
            )
        }

    # 2. Regex shortcut
    quick = _quick_thai(phrase, today)
    if quick:
        s, e, g = quick
        return {
            "time_range": TimeRange(
                start=s, end=e, granularity=g, confidence=0.95, raw_phrase=phrase
            )
        }

    # 3. dateparser
    try:
        import dateparser

        parsed = dateparser.parse(
            phrase,
            languages=["th", "en"],
            settings={"RELATIVE_BASE": _to_dt(today), "PREFER_DATES_FROM": "past"},
        )
        if parsed is not None:
            d = parsed.date() if hasattr(parsed, "date") else parsed
            # dateparser gives a single date — treat as the month containing it
            s, e = _month_range(d, 0)
            return {
                "time_range": TimeRange(
                    start=s, end=e, granularity="month", confidence=0.8, raw_phrase=phrase
                )
            }
    except Exception:
        pass

    # 4. LLM fallback
    llm_out = await _llm_resolve(phrase, today)
    if llm_out:
        s, e, g, conf = llm_out
        return {
            "time_range": TimeRange(
                start=s, end=e, granularity=g, confidence=conf, raw_phrase=phrase
            )
        }

    # 5. Give up — default to this month with low confidence
    s, e = _month_range(today, 0)
    return {
        "time_range": TimeRange(
            start=s, end=e, granularity="month", confidence=0.3, raw_phrase=phrase
        )
    }


def _to_dt(d: date):
    from datetime import datetime

    return datetime(d.year, d.month, d.day)
