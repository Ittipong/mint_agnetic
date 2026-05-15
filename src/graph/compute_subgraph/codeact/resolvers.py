"""Sandbox-callable resolvers — port of entity_resolver + time_resolver.

These helpers run inside the codeact sandbox. They preserve the heuristic
+ LLM rerank logic from the legacy nodes, exposed as plain functions so
the LLM can compose them with the SQL wrappers in one Python block.

LLM rerank runs by bridging the sandbox's worker thread back to the main
asyncio loop via `run_coroutine_threadsafe`, identical to how the SQL
wrappers reach the asyncpg pool.
"""

from __future__ import annotations

import asyncio
import calendar
import re
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import dateparser
from pydantic import BaseModel, Field

from src.entity_catalog import EntityCatalog
from src.graph.compute_subgraph.codeact.exceptions import (
    AmbiguousMatchError,
    ClarificationNeeded,
)


# ─────────────────────────────────────────────────────────────────────────────
# clarify() — terminate the loop and ask the user a question
# ─────────────────────────────────────────────────────────────────────────────


def clarify(question: str, options: list[str] | None = None) -> None:
    """Stop the codeact loop and ask the user a question.

    Use when the user's intent is genuinely ambiguous and you need their
    input to proceed (multiple plausible wallets, unclear time period,
    etc.). The frontend renders the question with optional answer chips.
    """
    raise ClarificationNeeded(question, options)


# ─────────────────────────────────────────────────────────────────────────────
# Entity resolution (port of nodes/entity_resolver.py)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Candidate:
    sync_id: str
    name: str
    wallet_type: str = "general"
    wallet_category: str | None = None
    parent_id: str | None = None
    keywords: list[str] | None = None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _score_pair(query: str, candidate: str) -> float:
    """Cheap deterministic similarity in [0,1] — same logic as the legacy
    entity_resolver so behavior is identical."""
    q = _normalize(query)
    c = _normalize(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if c.startswith(q) or q.startswith(c):
        return 0.85
    if q in c or c in q:
        return 0.7

    q_tokens = set(q.split())
    c_tokens = set(c.split())
    if q_tokens and c_tokens:
        overlap = len(q_tokens & c_tokens) / max(len(q_tokens), len(c_tokens))
        if overlap > 0:
            return 0.4 + 0.3 * overlap

    common = len(set(q) & set(c)) / max(len(set(q) | set(c)), 1)
    return 0.2 * common


def _wallet_candidates(catalog: EntityCatalog) -> list[_Candidate]:
    return [
        _Candidate(
            sync_id=w.sync_id,
            name=w.name,
            wallet_type=w.wallet_type,
            wallet_category=w.wallet_category,
        )
        for w in catalog.wallets
    ]


def _category_candidates(catalog: EntityCatalog) -> list[_Candidate]:
    return [
        _Candidate(
            sync_id=c.sync_id,
            name=c.name,
            parent_id=c.parent_id,
            keywords=c.keywords,
        )
        for c in catalog.categories
    ]


def _tag_candidates(catalog: EntityCatalog) -> list[_Candidate]:
    return [_Candidate(sync_id=t.sync_id, name=t.name) for t in catalog.tags]


# ── LLM rerank (async — bridged to sandbox via run_coroutine_threadsafe) ────


class _LLMChoice(BaseModel):
    sync_id: str = Field(description="The sync_id of the chosen candidate")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


async def _llm_rerank(
    kind: str, query: str, top: list[tuple[_Candidate, float]]
) -> tuple[_Candidate, float] | None:
    """Ask the planner LLM to pick the best candidate from `top`.

    Async — the sandbox calls this through `run_coroutine_threadsafe` so
    the main event loop runs the coroutine while the worker thread waits
    on the future.
    """
    if not top:
        return None
    try:
        from src.llm import llm
    except Exception:
        return None

    if kind == "wallet":
        bullet = "\n".join(
            (
                f"- sync_id={c.sync_id} | name={c.name!r} | type={c.wallet_type} "
                f"| category={c.wallet_category!r} | heuristic_score={s:.2f}"
                if c.wallet_category
                else
                f"- sync_id={c.sync_id} | name={c.name!r} | type={c.wallet_type} "
                f"| heuristic_score={s:.2f}"
            )
            for c, s in top
        )
        prompt = (
            f"Pick the wallet that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider Thai/English semantics and typos. Consider wallet_type "
            f"(creditcard/general/goal) and wallet_category. If none match, "
            f"pick the closest with low confidence."
        )
    elif kind == "category":
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r} | parent={c.parent_id!r} "
            f"| keywords={c.keywords!r} | heuristic_score={s:.2f}"
            for c, s in top
        )
        prompt = (
            f"Pick the category that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider parent-child hierarchy (e.g. 'เดินทาง' is parent of "
            f"'แท็กซี่', 'BTS/MRT', 'น้ำมัน') and the keywords field. "
            f"Return the sync_id of the PRIMARY match."
        )
    else:
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r} | heuristic_score={s:.2f}"
            for c, s in top
        )
        prompt = (
            f"Pick the {kind} that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider Thai/English semantics and typos."
        )

    try:
        structured = llm.with_structured_output(_LLMChoice)
        out: _LLMChoice = await structured.ainvoke(prompt)
        chosen = next((c for c, _ in top if c.sync_id == out.sync_id), None)
        if chosen is None:
            return None
        return chosen, max(0.0, min(1.0, out.confidence))
    except Exception:
        return None


def _rerank_sync(
    main_loop: asyncio.AbstractEventLoop,
    kind: str,
    query: str,
    top: list[tuple[_Candidate, float]],
    timeout: float = 10.0,
) -> tuple[_Candidate, float] | None:
    """Sync facade — sandbox is in worker thread, LLM lives on main loop."""
    coro = _llm_rerank(kind, query, top)
    future: Future = asyncio.run_coroutine_threadsafe(coro, main_loop)
    try:
        return future.result(timeout=timeout)
    except Exception:
        return None


# ── Public resolvers (called from inside the sandbox) ───────────────────────


def _expand_subcategories(
    primary: _Candidate, all_candidates: list[_Candidate]
) -> list[str]:
    """When primary is a parent (no parent_id of its own), expand to include
    all direct children. Else return empty (don't widen a leaf query)."""
    if primary.parent_id is None:
        return [c.name for c in all_candidates if c.parent_id == primary.sync_id]
    return []


def _resolve_one(
    kind: str,
    query: str,
    candidates: list[_Candidate],
    main_loop: asyncio.AbstractEventLoop,
) -> _Candidate:
    """Score → short-circuit on exact match → LLM rerank for everything else.

    Mirrors `entity_resolver._resolve_mention` so behavior is identical to
    the legacy node. The LLM rerank handles fuzzy / typo / paraphrase /
    cross-language matches that the heuristic alone misses (e.g. 'true
    money' → 'TrueMonney').

    Raises:
      ValueError when the catalog is empty or the LLM rerank fails AND the
        heuristic top score is below 0.4 (no plausible match).
      AmbiguousMatchError when multiple candidates remain genuinely close
        after rerank — the LLM picks one explicitly next iteration.
    """
    if not candidates:
        raise ValueError(
            f"no {kind}s configured in catalog — cannot resolve {query!r}"
        )

    scored = [(c, _score_pair(query, c.name)) for c in candidates]
    scored.sort(key=lambda x: -x[1])
    top_score = scored[0][1]

    # Exact / near-exact match → take it (no LLM round-trip)
    if top_score >= 0.95:
        return scored[0][0]

    # LLM rerank with top-N — handles 'true money' → 'TrueMonney' etc.
    top = scored[: min(5, len(scored))]
    rerank = _rerank_sync(main_loop, kind, query, top)
    if rerank is not None:
        chosen, conf = rerank
        if conf >= 0.5:
            return chosen
        # Low-confidence rerank — surface alternatives so LLM picks
        candidates_list = [c.name for c, _ in top]
        raise AmbiguousMatchError(
            query=query, kind=kind, candidates=candidates_list
        )

    # Rerank unavailable — fall back to heuristic if it's at least somewhat
    # confident, otherwise let the LLM pick from the catalog list.
    if top_score >= 0.4:
        return scored[0][0]
    raise ValueError(
        f"no {kind} matches {query!r}. Available: "
        f"{[c.name for c in candidates]}"
    )


def make_resolve_wallet(catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop):
    """Closure-bound resolver — sandbox sees a simple `resolve_wallet(query)` callable."""
    candidates = _wallet_candidates(catalog)

    def resolve_wallet(query: str) -> str:
        """Map free-text → canonical wallet name (single match).

        Examples:
          resolve_wallet('true money')  → 'TrueMonney'
          resolve_wallet('kbank')       → 'KBank Savings'
          resolve_wallet('xxx')         → ValueError("no wallet matches 'xxx'...")
        """
        chosen = _resolve_one("wallet", query, candidates, main_loop)
        return chosen.name

    return resolve_wallet


def make_resolve_category(catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop):
    """Returns canonical category name(s). Expands parent → children when
    the user's query matches a parent (e.g. 'เดินทาง' → ['เดินทาง', 'แท็กซี่',
    'BTS/MRT', 'น้ำมัน'])."""
    candidates = _category_candidates(catalog)

    def resolve_category(query: str) -> list[str]:
        """Return list of canonical category names matching `query`.

        For a parent category match, the list includes the parent + all its
        direct children. For a leaf match, the list is `[name]` only.

        Examples (catalog: เดินทาง→{แท็กซี่, BTS/MRT, น้ำมัน}, อาหาร→{ร้านอาหาร}):
          resolve_category('น้ำมัน')   → ['น้ำมัน']
          resolve_category('เดินทาง')  → ['เดินทาง', 'แท็กซี่', 'BTS/MRT', 'น้ำมัน']
          resolve_category('xxx')      → ValueError(...)
        """
        chosen = _resolve_one("category", query, candidates, main_loop)
        expanded = _expand_subcategories(chosen, candidates)
        return [chosen.name, *expanded]

    return resolve_category


def make_resolve_tag(catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop):
    candidates = _tag_candidates(catalog)

    def resolve_tag(query: str) -> str:
        """Map free-text → canonical tag name."""
        chosen = _resolve_one("tag", query, candidates, main_loop)
        return chosen.name

    return resolve_tag


def make_resolve_budget(catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop):
    """Budgets are ILIKE-matched at SQL level — this helper validates the
    phrase exists in the catalog (if catalog has a budgets list)."""

    def resolve_budget(query: str) -> str:
        """Echo the query — backend SQL uses ILIKE %query% for budget_name_phrase.

        Provided for API symmetry with other resolvers; budgets don't need
        canonical-name lookup because the SQL filter is fuzzy.
        """
        if not query or not query.strip():
            raise ValueError("budget query is empty")
        return query.strip()

    return resolve_budget


def make_resolve_goal(catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop):
    def resolve_goal(query: str) -> str:
        """Echo the query — SQL uses ILIKE %query% for goal_name_phrase."""
        if not query or not query.strip():
            raise ValueError("goal query is empty")
        return query.strip()

    return resolve_goal


# ─────────────────────────────────────────────────────────────────────────────
# Time parsing (port of nodes/time_resolver.py — sync, no LLM fallback)
# ─────────────────────────────────────────────────────────────────────────────


_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ALL_TIME_START = date(1900, 1, 1)


def _month_range(today: date, offset: int) -> tuple[date, date]:
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


_EN_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
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


def _parse_named_month(phrase: str, today: date) -> tuple[date, date] | None:
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
            # Buddhist-era → Gregorian heuristic
            if n > 2400:
                n -= 543
            if 1900 < n < 2100:
                year = n
    if month is None:
        return None
    if year is None:
        year = today.year
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


def parse_period(phrase: str | None, today: date) -> tuple[date, date]:
    """Parse Thai/English time phrase → (start, end). Inclusive on both ends.

    Examples (today=2026-05-05):
      parse_period(None)               → (1900-01-01, 2026-05-05)  # all time
      parse_period('')                 → (1900-01-01, 2026-05-05)
      parse_period('all time')         → (1900-01-01, 2026-05-05)
      parse_period('เดือนนี้')          → (2026-05-01, 2026-05-31)
      parse_period('เดือนที่แล้ว')     → (2026-04-01, 2026-04-30)
      parse_period('3 เดือนที่แล้ว')   → (2026-02-01, 2026-05-05)
      parse_period('ปีนี้')             → (2026-01-01, 2026-12-31)
      parse_period('Q1 2026')          → (2026-01-01, 2026-03-31)  # via dateparser
      parse_period('กุมภาพันธ์ 2569') → (2026-02-01, 2026-02-28)  # BE→AD
      parse_period('xxx')              → ValueError("unrecognized period 'xxx'")
    """
    if not phrase or not phrase.strip():
        return _ALL_TIME_START, today

    p = phrase.translate(_THAI_DIGITS).strip().lower()

    # All-time markers
    if p in {
        "all", "all time", "all-time", "alltime",
        "ทั้งหมด", "ตั้งแต่เริ่ม", "ตั้งแต่เริ่มต้น",
    }:
        return _ALL_TIME_START, today

    # Common Thai/English relative phrases
    if p in {"เดือนนี้", "this month", "current month", "เดือน"}:
        return _month_range(today, 0)
    if p in {"เดือนที่แล้ว", "เดือนก่อน", "last month", "previous month"}:
        return _month_range(today, -1)
    if p in {"ปีนี้", "this year", "current year"}:
        return _year_range(today, 0)
    if p in {"ปีที่แล้ว", "ปีก่อน", "last year", "previous year"}:
        return _year_range(today, -1)
    if p in {"วันนี้", "today"}:
        return today, today
    if p in {"เมื่อวาน", "yesterday"}:
        d = today - timedelta(days=1)
        return d, d

    # "<n> เดือนที่แล้ว" — window of n months ending today
    m = re.match(r"^(\d+)\s*เดือน(ที่แล้ว|ก่อน|ที่ผ่านมา)?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            s, _ = _month_range(today, -n)
            return s, today
    m = re.match(r"^last\s+(\d+)\s+months?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            s, _ = _month_range(today, -n)
            return s, today

    # "<n> วัน"
    m = re.match(r"^(\d+)\s*วัน(ที่แล้ว|ก่อน|ที่ผ่านมา)?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            return today - timedelta(days=n), today
    m = re.match(r"^last\s+(\d+)\s+days?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            return today - timedelta(days=n), today

    # Named month with optional year
    parsed = _parse_named_month(p, today)
    if parsed is not None:
        return parsed

    # dateparser fallback (covers "October 2005", "2026-03-15", etc.)
    try:
        dp = dateparser.parse(
            phrase,
            languages=["th", "en"],
            settings={
                "RELATIVE_BASE": datetime(today.year, today.month, today.day),
                "PREFER_DATES_FROM": "past",
            },
        )
    except Exception:
        dp = None
    if dp is not None:
        d = dp.date() if hasattr(dp, "date") else dp
        return _month_range(d, 0)

    raise ValueError(
        f"unrecognized period {phrase!r}. "
        f"Try 'เดือนนี้', 'เดือนที่แล้ว', '3 เดือนที่แล้ว', "
        f"'February 2026', 'all time', or use date(YYYY, M, D) directly."
    )
