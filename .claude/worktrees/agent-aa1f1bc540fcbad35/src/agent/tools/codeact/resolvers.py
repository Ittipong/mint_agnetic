"""Sandbox-callable resolvers — port of entity_resolver + time_resolver.

Ported from v2 codeact_subgraph in Wave 2 — namespace API is FROZEN to
prevent the 1,234.56 hallucination regression (per memory
`project_codeact_dual_impl`). Only import adjustments are allowed.

These helpers run inside the codeact sandbox. Entity resolution is
LLM-ONLY: the catalog entities are enumerated and handed to the planner
LLM, which picks the single best match. There is NO fuzzy / string-
similarity fast-path — every resolve is a guaranteed LLM call. This trades
latency/cost for correctness on cross-language matches (e.g. "food" →
"อาหาร", "true money" → "TrueMonney") that a cheap heuristic gets wrong.

The LLM choice runs by bridging the sandbox's worker thread back to the
main asyncio loop via `run_coroutine_threadsafe`, identical to how the SQL
wrappers reach the asyncpg pool. A timeout / unavailable LLM raises the
same `ValueError("no {kind} matches ...")` callers already handle.
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

from src.agent.entity_catalog import EntityCatalog
from src.agent.session_logger import slog
from .exceptions import (
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


# The structured adapter only tells the model to "match the schema the user
# describes" — so the rerank prompt MUST describe it, or the model omits the
# required `confidence` field and pydantic validation fails (silently → no-pick).
# Appended to the wallet/category prompts (NOT tag) so the model emits all
# three fields.
_CHOICE_SCHEMA_HINT = (
    '\n\nReturn ONLY this JSON object — no prose, no markdown:\n'
    '{"sync_id": "<one of the candidate sync_ids above>", '
    '"confidence": <number 0.0-1.0, how sure you are>, '
    '"reason": "<short reason>"}'
)


async def _llm_rerank(
    kind: str, query: str, candidates: list[_Candidate]
) -> tuple[_Candidate, float] | None:
    """Ask the planner LLM to pick the best candidate from `candidates`.

    This is the SOLE resolution mechanism — no fuzzy fast-path precedes it.
    Async — the sandbox calls this through `run_coroutine_threadsafe` so
    the main event loop runs the coroutine while the worker thread waits
    on the future.

    Returns `(chosen, confidence)` or `None` when the LLM is unavailable /
    picks a sync_id outside the candidate set.
    """
    if not candidates:
        return None
    try:
        from src.agent.llm import llm
    except Exception:
        return None

    if kind == "wallet":
        bullet = "\n".join(
            (
                f"- sync_id={c.sync_id} | name={c.name!r} | type={c.wallet_type} "
                f"| category={c.wallet_category!r}"
                if c.wallet_category
                else
                f"- sync_id={c.sync_id} | name={c.name!r} | type={c.wallet_type}"
            )
            for c in candidates
        )
        prompt = (
            f"Pick the wallet that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider Thai/English semantics and typos. Consider wallet_type "
            f"(creditcard/general/goal) and wallet_category. If none match, "
            f"pick the closest with low confidence."
            + _CHOICE_SCHEMA_HINT
        )
    elif kind == "category":
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r} | parent={c.parent_id!r} "
            f"| keywords={c.keywords!r}"
            for c in candidates
        )
        prompt = (
            f"Pick the category that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider parent-child hierarchy (e.g. 'เดินทาง' is parent of "
            f"'แท็กซี่', 'BTS/MRT', 'น้ำมัน') and the keywords field. "
            f"Return the sync_id of the PRIMARY match."
            + _CHOICE_SCHEMA_HINT
        )
    else:
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r}" for c in candidates
        )
        prompt = (
            f"Pick the {kind} that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider Thai/English semantics and typos."
        )

    try:
        structured = llm.with_structured_output(_LLMChoice)
        out: _LLMChoice = await structured.ainvoke(prompt)
    except Exception as e:
        # P2: NEVER swallow the LLM/validation error silently — that is what hid
        # the `confidence`-missing ValidationError that turned every rerank into
        # a no-pick. Log the real cause, then fall back to None as before. No-op
        # when no session logger is bound (sandbox worker-thread path).
        slog(
            "resolver",
            f"rerank {kind} FAILED for {query!r}: {type(e).__name__}: {e}",
        )
        return None
    chosen = next((c for c in candidates if c.sync_id == out.sync_id), None)
    # B3: surface the rerank decision in the session log so a downstream
    # "no-pick" is explainable (low conf vs sync_id outside the candidate set)
    # instead of a guess.
    slog(
        "resolver",
        f"rerank {kind} query={query!r} → "
        f"name={getattr(chosen, 'name', None)!r} "
        f"conf={out.confidence:.2f} in_set={chosen is not None} "
        f"reason={out.reason!r}",
    )
    if chosen is None:
        return None
    return chosen, max(0.0, min(1.0, out.confidence))


def _rerank_sync(
    main_loop: asyncio.AbstractEventLoop,
    kind: str,
    query: str,
    candidates: list[_Candidate],
    timeout: float = 10.0,
) -> tuple[_Candidate, float] | None:
    """Sync facade — sandbox is in worker thread, LLM lives on main loop.

    A timeout or any bridge failure returns None; the caller turns that into
    the same `ValueError("no {kind} matches ...")` it always has."""
    coro = _llm_rerank(kind, query, candidates)
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


def _decide(
    kind: str,
    query: str,
    candidates: list[_Candidate],
    rerank: tuple[_Candidate, float] | None,
) -> _Candidate:
    """Shared decision logic for both the sync (sandbox) and async (node)
    resolution paths. Turns a rerank result into a chosen candidate or the
    canonical error contract.

    Raises:
      ValueError — empty catalog OR LLM unavailable / no-pick / timeout.
      AmbiguousMatchError — LLM picked but with low confidence.
    """
    if not candidates:
        raise ValueError(
            f"no {kind}s configured in catalog — cannot resolve {query!r}"
        )
    if rerank is not None:
        chosen, conf = rerank
        if conf >= 0.5:
            return chosen
        # LLM picked but is unsure — surface alternatives so it commits next iter.
        raise AmbiguousMatchError(
            query=query, kind=kind, candidates=[c.name for c in candidates]
        )
    # LLM unavailable / no-pick / timeout → no plausible match.
    raise ValueError(
        f"no {kind} matches {query!r}. Available: "
        f"{[c.name for c in candidates]}"
    )


def _resolve_one(
    kind: str,
    query: str,
    candidates: list[_Candidate],
    main_loop: asyncio.AbstractEventLoop,
) -> _Candidate:
    """LLM-ONLY resolution (SYNC facade for the sandbox worker thread).

    Enumerates the catalog candidates and lets the planner LLM pick the
    single best match. No fuzzy / string-similarity fast-path — every call
    hits the LLM, handling typo / paraphrase / cross-language matches (e.g.
    'food' → 'อาหาร', 'true money' → 'TrueMonney') uniformly.

    Bridges to the main loop via `run_coroutine_threadsafe`; safe only from
    a worker thread. For the async node path use `_resolve_one_async`.
    """
    if not candidates:
        # Mirror _decide's empty-catalog error without a wasted bridge hop.
        raise ValueError(
            f"no {kind}s configured in catalog — cannot resolve {query!r}"
        )
    rerank = _rerank_sync(main_loop, kind, query, candidates)
    return _decide(kind, query, candidates, rerank)


async def _resolve_one_async(
    kind: str,
    query: str,
    candidates: list[_Candidate],
) -> _Candidate:
    """LLM-ONLY resolution (ASYNC — for callers already on the main loop,
    e.g. propose_transaction tool). Awaits the LLM choice directly instead
    of bridging through `run_coroutine_threadsafe`, which would deadlock when
    invoked from the same loop the coroutine must run on.
    """
    if not candidates:
        raise ValueError(
            f"no {kind}s configured in catalog — cannot resolve {query!r}"
        )
    try:
        rerank = await _llm_rerank(kind, query, candidates)
    except Exception:
        rerank = None
    return _decide(kind, query, candidates, rerank)


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
    'BTS/MRT', 'น้ำมัน']).

    This is the SANDBOX/codeact entry point — it widens to children so a
    QUERY over a parent category sums all subcategories. The ADD path must
    NOT widen; it uses `make_resolve_category_choice` instead.
    """
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


def make_resolve_category_choice(
    catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop
):
    """ADD-path category resolver — returns the SINGLE LLM-chosen `_Candidate`
    (name + sync_id) WITHOUT parent→children expansion.

    A recorded transaction is filed under exactly one category, so widening a
    parent to its children (as the sandbox resolver does for QUERY) is wrong
    here. The node maps `_Candidate.sync_id` → `category_sync_id` directly.
    """
    candidates = _category_candidates(catalog)

    def resolve_category_choice(query: str) -> _Candidate:
        """Return the single best `_Candidate` for `query` (no expansion)."""
        return _resolve_one("category", query, candidates, main_loop)

    return resolve_category_choice


# ── Async entry points for nodes already on the main loop (no sync bridge) ──
# `propose_transaction` (the ADD tool) runs on the main event loop, so it
# CANNOT use the sync `make_resolve_*` closures (they bridge via
# run_coroutine_threadsafe, which deadlocks when called from the loop the
# coroutine must run on). These coroutines await the LLM choice directly.


async def resolve_category_choice_async(
    catalog: EntityCatalog, query: str
) -> _Candidate:
    """ADD-path category resolver (async, single match, no expansion).

    Returns the LLM-chosen `_Candidate` (name + sync_id). Raises the same
    ValueError / AmbiguousMatchError contract as the sandbox resolver.
    """
    return await _resolve_one_async("category", query, _category_candidates(catalog))


async def resolve_category_for_add_async(
    catalog: EntityCatalog, query: str, wallet_sync_id: str | None,
    txn_type: str | None = None,
) -> _Candidate | None:
    """ADD-path category resolver — wallet-scoped, type-scoped, TOP-1, NO threshold.

    Differs from `resolve_category_choice_async` on purpose:
      - Candidates are scoped to ONE wallet (`catalog.categories_for`) so a word
        only resolves to a category that wallet actually owns.
      - Candidates are further scoped to the transaction's `txn_type`
        (expense/income). A wallet often has same-named categories of BOTH
        types (e.g. an expense "อื่นๆ" AND an income "อื่นๆ"); without this an
        expense could resolve to the income row (wrong type).
      - The LLM's single best pick is ALWAYS taken — there is no confidence
        floor and no AmbiguousMatchError. The user reviews/edits on the proposal
        card, so "closest available" beats "no category". This deliberately
        bypasses `_decide` (which the wallet + sandbox resolvers still use with
        their threshold intact).

    Returns the chosen `_Candidate` (name + sync_id), or None when there are no
    candidates, the query is empty, or the LLM call fails / picks out-of-set
    (the failure is already logged inside `_llm_rerank`).
    """
    if catalog is None or not query:
        return None
    entries = catalog.categories_for(wallet_sync_id)
    if txn_type:
        entries = [c for c in entries if c.type == txn_type]
    candidates = [
        _Candidate(
            sync_id=c.sync_id, name=c.name,
            parent_id=c.parent_id, keywords=c.keywords,
        )
        for c in entries
    ]
    if not candidates:
        return None
    rerank = await _llm_rerank("category", query, candidates)
    if rerank is None:
        return None
    chosen, _conf = rerank
    return chosen


async def resolve_wallet_choice_async(
    catalog: EntityCatalog, query: str
) -> _Candidate:
    """Wallet resolver (async, single match). Returns the LLM-chosen
    `_Candidate` so the node can read both `.name` and `.sync_id`."""
    return await _resolve_one_async("wallet", query, _wallet_candidates(catalog))


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
            raise ValueError(
                "resolve_budget needs a non-empty name phrase, e.g. "
                "resolve_budget('ค่ากิน'). To list budgets call budget_list()."
            )
        return query.strip()

    return resolve_budget


def make_resolve_goal(catalog: EntityCatalog, main_loop: asyncio.AbstractEventLoop):
    def resolve_goal(query: str) -> str:
        """Echo the query — SQL uses ILIKE %query% for goal_name_phrase."""
        if not query or not query.strip():
            raise ValueError(
                "resolve_goal needs a non-empty name phrase, e.g. "
                "resolve_goal('เที่ยวญี่ปุ่น'). To list goals call goal_list()."
            )
        return query.strip()

    return resolve_goal


# ─────────────────────────────────────────────────────────────────────────────
# Time parsing (port of nodes/time_resolver.py — sync, no LLM fallback)
# ─────────────────────────────────────────────────────────────────────────────


_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ALL_TIME_START = date(1900, 1, 1)

# Single source of truth for "what parse_period accepts". Three consumers MUST
# stay in lockstep (enforced by UT-NS08):
#   1. the `ValueError` message below (built from this tuple) — the ReAct
#      self-correction signal the model reads after a miss;
#   2. the `# CODEACT TOOLBOX` section in prompts.py — the LLM's contract, so it
#      emits a canonical form on the FIRST try (prevention > recovery);
#   3. parse_period itself — every form here MUST actually parse (no doc drift).
# Keep each entry a LITERAL string that both parses and appears verbatim in the
# prompt, so the guard test can assert all three by string match.
ACCEPTED_PERIOD_FORMS: tuple[str, ...] = (
    "เดือนนี้",
    "เดือนที่แล้ว",
    "3 เดือนที่แล้ว",
    "ปีนี้",
    "ปีที่แล้ว",
    "วันนี้",
    "เมื่อวาน",
    "ไตรมาสนี้",
    "ไตรมาสที่แล้ว",
    "ครึ่งปีแรก",
    "ครึ่งปีหลัง",
    "February 2026",
    "all time",
)


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


def _current_quarter(today: date) -> int:
    """1-based calendar quarter (Q1=Jan–Mar … Q4=Oct–Dec)."""
    return (today.month - 1) // 3 + 1


def _quarter_range(year: int, q: int) -> tuple[date, date]:
    """Inclusive (start, end) of calendar quarter `q` in `year`."""
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    last = calendar.monthrange(year, end_month)[1]
    return date(year, start_month, 1), date(year, end_month, last)


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
      parse_period('ไตรมาสนี้')         → (2026-04-01, 2026-06-30)  # Q2
      parse_period('ไตรมาสที่แล้ว')    → (2026-01-01, 2026-03-31)  # Q1
      parse_period('Q1 2026')          → (2026-01-01, 2026-03-31)
      parse_period('ครึ่งปีแรก')        → (2026-01-01, 2026-06-30)  # H1
      parse_period('กุมภาพันธ์ 2569') → (2026-02-01, 2026-02-28)  # BE→AD
      parse_period('xxx')              → ValueError("unrecognized period 'xxx'")
    """
    if not phrase or not phrase.strip():
        return _ALL_TIME_START, today

    # Normalize underscores → spaces and collapse whitespace. The LLM
    # frequently passes Python-identifier forms it derived from the slot value
    # ('this month' → 'this_month', 'last_month'); without this every query
    # wasted a whole ReAct step on a ValueError before retrying with the Thai
    # phrase. Accepting both forms makes step #1 succeed directly.
    p = phrase.translate(_THAI_DIGITS).strip().lower().replace("_", " ")
    p = " ".join(p.split())

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

    # Quarter — genuine semantic gap: the LLM wants "ไตรมาสนี้" but no surface
    # form existed, so it would burn a ReAct retry (or worse, silently fall back
    # to a wrong window via dateparser). Handle calendar quarters explicitly.
    if p in {"ไตรมาสนี้", "this quarter", "current quarter"}:
        return _quarter_range(today.year, _current_quarter(today))
    if p in {"ไตรมาสที่แล้ว", "ไตรมาสก่อน", "last quarter", "previous quarter"}:
        q = _current_quarter(today) - 1
        y = today.year
        if q == 0:
            q, y = 4, y - 1
        return _quarter_range(y, q)
    # "ไตรมาส 2" / "Q2" / "ไตรมาส 2 2026" / "Q1 2026" — explicit quarter + year.
    m = re.match(r"^(?:ไตรมาส|q)\s*([1-4])(?:\s+(\d{4}))?$", p)
    if m:
        q = int(m.group(1))
        y = int(m.group(2)) if m.group(2) else today.year
        if y > 2400:  # Buddhist era → Gregorian
            y -= 543
        return _quarter_range(y, q)

    # Half-year — same gap as quarters. Current calendar year only; for a past
    # year the LLM can use the explicit quarter/named-month forms instead.
    if p in {"ครึ่งปีแรก", "ครึ่งแรกของปี", "first half", "h1"}:
        return date(today.year, 1, 1), date(today.year, 6, 30)
    if p in {"ครึ่งปีหลัง", "ครึ่งหลังของปี", "second half", "h2"}:
        return date(today.year, 7, 1), date(today.year, 12, 31)

    # "<n> เดือนที่แล้ว" — window of n months ending today.
    # Accept every common Thai suffix the LLM emits for "the last n months":
    # ที่แล้ว / ก่อน / ที่ผ่านมา / ล่าสุด / ย้อนหลัง / หลังสุด. Missing one of
    # these (e.g. 'ล่าสุด') is what wasted a whole ReAct retry in trace 0015.
    m = re.match(r"^(\d+)\s*เดือน\s*(ที่แล้ว|ก่อน|ที่ผ่านมา|ล่าสุด|ย้อนหลัง|หลังสุด)?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            s, _ = _month_range(today, -n)
            return s, today
    m = re.match(r"^(?:last|past|recent)\s+(\d+)\s+months?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            s, _ = _month_range(today, -n)
            return s, today

    # "<n> วัน"
    m = re.match(r"^(\d+)\s*วัน\s*(ที่แล้ว|ก่อน|ที่ผ่านมา|ล่าสุด|ย้อนหลัง|หลังสุด)?$", p)
    if m:
        n = int(m.group(1))
        if n > 0:
            return today - timedelta(days=n), today
    m = re.match(r"^(?:last|past|recent)\s+(\d+)\s+days?$", p)
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

    # Loud, exhaustive, machine-followable. In a ReAct loop this message is the
    # observation the model reads to self-correct in ONE step — so it lists
    # EVERY accepted form verbatim (copy-pasteable) rather than a vague "try X".
    raise ValueError(
        f"unrecognized period {phrase!r}. parse_period accepts exactly: "
        + ", ".join(repr(f) for f in ACCEPTED_PERIOD_FORMS)
        + ". For any other range use date(YYYY, M, D) directly."
    )
