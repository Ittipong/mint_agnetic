"""Unit tests for src.agent.tools.codeact.namespace.

Covers UT-NS01..NS06.

UT-NS01 is the **lockstep regression guard** for the 1,234.56 hallucination
(memory `project_codeact_dual_impl`): every helper documented in the system
prompt's `# CODEACT TOOLBOX` section MUST be a key in
`build_namespace()`, and every callable namespace key MUST be mentioned in
the prompt. Wave 5 ships the prompt; until then the test is structured to
auto-skip with a TODO so this contract is enforced the day the prompt
lands.

UT-NS02..NS06 spot-check the high-traffic helpers (sum_expense, balance,
compare_periods, spending_pace, anomaly) with a mocked asyncpg pool — the
sandbox bridges async ↔ sync via run_coroutine_threadsafe, so the patch
hits `namespace._run_query` directly.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    TagEntry,
    WalletEntry,
)
from src.agent.tools.codeact import namespace as ns_mod
from src.agent.tools.codeact.namespace import build_namespace


# Repo root → docs path used by UT-NS01 to parse the system prompt design
# doc. Test file lives at `mint_money/mint_agentic_v3/tests/<this>` so
# parents[2] is `mint_money/` — sibling of `docs/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_DOC = (
    _REPO_ROOT / "docs" / "v3" / "phase2_system_prompt.md"
)


def _catalog() -> EntityCatalog:
    return EntityCatalog(
        wallets=[
            WalletEntry(sync_id="w-1", name="Cash", currency="THB"),
        ],
        categories=[
            CategoryEntry(sync_id="c-1", name="อาหาร", type="expense"),
        ],
        tags=[TagEntry(sync_id="t-1", name="งาน")],
    )


def _ns(main_loop: asyncio.AbstractEventLoop) -> dict:
    return build_namespace(
        user_id="u-1",
        catalog=_catalog(),
        today=date(2026, 5, 15),
        main_loop=main_loop,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS01 — Lockstep guard (prompt ⇔ namespace)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_toolbox_helpers(prompt_text: str) -> set[str]:
    """Pull function names out of the `# CODEACT TOOLBOX` section.

    Each helper line starts at column 0 with `name(`. We deliberately do
    NOT match across the whole prompt — the toolbox section is the
    contract, the MODE GUIDE / examples are free-text and may mention any
    helper.
    """
    start = prompt_text.find("# CODEACT TOOLBOX")
    if start == -1:
        return set()
    # The section ends at "# Output marker" or "# MODE GUIDE" (whichever first).
    rest = prompt_text[start:]
    end_marker = re.search(r"\n# (Output marker|MODE GUIDE|END OF SYSTEM PROMPT)", rest)
    if end_marker is not None:
        section = rest[: end_marker.start()]
    else:
        section = rest

    helpers: set[str] = set()
    # Match `name(` at start of a stripped line, capturing only ASCII names.
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z_0-9]*)\s*\(", stripped)
        if m:
            helpers.add(m.group(1))
    return helpers


def _filter_callable_keys(ns: dict) -> set[str]:
    """Namespace keys that the LLM is meant to CALL via the toolbox listing.

    Excludes (mentioned in the prompt's "Decimal-safe primitives" subsection,
    NOT in the function-style toolbox listing):
      - Type constants: Decimal, date, timedelta
      - Day helper: today  (line: `Decimal, date, timedelta, today()`)
      - Output marker: result
      - Python internal: __builtins__
    """
    skip = {"Decimal", "date", "timedelta", "today",
            "result", "__builtins__"}
    return {k for k in ns.keys() if k not in skip}


def test_UT_NS01_namespace_keys_match_prompt_toolbox():
    """UT-NS01 (regression guard for `project_codeact_dual_impl`): every
    callable in `build_namespace()` MUST be documented in the prompt's
    CODEACT TOOLBOX section, and the prompt MUST NOT mention helpers that
    aren't in the namespace.

    Until Wave 5 ships `prompts.py`, parse the design doc
    `docs/v3/phase2_system_prompt.md` — its `# CODEACT TOOLBOX` section is
    the locked contract per Phase 2 design.
    """
    if not _PROMPT_DOC.exists():
        pytest.skip(
            "phase2_system_prompt.md not found — Wave 5 will ship prompts.py; "
            "this test re-parses the source-of-truth then.",
        )
    prompt_text = _PROMPT_DOC.read_text(encoding="utf-8")
    prompt_helpers = _parse_toolbox_helpers(prompt_text)
    if not prompt_helpers:
        pytest.skip(
            "CODEACT TOOLBOX section parsed empty — likely a doc format "
            "drift. Once Wave 5 ships prompts.py this becomes a hard fail.",
        )

    loop = asyncio.new_event_loop()
    try:
        ns = _ns(loop)
    finally:
        loop.close()
    ns_callables = _filter_callable_keys(ns)

    missing_in_ns = prompt_helpers - ns_callables
    extra_in_ns = ns_callables - prompt_helpers

    # Hard fail on EITHER direction — both are R1 risks per the plan.
    assert not missing_in_ns, (
        f"prompt mentions helpers not in build_namespace: {missing_in_ns}"
    )
    assert not extra_in_ns, (
        f"build_namespace exposes helpers not in prompt: {extra_in_ns}"
    )


def test_UT_NS01b_namespace_returns_expected_18_data_keys():
    """Quick sanity: 18 data-query helpers + 7 entity/time resolvers +
    `clarify` + 4 primitives + `result` = 31 keys total today. The plan
    spec mentions "18 expected" (counting only the data-query/analytics
    callables, excluding resolvers + primitives + result marker)."""
    loop = asyncio.new_event_loop()
    try:
        ns = _ns(loop)
    finally:
        loop.close()
    # Data + analytics helpers — counted explicitly (cross-check § plan).
    data_keys = {
        "sum_income", "sum_expense", "sum_by_category", "sum_by_wallet",
        "sum_by_tag", "list_transactions", "balance", "budget_remaining",
        "budget_transactions", "budget_list", "creditcard_list", "goal_list",
        "goal_progress", "goal_transactions", "count_transactions",
        "wallet_list", "category_list", "tag_list", "spending_trend",
        "transaction_stats", "top_transactions", "currency_rate",
        "active_period", "compare_periods", "spending_pace", "anomaly",
    }
    assert data_keys.issubset(ns.keys())

    # Resolvers + clarify + primitives + marker.
    other_keys = {
        "resolve_wallet", "resolve_category", "resolve_tag",
        "resolve_budget", "resolve_goal", "parse_period", "clarify",
        "Decimal", "date", "timedelta", "today", "result",
    }
    assert other_keys.issubset(ns.keys())

    # Wave 5 — financial decision calculators (pure math, no DB).
    calculator_keys = {
        "compute_dti", "mortgage_payment", "affordability_check",
        "refi_payback_months", "debt_payoff_months", "emergency_fund_target",
    }
    assert calculator_keys.issubset(ns.keys())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS02 — sum_expense returns correct shape
# ─────────────────────────────────────────────────────────────────────────────


async def _async_setup_ns():
    """Build a namespace bound to the currently-running loop and patch
    `_run_query` so we never hit Postgres."""
    return build_namespace(
        user_id="u-1",
        catalog=_catalog(),
        today=date(2026, 5, 15),
        main_loop=asyncio.get_running_loop(),
    )


def test_UT_NS02_sum_expense_returns_rows(monkeypatch):
    """UT-NS02: sum_expense kwargs land in QuerySpec, _run_query returns
    rows shaped `[{currency, amount, cnt}]`."""

    async def fake_run_query(spec, user_id):
        # Assert the spec was constructed correctly.
        assert spec.metric == "sum_expense"
        assert spec.time_range.start == date(2026, 5, 1)
        assert spec.time_range.end == date(2026, 5, 31)
        assert user_id == "u-1"
        return [{"currency": "THB", "amount": Decimal("1234.56"), "cnt": 7}]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = await _async_setup_ns()
        # Sandbox calls sum_expense from a worker thread; mimic that here.
        rows = await asyncio.to_thread(
            ns["sum_expense"],
            start=date(2026, 5, 1),
            end=date(2026, 5, 31),
        )
        assert rows == [{"currency": "THB", "amount": Decimal("1234.56"), "cnt": 7}]

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS03 — balance with no transactions = 0.0
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_NS03_balance_empty_returns_zero(monkeypatch):
    """UT-NS03: when a user has no transactions, balance still returns the
    wallet row(s) with `amount=0` — initial_balance only, no NULL/Error."""

    async def fake_run_query(spec, user_id):
        assert spec.metric == "balance"
        return [{"wallet_sync_id": "w-1", "wallet_name": "Cash",
                 "currency": "THB", "amount": Decimal("0")}]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = await _async_setup_ns()
        rows = await asyncio.to_thread(ns["balance"])
        assert len(rows) == 1
        assert rows[0]["amount"] == Decimal("0")

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS04 — compare_periods returns delta as Decimal
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_NS04_compare_periods_preserves_decimal(monkeypatch):
    """UT-NS04: compare_periods does its own Decimal arithmetic in Python —
    the diff & pct must NEVER fall back to float (per memory
    `project_chat_money_column_float8`). We patch _run_query to feed
    Decimals and assert the resulting strings round-trip back to Decimal
    losslessly."""

    counter = {"n": 0}

    async def fake_run_query(spec, user_id):
        counter["n"] += 1
        if counter["n"] == 1:
            return [{"bucket": "อาหาร", "amount": Decimal("1000.00"), "cnt": 5}]
        return [{"bucket": "อาหาร", "amount": Decimal("1500.00"), "cnt": 6}]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = await _async_setup_ns()
        out = await asyncio.to_thread(
            ns["compare_periods"],
            period1_start=date(2026, 4, 1),
            period1_end=date(2026, 4, 30),
            period2_start=date(2026, 5, 1),
            period2_end=date(2026, 5, 31),
            by="category",
        )
        row = out["rows"][0]
        # Diff string MUST round-trip to Decimal cleanly.
        assert Decimal(row["diff"]) == Decimal("500.00")
        # 50% increase.
        assert Decimal(row["pct_change"]).quantize(Decimal("0.01")) \
            == Decimal("50.00")

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS04b — compare_periods defends its call surface (the "เทียบ 3 เดือน" misfire)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_NS04b_compare_periods_guards_mis_call():
    """UT-NS04b: the classic LLM misfire on "เทียบ N เดือน" — calling
    compare_periods with only period1 and/or by="month" — must raise an
    ACTIONABLE ValueError that names `spending_trend`, NOT a cryptic
    TypeError from Python's argument binding (trace 0005, thread 8e2887b2).
    """

    async def run():
        ns = await _async_setup_ns()
        fn = ns["compare_periods"]

        # Missing period2 → clear ValueError pointing at spending_trend.
        with pytest.raises(ValueError, match="spending_trend"):
            await asyncio.to_thread(
                fn,
                period1_start=date(2026, 3, 1),
                period1_end=date(2026, 5, 31),
                by="category",
            )

        # by="month" is not a bucket → clear ValueError, before period2 even
        # matters (this is the exact arg combo from the failing trace).
        with pytest.raises(ValueError, match="spending_trend"):
            await asyncio.to_thread(
                fn,
                period1_start=date(2026, 3, 1),
                period1_end=date(2026, 5, 31),
                by="month",
            )

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS05 — spending_pace handles incomplete month
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_NS05_spending_pace_projects_partial_month(monkeypatch):
    """UT-NS05: spending_pace from 1st → as_of computes daily_avg and
    projects to the full month. With today=May 15 (= day 15 of 31),
    spent_so_far=3000 → daily_avg=200 → projected_total=6200."""

    async def fake_run_query(spec, user_id):
        assert spec.metric == "sum_expense"
        return [{"currency": "THB", "amount": Decimal("3000.00"), "cnt": 15}]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = await _async_setup_ns()
        out = await asyncio.to_thread(
            ns["spending_pace"],
            as_of=date(2026, 5, 15),
        )
        assert out["days_elapsed"] == 15
        assert out["days_in_month"] == 31
        # spent_so_far comes back as a string of the Decimal.
        assert Decimal(out["spent_so_far"]) == Decimal("3000.00")
        # daily_avg = 3000 / 15 = 200.
        assert Decimal(out["daily_avg"]) == Decimal("200")
        # projected_total = 200 * 31 = 6200.
        assert Decimal(out["projected_total"]) == Decimal("6200")

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS06 — anomaly detection
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_NS06_anomaly_high_when_today_double_average(monkeypatch):
    """UT-NS06: anomaly tags 'high' when today's spend ≥ 2× the lookback
    daily average. We feed today=2000, window=300/day over 30d → ratio
    ≈ 6.67 → 'very_high'. (The exact bin boundaries are documented in the
    helper docstring; this asserts the ratio computation drives the
    bucketing, not a stale hardcoded value.)"""

    counter = {"n": 0}

    async def fake_run_query(spec, user_id):
        counter["n"] += 1
        if counter["n"] == 1:
            # today_spent for [as_of, as_of]
            return [{"currency": "THB", "amount": Decimal("2000.00"), "cnt": 4}]
        # window_total for the previous 30 days
        return [{"currency": "THB", "amount": Decimal("9000.00"), "cnt": 90}]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = await _async_setup_ns()
        out = await asyncio.to_thread(
            ns["anomaly"],
            lookback_days=30,
            as_of=date(2026, 5, 15),
        )
        assert out["lookback_days"] == 30
        assert Decimal(out["today_spent"]) == Decimal("2000.00")
        # ratio ≈ 2000 / (9000 / 30) = 2000 / 300 ≈ 6.67 → very_high (>= 3).
        assert out["anomaly_level"] == "very_high"

    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS07 — parse_period accepts common "last N months" phrasings (E2)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("phrase", [
    "3 เดือนล่าสุด",
    "3 เดือนย้อนหลัง",
    "3 เดือนที่ผ่านมา",
    "3 เดือนที่แล้ว",
    "3 เดือนก่อน",
    "past 3 months",
    "recent 3 months",
    "last 3 months",
])
def test_UT_NS07_parse_period_last_3_months_variants(phrase):
    """UT-NS07 (E2): eliminate the avoidable ValueError at the source. The LLM
    emits many surface forms for "the last 3 months"; each unrecognized one
    wastes a ReAct retry (trace 0015: 'ล่าสุด' was unrecognized). All MUST
    parse to the same Feb 1 → today window without raising."""
    from src.agent.tools.codeact.resolvers import parse_period
    start, end = parse_period(phrase, date(2026, 5, 15))
    assert start == date(2026, 2, 1), f"{phrase!r} → {start}"
    assert end == date(2026, 5, 15), f"{phrase!r} → {end}"


def test_UT_NS07b_parse_period_days_alias():
    from datetime import timedelta
    from src.agent.tools.codeact.resolvers import parse_period
    start, end = parse_period("7 วันล่าสุด", date(2026, 5, 15))
    assert start == date(2026, 5, 15) - timedelta(days=7)
    assert end == date(2026, 5, 15)


def test_UT_NS07c_parse_period_still_raises_on_garbage():
    from src.agent.tools.codeact.resolvers import parse_period
    with pytest.raises(ValueError, match="unrecognized period"):
        parse_period("ไม่รู้เรื่อง xyz", date(2026, 5, 15))


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS08 — Lockstep guard: ACCEPTED_PERIOD_FORMS ≡ parser ≡ error ≡ prompt
#
# The "prevention-first + self-correcting error" design has three consumers of
# one constant (the parser, the ValueError message, the prompt toolbox). If any
# drifts, the LLM gets contradictory signals — these tests fail the day that
# happens. They also assert "documented == actually works" (no doc drift).
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_NS08_every_accepted_form_actually_parses():
    """UT-NS08: every form advertised in ACCEPTED_PERIOD_FORMS MUST parse
    without raising — otherwise we document a form that burns a ReAct step."""
    from src.agent.tools.codeact.resolvers import (
        ACCEPTED_PERIOD_FORMS,
        parse_period,
    )
    today = date(2026, 5, 15)
    for form in ACCEPTED_PERIOD_FORMS:
        if form == "all time":
            continue  # validated by its own start sentinel below
        start, end = parse_period(form, today)
        assert start <= end, f"{form!r} → ({start}, {end})"


def test_UT_NS08b_error_message_lists_every_accepted_form():
    """UT-NS08b: the self-correction message MUST name every accepted form
    verbatim so the model can copy one back in a single retry."""
    from src.agent.tools.codeact.resolvers import (
        ACCEPTED_PERIOD_FORMS,
        parse_period,
    )
    try:
        parse_period("ช่วงไหนสักช่วง", date(2026, 5, 15))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        msg = str(exc)
    for form in ACCEPTED_PERIOD_FORMS:
        assert form in msg, f"{form!r} missing from error message"


def test_UT_NS08c_prompt_toolbox_mentions_every_accepted_form():
    """UT-NS08c (lockstep with prompts.py): the CODEACT TOOLBOX section MUST
    mention every accepted form, so the LLM emits a canonical phrase on the
    FIRST try (prevention) instead of relying on the error-recovery path."""
    from src.agent.prompts import CODEACT_TOOLBOX_SECTION
    from src.agent.tools.codeact.resolvers import ACCEPTED_PERIOD_FORMS
    for form in ACCEPTED_PERIOD_FORMS:
        assert form in CODEACT_TOOLBOX_SECTION, (
            f"{form!r} accepted by parse_period but not documented in the "
            f"prompt toolbox — LLM can't know to emit it"
        )


@pytest.mark.parametrize("phrase,start,end", [
    # today = 2026-05-15 → Q2
    ("ไตรมาสนี้", date(2026, 4, 1), date(2026, 6, 30)),
    ("this quarter", date(2026, 4, 1), date(2026, 6, 30)),
    ("ไตรมาสที่แล้ว", date(2026, 1, 1), date(2026, 3, 31)),
    ("ไตรมาส 4", date(2026, 10, 1), date(2026, 12, 31)),
    ("Q1 2026", date(2026, 1, 1), date(2026, 3, 31)),
    ("ครึ่งปีแรก", date(2026, 1, 1), date(2026, 6, 30)),
    ("ครึ่งปีหลัง", date(2026, 7, 1), date(2026, 12, 31)),
])
def test_UT_NS08d_quarter_and_half_year_ranges(phrase, start, end):
    """UT-NS08d: quarter / half-year forms (the genuine semantic gaps that had
    no surface form) resolve to the correct calendar window."""
    from src.agent.tools.codeact.resolvers import parse_period
    s, e = parse_period(phrase, date(2026, 5, 15))
    assert (s, e) == (start, end), f"{phrase!r} → ({s}, {e})"


def test_UT_NS08e_last_quarter_rolls_into_previous_year():
    """UT-NS08e: 'ไตรมาสที่แล้ว' in Q1 rolls back to Q4 of the prior year."""
    from src.agent.tools.codeact.resolvers import parse_period
    s, e = parse_period("ไตรมาสที่แล้ว", date(2026, 2, 10))  # today in Q1
    assert (s, e) == (date(2025, 10, 1), date(2025, 12, 31))


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS09 — Week phrases resolve to a Mon–Sun calendar week (today = Wed 2026-06-03)
# ─────────────────────────────────────────────────────────────────────────────
_TODAY_WED = date(2026, 6, 3)  # a Wednesday


@pytest.mark.parametrize("phrase,start,end", [
    ("สัปดาห์นี้", date(2026, 6, 1), date(2026, 6, 7)),
    ("อาทิตย์นี้", date(2026, 6, 1), date(2026, 6, 7)),
    ("this week", date(2026, 6, 1), date(2026, 6, 7)),
    ("สัปดาห์ที่แล้ว", date(2026, 5, 25), date(2026, 5, 31)),
    ("สัปดาห์ก่อน", date(2026, 5, 25), date(2026, 5, 31)),
    ("last week", date(2026, 5, 25), date(2026, 5, 31)),
    # "<n> weeks ago" = trailing n×7 days ending today
    ("2 สัปดาห์ที่แล้ว", date(2026, 5, 20), date(2026, 6, 3)),
    ("past 2 weeks", date(2026, 5, 20), date(2026, 6, 3)),
])
def test_UT_NS09_week_ranges(phrase, start, end):
    from src.agent.tools.codeact.resolvers import parse_period
    s, e = parse_period(phrase, _TODAY_WED)
    assert (s, e) == (start, end), f"{phrase!r} → ({s}, {e})"


def test_UT_NS10_this_week_is_not_silently_a_whole_month():
    """UT-NS10 (regression): the original bug expanded 'สัปดาห์นี้' to the WHOLE
    month via dateparser → a confidently wrong window. A week must span ≤ 7 days,
    never ~30."""
    from src.agent.tools.codeact.resolvers import parse_period
    s, e = parse_period("สัปดาห์นี้", _TODAY_WED)
    assert (e - s).days == 6, f"expected a 7-day span, got {(e - s).days + 1} days"
    # explicit anti-regression: must NOT be the June 1–30 whole-month window
    assert (s, e) != (date(2026, 6, 1), date(2026, 6, 30))


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS11 — Year-to-date + rolling 'past year'
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("phrase,start,end", [
    ("ตั้งแต่ต้นปี", date(2026, 1, 1), _TODAY_WED),
    ("year to date", date(2026, 1, 1), _TODAY_WED),
    ("ytd", date(2026, 1, 1), _TODAY_WED),
    ("ปีนี้ถึงตอนนี้", date(2026, 1, 1), _TODAY_WED),
    # rolling 365-day window, NOT the previous calendar year
    ("past year", date(2025, 6, 3), _TODAY_WED),
    ("ปีที่ผ่านมา", date(2025, 6, 3), _TODAY_WED),
])
def test_UT_NS11_ytd_and_past_year(phrase, start, end):
    from src.agent.tools.codeact.resolvers import parse_period
    s, e = parse_period(phrase, _TODAY_WED)
    assert (s, e) == (start, end), f"{phrase!r} → ({s}, {e})"


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS12 — Guard: fuzzy spans with NO date signal fail LOUDLY (no wrong window)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("phrase", [
    "ช่วงปลายปี", "เร็วๆ นี้", "ช่วงสงกรานต์", "เทศกาลปีใหม่",
    "ไม่นานมานี้", "หลายเดือนก่อน", "ช่วงนี้", "ต้นเดือน",
])
def test_UT_NS12_fuzzy_period_raises_instead_of_wrong_window(phrase):
    """UT-NS12: a vague span the parser cannot ground must raise ValueError so the
    ReAct model falls back to date(YYYY, M, D) — NEVER return a guessed range."""
    from src.agent.tools.codeact.resolvers import parse_period
    with pytest.raises(ValueError):
        parse_period(phrase, _TODAY_WED)


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS13 — Thai number words + quarter/half with an explicit year token
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("phrase,start,end", [
    ("สองเดือนก่อน", date(2026, 4, 1), _TODAY_WED),
    ("สามสัปดาห์ที่แล้ว", date(2026, 5, 13), _TODAY_WED),
    ("เมื่อสองวันก่อน", date(2026, 6, 1), _TODAY_WED),
    ("ไตรมาสแรก", date(2026, 1, 1), date(2026, 3, 31)),
    ("ไตรมาสที่ 4 ปีที่แล้ว", date(2025, 10, 1), date(2025, 12, 31)),
    ("ครึ่งปีแรกปีที่แล้ว", date(2025, 1, 1), date(2025, 6, 30)),
    ("ครึ่งหลังปี 2568", date(2025, 7, 1), date(2025, 12, 31)),
    ("first half of 2025", date(2025, 1, 1), date(2025, 6, 30)),
])
def test_UT_NS13_thai_numbers_and_year_tokens(phrase, start, end):
    from src.agent.tools.codeact.resolvers import parse_period
    s, e = parse_period(phrase, _TODAY_WED)
    assert (s, e) == (start, end), f"{phrase!r} → ({s}, {e})"


# ─────────────────────────────────────────────────────────────────────────────
# UT-NS-CB — category_breakdown card retired (no more block sink)
# ─────────────────────────────────────────────────────────────────────────────
#
# The per-category breakdown CARD was removed in v3; breakdowns render as a
# table/bullet list inside the answer again. So the Option C `__block_sink__`
# channel is gone entirely. These tests pin the new behaviour: the namespace
# carries no sink, and the data methods (sum_by_category /
# compare_periods(by="category")) return their rows UNCHANGED for the LLM to
# render — with NO block-staging side effect.


def _ns(loop):
    """Build a fresh namespace for a turn."""
    return build_namespace(
        user_id="u-1",
        catalog=_catalog(),
        today=date(2026, 5, 15),
        main_loop=loop,
    )


def test_UT_NS_CB01_namespace_has_no_block_sink():
    """UT-NS-CB01: build_namespace must NOT expose `__block_sink__` any more —
    the card channel was removed. A stale key would let a resurrected emitter
    silently ship a card again."""
    loop = asyncio.new_event_loop()
    try:
        ns = _ns(loop)
    finally:
        loop.close()
    assert "__block_sink__" not in ns


def test_UT_NS_CB02_sum_by_category_returns_rows_no_side_effect(monkeypatch):
    """UT-NS-CB02: sum_by_category returns its rows UNCHANGED (Decimal preserved
    for R3 prose grounding) and stages no block — there is no sink to mutate."""

    async def fake_run_query(spec, user_id):
        assert spec.metric == "sum_by_category"
        return [
            {"bucket": "อาหาร", "category_sync_id": "c-food", "currency": "THB",
             "amount": Decimal("900.00"), "cnt": 3},
            {"bucket": "เดินทาง", "category_sync_id": "c-trip", "currency": "THB",
             "amount": Decimal("800.50"), "cnt": 2},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = _ns(asyncio.get_running_loop())
        assert "__block_sink__" not in ns
        rows = await asyncio.to_thread(
            ns["sum_by_category"],
            start=date(2026, 5, 1), end=date(2026, 5, 31),
        )
        # Rows returned to the LLM are UNCHANGED (Decimal preserved for R3).
        assert rows[0]["amount"] == Decimal("900.00")
        assert {r["bucket"] for r in rows} == {"อาหาร", "เดินทาง"}

    asyncio.run(run())


def test_UT_NS_CB03_compare_periods_category_returns_rows_no_side_effect(
    monkeypatch,
):
    """UT-NS-CB03: compare_periods(by='category') returns the per-bucket diff
    rows for the LLM to render as a table — with no block-staging side effect.
    diff/pct math is unchanged (computed in-sandbox, Decimal-exact)."""

    async def fake_run_query(spec, user_id):
        # period1 = older, period2 = current (both sum_by_category fetches).
        if spec.time_range.start == date(2026, 4, 1):
            return [
                {"bucket": "อาหาร", "currency": "THB",
                 "amount": Decimal("1000"), "cnt": 4},
                {"bucket": "เดินทาง", "currency": "THB",
                 "amount": Decimal("500"), "cnt": 2},
            ]
        return [
            {"bucket": "อาหาร", "currency": "THB",
             "amount": Decimal("1200"), "cnt": 5},
            {"bucket": "เดินทาง", "currency": "THB",
             "amount": Decimal("450"), "cnt": 2},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns = _ns(asyncio.get_running_loop())
        result = await asyncio.to_thread(
            ns["compare_periods"],
            period1_start=date(2026, 4, 1), period1_end=date(2026, 4, 30),
            period2_start=date(2026, 5, 1), period2_end=date(2026, 5, 31),
            by="category",
        )
        assert result["by"] == "category"
        rows = {r["bucket"]: r for r in result["rows"]}
        # อาหาร: 1000 → 1200, diff +200, pct +20%.
        assert rows["อาหาร"]["period2_amount"] == "1200"
        assert rows["อาหาร"]["diff"] == "200"
        # เดินทาง: 500 → 450, diff -50.
        assert rows["เดินทาง"]["diff"] == "-50"

    asyncio.run(run())
