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
    # `__block_sink__` is a server-side block-emission channel (a list), not a
    # callable the LLM invokes — it is intentionally NOT documented in the
    # prompt toolbox and is unreachable from sandbox code (AST validator bans
    # dunder access). Exclude it from the prompt⇔namespace lockstep set.
    skip = {"Decimal", "date", "timedelta", "today",
            "result", "__builtins__", "__block_sink__"}
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
# UT-NS-CB — category_breakdown block sink (Option C list-sharing)
# ─────────────────────────────────────────────────────────────────────────────
#
# These verify the deterministic C2b guard + last-wins dedup + frozen wire
# schema produced by sum_by_category / compare_periods(by="category"). The
# sink is the SAME list reachable via the namespace dict's `__block_sink__`
# key, proving the cross-worker-thread channel works without state plumbing.


def _ns_with_sink(loop):
    """Build a namespace + return (ns, sink) where `sink` is the exact list
    the data methods append to (identity-shared with ns['__block_sink__'])."""
    ns = build_namespace(
        user_id="u-1",
        catalog=_catalog(),
        today=date(2026, 5, 15),
        main_loop=loop,
    )
    return ns, ns["__block_sink__"]


def test_UT_NS_CB01_block_sink_is_shared_object():
    """UT-NS-CB01: the list under ns['__block_sink__'] is the SAME object the
    _Wrappers method appends to — the Option C identity guarantee. If these
    ever diverge, run_python would read an empty sink and silently drop the
    card."""
    loop = asyncio.new_event_loop()
    try:
        ns, sink = _ns_with_sink(loop)
    finally:
        loop.close()
    # `__block_sink__` and the closure bound inside the wrappers must be one
    # object. We can't reach the wrapper directly, so assert the key exists,
    # is a list, and starts empty (a fresh turn).
    assert isinstance(sink, list) and sink == []
    assert ns["__block_sink__"] is sink


def test_UT_NS_CB02_sum_by_category_stages_block_schema(monkeypatch):
    """UT-NS-CB02: sum_by_category with 2+ buckets stages ONE block matching
    the frozen wire schema — type/title/items/total, each item name+amount+
    icon, NO diff/pct (single period). Numbers are JSON-native (int/float),
    NOT Decimal, so the plain block serializer won't crash."""

    async def fake_run_query(spec, user_id):
        assert spec.metric == "sum_by_category"
        return [
            {"bucket": "อาหาร", "category_sync_id": "c-food", "currency": "THB",
             "amount": Decimal("900.00"), "cnt": 3,
             "icon": {"type": "asset", "value": "food.png"}},
            {"bucket": "เดินทาง", "category_sync_id": "c-trip", "currency": "THB",
             "amount": Decimal("800.50"), "cnt": 2, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns, sink = _ns_with_sink(asyncio.get_running_loop())
        rows = await asyncio.to_thread(
            ns["sum_by_category"],
            start=date(2026, 5, 1), end=date(2026, 5, 31),
        )
        # Rows returned to the LLM are UNCHANGED (Decimal preserved for R3).
        assert rows[0]["amount"] == Decimal("900.00")
        # Exactly one staged block.
        assert len(sink) == 1
        block = sink[0]
        assert block["type"] == "category_breakdown"
        assert block["title"] == "หมวดที่ใช้จ่าย"
        assert "period_label" not in block  # single period
        # total = 900.00 + 800.50 = 1700.50 → float (non-integral).
        assert block["total"] == 1700.5
        assert len(block["items"]) == 2
        food = block["items"][0]
        assert food["name"] == "อาหาร"
        assert food["amount"] == 900  # integral Decimal → int
        assert food["icon"] == {"type": "asset", "value": "food.png"}
        assert "diff" not in food and "pct" not in food
        trip = block["items"][1]
        assert trip["amount"] == 800.5
        assert "icon" not in trip  # None icon dropped from wire item

    asyncio.run(run())


def test_UT_NS_CB03_single_item_does_not_emit(monkeypatch):
    """UT-NS-CB03 (C2b guard): a single-category result must NOT stage a block
    — the card needs ≥2 items to be worth showing; one row is prose-only."""

    async def fake_run_query(spec, user_id):
        return [
            {"bucket": "อาหาร", "category_sync_id": "c-food", "currency": "THB",
             "amount": Decimal("900"), "cnt": 3, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns, sink = _ns_with_sink(asyncio.get_running_loop())
        await asyncio.to_thread(
            ns["sum_by_category"],
            start=date(2026, 5, 1), end=date(2026, 5, 31),
        )
        assert sink == []  # guard suppressed the card

    asyncio.run(run())


def test_UT_NS_CB04_multi_currency_suppressed(monkeypatch):
    """UT-NS-CB04: a multi-currency result has NO single coherent total (can't
    sum THB+USD), so the card is suppressed — correctness over a faked total.
    Prose still renders both currencies."""

    async def fake_run_query(spec, user_id):
        return [
            {"bucket": "อาหาร", "category_sync_id": "c-food", "currency": "THB",
             "amount": Decimal("900"), "cnt": 3, "icon": None},
            {"bucket": "อาหาร", "category_sync_id": "c-food", "currency": "USD",
             "amount": Decimal("20"), "cnt": 1, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns, sink = _ns_with_sink(asyncio.get_running_loop())
        await asyncio.to_thread(
            ns["sum_by_category"],
            start=date(2026, 5, 1), end=date(2026, 5, 31),
        )
        assert sink == []

    asyncio.run(run())


def test_UT_NS_CB05_last_wins_dedup(monkeypatch):
    """UT-NS-CB05 (last-wins): when sum_by_category is called twice in ONE
    turn, only the LAST breakdown survives in the sink (the card shows at most
    one breakdown per turn). The sink is cleared before each append."""

    calls = {"n": 0}

    async def fake_run_query(spec, user_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return [
                {"bucket": "อาหาร", "currency": "THB",
                 "amount": Decimal("100"), "cnt": 1, "icon": None},
                {"bucket": "เดินทาง", "currency": "THB",
                 "amount": Decimal("200"), "cnt": 1, "icon": None},
            ]
        return [
            {"bucket": "ช้อปปิ้ง", "currency": "THB",
             "amount": Decimal("500"), "cnt": 1, "icon": None},
            {"bucket": "บิล", "currency": "THB",
             "amount": Decimal("700"), "cnt": 1, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns, sink = _ns_with_sink(asyncio.get_running_loop())
        await asyncio.to_thread(
            ns["sum_by_category"], start=date(2026, 4, 1), end=date(2026, 4, 30),
        )
        await asyncio.to_thread(
            ns["sum_by_category"], start=date(2026, 5, 1), end=date(2026, 5, 31),
        )
        # Only the SECOND breakdown remains.
        assert len(sink) == 1
        names = {it["name"] for it in sink[0]["items"]}
        assert names == {"ช้อปปิ้ง", "บิล"}
        assert sink[0]["total"] == 1200

    asyncio.run(run())


def test_UT_NS_CB06_compare_periods_category_stages_diff_pct(monkeypatch):
    """UT-NS-CB06: compare_periods(by='category') stages a breakdown whose
    items carry `amount` (current period), `diff`, and `pct` — and a
    `period_label`. diff/pct come from the SAME Decimal math compare_periods
    already computed (no head-math). Ordered by current spend desc."""

    async def fake_run_query(spec, user_id):
        # compare_periods fetches period1 then period2 (both sum_by_category).
        # period1 = older, period2 = current.
        if spec.time_range.start == date(2026, 4, 1):  # period1 (old)
            return [
                {"bucket": "อาหาร", "currency": "THB",
                 "amount": Decimal("1000"), "cnt": 4,
                 "icon": {"type": "asset", "value": "food.png"}},
                {"bucket": "เดินทาง", "currency": "THB",
                 "amount": Decimal("500"), "cnt": 2, "icon": None},
            ]
        # period2 (current)
        return [
            {"bucket": "อาหาร", "currency": "THB",
             "amount": Decimal("1200"), "cnt": 5,
             "icon": {"type": "asset", "value": "food.png"}},
            {"bucket": "เดินทาง", "currency": "THB",
             "amount": Decimal("450"), "cnt": 2, "icon": None},
        ]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns, sink = _ns_with_sink(asyncio.get_running_loop())
        result = await asyncio.to_thread(
            ns["compare_periods"],
            period1_start=date(2026, 4, 1), period1_end=date(2026, 4, 30),
            period2_start=date(2026, 5, 1), period2_end=date(2026, 5, 31),
            by="category",
        )
        assert result["by"] == "category"
        assert len(sink) == 1
        block = sink[0]
        assert block["type"] == "category_breakdown"
        assert block["title"] == "เปรียบเทียบรายหมวด"
        assert "period_label" in block
        # total = current period sum = 1200 + 450 = 1650.
        assert block["total"] == 1650
        # Ordered by current amount desc → อาหาร (1200) first.
        food = block["items"][0]
        assert food["name"] == "อาหาร"
        assert food["amount"] == 1200
        assert food["diff"] == 200          # 1200 - 1000
        assert food["pct"] == 20.0          # 200 / 1000 * 100
        assert food["icon"] == {"type": "asset", "value": "food.png"}
        trip = block["items"][1]
        assert trip["amount"] == 450
        assert trip["diff"] == -50          # 450 - 500
        assert trip["pct"] == -10.0

    asyncio.run(run())


def test_UT_NS_CB07_compare_periods_total_does_not_emit(monkeypatch):
    """UT-NS-CB07: compare_periods(by='total') must NOT stage a breakdown —
    only the category compare feeds the card (by='total' has no per-category
    rows to itemise)."""

    async def fake_run_query(spec, user_id):
        return [{"currency": "THB", "amount": Decimal("1000"), "cnt": 10}]

    monkeypatch.setattr(ns_mod, "_run_query", fake_run_query)

    async def run():
        ns, sink = _ns_with_sink(asyncio.get_running_loop())
        await asyncio.to_thread(
            ns["compare_periods"],
            period1_start=date(2026, 4, 1), period1_end=date(2026, 4, 30),
            period2_start=date(2026, 5, 1), period2_end=date(2026, 5, 31),
            by="total",
        )
        assert sink == []

    asyncio.run(run())


def test_UT_NS_CB08_block_is_json_serializable():
    """UT-NS-CB08: the staged block must survive a PLAIN json.dumps (the path
    sse_adapter uses — no Decimal default). A Decimal leaking into the wire
    block would raise here, exactly the production crash this coercion
    prevents."""
    import json as _json
    from src.agent.tools.codeact.namespace import _Wrappers

    loop = asyncio.new_event_loop()
    try:
        sink: list = []
        w = _Wrappers(
            user_id="u-1", catalog=_catalog(), today=date(2026, 5, 15),
            main_loop=loop, block_sink=sink,
        )
        # Stage directly with Decimal inputs (what the data methods pass).
        w._stage_breakdown_block(
            title="หมวดที่ใช้จ่าย",
            items=[
                {"name": "อาหาร", "amount": Decimal("900.00"), "icon": None},
                {"name": "เดินทาง", "amount": Decimal("800.50"),
                 "icon": None, "diff": Decimal("-50"), "pct": Decimal("-5.5")},
            ],
            total=Decimal("1700.50"),
        )
        assert len(sink) == 1
        # The crash-prevention assertion: plain dumps, no default=str.
        wire = _json.dumps(sink[0], ensure_ascii=False)
        assert "category_breakdown" in wire
        reparsed = _json.loads(wire)
        assert reparsed["total"] == 1700.5
        assert reparsed["items"][0]["amount"] == 900
        assert reparsed["items"][1]["diff"] == -50
        assert reparsed["items"][1]["pct"] == -5.5
    finally:
        loop.close()
