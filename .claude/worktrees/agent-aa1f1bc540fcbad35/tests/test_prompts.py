"""Unit tests for src.agent.prompts — the v3 system prompt and its helpers.

Covers Wave 4 UT-P01..UT-P03.

UT-P01 is the **binding-driven** lockstep guard for the 1,234.56 hallucination
(memory `project_codeact_dual_impl`). It complements UT-NS01 in
`test_codeact_namespace.py`:

  - UT-NS01 parses the design doc `docs/v3/phase2_system_prompt.md` and
    checks that every helper named there exists in `build_namespace()`.
  - UT-P01 parses the SAME boundary out of the runtime `SYSTEM_PROMPT`
    constant — the thing the agent actually sees at run time — and runs the
    same check.

Both pass today; once we deprecate the design doc (UT-NS01 can be deleted),
UT-P01 alone is sufficient because it inspects what ships, not what was
designed.

UT-P02 covers the `{today}` / `{user_id}` placeholder substitution.
UT-P03 enforces Q6 — `clarify_wallet` was dropped from the tool registry
and MUST NOT be named in the shipping prompt (no row in the status table,
no TOOLS entry, no example).
"""

from __future__ import annotations

import asyncio
import re
from datetime import date

import pytest

from src.agent.prompts import (
    CODEACT_TOOLBOX_SECTION,
    SYSTEM_PROMPT,
    render_system_prompt,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirror test_codeact_namespace.py for binding-vs-doc parity)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_toolbox_helpers(section_text: str) -> set[str]:
    """Pull function names out of the CODEACT TOOLBOX section.

    Matches `name(` at the start of a stripped line — same convention as
    UT-NS01 in test_codeact_namespace.py so the two tests stay aligned.
    """
    helpers: set[str] = set()
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z_0-9]*)\s*\(", stripped)
        if m:
            helpers.add(m.group(1))
    return helpers


def _filter_callable_keys(ns: dict) -> set[str]:
    """Namespace keys the LLM is meant to CALL (excludes primitives + marker).

    Same skip list as test_codeact_namespace.py so UT-P01 and UT-NS01 see
    identical sets.
    """
    # `__block_sink__` is a server-side block-emission channel (a list), not an
    # LLM-callable helper — intentionally undocumented in the prompt toolbox
    # and unreachable from sandbox code (AST validator bans dunder access).
    skip = {"Decimal", "date", "timedelta", "today",
            "result", "__builtins__", "__block_sink__"}
    return {k for k in ns.keys() if k not in skip}


# ─────────────────────────────────────────────────────────────────────────────
# UT-P01 — binding-driven lockstep (SYSTEM_PROMPT ⇔ build_namespace())
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_P01_prompt_toolbox_matches_namespace_keys():
    """UT-P01: every helper named in the SYSTEM_PROMPT's CODEACT TOOLBOX
    section MUST exist in `build_namespace()`, and `build_namespace()` MUST
    NOT expose callables the prompt fails to advertise.

    This is the R1 risk guard (per phase3_implementation_plan.md §7) — the
    binding-driven half of the lockstep. UT-NS01 covers the doc-driven half.
    """
    # Import inside the test to keep collection independent of an event loop
    # being available — `build_namespace` calls into asyncpg setup that
    # touches the running loop.
    from src.agent.entity_catalog import (
        CategoryEntry,
        EntityCatalog,
        TagEntry,
        WalletEntry,
    )
    from src.agent.tools.codeact.namespace import build_namespace

    catalog = EntityCatalog(
        wallets=[WalletEntry(sync_id="w-1", name="Cash", currency="THB")],
        categories=[
            CategoryEntry(sync_id="c-1", name="อาหาร", type="expense"),
        ],
        tags=[TagEntry(sync_id="t-1", name="งาน")],
    )

    # Pull helpers out of the SHIPPING prompt (not the design doc).
    prompt_helpers = _parse_toolbox_helpers(CODEACT_TOOLBOX_SECTION)
    assert prompt_helpers, (
        "CODEACT TOOLBOX section parsed empty — prompts.py extraction "
        "regex is out of sync with the section format."
    )

    # Build the namespace against a throwaway loop — we only need the keys.
    loop = asyncio.new_event_loop()
    try:
        ns = build_namespace(
            user_id="u-1",
            catalog=catalog,
            today=date(2026, 5, 15),
            main_loop=loop,
        )
    finally:
        loop.close()
    ns_callables = _filter_callable_keys(ns)

    missing_in_ns = prompt_helpers - ns_callables
    extra_in_ns = ns_callables - prompt_helpers
    assert not missing_in_ns, (
        f"SYSTEM_PROMPT names helpers not in build_namespace: {missing_in_ns}"
    )
    assert not extra_in_ns, (
        f"build_namespace exposes helpers not in SYSTEM_PROMPT: {extra_in_ns}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# UT-P02 — placeholder substitution
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_P02_render_substitutes_today_and_user_id():
    """UT-P02: `render_system_prompt(today, user_id)` MUST replace both
    placeholders. The raw template still carries `{today}` / `{user_id}`,
    but the rendered output MUST NOT contain either placeholder literal.
    """
    rendered = render_system_prompt(
        today="2026-05-29",
        user_id="user-abc-123",
    )
    assert "2026-05-29" in rendered, "today placeholder did not substitute"
    assert "user-abc-123" in rendered, "user_id placeholder did not substitute"
    # Placeholder literals must NOT survive into the rendered prompt — that
    # would mean the substitution missed one.
    assert "{today}" not in rendered
    assert "{user_id}" not in rendered


def test_UT_P02_render_is_pure_no_template_side_effects():
    """UT-P02 supporting: rendering DOES NOT mutate SYSTEM_PROMPT — repeated
    renders with different values are independent."""
    before = SYSTEM_PROMPT
    _r1 = render_system_prompt(today="2026-01-01", user_id="u-1")
    _r2 = render_system_prompt(today="2027-12-31", user_id="u-2")
    after = SYSTEM_PROMPT
    assert before == after, "render_system_prompt should not mutate the template"
    # Cross-check that {today}/{user_id} remain in the unrendered template.
    assert "{today}" in SYSTEM_PROMPT
    assert "{user_id}" in SYSTEM_PROMPT


def test_UT_P02_render_tolerates_unrelated_braces_in_examples():
    """UT-P02 supporting: the few-shot examples include Python dict /
    formatter snippets such as `{"income": ...}` and `{wallets,
    categories, ...}` — those LOOK like format placeholders but are NOT.
    Using `str.format` would raise KeyError on the unbound names; using
    `str.replace` is safe. This test guards against a regression to
    `str.format` by asserting the render succeeds AND that at least one of
    those literal brace tokens survives intact.
    """
    rendered = render_system_prompt(today="2026-05-29", user_id="u")
    # The get_user_context Return description carries a brace-y dict-like
    # spec the LLM is meant to see verbatim.
    assert "{wallets, goals, budgets" in rendered, (
        "expected the get_user_context return spec to survive rendering "
        "— if it doesn't, the rendering path may have broken its braces."
    )


# ─────────────────────────────────────────────────────────────────────────────
# UT-P03 — Q6 enforcement: no `clarify_wallet` in the shipping prompt
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_P03_no_clarify_wallet_in_system_prompt():
    """UT-P03: `clarify_wallet` was DROPPED per Q6 (memory
    `project_wallet_picker_disabled`). The shipping prompt MUST NOT mention
    it — no TOOLS row, no status table row, no example. Wallet ambiguity
    falls back to index-0 inside propose_transaction.

    Catches the most common regression: an implementer adds the tool back
    "for completeness" without realizing it was intentionally dropped.
    """
    assert "clarify_wallet" not in SYSTEM_PROMPT.lower(), (
        "SYSTEM_PROMPT must not mention `clarify_wallet` (Q6 amendment); "
        "wallet ambiguity → index-0 fallback in propose_transaction."
    )


def test_UT_P03_rendered_prompt_also_has_no_clarify_wallet():
    """UT-P03 supporting: rendering must not somehow re-inject the disabled
    tool name (e.g. via a placeholder that nudges the LLM). Defensive."""
    rendered = render_system_prompt(today="2026-05-29", user_id="u-1")
    assert "clarify_wallet" not in rendered.lower()


# ─────────────────────────────────────────────────────────────────────────────
# UT-P04 — dict-key lockstep (SYSTEM_PROMPT examples ⇔ tool return shapes)
# ─────────────────────────────────────────────────────────────────────────────
#
# Why this exists: UT-P01 already checks that every helper NAME the prompt
# mentions exists in build_namespace(). But it never inspects the example
# BODIES — the few-shot Python snippets that teach the LLM how to read tool
# outputs. The `amount_thb` regression (trace 019e7399-…) lived inside those
# bodies for months: the SQL aliased the column as `amount`, the prompt
# taught `rows[0]["amount_thb"]`, and every Analyst/Advisor query crashed
# with a silent ReAct loop until OOM.
#
# UT-P04 closes that loophole: scan every literal-key dict access inside
# SYSTEM_PROMPT (`x["key"]`, `x.get("key")`) and assert each key is one the
# tooling actually returns — either:
#   (a) a SQL alias extracted from sql_templates.py (any `AS keyname`), or
#   (b) a key produced by a Python composite in namespace.py
#       (compare_periods / spending_pace / anomaly), or
#   (c) a key returned by a non-SQL tool (memory_recall / get_user_context
#       / propose_transaction's ToolMessage body).
#
# When this test fails because the prompt added a legitimate new tool, the
# fix is to extend `_PYTHON_COMPOSITE_KEYS` below — NOT to silence the test.


_KEY_PATTERN = re.compile(
    r"""(?x)
    (?:
        \[                              # subscript open
        \s*
        ["']([a-zA-Z_][a-zA-Z_0-9]*)["'] # captured key
        \s*
        \]                              # subscript close
      |
        \.get\(                         # .get( call
        \s*
        ["']([a-zA-Z_][a-zA-Z_0-9]*)["'] # captured key
    )
    """
)


# Keys returned by hand-built python composites in `namespace.py` plus the
# non-SQL tools (get_user_context, memory_recall, propose_transaction's
# JSON ToolMessage payloads). These won't show up in `sql_templates.py`
# because they're built in Python — so we list them explicitly.
_PYTHON_COMPOSITE_KEYS: frozenset[str] = frozenset({
    # compare_periods — top-level and per-row
    "by", "period1", "period2", "rows",
    "period1_amount", "period2_amount", "diff", "pct_change",
    # spending_pace
    "month_start", "as_of", "month_end",
    "days_elapsed", "days_in_month",
    "spent_so_far", "daily_avg", "projected_total", "projected_remaining",
    # anomaly
    "lookback_days", "today_spent", "lookback_avg",
    "ratio", "anomaly_level",
    # get_user_context — catalog shape
    "wallets", "categories", "goals", "budgets", "tags",
    "wallet_id", "default_currency_code",
    # EntityCatalog row shapes
    "wallet_type", "is_default", "parent_sync_id",
    # memory_recall return
    "memories", "ts",
    # propose_transaction JSON body
    "proposal_id", "transaction_sync_id", "discarded_proposal_id",
    # generic tool error envelope
    "error",
})


def _extract_sql_aliases() -> set[str]:
    """Pull every `AS keyname` alias out of `sql_templates.py`.

    The regex is intentionally loose — we want every alias the SQL ever emits
    so the allow-list catches up automatically when a builder is extended.
    """
    import pathlib

    sql_path = (
        pathlib.Path(__file__).parent.parent
        / "src" / "agent" / "tools" / "codeact" / "sql_templates.py"
    )
    text = sql_path.read_text()
    # Match `AS alias` and `AS  alias` — ignore quoted aliases (we have none).
    return set(re.findall(r"\bAS\s+([a-zA-Z_][a-zA-Z_0-9]*)", text))


def test_UT_P04_prompt_dict_keys_match_tool_return_shapes():
    """UT-P04: every `["key"]` / `.get("key")` access inside SYSTEM_PROMPT
    must reference a key some tool actually returns.

    This guards against the `amount_thb` class of bug: a prompt example that
    accesses a field name the SQL never aliases. The LLM dutifully follows
    the prompt → `KeyError` → ReAct retries with identical code → loop until
    timeout. The trace `019e7399-4c2b-71e1-bdb0-05c0b619bd20` is the canonical
    failure that motivated this test.

    When extending: if you add a tool that returns a new key the prompt
    mentions, add it to `_PYTHON_COMPOSITE_KEYS` (for Python-built dicts) or
    add an `AS newkey` clause in `sql_templates.py` (for SQL-emitted columns).
    Do NOT silence the assertion.
    """
    sql_aliases = _extract_sql_aliases()
    allowed = sql_aliases | _PYTHON_COMPOSITE_KEYS

    # Find every literal-key dict access. Each match yields one of two
    # capture groups (the other is empty for that match).
    keys_used: set[str] = set()
    for subscript_key, get_key in _KEY_PATTERN.findall(SYSTEM_PROMPT):
        key = subscript_key or get_key
        if key:
            keys_used.add(key)

    # Sanity: we expect at least a few keys (the few-shot Python examples
    # exercise dict access heavily). An empty set means the regex broke.
    assert keys_used, (
        "UT-P04: no dict-key access patterns found in SYSTEM_PROMPT — the "
        "regex `_KEY_PATTERN` may be out of sync with the example syntax."
    )

    unknown = keys_used - allowed
    assert not unknown, (
        f"UT-P04: SYSTEM_PROMPT teaches the LLM to access dict keys that no "
        f"tool returns: {sorted(unknown)}. Either the prompt is wrong (will "
        f"cause a KeyError → ReAct loop, see trace 019e7399-…) or the "
        f"allow-list in this test is stale. Cross-check against "
        f"`sql_templates.py` AS-aliases and `_PYTHON_COMPOSITE_KEYS`."
    )


# ─────────────────────────────────────────────────────────────────────────────
# UT-P05 — no `get_user_context()` call in any few-shot example body
# ─────────────────────────────────────────────────────────────────────────────
#
# Why this exists: `propose_transaction` and `run_python` both auto-load the
# catalog when `state.user_context` is missing, so an explicit
# `get_user_context()` call in an EX-* Plan is pure overhead — it costs a
# whole extra LLM round (+~2s, see log 0023, trace 019e741a-…) without
# changing the result. A previous version of the prompt taught the LLM to
# call `get_user_context()` as step 1 of every ADD turn; combined with the
# `missing_context` gate the LLM frequently failed to follow, this turned a
# 3-second turn into a 5-second turn (or a hard failure when the LLM skipped
# it entirely). The Option-4 rewrite deleted those calls and this test
# pins the fix.
#
# Allowed: prose mentions of the tool name in TOOLS docs, the deprecation
# guidance ("DO NOT call get_user_context first"), the status-message table,
# and the progress-narration example. Only `get_user_context(` as a literal
# Python call inside an EX-* block is the regression we guard against.


def test_UT_P05_no_get_user_context_calls_in_few_shot_examples():
    """UT-P05: no EX-* example body may contain a literal
    `get_user_context(...)` call.

    Slices the prompt by `## EX-` markers and scans each example block for
    the call pattern. Prose outside example blocks (TOOLS doc, status table,
    deprecation notes) is intentionally NOT scanned — those references are
    legit and reading them does not nudge the LLM into emitting a redundant
    tool call.
    """
    # Split into pre-examples + per-example chunks. The marker `## EX-` is
    # what the prompt uses to introduce every few-shot.
    parts = re.split(r"(?m)^## EX-", SYSTEM_PROMPT)
    if len(parts) <= 1:
        pytest.fail(
            "UT-P05: no `## EX-` markers found in SYSTEM_PROMPT — the prompt "
            "structure changed and this test needs updating."
        )

    offenders: list[str] = []
    # parts[0] is everything BEFORE the first EX-* (skip — it's TOOLS docs +
    # MODE GUIDE etc., where mentioning the tool name is OK).
    for chunk in parts[1:]:
        # First line of the chunk is the EX-* label (e.g. "ADD-1: Simple expense").
        label = chunk.splitlines()[0].strip() if chunk else "<unknown>"
        # `get_user_context(` is the call pattern. We ignore back-tick-quoted
        # prose mentions (` `get_user_context` `) because those don't render
        # as code calls in the LLM's eye.
        if re.search(r"(?<!`)\bget_user_context\s*\(", chunk):
            offenders.append(f"EX-{label}")

    assert not offenders, (
        f"UT-P05: the following few-shot examples still call "
        f"`get_user_context()`: {offenders}. "
        f"`propose_transaction` and `run_python` auto-load the catalog, so "
        f"this call is pure overhead — a wasted ~2-second LLM round. Drop "
        f"the call from each Plan and rely on the auto-load path."
    )


# ─────────────────────────────────────────────────────────────────────────────
# UT-P06 — ADD-emission rule (Fix B2 for trace 8e2887b2)
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_P06_prompt_forbids_confirmation_without_propose_call():
    """UT-P06: the ADD MODE GUIDE must carry the hard rule that the ADD
    confirmation ("ขอยืนยัน…" / "กดยืนยันด้านล่าง") is only valid AFTER a
    `propose_transaction` call THIS turn. This is the prompt half of the fix
    for the long-thread poisoning bug (trace 8e2887b2), where the model
    imitated its own text-only acknowledgements and skipped the tool.
    """
    # The confirmation few-shot wording still exists. It must ask for
    # confirmation ("กดยืนยันด้านล่าง") — never claim the save already happened.
    assert "กดยืนยันด้านล่าง" in SYSTEM_PROMPT
    # The guardrail line itself — keyed on its distinctive phrasing so a future
    # reword keeps the intent visible.
    assert "ONLY valid AFTER you called" in SYSTEM_PROMPT, (
        "UT-P06: the ADD section must explicitly forbid the confirmation text "
        "without a propose_transaction call this turn."
    )
    # And it must warn against imitating an earlier ADD answer's wording.
    assert "do NOT imitate an" in SYSTEM_PROMPT
