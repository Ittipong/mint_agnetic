"""Unit tests for the durable user-preferences feature.

Covers the pure, DB-free surface:
  - `format_about_user_block` — the [about_user] prompt block renderer
    (consent gating, occupation/AI-style always-on, null skipping, footer).
  - `classify_and_coerce` — the write tool's validation/coercion.
  - `set_user_preference` error paths (consent gate, unknown field) that
    short-circuit before touching the DB.
  - `render_system_prompt` placeholder substitution + collapse.
  - loader shaping via an injected fake (no real Postgres).

The live DB upsert path is exercised by a manual smoke (scripts), not here —
unit tests stay connection-free per tests/test_tools_registry.py::UT-T00 intent.
"""

from __future__ import annotations

import asyncio
import json

from src.agent.prompts import render_system_prompt
from src.agent.tools.user_preferences_tool import (
    classify_and_coerce,
    set_user_preference,
)
from src.agent.user_preferences import (
    format_about_user_block,
    get_preferences_pool,
    load_user_preferences,
    set_preferences_loader,
    set_preferences_pool,
)


# ─────────────────────────────────────────────────────────────────────────────
# format_about_user_block
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_UP01_empty_prefs_render_to_empty_string():
    """No prefs (None or empty dict) → "" so the {about_user} placeholder
    collapses and the prompt is byte-identical to a no-preferences user."""
    assert format_about_user_block(None) == ""
    assert format_about_user_block({}) == ""


def test_UT_UP02_occupation_and_ai_style_render_without_consent():
    """Occupation (identity) and AI style (non-sensitive) ALWAYS surface —
    they are not gated behind memory_consent. Financial context must NOT."""
    prefs = {
        "occupation": "วิศวกร",
        "memory_consent": False,
        "financial_literacy_level": "beginner",   # financial → must be hidden
        "ai_preferences": {"ai_tone": "friendly", "ai_language": "th"},
    }
    block = format_about_user_block(prefs)
    assert "วิศวกร" in block
    assert "ai_tone=friendly" in block
    # Consent off → no financial line leaks.
    assert "beginner" not in block
    assert "Financial literacy" not in block


def test_UT_UP03_financial_context_requires_consent():
    """With consent, financial fields render; null fields are skipped."""
    prefs = {
        "memory_consent": True,
        "financial_literacy_level": "beginner",
        "housing_status": "mortgage",
        "dependents_count": 2,
        "risk_tolerance": None,            # null → skipped
        "declared_monthly_income": 45000,
        "field_sources": {},
    }
    block = format_about_user_block(prefs)
    assert "Financial literacy: beginner" in block
    assert "Housing: mortgage" in block
    assert "Dependents: 2" in block
    assert "Risk tolerance" not in block          # null skipped, no empty slot
    assert "45,000" in block                      # numeric formatted w/ commas


def test_UT_UP04_ai_inferred_footer_only_when_inferred_present():
    """The 'verify before relying' footer appears only when at least one
    field is AI-inferred (so the LLM trusts user-stated facts more)."""
    base = {
        "memory_consent": True,
        "financial_literacy_level": "advanced",
    }
    stated = dict(base, field_sources={"financial_literacy_level": "user_stated"})
    inferred = dict(base, field_sources={"financial_literacy_level": "ai_inferred"})
    # Assert on the footer-unique phrase — the header always mentions
    # "AI-inferred" (the trust rule), so match "not yet confirmed" instead.
    assert "not yet confirmed" not in format_about_user_block(stated)
    assert "not yet confirmed" in format_about_user_block(inferred)


# ─────────────────────────────────────────────────────────────────────────────
# classify_and_coerce
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_UP05_enum_validation():
    kind, val, _ = classify_and_coerce("risk_tolerance", "moderate")
    assert (kind, val) == ("column", "moderate")
    kind, _, msg = classify_and_coerce("risk_tolerance", "yolo")
    assert kind == "invalid_value" and "conservative" in msg


def test_UT_UP06_int_range_and_numeric_coercion():
    assert classify_and_coerce("salary_day", "25")[:2] == ("column", 25)
    assert classify_and_coerce("salary_day", 40)[0] == "invalid_value"   # >31
    assert classify_and_coerce("declared_monthly_income", "30000")[:2] == ("column", 30000.0)
    assert classify_and_coerce("declared_monthly_income", -5)[0] == "invalid_value"


def test_UT_UP07_bool_meta_and_ai_style():
    assert classify_and_coerce("memory_consent", "true")[:2] == ("column", True)
    assert classify_and_coerce("ai_tone", "friendly")[:2] == ("ai_style", "friendly")
    assert classify_and_coerce("use_emoji", "yes")[:2] == ("ai_style", True)


def test_UT_UP08_derivable_or_unknown_fields_rejected():
    """Derivable numbers (actual income/debt/savings) and typos are rejected —
    the whitelist enforces the 'derive, don't store' rule."""
    for bad in ("monthly_income", "total_debt", "current_savings", "nonsense"):
        kind, _, msg = classify_and_coerce(bad, 1)
        assert kind == "invalid", f"{bad} should be rejected"
        assert "run_python" in msg                # points the agent at deriving


# ─────────────────────────────────────────────────────────────────────────────
# set_user_preference — DB-free error paths
# ─────────────────────────────────────────────────────────────────────────────


def _msg_payload(command) -> dict:
    return json.loads(command.update["messages"][0].content)


async def _invoke_set(*, field, value, state, source=None, tool_call_id="tc-x"):
    """Invoke set_user_preference via the ToolCall protocol (required because
    the tool has an InjectedToolCallId arg — state goes in `args`, id in `id`)."""
    args = {"field": field, "value": value, "state": state}
    if source is not None:
        args["source"] = source
    return await set_user_preference.ainvoke({
        "name": "set_user_preference",
        "args": args,
        "type": "tool_call",
        "id": tool_call_id,
    })


def test_UT_UP09_consent_gate_blocks_financial_write_without_consent():
    """A financial field with no prior consent → consent_required error,
    short-circuiting before any DB write."""
    async def run():
        cmd = await _invoke_set(
            field="financial_literacy_level", value="beginner",
            state={"user_id": "u-1", "user_preferences": {"memory_consent": False}},
        )
        payload = _msg_payload(cmd)
        assert payload["kind"] == "consent_required"
        # No state write-back (no preferences mutation on a blocked write).
        assert "user_preferences" not in cmd.update
    asyncio.run(run())


def test_UT_UP10_unknown_field_rejected_before_db():
    async def run():
        cmd = await _invoke_set(
            field="current_savings", value=99999, state={"user_id": "u-1"},
        )
        assert _msg_payload(cmd)["kind"] == "invalid_field"
    asyncio.run(run())


def test_UT_UP11_ai_style_not_consent_gated():
    """AI-style write is NOT consent-gated; with no pool wired it falls through
    to store_unavailable (NOT consent_required) — proving the gate is skipped.

    Force the pool to None for this test: a prior server-lifespan test may have
    installed the module-global backend pool, which would otherwise route this
    write to the DB instead of the store_unavailable branch."""
    async def run():
        saved = get_preferences_pool()
        set_preferences_pool(None)
        try:
            cmd = await _invoke_set(
                field="ai_tone", value="formal", state={"user_id": "u-1"},  # no consent
            )
            assert _msg_payload(cmd)["kind"] == "store_unavailable"
        finally:
            set_preferences_pool(saved)
    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# render_system_prompt + loader shaping
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_UP12_prompt_placeholder_substitutes_and_collapses():
    empty = render_system_prompt(today="2026-06-02", user_id="u-1")
    assert "{about_user}" not in empty            # placeholder always resolved
    assert "# ABOUT THIS USER" not in empty       # nothing injected when ""
    block = "\n# ABOUT THIS USER\n- Occupation: หมอ"
    withp = render_system_prompt(today="2026-06-02", user_id="u-1", about_user=block)
    assert "Occupation: หมอ" in withp


def test_UT_UP13_loader_uses_injected_fake():
    """`load_user_preferences` resolves through an injected test loader without
    any Postgres connection."""
    async def fake(user_id):
        return {"occupation": "นักบิน", "memory_consent": False, "ai_preferences": {}}

    async def run():
        set_preferences_loader(fake)
        try:
            out = await load_user_preferences("u-1")
            assert out["occupation"] == "นักบิน"
        finally:
            set_preferences_loader(None)
    asyncio.run(run())
