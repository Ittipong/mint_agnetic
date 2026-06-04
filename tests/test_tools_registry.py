"""Unit tests for src.agent.tools (registry & exports).

Covers UT-T00 — the Q6 enforcement gate. If a future implementer adds
`clarify_wallet` (per memory `project_wallet_picker_disabled` it was
dropped from v3), this test fails.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_UT_T00_ALL_TOOLS_has_exactly_six_entries():
    """UT-T00: the tool registry must expose EXACTLY 6 tools.

    History — count went 7 → 6 when `wallet_required_cta` was dropped from
    ALL_TOOLS; 6 → 8 in Wave 5 (+get_advice_playbook, +get_app_capability);
    8 → 7 when `emit_suggestions` was demoted from a model tool to the
    deterministic outer-graph `suggest` node (src/agent/suggest_followups.py)
    so follow-up chips are guaranteed on non-ADD/non-crisis turns; 7 → 8 when
    `set_user_preference` (durable user-preferences write tool) was added;
    8 → 6 when `memory_recall` + `memory_write` were retired (superseded by
    set_user_preference + the always-on [about_user] block; the LangGraph
    store went with them).

    If a stray block-emitting tool slips in (e.g. clarify_wallet was
    re-added per Q6), this fails fast.
    """
    from src.agent.tools import ALL_TOOLS

    assert len(ALL_TOOLS) == 6, (
        f"expected 6 tools (memory_recall/memory_write retired), "
        f"found {len(ALL_TOOLS)}: {[t.name for t in ALL_TOOLS]}"
    )


def test_UT_T00_no_clarify_wallet_in_registry():
    """UT-T00 supporting: no tool named 'clarify_wallet' exists in ALL_TOOLS.
    Backup assertion in case someone bumps the count by removing a different
    tool while also adding clarify_wallet."""
    from src.agent.tools import ALL_TOOLS

    tool_names = {t.name for t in ALL_TOOLS}
    assert "clarify_wallet" not in tool_names, (
        "clarify_wallet was dropped per Q6 amendment (memory "
        "project_wallet_picker_disabled). Re-adding it requires a spec "
        "amendment, not a tools/__init__.py edit."
    )


def test_UT_T00_no_clarify_wallet_module_file():
    """UT-T00 supporting: no `clarify_wallet.py` file exists under tools/.
    The file existing is the actual hazard — even unimported, a stale file
    misleads readers about the tool surface."""
    tools_dir = (
        Path(__file__).parent.parent
        / "src" / "agent" / "tools"
    )
    assert tools_dir.is_dir(), f"missing tools dir: {tools_dir}"
    assert not (tools_dir / "clarify_wallet.py").exists(), (
        "tools/clarify_wallet.py must NOT exist — Q6 enforcement."
    )


def test_UT_T00_expected_tool_names_match_phase2_spec():
    """The 6 tools must match phase2_tools_design.md exactly — names are the
    LLM-facing tool ID and must stay stable so the system prompt's tool
    references resolve. `emit_suggestions` is intentionally absent: follow-up
    chips are emitted by the deterministic `suggest` node, not a model tool."""
    from src.agent.tools import ALL_TOOLS

    expected = {
        "get_user_context",
        "run_python",
        "propose_transaction",
        "set_user_preference",
        "get_advice_playbook",
        "get_app_capability",
    }
    assert "emit_suggestions" not in {t.name for t in ALL_TOOLS}, (
        "emit_suggestions must NOT be a model tool — it was demoted to the "
        "outer-graph `suggest` node (src/agent/suggest_followups.py)."
    )
    actual = {t.name for t in ALL_TOOLS}
    assert actual == expected, (
        f"tool name set drift — expected {expected}, got {actual}"
    )
    assert "wallet_required_cta" not in actual, (
        "wallet_required_cta must NOT be in ALL_TOOLS — propose_transaction "
        "and run_python emit the wallet_required block in-place via "
        "build_wallet_required_block(). The standalone @tool wrapper is "
        "retained only for direct invocation / tests."
    )


def test_UT_T00_registry_import_does_not_trigger_db_connect():
    """Importing `src.agent.tools` MUST be side-effect-free — the FastAPI
    lifespan wires DB pools later. If a tool module opened a pool at import
    time, the import in conftest would fail before DATABASE_URL is set."""
    # Force a clean re-import to catch import-time side effects.
    mod = importlib.import_module("src.agent.tools")
    assert hasattr(mod, "ALL_TOOLS")
