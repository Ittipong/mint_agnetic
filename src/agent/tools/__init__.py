"""LangGraph tool registry — exactly 6 tools (Q6 enforcement).

NEW in Wave 3. Phase 2 spec: docs/v3/phase2_tools_design.md "Tool Registry".

The ReAct agent is created via `create_react_agent(..., tools=ALL_TOOLS)`.
Any tool not in this list is invisible to the LLM, so this file is the
single source of truth for the tool surface.

CRITICAL — Q6 enforcement (memory `project_wallet_picker_disabled`):
    DO NOT add `clarify_wallet` here. The wallet-picker tool was dropped
    per the Q6 amendment in docs/mint_agentic_v3_pure_react_spec.html
    section 17. Wallet ambiguity falls back to index-0 INSIDE
    propose_transaction (general → creditcard → goal ordering). If a
    future implementer adds clarify_wallet by reflex, UT-T00 fails.

CRITICAL — wallet_required_cta NOT in ALL_TOOLS:
    The onboarding CTA is now emitted in-place by `propose_transaction`
    (ADD path) and `run_python` (analyst path) via
    `build_wallet_required_block()`. The standalone `@tool` wrapper is
    retained for tests/direct invocation but the LLM cannot reach it.

CRITICAL — emit_suggestions NOT in ALL_TOOLS:
    Follow-up `suggestions` chips were demoted from a model tool to the
    deterministic outer-graph `suggest` node (src/agent/suggest_followups.py),
    so chips are GUARANTEED on non-ADD/non-crisis turns and a wildcard slot is
    locked in code — instead of relying on the LLM optionally calling a tool.
    The `@tool` wrapper is retained for tests/direct invocation only.

RETIRED — memory_recall / memory_write (2026-06-02):
    The free-text LangGraph-store memory tools were retired once
    `set_user_preference` + the always-on `[about_user]` block superseded
    them. They were optional (the LLM rarely called memory_recall), unverified
    (no consent/provenance), and overlapped with user_preferences'
    `financial_notes`. Their only consumer was the BaseStore, so the store
    (utils/store_factory.py) was removed with them. See
    docs/user_preferences.md.

The tool count (6) is enforced by `tests/test_tools_registry.py::UT_T00`.
"""

# DO NOT add clarify_wallet — dropped per Q6 amendment in
# docs/mint_agentic_v3_pure_react_spec.html §17. Wallet ambiguity falls
# back to index-0 inside propose_transaction (memory: project_wallet_picker_disabled).

from .advice_playbook import get_advice_playbook
from .app_capability import get_app_capability
from .codeact import run_python
from .emit_suggestions import emit_suggestions
from .get_user_context import get_user_context
from .propose_transaction import propose_transaction
from .user_preferences_tool import set_user_preference
from .wallet_required_cta import wallet_required_cta


ALL_TOOLS = [
    # Data tools (call when intent matches — get_user_context only on
    # wallet/category/tag queries; run_python loads its own catalog inline).
    get_user_context,
    run_python,
    # Block-emitting tool — wallet_required is emitted in-place by
    # propose_transaction / run_python, not via wallet_required_cta tool.
    propose_transaction,
    # NOTE: follow-up `suggestions` chips are NO LONGER a model tool. They are
    # emitted deterministically by the outer-graph `suggest` node
    # (src/agent/suggest_followups.py) AFTER the answer, so they are guaranteed
    # on non-ADD/non-crisis turns and a wildcard slot is locked in code.
    # Durable user preferences (financial context / AI style). Read path is the
    # [about_user] prompt block injected every turn (src/agent/user_preferences.py);
    # this tool is the WRITE side. PDPA consent-gated for personal fields.
    set_user_preference,
    # Advisor playbook (Wave 5) — static framework lookup; load JIT for
    # major-decision topics (home/refi/car/debt/tax/invest/discipline).
    get_advice_playbook,
    # App capability FAQ (Wave 5) — what chat can / can't do.
    get_app_capability,
]


__all__ = [
    "ALL_TOOLS",
    "get_user_context",
    "run_python",
    "propose_transaction",
    "set_user_preference",
    "wallet_required_cta",
    "emit_suggestions",
    "get_advice_playbook",
    "get_app_capability",
]
