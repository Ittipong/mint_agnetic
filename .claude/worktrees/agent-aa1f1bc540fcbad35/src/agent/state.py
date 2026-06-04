"""AgentState — the v3 ReAct loop's typed state container.

NEW in Wave 1. Sourced from `docs/v3/phase2_state_and_graph.md` §1 + §2.

Key design points (vs v2):
- No `intents`/`intent_queue`/`confidence`/`is_multi_intent` — ReAct decides
  intent implicitly via tool calls.
- `response_blocks` split into per-turn `emitted_blocks_this_turn` (cleared
  every turn) + thread history (persisted via `messages_repo`).
- `entity_catalog` replaced by typed `user_context` TypedDict so the value
  survives the LangGraph checkpoint serializer (per memory
  `project_codeact_dual_impl` + KEEP `entity_catalog.py` `from_dict`).
- Append-only reducer on per-turn lists: pre-turn hook resets to [], each
  tool node appends, validator + SSE adapter read the union.

Persisted vs transient — see §2 of phase2_state_and_graph.md:
- Persisted across turns: messages, proposals, pending_*, last_*, onboarding_stage
- Transient (reset by `_pre_turn_hook`): user_context, tool_outputs_this_turn,
  emitted_blocks_this_turn, __validator_*__
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps


# ─────────────────────────────────────────────────────────────────────────────
# Sub-schemas (TypedDicts so they survive the LangGraph checkpoint serializer)
# ─────────────────────────────────────────────────────────────────────────────


class Proposal(TypedDict, total=False):
    """One ADD-transaction proposal entry (matches v2 ProposalEntry).

    Lifecycle: pending -> confirmed (via /transactions/confirm) OR cancelled
    (via /transactions/cancel) OR discarded (by propose_transaction atomic
    guard, per memory `project_proposal_discard_state_machine`).
    """
    proposal_id: str
    intent_type: str                       # "ADD_TRANSACTION"
    type: str                              # legacy field, == intent_type
    payload: dict[str, Any]                # mobile write contract
    status: Literal["pending", "confirmed", "cancelled", "discarded"]
    created_at: str                        # ISO 8601
    confirmed_at: Optional[str]
    cancelled_at: Optional[str]
    discarded_at: Optional[str]


class PendingClarification(TypedDict, total=False):
    """Parked clarification awaiting user answer.

    Per memory `project_chat_clarification_loop`: known_slots is SEPARATE
    from any proposal — bare answers (e.g. "99", "ค่าน้ำ") merge known_slots
    back into a new ADD without re-asking.
    """
    kind: Literal["wallet", "slot"]        # only "wallet" used currently
    intent: str                            # "ADD_TRANSACTION" (resume target)
    ask: str                               # which slot is missing ("category", "amount")
    question: str                          # Thai prompt shown to user
    options: list[dict]                    # [{"sync_id", "name", "icon"}] or [{"label", "send"}]
    known_slots: dict[str, Any]            # already-collected slots from prior turns


class UserContext(TypedDict, total=False):
    """Per-turn cache of the user's entity catalog + defaults.

    Populated by whichever data tool runs first this turn:
      - `get_user_context` — only when the user asks about the catalog
        directly (wallet/category/tag list).
      - `propose_transaction` / `run_python` — both load the catalog inline
        and write back to this slot, so subsequent tools in the same turn
        skip the DB round-trip.
    """
    wallets: list[dict]                    # ordered general -> creditcard -> goal
    categories: list[dict]
    budgets: list[dict]
    goals: list[dict]
    wallet_id: Optional[str]               # catalog default wallet (computed fallback)
    default_currency_code: str
    fetched_at: str                        # ISO 8601


class ToolOutput(TypedDict, total=False):
    """One record of a tool's output for the numerical validator.

    The numerical validator (`validators/numerical.py`) reads every entry's
    `result` + `stdout` to extract money numbers and cross-check against
    the final answer text.
    """
    tool: str                              # "run_python", "propose_transaction", ...
    result: str                            # repr of return value
    stdout: str                            # captured print() output (run_python only)


# ─────────────────────────────────────────────────────────────────────────────
# Reducers (append-only for per-turn lists)
# ─────────────────────────────────────────────────────────────────────────────


_RESET_SENTINEL = "__RESET__"


def append_reducer(left: list, right: list) -> list:
    """Append-with-explicit-reset reducer for per-turn lists.

    The pre-turn hook needs to wipe these lists at the START of every turn
    while every tool's Command(update=[...]) within the turn must
    APPEND (the inner ReAct subgraph doesn't propagate in-place list
    mutations — only channel writes show up across the boundary). Naive
    `list + list` concat would never let us reset.

    Convention: a `right` whose ONLY element is the sentinel string
    `"__RESET__"` clears the channel (pre-turn hook uses this). Any other
    `right` (including empty `[]`) appends to `left`.

    Semantics:
      - left=None              -> treat as []
      - right is None          -> no-op, return left
      - right == ["__RESET__"] -> CLEAR (returns [])
      - both real lists        -> concat (new list, never mutate inputs)
    """
    if left is None:
        left = []
    if right is None:
        return list(left)
    if right == [_RESET_SENTINEL]:
        return []
    return list(left) + list(right)


# ─────────────────────────────────────────────────────────────────────────────
# Main AgentState
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    # — conversation —
    messages: Annotated[list, add_messages]    # ReAct's working memory; persisted
    # `remaining_steps` is required by langgraph.prebuilt.create_react_agent —
    # it counts down through the ReAct loop and short-circuits when it hits 0.
    # The graph's `recursion_limit` config provides the hard outer bound; this
    # field is the soft per-loop counter the prebuilt agent maintains.
    remaining_steps: RemainingSteps
    thread_id: str
    user_id: str

    # — client-chosen wallet for THIS turn (mobile chat dropdown) —
    # Seeded by the server / fast-path / voice entrypoints from the request's
    # `wallet_id`. MUST be declared here or LangGraph drops it at the graph
    # boundary and propose_transaction never sees the user's pick. The wallet
    # cascade reads this first, before the catalog default in user_context.
    wallet_id: Optional[str]

    # — proposal lifecycle (decision X / memory project_chat_edit_repropose) —
    proposals: list[Proposal]                  # all entries — pending/confirmed/cancelled/discarded
    pending_proposal: Optional[Proposal]       # convenience pointer — the one currently pending
    pending_clarification: Optional[PendingClarification]

    # — per-turn cache (cleared at turn start by pre-turn hook) —
    user_context: Optional[UserContext]

    # — per-turn outputs (validator reads, SSE adapter forwards, append-only) —
    tool_outputs_this_turn: Annotated[list[ToolOutput], append_reducer]
    emitted_blocks_this_turn: Annotated[list[dict], append_reducer]

    # — follow-up suggestion chips —
    # Written by the `gen_suggestions` node (runs last, after the answer is in
    # `messages`): a ready-to-emit `{"type":"suggestions", "items":[…]}` block,
    # or None to emit nothing (ADD / crisis / no-answer / LLM-skip). Scalar
    # replace channel so it merges to the outer graph (unlike the append
    # `emitted_blocks_this_turn`, which the subgraph boundary doesn't merge for
    # a downstream reader). The SSE adapter reads it off the `updates` stream
    # and emits it AFTER the answer block — keeping generation in the graph and
    # the adapter a pure ordering/forwarding layer. Reset each turn by pre_turn.
    suggestions_block: Optional[dict]

    # — observability —
    trace_id: str                              # LangSmith run_id
    session_log_path: str                      # mint_agentic_v3/logs/session_<thread>.log

    # — validator scratch (internal — not part of mobile contract) —
    __validator_retries__: int                 # 0 or 1; reset per turn
    __validator_failed__: bool                 # set True on second failure
    __validator_failure_detail__: dict[str, Any]
    __tool_error_retries__: int                # tool-error guard; reset per turn

    # — short-term memory (carried across turns within thread) —
    last_txn: Optional[dict[str, Any]]         # last confirmed/proposed txn
    last_query: Optional[dict[str, Any]]       # last analyst query (for follow-ups)

    # — onboarding state —
    onboarding_stage: str                      # "no_wallet" | "no_transaction" | "done"

    # — classify-router shortcut scratch (Wave 8; reset per turn) —
    # `classify_intent` writes the route decision + extracted ADD slots so the
    # conditional edge + `direct_propose` node can read them. The classifier is
    # stateful — it sees windowed conversation history (so a bare amount that
    # answers a prior advisor question stays on the advisor flow, not an ADD).
    # `__classify_route__` steers the edge ("direct_propose" | "react").
    # `__classified_add__` carries {amount, type, category_label, wallet_label,
    # note, date_iso} for direct_propose when route=="direct_propose".
    __classify_route__: str
    __classified_add__: dict[str, Any]
