"""Unit tests for src.agent.state.

Covers:
- UT-ST01: AgentState TypedDict roundtrips through a plain-dict copy
  (proxy for LangGraph MsgPack serde — full serde test lives in
  test_checkpoint_serde.py which is KEEP from v2 / runs in a later wave).
- UT-ST02: `append_reducer` semantics for per-turn lists.
"""

from __future__ import annotations

from src.agent.state import (
    AgentState,
    PendingClarification,
    Proposal,
    ToolOutput,
    UserContext,
    append_reducer,
)


# ─────────────────────────────────────────────────────────────────────────────
# UT-ST01 — TypedDict roundtrip
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentStateRoundtrip:
    def test_UT_ST01_a_empty_state_is_valid_dict(self):
        """UT-ST01a: an empty AgentState is just an empty dict — total=False
        means every field is optional, so the simplest construction works."""
        state: AgentState = {}
        assert state == {}

    def test_UT_ST01_b_full_state_round_trips_as_dict(self):
        """UT-ST01b: a maximally-populated AgentState marshals to a plain
        dict and back without losing fields — the serde shape LangGraph
        MsgPack will see is exactly this dict."""
        proposal: Proposal = {
            "proposal_id": "p1",
            "intent_type": "ADD_TRANSACTION",
            "type": "ADD_TRANSACTION",
            "payload": {"amount": 250, "category": "ค่าอาหาร"},
            "status": "pending",
            "created_at": "2026-05-28T10:00:00Z",
            "confirmed_at": None,
            "cancelled_at": None,
            "discarded_at": None,
        }
        clarification: PendingClarification = {
            "kind": "wallet",
            "intent": "ADD_TRANSACTION",
            "ask": "wallet",
            "question": "เลือก wallet:",
            "options": [{"sync_id": "w1", "name": "เงินสด"}],
            "known_slots": {"amount": 250},
        }
        user_context: UserContext = {
            "wallets": [{"sync_id": "w1", "name": "เงินสด"}],
            "categories": [{"sync_id": "c1", "name": "ค่าอาหาร"}],
            "budgets": [],
            "goals": [],
            "wallet_id": "w1",
            "default_currency_code": "THB",
            "fetched_at": "2026-05-28T10:00:00Z",
        }
        tool_out: ToolOutput = {
            "tool": "run_python",
            "result": "{'total': 12500}",
            "stdout": "",
        }

        state: AgentState = {
            "messages": [],
            "thread_id": "t-1",
            "user_id": "u-1",
            "proposals": [proposal],
            "pending_proposal": proposal,
            "pending_clarification": clarification,
            "user_context": user_context,
            "tool_outputs_this_turn": [tool_out],
            "emitted_blocks_this_turn": [{"type": "answer", "text": "ok"}],
            "trace_id": "trace-1",
            "session_log_path": "logs/session_t-1.log",
            "__validator_retries__": 0,
            "__validator_failed__": False,
            "__validator_failure_detail__": {},
            "last_txn": {"id": "p1", "amount": 250, "pending": True},
            "last_query": None,
            "onboarding_stage": "done",
        }

        # Round-trip through a plain dict copy — proxy for MsgPack serde
        # (TypedDict is dict at runtime; LangGraph's serializer treats it
        # as such).
        as_dict = dict(state)
        assert as_dict["proposals"][0]["proposal_id"] == "p1"
        assert as_dict["pending_clarification"]["known_slots"] == {"amount": 250}
        assert as_dict["user_context"]["wallet_id"] == "w1"
        assert as_dict["tool_outputs_this_turn"][0]["tool"] == "run_python"
        assert as_dict["__validator_retries__"] == 0

    def test_UT_ST01_c_typeddict_total_false_allows_partial(self):
        """UT-ST01c: total=False means a TypedDict instance only needs the
        fields present in actual use. Critical for pre_turn_hook returning
        a partial-update dict rather than the full schema."""
        partial: AgentState = {"thread_id": "t-1"}
        assert "thread_id" in partial
        assert "messages" not in partial


# ─────────────────────────────────────────────────────────────────────────────
# UT-ST02 — append_reducer behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestAppendReducer:
    def test_UT_ST02_a_appends_right_to_left(self):
        """UT-ST02a: classic append — left + right concatenates in order."""
        out = append_reducer([{"tool": "a"}], [{"tool": "b"}])
        assert out == [{"tool": "a"}, {"tool": "b"}]

    def test_UT_ST02_b_left_none_treated_as_empty(self):
        """UT-ST02b: None on left -> treat as []. Important because LangGraph
        invokes reducers with `None` as the initial-state placeholder."""
        out = append_reducer(None, [{"tool": "a"}])
        assert out == [{"tool": "a"}]

    def test_UT_ST02_c_right_none_returns_left_unchanged(self):
        """UT-ST02c: None on right -> no-op. A node that doesn't touch the
        list returns None for that delta; reducer must not append None."""
        left = [{"tool": "a"}]
        out = append_reducer(left, None)
        assert out == [{"tool": "a"}]

    def test_UT_ST02_d_both_none_returns_empty(self):
        """UT-ST02d: both None -> []. First reducer call when neither
        existing state nor delta carries the key."""
        out = append_reducer(None, None)
        assert out == []

    def test_UT_ST02_e_empty_lists_concat_to_empty(self):
        """UT-ST02e: both empty lists -> []. The pre-turn hook resets these
        to []; a node that doesn't append should leave state as []."""
        out = append_reducer([], [])
        assert out == []

    def test_UT_ST02_f_does_not_mutate_inputs(self):
        """UT-ST02f: reducer returns a NEW list — must not mutate either
        operand (would corrupt LangGraph's checkpoint diffing)."""
        left = [{"tool": "a"}]
        right = [{"tool": "b"}]
        out = append_reducer(left, right)
        out.append({"tool": "leaked"})
        assert left == [{"tool": "a"}]
        assert right == [{"tool": "b"}]

    def test_UT_ST02_g_idempotent_on_empty_appends(self):
        """UT-ST02g: appending [] over and over keeps the list intact."""
        left = [{"tool": "a"}, {"tool": "b"}]
        out = left
        for _ in range(5):
            out = append_reducer(out, [])
        assert out == [{"tool": "a"}, {"tool": "b"}]

    def test_UT_ST02_h_reset_sentinel_clears_channel(self):
        """UT-ST02h: a write of `["__RESET__"]` clears the channel — used by
        the pre-turn hook to reset per-turn lists at turn boundaries.
        Without this sentinel, the append_reducer would keep accumulating
        across turns and per-turn views would leak."""
        left = [{"tool": "a"}, {"tool": "b"}]
        out = append_reducer(left, ["__RESET__"])
        assert out == []

    def test_UT_ST02_i_reset_sentinel_only_matches_exact_single_item(self):
        """UT-ST02i: the sentinel match must be STRICT — a real per-turn
        write that happens to include the string `"__RESET__"` as part of
        a multi-item list MUST be treated as an append, not a reset."""
        left = [{"tool": "a"}]
        # A two-item right containing the sentinel + another value is a
        # legitimate append, not a reset.
        out = append_reducer(left, ["__RESET__", {"tool": "b"}])
        assert out == [{"tool": "a"}, "__RESET__", {"tool": "b"}]
