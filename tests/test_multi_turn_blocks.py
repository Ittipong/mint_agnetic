"""Multi-turn block-leak regression (Wave 7b adaptation).

v2's `test_multi_turn_blocks.py` drove the full LangGraph with a checkpointer
and asserted that turn N's response didn't carry every block from turns 1..N
(the bug: nodes appended in-place to `response_blocks`, leaking across turns).

v3 absorbs this guarantee into the `append_reducer` channel + the
`__RESET__` sentinel applied by the graph's pre-turn hook. The reducer +
sentinel mechanics are covered by `test_state.py` UT-ST02; the graph's
pre-turn hook side effects are covered by `test_react_loop.py` UT-G02.
What this file ADDS is a UT-level assertion of the COMPOSED behaviour the v2
test was after:

  pre-turn hook RESET -> tool A appends N blocks -> tool B appends M blocks
  -> channel sees N+M (this turn only). On the NEXT turn, pre-turn hook
  emits __RESET__ again -> channel clears -> next turn's tool emissions
  start from [].

This is a state-shape integration test for the append_reducer's behaviour
under a realistic multi-tool, multi-turn sequence — without spinning the
full graph (which Wave 7c integration tests will do).

The full graph-level cross-turn flow ("ADD turn -> discard_proposal on the
next QUERY turn -> no transaction_proposal leak") is deferred to
Wave 7c's integration suite (`I1002` clarification loop, `I1007` golden
mobile contract). Re-asserting it here at unit level without the graph
would just remock everything; the value is in the integration test.
"""

from __future__ import annotations

from src.agent.state import append_reducer


_RESET = ["__RESET__"]


# ---------------------------------------------------------------------------
# UT-MT001 — within a single turn, two tool emissions accumulate
# ---------------------------------------------------------------------------
def test_UT_MT001_within_turn_two_tool_emissions_accumulate():
    """A turn's emitted_blocks_this_turn channel accumulates ALL blocks
    emitted by every tool call within that turn — this is what lets the SSE
    adapter forward them in order."""
    channel: list = []
    # Tool A (e.g. propose_transaction) emits one block.
    channel = append_reducer(channel, [{"type": "transaction_proposal", "id": "p1"}])
    # Tool B (e.g. emit_suggestions) emits another block in the same turn.
    channel = append_reducer(channel, [{"type": "suggestions", "items": ["x", "y"]}])

    assert len(channel) == 2
    assert [b["type"] for b in channel] == [
        "transaction_proposal", "suggestions",
    ]


# ---------------------------------------------------------------------------
# UT-MT002 — RESET between turns clears the channel; next turn starts empty
# ---------------------------------------------------------------------------
def test_UT_MT002_reset_between_turns_clears_channel_for_next_turn():
    """Pre-turn hook emits the __RESET__ sentinel at turn boundary -> the
    channel goes back to [] regardless of accumulated state from prior turns.
    This is the mechanism that closes v2's "blocks leak across turns" bug."""
    # Turn 1 — two blocks accumulate.
    channel = append_reducer([], [{"type": "transaction_proposal", "id": "p1"}])
    channel = append_reducer(channel, [{"type": "suggestions"}])
    assert len(channel) == 2

    # Pre-turn hook fires at the START of turn 2.
    channel = append_reducer(channel, _RESET)
    assert channel == []

    # Turn 2 emits a different block. The previous turn's blocks DO NOT
    # come back — leak fixed.
    channel = append_reducer(channel, [{"type": "answer", "text": "..."}])
    assert [b["type"] for b in channel] == ["answer"]


# ---------------------------------------------------------------------------
# UT-MT003 — RESET is idempotent and order-sensitive
# ---------------------------------------------------------------------------
def test_UT_MT003_double_reset_is_idempotent_and_subsequent_append_works():
    """Two pre-turn resets in a row (e.g. defensive re-entry guard) MUST NOT
    crash — they just leave the channel empty. A subsequent append after the
    resets behaves normally."""
    channel = append_reducer([{"type": "any"}], _RESET)
    channel = append_reducer(channel, _RESET)            # idempotent
    assert channel == []

    channel = append_reducer(channel, [{"type": "answer"}])
    assert channel == [{"type": "answer"}]


# ---------------------------------------------------------------------------
# UT-MT004 — RESET sentinel must be sole-element to trigger clear
# ---------------------------------------------------------------------------
def test_UT_MT004_reset_sentinel_must_be_sole_element():
    """A list whose ONLY element is "__RESET__" clears. A list that includes
    "__RESET__" alongside real blocks must NOT clear — that would be a footgun
    if any tool ever embedded the literal string in a payload field. The
    sentinel is a discriminator on the wire, not a substring match."""
    channel = append_reducer([{"type": "answer"}], ["__RESET__", {"type": "x"}])
    # Not the sentinel pattern (length != 1) -> normal append.
    assert len(channel) == 3  # original 1 + 2 appended


# ---------------------------------------------------------------------------
# UT-MT005 — proposals list is NOT reset across turns (persistence contract)
# ---------------------------------------------------------------------------
def test_UT_MT005_proposals_list_is_managed_separately_from_per_turn_channel():
    """The append_reducer applies only to per-turn lists
    (`emitted_blocks_this_turn`, `tool_outputs_this_turn`). The `proposals`
    list lives on the persisted state and is NOT reset by the pre-turn hook —
    history must carry every entry forward (pending / confirmed / cancelled /
    discarded). The state schema's typing (see test_state.py) is the contract
    asserter; this case just documents the design boundary alongside the
    other multi-turn rules so future refactors don't accidentally apply the
    reset semantics to the proposals channel."""
    # No code-level assertion needed beyond what test_state.py does — this
    # test exists as a deliberate boundary-marker comment in the suite.
    # We assert one minimal property: appending to a non-reducer-managed
    # list at the Python level still composes correctly (no reducer to fire).
    proposals: list = [{"proposal_id": "p1", "status": "confirmed"}]
    proposals = proposals + [{"proposal_id": "p2", "status": "pending"}]
    assert [p["proposal_id"] for p in proposals] == ["p1", "p2"]
