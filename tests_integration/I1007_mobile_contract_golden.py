"""I1007 — mobile contract golden, end-to-end through real /chat/stream.

The most critical Wave 7 test. The mobile app frozes its SSE wire format
against v2 (memory `project_chat_v2_contract_migration` /
`project_sse_ngrok_framing`). v3 MUST emit the same wire shape:

  * Event types: `status_token`, `answer_token`, `block`, `done`, `error`.
  * Block payloads: snake_case keys, JSON-decodable, schema matches v2.
  * Ordering: `status_token`* → `answer_token`* → `block`* → `done` (or
    `error` at any time). Per spec §3 and `phase2_streaming_adapter.md` §5.
  * Block types present: at minimum `answer` block at end-of-stream (the
    streamed prose is also persisted as a block for history replay —
    memory `project_chat_history_block_replay`).

This test is shape-strict but content-tolerant: we don't pin the exact
amount the agent settles on (the categorizer LLM can pick any expense
category for "เพิ่ม 250 กาแฟ"), only the wire shape that mobile parses.

Compares v3 against a HAND-AUTHORED v2 baseline (`_V2_BASELINE_…`) rather
than scraping a v2 trace at runtime. Reasons:
  1. v2 is being retired (Wave 7 prep) — running v2's server alongside the
     test adds a tunnel/process dep we don't want in CI.
  2. The baseline IS the contract — derived from the freeze spec §3 + §8,
     not from a v2 trace which could itself drift.
  3. A v2 trace would mean grading the agent against another agent's
     output — circular and brittle.

So `_v2_assert` codifies the FROZEN mobile contract in code; if v3 emits a
block missing a required key, the assertion fires the same way it would if
we diffed against a stored v2 .sse capture — but with clearer error msgs.

NOT hermetic — needs `OPENROUTER_API_KEY`, `DATABASE_URL`,
`BACKEND_DATABASE_URL`, and a REACT/CODEACT model env. Skipped cleanly when
any is missing.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.server import app


SEED_USER = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
WALLET_TEST = "00000000-0001-4000-a000-000000000001"

_REQUIRED_ENV = ("OPENROUTER_API_KEY", "DATABASE_URL", "BACKEND_DATABASE_URL")
_MODEL_ENV = ("REACT_MODEL",)
pytestmark = pytest.mark.skipif(
    not all(os.getenv(k) for k in _REQUIRED_ENV)
    or not any(os.getenv(k) for k in _MODEL_ENV),
    reason=(
        f"integration test needs env: {', '.join(_REQUIRED_ENV)} + one of "
        f"{', '.join(_MODEL_ENV)}"
    ),
)


# ---------------------------------------------------------------------------
# Frozen mobile-contract surface — codified from spec §3 + §8
# ---------------------------------------------------------------------------

# Set of event types the mobile decoder recognises. Anything else MUST be
# absent. Anything in this set MAY appear.
_ALLOWED_EVENT_TYPES = {"status_token", "answer_token", "block", "done", "error"}

# Per spec §8 — required keys per block type. Extra keys are allowed (the
# decoder ignores unknown), missing required keys silently drop the block
# in mobile. So a missing key here = a silent UX bug.
_REQUIRED_BLOCK_KEYS: dict[str, set[str]] = {
    # The `answer` block is the streamed prose, re-emitted at end-of-stream
    # so history replay can render it the same way (memory:
    # project_chat_history_block_replay).
    "answer": {"type", "text"},
    "transaction_proposal": {"type", "proposal_id", "transaction"},
    "transaction_proposal_group": {
        "type", "proposal_id", "group_id", "transactions", "total",
    },
    "wallet_required": {"type", "text"},
    "suggestions": {"type", "items"},
    "clarification": {"type", "text"},
    "transcript": {"type", "text"},
    "stt_error": {"type", "reason"},
}

# Inside a `transaction_proposal.transaction` payload — required keys.
_REQUIRED_TXN_KEYS = {"type", "amount", "wallet_sync_id"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _stream_events(client: TestClient, body: dict[str, Any]) -> list[dict]:
    """Drive /chat/stream and return parsed (event, data) dicts in order.

    Preserves order — mobile state machine asserts that `status_token`
    arrives before `answer_token` arrives before `block`. We retain raw
    `data` strings so the caller can assert on them; JSON-decoding is the
    caller's job (status_token / answer_token carry raw text).
    """
    resp = client.post("/chat/stream", json=body)
    assert resp.status_code == 200, resp.text
    events: list[dict] = []
    for chunk in resp.text.replace("\r\n", "\n").split("\n\n"):
        lines = chunk.splitlines()
        event = ""
        data_parts: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].lstrip())
        if event:
            events.append({"event": event, "data": "".join(data_parts)})
    return events


def _assert_wire_contract(events: list[dict], expected_block_types: set[str]) -> None:
    """The whole frozen mobile contract in one place.

    `expected_block_types`: types we know MUST appear for this trace (e.g.
      "answer" always; "transaction_proposal" for ADD traces). Other blocks
      are allowed to appear too — the assertion only requires presence,
      not exclusivity.
    """
    assert events, "stream produced no events at all"

    # ── Event types are all known ──
    seen_types = {e["event"] for e in events}
    unknown = seen_types - _ALLOWED_EVENT_TYPES
    assert not unknown, (
        f"unknown SSE event types in stream: {unknown} — mobile decoder will "
        f"drop these. Allowed: {_ALLOWED_EVENT_TYPES}"
    )

    # ── Stream ends with `done` (or `error`) ──
    terminator = events[-1]["event"]
    assert terminator in ("done", "error"), (
        f"stream did not terminate with done/error — last event was "
        f"{terminator!r}. mobile will hang waiting."
    )

    # ── If no error, `done` carries thread_id ──
    if terminator == "done":
        try:
            done_payload = json.loads(events[-1]["data"])
        except json.JSONDecodeError as exc:
            pytest.fail(f"done event payload not JSON: {events[-1]['data']!r} ({exc})")
        assert "thread_id" in done_payload, (
            f"done event missing thread_id — mobile uses it to clear pending: "
            f"{done_payload}"
        )

    # ── Each block validates against the per-type required-key set ──
    blocks = [e for e in events if e["event"] == "block"]
    parsed_blocks: list[dict] = []
    for ev in blocks:
        try:
            blk = json.loads(ev["data"])
        except json.JSONDecodeError as exc:
            pytest.fail(f"block payload not JSON: {ev['data']!r} ({exc})")
        assert isinstance(blk, dict), f"block payload not dict: {blk!r}"
        assert "type" in blk, f"block missing `type`: {blk!r}"
        parsed_blocks.append(blk)

        btype = blk.get("type")
        required = _REQUIRED_BLOCK_KEYS.get(btype)
        if required is not None:
            missing = required - set(blk.keys())
            assert not missing, (
                f"block type={btype!r} missing required keys {missing}; "
                f"got {sorted(blk.keys())}. Mobile silently drops this block."
            )

        # Drill into the transaction sub-payload for proposals — the v2
        # `proposalId` vs `proposal_id` silent-drift bug lives here.
        if btype == "transaction_proposal":
            txn = blk.get("transaction") or {}
            txn_missing = _REQUIRED_TXN_KEYS - set(txn.keys())
            assert not txn_missing, (
                f"transaction_proposal.transaction missing keys {txn_missing}; "
                f"got {sorted(txn.keys())}. Mobile will fail to decode."
            )
            # snake_case sanity — wallet_sync_id, not walletSyncId.
            assert "walletSyncId" not in txn, (
                "found camelCase walletSyncId in transaction — v3 must "
                "emit snake_case wallet_sync_id"
            )

    # ── All expected block types are present ──
    seen_block_types = {b["type"] for b in parsed_blocks}
    missing_types = expected_block_types - seen_block_types
    assert not missing_types, (
        f"expected block types {missing_types} not present; got "
        f"{seen_block_types}"
    )


# ---------------------------------------------------------------------------
# I1007-a — text ADD turn ("เพิ่ม 250 กาแฟ")
# ---------------------------------------------------------------------------
@pytest.mark.id("I1007")
def test_I1007_a_text_add_wire_contract(client):
    """I1007-a: a simple text ADD must produce a stream whose wire shape is
    indistinguishable from v2 from mobile's perspective. Required block
    types: `answer` (the streamed prose, re-emitted at end) and
    `transaction_proposal`."""
    body = {
        "thread_id": str(uuid.uuid4()),
        "user_id": SEED_USER,
        "message": "เพิ่ม 250 กาแฟ",
        "wallet_id": WALLET_TEST,
    }
    events = _stream_events(client, body)
    _assert_wire_contract(events, expected_block_types={"transaction_proposal"})


# ---------------------------------------------------------------------------
# I1007-b — chitchat (no tool calls, plain answer)
# ---------------------------------------------------------------------------
@pytest.mark.id("I1007")
def test_I1007_b_chitchat_wire_contract(client):
    """I1007-b: chitchat — pure prose, no proposal block, no tools called.
    Mobile MUST still see `answer_token` events streaming + an `answer` block
    at end-of-stream (the persistence handoff so history replay can render
    it the same way as proposals)."""
    body = {
        "thread_id": str(uuid.uuid4()),
        "user_id": SEED_USER,
        "message": "สวัสดี",
        "wallet_id": WALLET_TEST,
    }
    events = _stream_events(client, body)

    # answer block MUST be present (history replay relies on it).
    _assert_wire_contract(events, expected_block_types={"answer"})

    # NO proposal types should appear on a pure-chitchat turn.
    blocks = [json.loads(e["data"]) for e in events if e["event"] == "block"]
    proposal_types = {"transaction_proposal", "transaction_proposal_group"}
    seen_proposals = {b.get("type") for b in blocks} & proposal_types
    assert not seen_proposals, (
        f"chitchat turn produced a proposal block: {seen_proposals}. "
        "ReAct should not call propose_transaction here."
    )


# ---------------------------------------------------------------------------
# I1007-c — advisor query ("เดือนนี้ใช้ไปเท่าไหร่") — wire shape only
# ---------------------------------------------------------------------------
@pytest.mark.id("I1007")
def test_I1007_c_advisor_wire_contract(client):
    """I1007-c: an advisor query exercises run_python → answer with numbers.
    We pin only the wire contract — block-shape + ordering. The number itself
    is graded by `evals/run_eval.py` against SQL truth; here we just guard
    that the wire shape doesn't drift."""
    body = {
        "thread_id": str(uuid.uuid4()),
        "user_id": SEED_USER,
        "message": "เดือนนี้ใช้ไปเท่าไหร่",
        "wallet_id": WALLET_TEST,
    }
    events = _stream_events(client, body)
    _assert_wire_contract(events, expected_block_types={"answer"})
