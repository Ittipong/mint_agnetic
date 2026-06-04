"""I1002 — multi-turn clarification slot accumulation (v3 integration, real LLM + DB).

In v2 the clarification loop was a dedicated `pending_clarification` slot in
state, populated/cleared by `_handle_add`. v3 deletes that explicit state —
pure ReAct + the conversation history (HumanMessage / AIMessage) carries the
context. The clarification path is now:

  Turn 1: ReAct decides the input is incomplete (amount only, or category only)
          → it emits a plain answer asking for the missing field, NO
          `transaction_proposal` block, and no `propose_transaction` tool call.
  Turn 2: same thread_id → the checkpointer replays the prior turns into
          ReAct's prompt → ReAct sees the prior question + the bare answer
          and emits `propose_transaction`.

So this test asserts the USER-VISIBLE OUTCOME (same as v2 — invariants 1-3 on
the proposal), not which mechanism carried the slot. If the v3 ReAct loop
loses turn-1 context, turn 2 will either re-ask the same question (loop bug)
or fabricate the missing slot from nothing (silent-loss bug).

NOT hermetic — needs `OPENROUTER_API_KEY`, `DATABASE_URL`, `BACKEND_DATABASE_URL`,
and a REACT/CODEACT model env. Skipped cleanly when any is missing so the
offline unit run stays green.

Async convention (memory `project_e2e_async_pattern`): sync tests, asyncio.run
for async helpers. Multi-turn flows REUSE one thread_id so the checkpointer
replays prior messages back into ReAct's prompt.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.llm_judge import judge
from src.agent.server import app


SEED_USER = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
WALLET_TEST = "90cb44d7-a0de-4598-a206-9d8de6b28289"

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


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _parse_sse(raw: str) -> tuple[list[dict], str]:
    """Return (blocks, answer_text) from an SSE response body.

    `answer_text` is the concatenation of every `answer_token` data field —
    that is the user-visible reply prose. v3 streams the answer as raw text
    over `answer_token` events (memory `project_chat_answer_streaming_writer`).
    """
    blocks: list[dict] = []
    answer_parts: list[str] = []
    for chunk in raw.replace("\r\n", "\n").split("\n\n"):
        lines = chunk.splitlines()
        event = ""
        data_parts: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].lstrip())
        data = "".join(data_parts)
        if event == "block" and data:
            try:
                blocks.append(json.loads(data))
            except json.JSONDecodeError:
                continue
        elif event == "answer_token" and data:
            answer_parts.append(data)
    return blocks, "".join(answer_parts)


def _post(
    client: TestClient,
    thread_id: str,
    message: str,
    wallet_id: str | None = None,
) -> tuple[list[dict], str]:
    """POST /chat/stream on a FIXED thread_id; return (blocks, answer_text).

    Reusing thread_id is the whole point of the multi-turn test: the
    checkpointer keys state by `configurable.thread_id`, so prior messages
    are replayed into ReAct's prompt on the next call.
    """
    body: dict[str, Any] = {"thread_id": thread_id, "user_id": SEED_USER, "message": message}
    if wallet_id:
        body["wallet_id"] = wallet_id
    resp = client.post("/chat/stream", json=body)
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text)


def _types(blocks: list[dict]) -> list[str]:
    return [b.get("type") for b in blocks]


def _any_proposal(blocks: list[dict]) -> dict | None:
    """Match both the `transaction_proposal` (high-conf) and any future
    low-conf variant — both carry a `transaction` payload."""
    return next((b for b in blocks if "proposal" in (b.get("type") or "")), None)


def _amount_eq(payload_amount, expected: str) -> bool:
    """Exact money compare via Decimal-on-str — amount column is float8
    (memory `project_chat_money_column_float8`). Decimal(str(x)) keeps
    integer-valued comparisons exact."""
    return Decimal(str(payload_amount)) == Decimal(expected)


# ---------------------------------------------------------------------------
# I1002-a — two-turn slot accumulation (the headline regression guard)
# ---------------------------------------------------------------------------
@pytest.mark.id("I1002")
def test_I1002_a_two_turn_amount_then_label(client):
    """I1002-a: turn 1 carries the AMOUNT only ("99 บาท"). The ReAct loop must
    ask what the 99 was for (no proposal yet). Turn 2 names it ("ซื้อสบู่").
    Turn 2 must produce a proposal whose amount is exactly 99 — proving the
    turn-1 amount survived in the conversation context."""
    thread_id = str(uuid.uuid4())

    # Turn 1: amount only → ask category, no proposal.
    blocks1, _ = _post(client, thread_id, "99 บาท", wallet_id=WALLET_TEST)
    assert _any_proposal(blocks1) is None, (
        f"turn 1 (bare amount) must NOT propose; got {_types(blocks1)}"
    )

    # Turn 2: name it → propose, carrying the turn-1 amount.
    blocks2, _ = _post(client, thread_id, "ซื้อสบู่", wallet_id=WALLET_TEST)
    prop = _any_proposal(blocks2)
    assert prop is not None, (
        f"turn 2 (the label) must produce a proposal; got {_types(blocks2)}"
    )
    txn = prop["transaction"]
    assert _amount_eq(txn["amount"], "99"), (
        f"amount=99 (from turn 1) must survive to turn 2, got {txn['amount']!r}"
    )
    assert txn["type"] == "expense", f"bare amount must default to expense, got {txn['type']!r}"
    label_blob = " ".join(
        str(txn.get(k) or "") for k in ("note", "description", "category", "category_name")
    ).lower()
    assert "สบู่" in label_blob, (
        "turn-2 label (สบู่) was lost. "
        f"note={txn.get('note')!r} description={txn.get('description')!r} "
        f"category={txn.get('category') or txn.get('category_name')!r}"
    )


# ---------------------------------------------------------------------------
# I1002-b — label-only turn 1, amount turn 2 (the inverse direction)
# ---------------------------------------------------------------------------
@pytest.mark.id("I1002")
def test_I1002_b_two_turn_label_then_amount(client):
    """I1002-b: turn 1 carries a LABEL only ("จ่าย youtube"). Turn 2 answers
    "99". Turn 2 must produce an expense/99 proposal whose label still
    references youtube — the v2 regression where bare amounts dropped the
    accumulated slots translates to v3 as: ReAct must keep turn-1 context
    in the message history when interpreting turn 2's bare answer."""
    thread_id = str(uuid.uuid4())

    # Turn 1: label only → ask amount.
    blocks1, _ = _post(client, thread_id, "จ่าย youtube", wallet_id=WALLET_TEST)
    assert _any_proposal(blocks1) is None, (
        f"turn 1 (no amount) must NOT propose; got {_types(blocks1)}"
    )

    # Turn 2: bare amount → propose with surviving turn-1 label.
    blocks2, _ = _post(client, thread_id, "99", wallet_id=WALLET_TEST)
    prop = _any_proposal(blocks2)
    assert prop is not None, (
        f"turn 2 (bare 99) must produce a proposal; got {_types(blocks2)}"
    )
    txn = prop["transaction"]
    assert _amount_eq(txn["amount"], "99"), (
        f"amount=99 (from turn 2) must be in proposal, got {txn['amount']!r}"
    )
    assert txn["type"] == "expense", f"expected expense, got {txn['type']!r}"
    label_blob = " ".join(
        str(txn.get(k) or "")
        for k in ("note", "description", "category", "category_name")
    ).lower()
    assert "youtube" in label_blob, (
        "the turn-1 youtube label was lost across the clarification — "
        "ReAct dropped the context. "
        f"note={txn.get('note')!r} description={txn.get('description')!r} "
        f"category={txn.get('category') or txn.get('category_name')!r}"
    )


# ---------------------------------------------------------------------------
# I1002-c — bare amount on a fresh thread must NOT fabricate a proposal
# ---------------------------------------------------------------------------
@pytest.mark.id("I1002")
def test_I1002_c_bare_amount_fresh_thread_does_not_propose(client):
    """I1002-c: the negative-guard regression. A bare "99" on a FRESH thread
    (no prior history) carries no category and no inference signal. ReAct
    MUST ask for clarification, never fabricate an amount-only expense.

    This guards against the symmetric over-fire of the slot-accumulation
    behaviour — counterpart to v2 I1002-d."""
    thread_id = str(uuid.uuid4())

    blocks, _answer = _post(client, thread_id, "99", wallet_id=WALLET_TEST)
    assert _any_proposal(blocks) is None, (
        "a bare 99 on a fresh thread must NOT produce a proposal — "
        f"the ReAct loop fabricated one. blocks={_types(blocks)}"
    )


# ---------------------------------------------------------------------------
# I1002-d — income type survives the clarification resume
# ---------------------------------------------------------------------------
@pytest.mark.id("I1002")
def test_I1002_d_income_type_survives_clarification(client):
    """I1002-d: turn 1 says "ได้เงินจากงานเสริม" (income label, no amount) →
    ask amount. Turn 2 says "500" → propose income/500 with the side-job
    label surviving. Catches the regression where income silently flips to
    expense (the common default) during the resume."""
    thread_id = str(uuid.uuid4())

    blocks1, _ = _post(client, thread_id, "ได้เงินจากงานเสริม", wallet_id=WALLET_TEST)
    assert _any_proposal(blocks1) is None, (
        f"turn 1 (income label, no amount) must NOT propose; got {_types(blocks1)}"
    )

    blocks2, _ = _post(client, thread_id, "500", wallet_id=WALLET_TEST)
    prop = _any_proposal(blocks2)
    assert prop is not None, (
        f"turn 2 (the amount) must produce a proposal; got {_types(blocks2)}"
    )
    txn = prop["transaction"]
    assert _amount_eq(txn["amount"], "500"), (
        f"amount must be exactly 500, got {txn['amount']!r}"
    )
    assert txn["type"] == "income", (
        f"type must survive as income across the clarification, got {txn['type']!r}"
    )

    rubric = (
        "You grade a transaction proposal a finance assistant built after a "
        "two-turn clarification: turn 1 the user said they earned money from a "
        "side job (income, no amount), turn 2 they answered '500'.\n"
        "PASS if the proposal is INCOME of amount 500 that still refers to the "
        "side-job/earning context from turn 1.\n"
        "FAIL if the amount is not 500, the type is expense, or the proposal "
        "dropped the side-job context entirely."
    )
    answer = (
        f"proposal: type={txn['type']} amount={txn['amount']} "
        f"note={txn.get('note')!r} "
        f"category={txn.get('category') or txn.get('category_name')!r}"
    )
    verdict = asyncio.run(judge("turn1: ได้เงินจากงานเสริม | turn2: 500", answer, rubric))
    assert verdict["passed"], f"judge FAIL — {verdict['reason']}"
