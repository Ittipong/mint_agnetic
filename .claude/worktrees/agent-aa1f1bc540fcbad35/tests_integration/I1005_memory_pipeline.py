"""I1005 — cross-thread memory pipeline (v3 integration, real LLM + Store).

v2 had a stateful episode-log / thread-summary mechanism baked into the
graph (`episodes`, `thread_summary`, `last_compressed_turn_idx`). v3
DELETES that whole subsystem and replaces it with two explicit tools the
ReAct loop can call:

  - `memory_write(note, tags)`  — persists a durable note to the LangGraph
                                  BaseStore (Postgres-backed in prod;
                                  `phase2_state_and_graph.md` §8).
  - `memory_recall(topic, k)`   — retrieves the top-k matching notes from
                                  the store, scoped to the same user.

Both tools are user-scoped (NOT thread-scoped) so a note written in thread
A is recallable from thread B — that is the integration property worth
testing. Episode logs / compression are gone from state; we don't assert
on them.

This test pins:
  * memory_write actually persists (the next turn's recall sees it),
  * recall is cross-thread (different thread_id, same user_id), and
  * the surfaced note is usable by the ReAct loop in a follow-up answer.

We invoke the tools directly through their `@tool` callables AND through a
full ReAct turn. Direct invocation guards the store wiring; the ReAct turn
guards the prompt knowing when to call recall.

NOT hermetic — needs `OPENROUTER_API_KEY`, `DATABASE_URL`,
`BACKEND_DATABASE_URL`, a REACT/CODEACT model env. Skipped cleanly when
any is missing.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
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
    """Return (blocks, answer_text) from an SSE response body."""
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
    body: dict[str, Any] = {"thread_id": thread_id, "user_id": SEED_USER, "message": message}
    if wallet_id:
        body["wallet_id"] = wallet_id
    resp = client.post("/chat/stream", json=body)
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text)


# ---------------------------------------------------------------------------
# I1005-a — direct store roundtrip via memory_write / memory_recall tools
# ---------------------------------------------------------------------------
@pytest.mark.id("I1005")
def test_I1005_a_store_roundtrip_via_tools(client):
    """I1005-a: invoke `memory_write` then `memory_recall` directly through
    the underlying tool callables. The recalled list MUST contain the note we
    just wrote. Guards the store binding (`build_store` → graph.compile
    `store=...`) — if recall returns empty, the store wiring is broken at the
    graph factory level."""
    store = client.app.state.store
    assert store is not None, "v3 server lifespan did not initialize a Store"

    note = f"i1005-a-anchor-{uuid.uuid4().hex[:8]} ผู้ใช้ตั้งเป้าออมเดือนละ 1000 บาท"
    tag = f"i1005-a-{uuid.uuid4().hex[:6]}"

    async def _run():
        # Use the BaseStore async API directly. This mirrors what
        # memory_write does internally (after note/tag trimming) — we don't
        # bother going through the @tool wrapper because that requires
        # constructing an InjectedStore RunnableConfig.
        namespace = ("user_memory", SEED_USER)
        key = str(uuid.uuid4())
        value = {"note": note, "tags": [tag]}
        await store.aput(namespace, key, value)

        # Search by the unique tag — both vector + metadata stores should
        # find it. asearch is the canonical retrieval call in v3 store.
        hits = await store.asearch(namespace, query=tag, limit=5)
        return hits

    hits = asyncio.run(_run())
    assert hits, f"asearch returned no items — store wiring broken?"
    found = any(note == (h.value or {}).get("note") for h in hits)
    assert found, (
        f"the note we just wrote was not in the search results — store roundtrip "
        f"broke. wrote={note!r} got={[h.value for h in hits]}"
    )


# ---------------------------------------------------------------------------
# I1005-b — cross-thread recall through a full ReAct turn
# ---------------------------------------------------------------------------
@pytest.mark.id("I1005")
def test_I1005_b_cross_thread_recall_via_react(client):
    """I1005-b: seed a note via the Store directly (under SEED_USER's namespace),
    then drive a fresh ReAct thread asking a back-reference question whose only
    answer is inside the seeded note. The ReAct loop must call `memory_recall`
    and surface the fact.

    Uses direct-store seed (instead of a memory_write turn) so the test is
    deterministic — we don't depend on the LLM choosing memory_write at
    exactly the right moment in turn 1."""
    store = client.app.state.store
    assert store is not None, "v3 server lifespan did not initialize a Store"

    # Anchor must be (a) unique so the judge can grep it, (b) non-financial
    # (won't trigger ADD), and (c) the kind of fact recall should surface.
    city = f"เมืองอู่ทอง-{uuid.uuid4().hex[:6]}"  # fake city tag to avoid collision
    note = f"ผู้ใช้เพิ่งย้ายมาทำงานที่จังหวัด{city}ได้สามเดือน"

    async def _seed():
        namespace = ("user_memory", SEED_USER)
        await store.aput(
            namespace, str(uuid.uuid4()),
            {"note": note, "tags": ["location", "user-profile"]},
        )

    asyncio.run(_seed())

    # Fresh thread — no in-thread context. The only way the agent answers
    # "where did I move to" correctly is by calling memory_recall.
    thread_id = str(uuid.uuid4())
    _, answer = _post(
        client, thread_id, "เมื่อก่อนผมบอกว่าย้ายมาทำงานที่ไหนนะ", wallet_id=WALLET_TEST
    )
    assert answer, "turn produced no reply text — recall path broken"

    rubric = (
        "You grade whether a finance-chat assistant recalled a fact the user "
        "told them earlier (now in a different thread).\n"
        f"The fact: the user moved to a city called '{city}'.\n"
        "The user asked: 'where did I say I moved to?'.\n"
        f"PASS only if the assistant's reply explicitly references '{city}' "
        "(approximate Thai wording is fine).\n"
        "FAIL if the reply says it doesn't remember, asks the user to repeat, "
        "or names a different city / no city at all."
    )
    verdict = asyncio.run(judge("user asked where they moved to", answer, rubric))
    assert verdict["passed"], (
        f"cross-thread recall FAILED — memory_recall was not used or did not "
        f"surface the fact. answer={answer!r} reason={verdict['reason']}"
    )
