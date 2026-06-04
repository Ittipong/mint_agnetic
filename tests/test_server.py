"""Unit tests for `src.agent.server` — endpoint wiring + lifespan version gate.

Covers UT-SR01..UT-SR09 from `docs/v3/phase3_implementation_plan.md` §3.

Test strategy:
  - The FastAPI app is imported once; tests skip the production lifespan
    (`with TestClient(app)` would open a real Postgres pool) and inject a
    fake `app.state.*` set per test via a helper context manager.
  - Mocks replace: `agent_graph` (astream / aget_state / aupdate_state),
    `vision_call`, `stt_call`, `threads`, `messages`, `repo`.
  - We assert on the SSE wire (event sequence) + on whether the mocks were
    called with the right arguments (e.g. `as_node="finalize"` on
    confirm/cancel — memory `project_slip_vision_as_node`).
  - No real network / DB / LLM is touched. The TestClient uses ASGI's
    in-process transport, so SSE chunks land as a streaming response.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.agent import server
from src.agent.entity_catalog import CategoryEntry, EntityCatalog, WalletEntry


# ---------------------------------------------------------------------------
# Fake state — minimal mocks for app.state slots the handlers read
# ---------------------------------------------------------------------------


class FakeThreads:
    """Mimics ThreadsRepo's public methods. All async; collect calls in lists
    so tests can assert on side effects."""

    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.created: list[dict] = []
        self.renamed: list[dict] = []
        self.deleted: list[str] = []
        # Pre-seed state — tests poke `_threads` directly when they need a
        # specific row to be findable.
        self._threads: dict[str, dict] = {}

    async def upsert(self, *, thread_id: str, user_id: str, title: Optional[str] = None) -> None:
        self.upserts.append({"thread_id": thread_id, "user_id": user_id, "title": title})

    async def create(self, *, user_id: str, title: Optional[str] = None) -> dict:
        row = {
            "thread_id": "new-thread-123",
            "user_id": user_id,
            "title": title or "แชตใหม่",
            "created_at": None,
            "updated_at": None,
            "message_count": 0,
            "last_message_preview": None,
        }
        self.created.append(row)
        return row

    async def rename(self, thread_id: str, title: str) -> Optional[dict]:
        if thread_id not in self._threads:
            return None
        row = {"thread_id": thread_id, "title": title, "updated_at": None}
        self.renamed.append(row)
        return row

    async def get(self, thread_id: str) -> Optional[dict]:
        return self._threads.get(thread_id)

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        return [r for r in self._threads.values() if r.get("user_id") == user_id]

    async def delete(self, thread_id: str) -> bool:
        if thread_id in self._threads:
            del self._threads[thread_id]
            self.deleted.append(thread_id)
            return True
        return False


class FakeMessages:
    """Mimics MessagesRepo's public methods."""

    def __init__(self) -> None:
        self.appended: list[dict] = []
        self._by_thread: dict[str, list[dict]] = {}

    async def append(self, *, thread_id: str, user_id: str, role: str, content: dict) -> None:
        row = {"role": role, "content": content, "thread_id": thread_id, "user_id": user_id}
        self.appended.append(row)
        self._by_thread.setdefault(thread_id, []).append(
            {"role": role, "content": content, "created_at": None}
        )

    async def list_for_thread(self, thread_id: str) -> list[dict]:
        return list(self._by_thread.get(thread_id, []))

    async def delete_for_thread(self, thread_id: str) -> None:
        self._by_thread.pop(thread_id, None)


class FakeRepo:
    """Mimics ProposalRepo.insert_pending_proposal — only that method is
    called from the server's slip-persistence path."""

    def __init__(self) -> None:
        self.inserts: list[dict] = []

    async def insert_pending_proposal(
        self, *, proposal_id: str, user_id: str, kind: str, payload: dict
    ) -> str:
        self.inserts.append(
            {"proposal_id": proposal_id, "user_id": user_id,
             "kind": kind, "payload": payload}
        )
        return proposal_id


class FakeGraph:
    """Stub LangGraph compiled graph for tests.

    Customise `events` per test — they're yielded as `(stream_mode, chunk)`
    tuples by `astream` exactly the way the real graph would. Customise
    `state_values` to control what `aget_state` returns (for the confirm
    handler tests). `updates` collects every `aupdate_state` call so tests
    can verify `as_node="finalize"` was passed.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.state_values: dict = {}
        self.updates: list[dict] = []

    def astream(self, init_state: dict, config: dict, stream_mode: list[str]):
        events_copy = list(self.events)

        async def gen():
            for ev in events_copy:
                yield ev

        return gen()

    async def aget_state(self, cfg: dict):
        return _StateSnapshot(self.state_values)

    async def aupdate_state(self, cfg: dict, update: dict, *, as_node: Optional[str] = None):
        self.updates.append({"cfg": cfg, "update": update, "as_node": as_node})


class _StateSnapshot:
    def __init__(self, values: dict):
        self.values = values


# ---------------------------------------------------------------------------
# Fixture: inject fake state without running the real lifespan
# ---------------------------------------------------------------------------


@contextmanager
def fake_app_state(
    *,
    graph: Optional[FakeGraph] = None,
    threads: Optional[FakeThreads] = None,
    messages: Optional[FakeMessages] = None,
    repo: Optional[FakeRepo] = None,
    vision_call: Any = None,
    stt_call: Any = None,
    backend_pool: Any = None,
) -> Iterator[TestClient]:
    """Patch `app.state.*` slots + yield a TestClient.

    We skip `with TestClient(app):` (which would invoke lifespan and open a
    real Postgres pool) by constructing the client and directly assigning
    state. The handlers read everything via `req.app.state.*`, so they
    can't tell the difference.
    """
    app = server.app
    # Save and restore prior values in case anything else set them.
    saved = {}
    for k in ("agent_graph", "threads", "messages", "repo",
              "vision_call", "stt_call", "backend_pool", "saver", "store"):
        saved[k] = getattr(app.state, k, None)
    app.state.agent_graph = graph or FakeGraph()
    app.state.threads = threads or FakeThreads()
    app.state.messages = messages or FakeMessages()
    app.state.repo = repo or FakeRepo()
    app.state.vision_call = vision_call
    app.state.stt_call = stt_call
    app.state.backend_pool = backend_pool
    app.state.saver = None
    # raise_server_exceptions=False — when a handler raises, we want to see
    # the 500 response and assert on the JSON body rather than have pytest
    # re-raise the exception out of the client call.
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        for k, v in saved.items():
            setattr(app.state, k, v)


# ---------------------------------------------------------------------------
# Helpers — parse SSE responses from TestClient (sse_starlette format)
# ---------------------------------------------------------------------------


def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE wire bytes into a list of `{event, data}` dicts.

    sse_starlette emits one event per `\\r\\n\\r\\n`-separated block in the
    form:
        event: <name>
        data: <payload>

    We normalize CRLF -> LF (mobile decoder does the same — memory
    `project_sse_ngrok_framing`) and split.
    """
    text = raw.replace("\r\n", "\n")
    out: list[dict] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data_parts: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].strip())
        if event:
            out.append({"event": event, "data": "\n".join(data_parts)})
    return out


# ---------------------------------------------------------------------------
# UT-SR01 — POST /chat/stream text-only routes to ReAct stream
# ---------------------------------------------------------------------------


def test_UT_SR01_chat_stream_text_routes_to_react() -> None:
    """UT-SR01: a text-only `/chat/stream` body drives the graph through
    sse_adapter — wire produces status_token + answer_token + answer block
    + done. The graph is a FakeGraph yielding one AIMessageChunk through
    the messages stream mode."""
    from langchain_core.messages import AIMessageChunk

    g = FakeGraph()
    # Mimic what stream_chat consumes: ("messages", (chunk, metadata)) for
    # answer tokens; ("updates", {node: {...}}) for blocks.
    g.events = [
        ("messages", (AIMessageChunk(content="สวัสดี"), {})),
        ("messages", (AIMessageChunk(content="ครับ"), {})),
    ]
    with fake_app_state(graph=g) as client:
        resp = client.post(
            "/chat/stream",
            json={
                "thread_id": "t-sr1",
                "user_id": "u-1",
                "message": "hello",
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        # Status tokens fire first (one per word of "กำลังคิด...").
        assert any(e["event"] == "status_token" for e in events), events
        # Answer tokens carry the streamed deltas.
        answer_tokens = [e for e in events if e["event"] == "answer_token"]
        assert "".join(e["data"] for e in answer_tokens) == "สวัสดีครับ"
        # End-of-stream answer block.
        block_events = [e for e in events if e["event"] == "block"]
        assert any(
            json.loads(b["data"]).get("type") == "answer"
            for b in block_events
        ), block_events
        # Final done event with thread_id.
        assert events[-1]["event"] == "done"
        assert json.loads(events[-1]["data"]) == {"thread_id": "t-sr1"}


# ---------------------------------------------------------------------------
# UT-SR02 — POST /chat/stream with image_b64s routes to slip_handler
# ---------------------------------------------------------------------------


def _slip_catalog() -> EntityCatalog:
    return EntityCatalog(
        wallets=[
            WalletEntry(
                sync_id="w-cash", name="เงินสด", currency="THB",
                wallet_type="general", is_default=True,
            ),
        ],
        categories=[
            CategoryEntry(sync_id="c-coffee", name="กาแฟ", type="expense"),
            CategoryEntry(sync_id="c-other-exp", name="อื่นๆ", type="expense"),
            CategoryEntry(sync_id="c-other-inc", name="อื่นๆ", type="income"),
        ],
    )


def test_UT_SR02_chat_stream_image_routes_to_slip(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-SR02: when `image_b64s` is present, the slip branch fires — vision
    is invoked and the response contains a `transaction_proposal_group`
    block. The graph's astream is NEVER called (slip skips ReAct per
    decision α).

    We monkeypatch `load_entity_catalog` to return a hand-built catalog —
    keeping the test off the real backend pool.
    """
    async def fake_vision(messages: list[dict]) -> str:
        return json.dumps({
            "readable": True,
            "transactions": [
                {
                    "type": "expense", "amount": 120,
                    "category_sync_id": "c-coffee",
                    "note": "ค่ากาแฟ", "include_in_report": True,
                }
            ],
        })

    async def fake_load_catalog(pool: Any, user_id: str) -> EntityCatalog:
        return _slip_catalog()

    monkeypatch.setattr(server, "load_entity_catalog", fake_load_catalog)

    g = FakeGraph()
    g.events = []  # graph never called
    g.astream = MagicMock(side_effect=AssertionError("graph.astream must NOT be called on slip turn"))

    with fake_app_state(graph=g, vision_call=fake_vision) as client:
        resp = client.post(
            "/chat/stream",
            json={
                "thread_id": "t-sr2",
                "user_id": "u-1",
                "message": "[INTENT:parse_transaction_from_slip]",
                "image_b64s": ["fake-jpeg-b64"],
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        block_events = [
            json.loads(e["data"]) for e in events if e["event"] == "block"
        ]
        groups = [b for b in block_events if b.get("type") == "transaction_proposal_group"]
        assert len(groups) == 1, block_events
        group = groups[0]
        assert group["wallet_sync_id"] == "w-cash"
        assert group["total"] == 120
        assert group["proposal_id"] == group["group_id"]  # memory: lockstep
        # The slip handler short-circuited — `aupdate_state` was called for
        # persistence with `as_node="finalize"` (memory: project_slip_vision_as_node).
        assert any(u.get("as_node") == "finalize" for u in g.updates), g.updates


# ---------------------------------------------------------------------------
# UT-SR03 — POST /chat/voice routes to voice_handler then ReAct
# ---------------------------------------------------------------------------


def test_UT_SR03_chat_voice_starts_with_transcript_block() -> None:
    """UT-SR03: `/chat/voice` runs STT, emits a `transcript` block before
    the graph stream. Graph yields nothing here so the response ends with
    just the transcript + done."""

    async def fake_stt(messages: list[dict]) -> str:
        return "ใช้กาแฟห้าสิบบาท"

    g = FakeGraph()  # empty events — stream_chat emits done.
    with fake_app_state(graph=g, stt_call=fake_stt) as client:
        resp = client.post(
            "/chat/voice",
            data={"user_id": "u-1", "thread_id": "t-sr3"},
            files={"audio": ("voice.m4a", b"\x00\x01\x02", "audio/m4a")},
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        block_events = [json.loads(e["data"]) for e in events if e["event"] == "block"]
        # First block is the transcript (heard text).
        transcripts = [b for b in block_events if b.get("type") == "transcript"]
        assert len(transcripts) == 1
        assert transcripts[0]["text"] == "ใช้กาแฟห้าสิบบาท"
        # The terminal done event carries the thread_id.
        assert events[-1]["event"] == "done"


# ---------------------------------------------------------------------------
# UT-SR04 — POST /transactions/confirm updates checkpoint with as_node=finalize
# ---------------------------------------------------------------------------


def test_UT_SR04_confirm_updates_state_with_finalize_node() -> None:
    """UT-SR04: confirming a pending proposal flips status to "confirmed"
    via `aupdate_state(cfg, update, as_node="finalize")`. The `as_node`
    argument is MANDATORY — memory `project_slip_vision_as_node`.

    Also writes `last_txn` so the next turn's resolver can see the freshest
    confirmed txn.
    """
    g = FakeGraph()
    g.state_values = {
        "proposals": [
            {
                "proposal_id": "p-1",
                "intent_type": "ADD_TRANSACTION",
                "type": "ADD_TRANSACTION",
                "payload": {
                    "type": "expense", "amount": 50, "category": "coffee",
                    "category_sync_id": "c-1", "wallet_sync_id": "w-1",
                    "sync_id": "tx-sync-1",
                },
                "status": "pending",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
    }
    with fake_app_state(graph=g) as client:
        resp = client.post(
            "/transactions/confirm",
            json={"proposal_id": "p-1", "user_id": "u-1", "thread_id": "t-sr4"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "confirmed"
        assert body["proposal_id"] == "p-1"
        # The graph was updated with as_node="finalize".
        assert len(g.updates) == 1
        update_call = g.updates[0]
        assert update_call["as_node"] == "finalize", (
            "as_node='finalize' is REQUIRED — memory project_slip_vision_as_node"
        )
        # last_txn is written for an ADD_TRANSACTION confirm.
        assert update_call["update"].get("last_txn", {}).get("id") == "p-1"
        # Proposals list now has the entry flipped to confirmed.
        flipped = update_call["update"]["proposals"][0]
        assert flipped["status"] == "confirmed"


# ---------------------------------------------------------------------------
# UT-SR05 — GET /threads/{id}/messages returns rich blocks
# ---------------------------------------------------------------------------


def test_UT_SR05_messages_returns_rich_blocks_with_status() -> None:
    """UT-SR05: `/threads/{id}/messages` returns assistant content as
    `{blocks, answer}`. Proposal blocks are annotated with the live
    checkpointer status so a reopened thread shows finalized cards instead
    of stale-pending (memory `project_chat_history_block_replay`)."""
    m = FakeMessages()
    m._by_thread["t-sr5"] = [
        {"role": "user", "content": {"text": "ซื้อกาแฟ"}, "created_at": None},
        {
            "role": "assistant",
            "content": {
                "blocks": [
                    {
                        "type": "transaction_proposal",
                        "proposal_id": "p-1",
                        "amount": 50,
                    },
                ],
                "answer": None,
            },
            "created_at": None,
        },
    ]
    g = FakeGraph()
    g.state_values = {
        "proposals": [
            {"proposal_id": "p-1", "status": "confirmed"},
        ],
    }
    with fake_app_state(graph=g, messages=m) as client:
        resp = client.get("/threads/t-sr5/messages")
        assert resp.status_code == 200
        body = resp.json()
        msgs = body["messages"]
        assert len(msgs) == 2
        assistant = msgs[1]
        blocks = assistant["content"]["blocks"]
        # Status was annotated from live state.
        assert blocks[0]["status"] == "confirmed"


# ---------------------------------------------------------------------------
# UT-SR06 — CHAT_AGENT_VERSION gate refuses non-v3 startup
# ---------------------------------------------------------------------------


def test_UT_SR06_lifespan_refuses_when_not_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-SR06: when CHAT_AGENT_VERSION!=v3, the lifespan raises with the
    REFUSE_REASON message. We test the lifespan coroutine directly because
    TestClient swallows lifespan exceptions into a startup-shutdown event.
    """
    from src.agent.utils.version_router import REFUSE_REASON
    # Reset one-shot warning so the gate sees a clean default.
    from src.agent.utils import version_router as vr
    vr._WARNED["emitted"] = False
    monkeypatch.delenv("CHAT_AGENT_VERSION", raising=False)

    async def run_lifespan() -> None:
        # Driving the lifespan async context manager directly — it raises
        # before entering the `yield`.
        async with server.lifespan(server.app):  # type: ignore[arg-type]
            pass

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(run_lifespan())
    assert REFUSE_REASON in str(exc_info.value)


def test_UT_SR06b_lifespan_proceeds_when_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    """UT-SR06b: with CHAT_AGENT_VERSION=v3 and a mocked pool/graph chain,
    the lifespan proceeds past the gate. We can't run the full lifespan
    without Postgres, so we patch the post-gate calls to no-ops and just
    verify the gate passes.

    The simpler proof: `is_v3_active()` flips True and the very first
    statement past the gate (session log dir setup) is exercised without
    raising the REFUSE_REASON RuntimeError.
    """
    from src.agent.utils import version_router as vr
    vr._WARNED["emitted"] = False
    monkeypatch.setenv("CHAT_AGENT_VERSION", "v3")
    # Patch heavy boot dependencies with no-ops so the lifespan can run.
    monkeypatch.setattr(server, "make_pool", lambda: _FakePool())
    monkeypatch.setattr(server, "make_backend_pool", lambda: _FakePool())

    async def fake_build_graph(*args, **kwargs):
        return FakeGraph()

    monkeypatch.setattr(server, "build_graph", fake_build_graph)

    # Patch make_multimodal_call so it doesn't blow up on missing API key.
    def fake_mm(role: str):
        async def _call(messages: list[dict]) -> str:
            return ""

        return _call

    monkeypatch.setattr(server, "make_multimodal_call", fake_mm)
    # Patch AsyncPostgresSaver import inside the lifespan body — set
    # DATABASE_URL empty so the saver branch is skipped entirely.
    monkeypatch.delenv("DATABASE_URL", raising=False)

    async def run_lifespan() -> None:
        async with server.lifespan(server.app):
            # Past the gate — that's all this test asserts.
            assert server.app.state.agent_graph is not None

    asyncio.run(run_lifespan())


class _FakePool:
    """Minimal pool stub used for the lifespan gate-pass test."""

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def connection(self):
        return _FakeConn()


class _FakeConn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def cursor(self):
        return _FakeCursor()


class _FakeCursor:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def execute(self, *args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# UT-SR07 — POST /transactions/cancel (mirror of SR04)
# ---------------------------------------------------------------------------


def test_UT_SR07_cancel_updates_state_with_finalize_node() -> None:
    """UT-SR07: cancelling a pending proposal flips status to "cancelled"
    via `aupdate_state(cfg, update, as_node="finalize")` and clears
    `last_txn` when it points at the cancelled proposal (so a follow-up
    "แก้..." doesn't latch on)."""
    g = FakeGraph()
    g.state_values = {
        "proposals": [
            {
                "proposal_id": "p-2", "intent_type": "ADD_TRANSACTION",
                "payload": {"amount": 50}, "status": "pending",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "last_txn": {"id": "p-2", "amount": 50},
    }
    with fake_app_state(graph=g) as client:
        resp = client.post(
            "/transactions/cancel",
            json={"proposal_id": "p-2", "user_id": "u-1", "thread_id": "t-sr7"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "cancelled"
        update_call = g.updates[0]
        assert update_call["as_node"] == "finalize"
        # last_txn cleared because it pointed at the cancelled proposal.
        assert "last_txn" in update_call["update"]
        assert update_call["update"]["last_txn"] is None


# ---------------------------------------------------------------------------
# UT-SR08 — GET /healthz reports ok + version
# ---------------------------------------------------------------------------


def test_UT_SR08_healthz_returns_ok_v3() -> None:
    """UT-SR08: `/healthz` is the cheap liveness probe Cloudflare tunnel +
    uvicorn watcher hit. Reports `ok=True` and the active version so an
    operator can verify the cutover at a glance."""
    with fake_app_state() as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "version": "v3"}


# ---------------------------------------------------------------------------
# UT-SR09 — malformed body returns 4xx with clear error, no stack trace
# ---------------------------------------------------------------------------


def test_UT_SR09_chat_stream_malformed_body_returns_validation_error() -> None:
    """UT-SR09: a body missing required fields returns a 422 validation
    error from FastAPI with field-level detail — no stack trace, no 500.
    The mobile decoder treats 4xx as a hard client error (retry won't
    help) vs 5xx as a transient (retry might)."""
    with fake_app_state() as client:
        resp = client.post(
            "/chat/stream",
            json={"thread_id": "t-only"},  # missing user_id + message
        )
        assert resp.status_code == 422, resp.text
        body = resp.json()
        # FastAPI surfaces validation errors under `detail` — assert the
        # missing field is named so mobile can show a specific message.
        assert "detail" in body
        assert any(
            "user_id" in str(item).lower() or "message" in str(item).lower()
            for item in body["detail"]
        )


# ---------------------------------------------------------------------------
# Bonus — endpoint inventory check (exactly the mobile-contract surface)
# ---------------------------------------------------------------------------


def test_UT_SR10_endpoint_inventory() -> None:
    """Every endpoint listed in `phase3_implementation_plan.md` §2 Wave 6
    + the mobile contract spec section 3 is present on `app`. Guards
    against accidental endpoint deletion during refactors."""
    endpoints = {
        ("/chat/stream", "POST"),
        ("/chat/voice", "POST"),
        ("/transactions/confirm", "POST"),
        ("/transactions/cancel", "POST"),
        ("/threads", "GET"),
        ("/threads", "POST"),
        ("/threads/{thread_id}", "PATCH"),
        ("/threads/{thread_id}", "GET"),
        ("/threads/{thread_id}", "DELETE"),
        ("/threads/{thread_id}/messages", "GET"),
        ("/chat/{thread_id}", "GET"),
        ("/healthz", "GET"),
    }
    found = set()
    for route in server.app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            for m in route.methods:
                found.add((route.path, m))
    missing = endpoints - found
    assert not missing, f"missing endpoints: {sorted(missing)}"
