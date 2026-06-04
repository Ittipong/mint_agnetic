"""I1004 — voice STT → ReAct proposal (v3 integration, real STT_MODEL + DB).

Adapted from v2 `I1004_voice_stt.py`. v3 keeps the v2 mobile contract for
voice (multipart `audio` upload to `/chat/voice`) but the internals changed:
the voice endpoint now calls `endpoints/voice_handler.handle_voice_chat`
which does STT then feeds the transcript into the ReAct graph via the same
`streaming.stream_chat` adapter the text path uses. There is no longer a
"voice_fast_path" merged classifier (memory `project_voice_fast_path` was
v2-only).

This test pins:
  * the REAL STT (not the canned mock — guards the v2 regression where
    `/chat/voice` returned "จ่ายค่ากาแฟ 120 บาท" regardless of audio),
  * the transcript block is emitted before any answer/proposal,
  * the ReAct loop produces a `transaction_proposal` matching the spoken
    amount (200 baht) and a sensible category for "กินข้าว",
  * the empty-audio reject path takes `stt_error` with `reason=no_speech`.

NOT hermetic — needs `OPENROUTER_API_KEY`, `DATABASE_URL`,
`BACKEND_DATABASE_URL`, `STT_MODEL`, plus a REACT/CODEACT model and the
committed `eat_rice_200.m4a` fixture. Skipped cleanly when any is missing.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agent.entity_catalog import load_entity_catalog
from src.agent.llm_judge import judge
from src.agent.server import app


SEED_USER = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"

_FIXTURE = Path(__file__).parent / "fixtures" / "voice" / "eat_rice_200.m4a"

_REQUIRED_ENV = ("OPENROUTER_API_KEY", "DATABASE_URL", "BACKEND_DATABASE_URL", "STT_MODEL")
_MODEL_ENV = ("REACT_MODEL",)
pytestmark = pytest.mark.skipif(
    not all(os.getenv(k) for k in _REQUIRED_ENV)
    or not any(os.getenv(k) for k in _MODEL_ENV)
    or not _FIXTURE.exists(),
    reason=(
        f"integration test needs env {', '.join(_REQUIRED_ENV)} + one of "
        f"{', '.join(_MODEL_ENV)} + voice fixture at {_FIXTURE}"
    ),
)

_RUBRIC = (
    "You grade a category chosen by a finance assistant for one spoken expense.\n"
    "PASS if the chosen category is a reasonable fit for the note, picked from "
    "ONLY the wallet's available categories listed in the answer.\n"
    "- 'กินข้าว' (eating a meal) fits a food category ('อาหาร') — PASS.\n"
    "- A general category ('อื่นๆ') passes only if no food category exists.\n"
    "- An unrelated specific category fails (e.g. food -> fuel)."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def catalog():
    return asyncio.run(_load_catalog())


@pytest.fixture(scope="module")
def fallback_wallet(catalog):
    """The index-0 fallback voice files into (no client wallet selection)."""
    w = catalog.slip_wallet(None)
    assert w is not None, "SEED_USER has no wallets — cannot run voice integration"
    return w


async def _load_catalog():
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        os.environ["BACKEND_DATABASE_URL"], min_size=1, max_size=1, open=False
    )
    await pool.open()
    try:
        return await load_entity_catalog(pool, SEED_USER)
    finally:
        await pool.close()


def _parse_sse_blocks(raw: str) -> list[dict]:
    """Return JSON-decoded `event: block` payloads from an SSE response."""
    blocks: list[dict] = []
    for chunk in raw.replace("\r\n", "\n").split("\n\n"):
        lines = chunk.splitlines()
        event = ""
        data_parts: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_parts.append(line[len("data:"):].lstrip())
        if event == "block" and data_parts:
            try:
                blocks.append(json.loads("".join(data_parts)))
            except json.JSONDecodeError:
                continue
    return blocks


def _voice_blocks(client: TestClient, audio_bytes: bytes) -> list[dict]:
    """POST a voice turn and return the emitted blocks."""
    resp = client.post(
        "/chat/voice",
        data={"user_id": SEED_USER, "thread_id": str(uuid.uuid4())},
        files={"audio": ("voice.m4a", audio_bytes, "audio/m4a")},
    )
    assert resp.status_code == 200, resp.text
    return _parse_sse_blocks(resp.text)


@pytest.mark.id("I1004")
def test_I1004_voice_transcribes_and_proposes(client, catalog, fallback_wallet):
    blocks = _voice_blocks(client, _FIXTURE.read_bytes())
    types = [b.get("type") for b in blocks]

    # 1. A transcript block carries the spoken text — NOT the old canned mock.
    transcripts = [b for b in blocks if b.get("type") == "transcript"]
    assert transcripts, f"no transcript block; got {types}"
    text = transcripts[0].get("text", "")
    assert text and "120" not in text, f"looks like the old canned mock: {text!r}"
    assert "200" in text, f"transcript missing the spoken amount: {text!r}"

    # 2. Exactly one transaction_proposal, for 200 THB expense.
    props = [b for b in blocks if b.get("type") == "transaction_proposal"]
    assert len(props) == 1, f"expected ONE proposal, got {len(props)}; types={types}"
    txn = props[0]["transaction"]
    assert txn["amount"] == pytest.approx(200), txn
    assert txn["type"] == "expense", txn
    assert txn["wallet_sync_id"] == fallback_wallet.sync_id, txn

    # 3. Category fits the spoken note.
    sid = txn.get("category_sync_id")
    if sid is not None:
        by_id = {c.sync_id: c for c in catalog.categories_for(fallback_wallet.sync_id)}
        assert sid in by_id, f"category {sid} not in wallet (+global)"
        assert by_id[sid].type == "expense", f"type-leak: {by_id[sid].type!r}"
        available = [c.name for c in by_id.values() if c.type == "expense"]
        answer = (
            f"spoken note: {txn.get('note')}\n"
            f"chosen category: {by_id[sid].name}\n"
            f"wallet's available expense categories: {available}"
        )
        verdict = asyncio.run(judge(str(txn.get("note") or ""), answer, _RUBRIC))
        assert verdict["passed"], f"judge FAIL — {verdict['reason']}"


@pytest.mark.id("I1004")
def test_I1004_empty_audio_takes_no_speech_path(client):
    """A 0-byte upload must reject before any STT spend: stt_error no_speech,
    no proposal."""
    blocks = _voice_blocks(client, b"")
    assert not [b for b in blocks if b.get("type") == "transaction_proposal"], (
        f"empty audio produced a proposal: {blocks}"
    )
    errs = [b for b in blocks if b.get("type") == "stt_error"]
    assert errs, f"expected stt_error block; got {blocks}"
    assert errs[0].get("reason") == "no_speech", f"got {blocks}"
