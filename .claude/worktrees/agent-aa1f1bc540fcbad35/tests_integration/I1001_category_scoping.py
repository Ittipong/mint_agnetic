"""I1001 — wallet + type-scoped category resolution (v3 integration, real LLM + DB).

Adapted from v2 `tests_integration/I1001_category_scoping.py`. The contract
under test is identical (the category resolver must scope by wallet AND by
transaction type), but the v3 plumbing differs:

  - v2 exposed a JSON `/chat` returning `{"blocks": [...]}`.
  - v3 exposes ONLY `/chat/stream` (SSE) per the mobile contract freeze; the
    text branch goes through ReAct, which calls `get_user_context` then
    `propose_transaction`. The proposal block is emitted by the tool and
    rides the SSE `block` event.

So this test:
  1. POSTs to `/chat/stream` and parses the SSE stream into block dicts.
  2. Extracts the `transaction_proposal` block.
  3. Asserts wallet + category scoping invariants (same as v2).
  4. Uses the LLM-judge for fuzzy semantic fit (same rubric as v2).

NOT hermetic — needs `OPENROUTER_API_KEY`, `DATABASE_URL`, `BACKEND_DATABASE_URL`,
and a `REACT_MODEL` / `CODEACT_MODEL` so the ReAct loop can boot.
Skipped when any is missing so the offline unit run stays green.

Async convention (memory `project_e2e_async_pattern`): test functions are
SYNC and drive async helpers via `asyncio.run(...)`. TestClient spins its own
loop — adding `@pytest.mark.asyncio` would deadlock.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Import server FIRST so `load_dotenv()` populates os.environ before the
# module-level skipif sees the env (memory `project_integration_test_pattern`).
from src.agent.entity_catalog import load_entity_catalog
from src.agent.llm_judge import judge
from src.agent.server import app


# Seeded dev user whose wallets carry the categories under test.
SEED_USER = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
WALLET_TEST = "90cb44d7-a0de-4598-a206-9d8de6b28289"   # general; HAS เฟอร์นิเจอร์
WALLET_CARDX = "d7e132d4-f472-4bbf-9dfc-2923e77305bb"  # creditcard; NO เฟอร์นิเจอร์

# v3 needs a REACT model env to construct the LLM (graph._make_model). Either
# of REACT_MODEL satisfies the build_graph
# contract, so we check the union.
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

_RUBRIC = (
    "You grade a category chosen by a finance assistant for a transaction.\n"
    "PASS if the chosen category is a reasonable fit for the item in the user's "
    "message, picked from ONLY the wallet's available categories listed in the "
    "answer.\n"
    "- An exact or clearly-related category passes (e.g. chair -> furniture).\n"
    "- If the wallet has NO exact category, the closest general one "
    "(e.g. 'ช้อปปิ้ง' / 'อื่นๆ') passes.\n"
    "- An unrelated category fails (e.g. chair -> fuel).\n"
    "The chosen category MUST appear in the available list."
)


@pytest.fixture(scope="module")
def client():
    """Real app via TestClient — entering the context runs the production
    lifespan (DB pools + graph build). No external uvicorn needed."""
    with TestClient(app) as c:
        yield c


def _parse_sse_blocks(raw: str) -> list[dict]:
    """Split an SSE response body into JSON-decoded `block` event dicts.

    sse_starlette emits CRLF-framed events (`event: block\\r\\ndata: {...}\\r\\n\\r\\n`).
    We normalise to LF, split on blank-line event boundaries, then collect only
    `event: block` payloads. Other events (`status_token`, `answer_token`,
    `done`, `error`) are dropped — this test only asserts block-level state.

    Memory `project_sse_ngrok_framing` warns about coalescing through proxies;
    TestClient runs in-process so we don't see that here, but the parser is
    deliberately robust to both CRLF and LF.
    """
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
            payload = "".join(data_parts)
            try:
                blocks.append(json.loads(payload))
            except json.JSONDecodeError:
                # Malformed block — skip; test will fail downstream on
                # "no proposal" if this was load-bearing.
                continue
    return blocks


def _propose(client: TestClient, message: str, wallet_id: str | None) -> dict:
    """POST /chat/stream and return the transaction payload of the proposal block."""
    body: dict[str, Any] = {
        "thread_id": str(uuid.uuid4()),
        "user_id": SEED_USER,
        "message": message,
    }
    if wallet_id:
        body["wallet_id"] = wallet_id
    resp = client.post("/chat/stream", json=body)
    assert resp.status_code == 200, resp.text
    blocks = _parse_sse_blocks(resp.text)
    prop = next((b for b in blocks if "proposal" in b.get("type", "")), None)
    assert prop is not None, (
        f"no proposal for {message!r}; got blocks={[b.get('type') for b in blocks]}"
    )
    return prop["transaction"]


async def _load_catalog():
    """Load the real catalog (same loader the agent uses) so the test can read
    the index-0 wallet (fallback when the caller sends no wallet) and each
    wallet's scoped category set."""
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        os.environ["BACKEND_DATABASE_URL"], min_size=1, max_size=1, open=False
    )
    await pool.open()
    try:
        return await load_entity_catalog(pool, SEED_USER)
    finally:
        await pool.close()


@pytest.mark.id("I1001")
@pytest.mark.parametrize(
    "message, wallet_id, txn_type",
    [
        ("เก้าอี้ 4000", WALLET_TEST, "expense"),    # Test owns เฟอร์นิเจอร์
        ("เก้าอี้ 4000", WALLET_CARDX, "expense"),   # CardX has none -> closest
        ("ข้าวเที่ยง 120", WALLET_CARDX, "expense"),
        ("เติมน้ำมัน 800", WALLET_CARDX, "expense"),
        ("เงินเดือน 50000", WALLET_CARDX, "income"),  # type-scope: income side
        # No wallet AND no caller-fixed type — the ReAct loop infers BOTH. The
        # test derives the expected wallet (index-0 fallback) and reads the
        # inferred type from the proposal, then still enforces wallet-scope +
        # type-consistency + semantic fit.
        ("ซื้อกาแฟ 60", None, None),
        ("ได้โบนัส 5000", None, None),
    ],
)
def test_I1001_category_is_wallet_and_type_scoped(client, message, wallet_id, txn_type):
    txn = _propose(client, message, wallet_id)
    catalog = asyncio.run(_load_catalog())

    # --- wallet: explicit if the caller sent one, else the index-0 fallback
    # (memory `project_wallet_index0_ordering`) ------------------------------
    expected_wallet = wallet_id or catalog.wallets[0].sync_id
    assert txn["wallet_sync_id"] == expected_wallet, (
        f"wallet mismatch: got {txn['wallet_sync_id']}, expected {expected_wallet}"
    )

    # --- type: assert the agent's classification when the caller fixed it;
    # otherwise take whatever the agent inferred and enforce self-consistency
    # against the resolved category below ------------------------------------
    if txn_type is not None:
        assert txn["type"] == txn_type, f"expected {txn_type} txn, got {txn['type']}"
    effective_type = txn_type or txn["type"]
    assert effective_type in ("expense", "income"), f"bad type {effective_type!r}"

    # --- category: scoped to this wallet (+ global) AND the SAME type --------
    entries = catalog.categories_for(expected_wallet)
    by_id = {c.sync_id: c for c in entries}
    sid = txn.get("category_sync_id")
    # v3 may emit null category when the resolver can't find a fit (Other floor
    # per memory `project_chat_null_category_other_floor`). When null we skip
    # the sid lookup — the Other-floor name match is asserted via category_name.
    if sid is not None:
        assert sid in by_id, (
            f"category_sync_id {sid} is not a category of wallet "
            f"{expected_wallet} (+global)"
        )
        # No type-leak: resolved category type == transaction type.
        assert by_id[sid].type == effective_type == txn["type"], (
            f"type-leak: {txn.get('category')!r} ({str(sid)[:8]}) is "
            f"{by_id[sid].type!r}; txn type={txn['type']!r}, "
            f"effective={effective_type!r}"
        )

    # --- LLM-as-judge: semantic fit among the wallet's options ---------------
    available = [c.name for c in entries if c.type == effective_type]
    answer = (
        f"message: {message}\n"
        f"chosen category: {txn.get('category') or txn.get('category_name')}\n"
        f"wallet's available {effective_type} categories: {available}"
    )
    verdict = asyncio.run(judge(message, answer, _RUBRIC))
    assert verdict["passed"], f"judge FAIL — {verdict['reason']}"
