"""I1003 — slip / receipt vision parsing (v3 integration, real VISION_MODEL + DB).

Adapted from v2 `I1003_slip_vision.py`. The same end-to-end contract holds —
a slip image must yield a `transaction_proposal_group` block with wallet- and
type-scoped categories — but the v3 plumbing routes through the dedicated
`endpoints/slip_handler.py` (decision α: slip flow skips ReAct entirely; one
vision call → one group block, per `docs/v3/phase2_state_and_graph.md`).

Wire:
  POST /chat/stream with `image_b64s = [<base64>]` → SSE stream containing a
  `block` event whose payload is the group block. The handler emits exactly
  one group block per readable slip; the v3 mobile decoder reads
  `block["proposal_id"] == block["group_id"]` for history replay (memory
  `project_slip_vision_as_node`).

NOT hermetic — needs `OPENROUTER_API_KEY`, `DATABASE_URL`,
`BACKEND_DATABASE_URL`, `VISION_MODEL`, and the committed slip fixtures.
Skipped cleanly when any is missing so the offline unit run stays green.

Two layers per case:
  * Deterministic invariants (asserts):
      - exactly one transaction_proposal_group block emitted,
      - group_id == proposal_id (same memory),
      - every proposal row has amount > 0 and a valid expense/income type,
      - the wallet is fixed to the client's selection,
      - the chosen category_sync_id (when not null) is a real category of
        THAT wallet (+ global) — never a hallucinated or cross-wallet id.
  * LLM-as-judge (fuzzy): each row's category fits its note among the
    wallet's available categories.

Plus a reject case: a content-free image takes the unreadable path
(`SLIP_UNREADABLE_MESSAGE` returned as an `answer` block).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.agent.endpoints.slip_handler import SLIP_UNREADABLE_MESSAGE
from src.agent.entity_catalog import load_entity_catalog
from src.agent.llm_judge import judge
from src.agent.server import app


SEED_USER = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
SLIP_MARKER = "[INTENT:parse_transaction_from_slip]"

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "slips"
_SLIP_FILES = (
    sorted(p.name for p in _FIXTURE_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if _FIXTURE_DIR.exists()
    else []
)

# A 1×1 transparent PNG — content-free, must be classified unreadable.
_BLANK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_REQUIRED_ENV = ("OPENROUTER_API_KEY", "DATABASE_URL", "BACKEND_DATABASE_URL", "VISION_MODEL")
pytestmark = pytest.mark.skipif(
    not all(os.getenv(k) for k in _REQUIRED_ENV) or not _SLIP_FILES,
    reason=(
        f"integration test needs env {', '.join(_REQUIRED_ENV)} + slip fixtures in "
        f"{_FIXTURE_DIR}"
    ),
)

_RUBRIC = (
    "You grade a category chosen by a slip-parsing assistant for one line of a "
    "receipt/slip.\n"
    "PASS if the chosen category is a reasonable fit for the line's note, picked "
    "from ONLY the wallet's available categories listed in the answer.\n"
    "- An exact or clearly-related category passes.\n"
    "- A general category ('ช้อปปิ้ง'/'อาหาร'/'อื่นๆ') passes when nothing exact fits.\n"
    "- 'other'/null is acceptable when nothing fits.\n"
    "- An unrelated specific category fails (e.g. coffee -> fuel)."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def catalog():
    return asyncio.run(_load_catalog())


@pytest.fixture(scope="module")
def target_wallet(catalog):
    """A REAL wallet the slip handler will pick. Index-0 fallback is the
    handler's contract when wallet_id is None."""
    w = catalog.slip_wallet(None)
    assert w is not None, "SEED_USER has no wallets — cannot run slip integration"
    return w


def _b64(name: str) -> str:
    return base64.b64encode((_FIXTURE_DIR / name).read_bytes()).decode("ascii")


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


def _parse_sse(raw: str) -> list[dict]:
    """Return the list of `event: block` JSON payloads from an SSE response."""
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
            data = "".join(data_parts)
            try:
                blocks.append(json.loads(data))
            except json.JSONDecodeError:
                continue
    return blocks


def _slip_blocks(client: TestClient, image_b64: str, wallet_id: str | None) -> list[dict]:
    """POST /chat/stream with an attached image and return emitted blocks."""
    body: dict = {
        "thread_id": str(uuid.uuid4()),
        "user_id": SEED_USER,
        "message": SLIP_MARKER,
        "image_b64s": [image_b64],
    }
    if wallet_id:
        body["wallet_id"] = wallet_id
    resp = client.post("/chat/stream", json=body)
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text)


def _signed_total(rows: list[dict]) -> float:
    """Σexpense − Σincome — the slip group card's `total` convention."""
    from decimal import Decimal

    t = sum(
        (
            Decimal(str(r["amount"]))
            * (Decimal(-1) if r["type"] == "income" else Decimal(1))
            for r in rows
        ),
        Decimal(0),
    )
    return float(t)


@pytest.mark.id("I1003")
@pytest.mark.parametrize("slip_file", _SLIP_FILES)
def test_I1003_slip_produces_valid_scoped_group(client, catalog, target_wallet, slip_file):
    wallet_id = target_wallet.sync_id
    blocks = _slip_blocks(client, _b64(slip_file), wallet_id)
    groups = [b for b in blocks if b.get("type") == "transaction_proposal_group"]
    assert groups, (
        f"{slip_file}: no group; got blocks={[b.get('type') for b in blocks]}"
    )
    assert len(groups) == 1, f"{slip_file}: expected ONE group, got {len(groups)}"
    group = groups[0]

    rows = group.get("transactions") or []
    assert rows, f"{slip_file}: group has no rows"
    assert group.get("group_id"), f"{slip_file}: group missing group_id"

    # group_id == proposal_id — memory project_slip_vision_as_node. The mobile
    # history-replay path keys status by proposal_id; the slip handler emits
    # both with the same value on purpose.
    assert group.get("proposal_id") == group.get("group_id"), (
        f"{slip_file}: proposal_id ({group.get('proposal_id')!r}) must equal "
        f"group_id ({group.get('group_id')!r})"
    )

    # Group total == Σexpense − Σincome of rows.
    assert group["total"] == pytest.approx(_signed_total(rows)), (
        f"{slip_file}: total {group['total']} != signed row sum {_signed_total(rows)}"
    )

    by_id = {c.sync_id: c for c in catalog.categories_for(wallet_id)}

    for txn in rows:
        assert isinstance(txn["amount"], (int, float)) and txn["amount"] > 0, txn
        assert txn["type"] in ("expense", "income"), txn
        assert txn["wallet_sync_id"] == wallet_id, txn

        sid = txn.get("category_sync_id")
        if sid is not None:
            assert sid in by_id, (
                f"{slip_file}: category {sid} not in wallet {wallet_id} (+global)"
            )
            assert by_id[sid].type == txn["type"], (
                f"{slip_file}: type-leak {txn.get('category')!r} is "
                f"{by_id[sid].type!r} but txn is {txn['type']!r}"
            )

            available = [c.name for c in by_id.values() if c.type == txn["type"]]
            answer = (
                f"slip line note: {txn.get('note')}\n"
                f"chosen category: {txn.get('category')}\n"
                f"wallet's available {txn['type']} categories: {available}"
            )
            verdict = asyncio.run(judge(str(txn.get("note") or ""), answer, _RUBRIC))
            assert verdict["passed"], f"{slip_file}: judge FAIL — {verdict['reason']}"


@pytest.mark.id("I1003")
def test_I1003_blank_image_takes_unreadable_path(client, target_wallet):
    """A content-free image must take the unreadable path: no proposal, the
    exact Thai sentence rendered as an `answer` block."""
    blocks = _slip_blocks(client, _BLANK_PNG_B64, target_wallet.sync_id)
    assert not [b for b in blocks if b.get("type") in ("transaction_proposal", "transaction_proposal_group")], (
        f"blank image produced a proposal: {blocks}"
    )
    answers = [b for b in blocks if b.get("type") == "answer"]
    assert answers, f"no answer block for blank image; got {blocks}"
    assert any(SLIP_UNREADABLE_MESSAGE in (a.get("text") or "") for a in answers), (
        f"missing unreadable sentence — got answer texts: "
        f"{[a.get('text') for a in answers]}"
    )
