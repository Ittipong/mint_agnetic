"""Integration tests for the quick-add (text → propose_transaction) flow.

Chat is **add-only**. Editing a still-pending proposal is delegated
to the mobile in-app editor: when the user types an edit-style
phrase while a card is on screen, the server dispatches an
`open_edit_sheet` SSE event (no LLM call) and mobile opens its
existing transaction editor pre-filled with the proposal. This
keeps the chat lane small and reliable.

Coverage:

1. **Add transaction** — user types "กินกาแฟ 100 บาท" → server
   emits a `propose_transaction_group` SSE payload with a fresh
   `group_id`. Asserts every field of the payload against the live
   catalog so wallet_id / category_id can't drift.

2. **Pending-state** — multiple un-confirmed adds coexist as
   independent cards (S2), and saving one doesn't poison the next
   (S3). Dismissed cards are closed (no carry-over).

3. **Open-edit-sheet redirect** — when a pending card is on screen
   and the user types an edit-style phrase, the server fires an
   `open_edit_sheet` event carrying the pending `group_id`. No new
   proposal is generated.

Tests require a live LangGraph server at `STUDIO_URL` (defaults to
http://localhost:8000) and a test user with at least one general
wallet + one expense category.

Run:
    pytest mint_agentic/evaluation/integration/test_quick_add_flow.py -m live -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date
from typing import Any

import httpx
import pytest
import pytest_asyncio

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _send_message(
    client: httpx.AsyncClient,
    studio_url: str,
    user_id: str,
    thread_id: str,
    message: str,
    *,
    currency_code: str = "THB",
    currency_symbol: str = "฿",
) -> dict[str, list[dict[str, Any]]]:
    """POST /chat/stream and bucket the structured-data events that arrive.

    Returns a dict keyed by payload `type`:
      - `propose_transaction_group`: fresh add cards
      - `open_edit_sheet`: edit-redirect events (mobile opens its
        in-app editor for the listed `group_id`)

    Each value is a list of the inner `data` blocks in arrival order.
    """
    buckets: dict[str, list[dict[str, Any]]] = {
        "propose_transaction_group": [],
        "open_edit_sheet": [],
    }
    body = {
        "user_id": user_id,
        "thread_id": thread_id,
        "message": message,
        "default_currency_code": currency_code,
        "default_currency_symbol": currency_symbol,
    }
    async with client.stream(
        "POST", f"{studio_url}/chat/stream", json=body, timeout=120
    ) as resp:
        assert resp.status_code == 200, f"chat/stream failed: {resp.status_code}"
        async for raw in resp.aiter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            try:
                event = json.loads(raw[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if event.get("type") != "data":
                continue
            payload = event.get("payload") or {}
            ptype = payload.get("type")
            if ptype in buckets:
                buckets[ptype].append(payload.get("data") or {})
    return buckets


async def _send_message_groups(
    client: httpx.AsyncClient,
    studio_url: str,
    user_id: str,
    thread_id: str,
    message: str,
) -> list[dict[str, Any]]:
    """Shortcut for tests that only care about `propose_transaction_group`
    payloads — keeps the test bodies readable."""
    buckets = await _send_message(
        client, studio_url, user_id, thread_id, message
    )
    return buckets["propose_transaction_group"]


async def _send_intent(
    client: httpx.AsyncClient,
    studio_url: str,
    user_id: str,
    thread_id: str,
    action: str,
    group_id: str,
) -> None:
    """Simulate mobile tapping Save / Discard on a proposal card.

    `action` is `transaction_saved` or `transaction_dismissed`.
    """
    resp = await client.post(
        f"{studio_url}/chat/intent",
        json={
            "user_id": user_id,
            "thread_id": thread_id,
            "action": action,
            "group_id": group_id,
        },
        timeout=30,
    )
    assert resp.status_code == 200, f"chat/intent failed: {resp.status_code} {resp.text}"


async def _delete_thread(
    client: httpx.AsyncClient,
    studio_url: str,
    user_id: str,
    thread_id: str,
) -> None:
    """Clean up a test thread + its LangGraph checkpoint. Best-effort —
    tests don't fail if cleanup fails, the next run just inherits a stale
    thread row (the per-test uuid4 thread_id keeps collisions out)."""
    try:
        await client.delete(
            f"{studio_url}/threads/{thread_id}",
            params={"user_id": user_id},
            timeout=15,
        )
    except Exception:
        pass


# Module-level cache so we only hit the DB on the very first test.
# `asyncpg` pools bind to the asyncio loop that created them, and
# pytest-asyncio spins a new loop per test function — so calling
# `fetch_user_catalog` again from a later test would deadlock on a
# dead loop. Catalog data is static for the lifetime of a test run.
_CATALOG_CACHE: dict[str, Any] | None = None


async def _fetch_catalog(user_id: str) -> dict[str, Any]:
    """Pull the live wallet + category catalog for the test user.

    Returns a dict with `wallet_ids: set[str]`, `category_ids: set[str]`,
    `wallets: list[dict]`, `categories_by_wallet: dict[wallet_id, list[cat]]`.
    Tests use this to assert that the LLM picked a sync_id that actually
    exists for this user (not a hallucinated UUID).
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    from src.entity_catalog import (
        fetch_categories_by_wallet,
        fetch_user_catalog,
    )

    catalog = await fetch_user_catalog(user_id)
    by_wallet = await fetch_categories_by_wallet(user_id)

    # `EntityCatalog` is a frozen dataclass; quick-add only surfaces
    # general wallets (slip_node.render_for_slip enforces the same
    # filter), so mirror that here.
    wallets = [
        {"sync_id": w.sync_id, "name": w.name, "currency": w.currency}
        for w in catalog.wallets
        if w.wallet_type == "general"
    ]
    wallet_ids = {w["sync_id"] for w in wallets}

    # categories_by_wallet may include entries for non-general wallets;
    # restrict to the ones we accept so assertions don't pass on a
    # category that belongs to a hidden wallet.
    filtered_by_wallet = {
        wid: cats for wid, cats in by_wallet.items() if wid in wallet_ids
    }
    category_ids: set[str] = set()
    for cats in filtered_by_wallet.values():
        for c in cats:
            category_ids.add(c["sync_id"])

    _CATALOG_CACHE = {
        "wallets": wallets,
        "wallet_ids": wallet_ids,
        "categories_by_wallet": filtered_by_wallet,
        "category_ids": category_ids,
    }
    return _CATALOG_CACHE


def _assert_full_payload(
    group: dict[str, Any],
    *,
    expected_amount: float,
    expected_type: str,
    catalog: dict[str, Any],
    expected_currency_code: str = "THB",
    expected_currency_symbol: str = "฿",
    note_contains: str | None = None,
    merchant_contains: str | None = None,
    expected_date: str | None = None,
    expected_include_in_report: bool = True,
) -> None:
    """Strict assertions on a `propose_transaction_group` payload.

    Validates:
      - `group_id` is a UUID string
      - `wallet_id` ∈ catalog
      - currency_code / currency_symbol match defaults the client sent
      - `total` lines up with the single line item's amount
      - exactly one transaction (quick-add is single-line by design)
      - every transaction field is present and consistent
    """
    # group-level
    assert "group_id" in group and group["group_id"], "missing group_id"
    uuid.UUID(group["group_id"])  # validates UUID shape

    assert group.get("wallet_id") in catalog["wallet_ids"], (
        f"wallet_id {group.get('wallet_id')!r} not in user catalog "
        f"({len(catalog['wallet_ids'])} wallets)"
    )
    assert group.get("currency_code") == expected_currency_code
    assert group.get("currency_symbol") == expected_currency_symbol

    # Quick-add ALWAYS emits exactly one line item.
    txs = group.get("transactions") or []
    assert len(txs) == 1, f"expected 1 transaction, got {len(txs)}"
    tx = txs[0]

    # per-transaction
    assert tx.get("type") == expected_type, (
        f"expected type={expected_type}, got {tx.get('type')}"
    )
    assert abs(float(tx.get("amount") or 0) - expected_amount) < 0.01, (
        f"expected amount≈{expected_amount}, got {tx.get('amount')}"
    )
    # group total = single-line amount for expense, -amount for income.
    expected_total = expected_amount if expected_type == "expense" else -expected_amount
    assert abs(float(group.get("total") or 0) - expected_total) < 0.01, (
        f"expected total≈{expected_total}, got {group.get('total')}"
    )

    assert tx.get("date"), "missing tx.date"
    if expected_date is not None:
        assert tx["date"].startswith(expected_date), (
            f"expected date starting with {expected_date}, got {tx['date']}"
        )

    assert tx.get("wallet_id") in catalog["wallet_ids"], "tx wallet_id not in catalog"
    assert tx.get("wallet_id") == group.get("wallet_id"), (
        "tx wallet_id must equal group wallet_id for quick-add"
    )
    assert tx.get("category_id") in catalog["category_ids"], (
        f"category_id {tx.get('category_id')!r} not in user catalog"
    )
    # Category must belong to the matched wallet (catalog is wallet-scoped).
    cats_for_wallet = catalog["categories_by_wallet"].get(tx["wallet_id"], [])
    cat_ids_for_wallet = {c["sync_id"] for c in cats_for_wallet}
    assert tx["category_id"] in cat_ids_for_wallet, (
        f"category_id {tx['category_id']!r} does not belong to "
        f"wallet {tx['wallet_id']!r}"
    )

    assert tx.get("currency_code") == expected_currency_code
    assert tx.get("currency_symbol") == expected_currency_symbol
    # Server emits camelCase `includeInReport` (mobile JSON convention)
    # even though the tool arg is `include_in_report`.
    assert tx.get("includeInReport") is expected_include_in_report

    # Server merges `merchant_name` tool arg into the final `note`,
    # so both note keywords AND merchant keywords are asserted against
    # the same `note` field.
    if note_contains is not None:
        note = (tx.get("note") or "").lower()
        assert note_contains.lower() in note, (
            f"expected note to contain {note_contains!r}, got {tx.get('note')!r}"
        )

    if merchant_contains is not None:
        note = (tx.get("note") or "").lower()
        assert merchant_contains.lower() in note, (
            f"expected merchant ({merchant_contains!r}) to appear in note, "
            f"got {tx.get('note')!r}"
        )


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def http_client() -> httpx.AsyncClient:
    """Shared async client. One per test so streams don't cross-contaminate."""
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture
def thread_id() -> str:
    """Fresh thread per test to keep checkpoints isolated."""
    return f"test-quickadd-{uuid.uuid4()}"


@pytest_asyncio.fixture
async def live_catalog(default_user_id: str) -> dict[str, Any]:
    """Fetch the test user's wallet + category catalog.

    Skips every test in this module if the user has no general wallets —
    quick-add can't propose anything without them.
    """
    try:
        catalog = await _fetch_catalog(default_user_id)
    except Exception as exc:
        pytest.skip(f"could not fetch catalog for test user: {exc}")
    if not catalog["wallet_ids"]:
        pytest.skip("test user has no general wallets — quick-add cannot run")
    if not catalog["category_ids"]:
        pytest.skip("test user has no categories — quick-add cannot run")
    return catalog


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_thread(
    request,
    http_client: httpx.AsyncClient,
    studio_url: str,
    default_user_id: str,
    thread_id: str,
):
    """Delete the test thread after each test so checkpoints don't pile up."""
    yield
    await _delete_thread(http_client, studio_url, default_user_id, thread_id)


# ── Add-transaction intent (S1) ──────────────────────────────────────────────


class TestAddTransactionIntent:
    """User types a journal entry → agent must call propose_transaction."""

    async def test_S1_simple_add_then_save(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """S1: add → propose → save. Single fresh proposal."""
        groups = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(groups) == 1, f"expected 1 proposal, got {len(groups)}"
        _assert_full_payload(
            groups[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            note_contains="กาแฟ",
            expected_date=date.today().isoformat(),
        )

        # Save it — should not throw and should leave thread in a clean state.
        await _send_intent(
            http_client, studio_url, default_user_id, thread_id,
            "transaction_saved", groups[0]["group_id"],
        )

    async def test_add_income(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """Income keywords (เงินเดือน) → type=income."""
        groups = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "เงินเดือนเข้า 30000",
        )
        assert len(groups) == 1
        _assert_full_payload(
            groups[0],
            expected_amount=30000.0,
            expected_type="income",
            catalog=live_catalog,
            note_contains="เงินเดือน",
        )

    async def test_add_with_merchant(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """Merchant name in prompt → propagated to merchant_name."""
        groups = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว ที่ KFC 150 บาท",
        )
        assert len(groups) == 1
        _assert_full_payload(
            groups[0],
            expected_amount=150.0,
            expected_type="expense",
            catalog=live_catalog,
            note_contains="ข้าว",
            merchant_contains="KFC",
        )


# ── Pending-state behaviour (S2, S3) ─────────────────────────────────────────


class TestPendingProposals:
    """Multiple un-confirmed cards may coexist; saving / dismissing one
    must not contaminate the next."""

    async def test_S2_two_adds_both_pending(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """S2: add coffee 100 → (no save) → add rice 20. Each must be
        an independent card — mobile shows 2 pending. Cards do not
        cross-reference each other."""
        first = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1
        _assert_full_payload(
            first[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            note_contains="กาแฟ",
        )

        # NOTE: deliberately DON'T call /chat/intent — the first card
        # stays pending. Use "เพิ่มรายการ" so the LLM clearly hears
        # this as a new add, not a clarification of the prior one.
        second = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "เพิ่มรายการ กินข้าว 20 บาท",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=20.0,
            expected_type="expense",
            catalog=live_catalog,
            note_contains="ข้าว",
        )
        assert second[0]["group_id"] != first[0]["group_id"], (
            "second proposal must have its own group_id"
        )

    async def test_S3_save_first_then_add_second(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """S3: add → save → add. Second proposal is a fresh card."""
        first = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        await _send_intent(
            http_client, studio_url, default_user_id, thread_id,
            "transaction_saved", first[0]["group_id"],
        )

        second = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว 20 บาท",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=20.0,
            expected_type="expense",
            catalog=live_catalog,
            note_contains="ข้าว",
        )

    async def test_dismissed_proposal_clears_state(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """Dismissed card is closed — next add is a normal fresh proposal."""
        first = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        await _send_intent(
            http_client, studio_url, default_user_id, thread_id,
            "transaction_dismissed", first[0]["group_id"],
        )

        second = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว 50 บาท",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=50.0,
            expected_type="expense",
            catalog=live_catalog,
            note_contains="ข้าว",
        )


# ── Edit redirect (chat-edit is delegated to mobile) ─────────────────────────


class TestEditOpensEditSheet:
    """When a proposal card is on screen and the user types an edit-style
    phrase, the server must dispatch an `open_edit_sheet` SSE event
    carrying the pending `group_id` — no new proposal, no LLM call.
    Mobile responds by opening its in-app transaction editor."""

    async def test_edit_amount_phrasing_opens_edit_sheet(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """'แก้ยอดจาก 100 เป็น 300' while pending → open_edit_sheet event,
        no new propose_transaction_group."""
        first = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1
        first_group_id = first[0]["group_id"]

        buckets = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "แก้ยอดจาก 100 เป็น 300",
        )
        assert buckets["propose_transaction_group"] == [], (
            f"expected NO new proposal, got "
            f"{len(buckets['propose_transaction_group'])}"
        )
        edits = buckets["open_edit_sheet"]
        assert len(edits) == 1, (
            f"expected 1 open_edit_sheet event, got {len(edits)}"
        )
        assert edits[0].get("group_id") == first_group_id, (
            f"open_edit_sheet must reference the pending card; "
            f"expected {first_group_id}, got {edits[0].get('group_id')}"
        )

    @pytest.mark.parametrize("phrase", [
        "ขอแก้ category เป็น shopping",
        "ผิดแล้ว เป็น 200 ต่างหาก",
        "เปลี่ยน wallet เป็น kbank",
        "ไม่ใช่ 100 เป็น 200",
        "ที่จริงเป็น 200",
    ])
    async def test_other_edit_phrasings_also_redirect(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
        phrase: str,
    ):
        """All correction keywords must trigger the open_edit_sheet
        redirect, not a new proposal — the LLM is bypassed entirely."""
        first = await _send_message_groups(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        buckets = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            phrase,
        )
        assert buckets["propose_transaction_group"] == [], (
            f"phrase {phrase!r} produced an unexpected new proposal"
        )
        assert len(buckets["open_edit_sheet"]) == 1, (
            f"phrase {phrase!r} did not trigger open_edit_sheet"
        )
        assert buckets["open_edit_sheet"][0].get("group_id") == first[0]["group_id"]

    async def test_edit_phrasing_without_pending_card_still_adds(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """Sanity: when there's NO pending card, an 'edit-ish' message
        shouldn't open an edit sheet — it should either be a normal add
        (or a normal ask-back). The redirect is gated on a real pending
        proposal."""
        buckets = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "แก้ยอดเป็น 300 บาท",
        )
        assert buckets["open_edit_sheet"] == [], (
            "open_edit_sheet must not fire without a pending card"
        )
