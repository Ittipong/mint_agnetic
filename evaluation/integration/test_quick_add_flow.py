"""Integration tests for the quick-add (text → propose_transaction) flow.

Two intents are exercised:

1. **Add transaction** — user types a journal entry ("กินกาแฟ 100 บาท") and
   the agent must call `propose_transaction`, producing a
   `propose_transaction_group` SSE payload with `corrects_group_id == None`.
2. **Edit last transaction** — user corrects a still-pending proposal in the
   same thread ("ไม่ใช่ 100 เป็น 200", "แก้ category อาหาร -> shopping",
   etc.) and the agent must emit a NEW proposal whose `corrects_group_id`
   equals the prior proposal's `group_id`.

Scenarios S1-S4 mirror the user's hand-drawn pending-state spec:

    S1: t1 add → t2 propose → t3 SAVE
    S2: t1 add → t2 propose → t3 add → t4 propose (both pending)
    S3: S2 + SAVE t2 → t4 propose is the lone pending proposal
    S4: t1 add → t2 propose → t3 "ไม่ใช่ 100 เป็น 200" → t4 propose with
        corrects_group_id == t2.group_id  (edit, not add)

Edit-field coverage (E1-E6) covers amount, category, note, merchant,
wallet, and date — every field the user might correct on the card.

Tests require a live LangGraph server at `STUDIO_URL` (defaults to
http://localhost:8080) and a test user with at least one general wallet +
one expense category. Tests that need a second wallet auto-skip when the
catalog only has one.

Run:
    pytest mint_agentic/evaluation/integration/test_quick_add_flow.py -m live -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, timedelta
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
) -> list[dict[str, Any]]:
    """POST /chat/stream and return all `propose_transaction_group` payloads
    that arrived on the SSE stream, in order.

    Each entry is the inner `data` block (already unwrapped from
    `{"type": "propose_transaction_group", "data": {...}}`), so tests can
    read `payload["group_id"]`, `payload["transactions"]`,
    `payload["corrects_group_id"]` directly.
    """
    groups: list[dict[str, Any]] = []
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
            if payload.get("type") == "propose_transaction_group":
                groups.append(payload["data"])
    return groups


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
    expected_corrects_group_id: str | None,
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
      - `corrects_group_id` matches expectation (None for fresh, UUID for edit)
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

    if expected_corrects_group_id is None:
        assert group.get("corrects_group_id") in (None, ""), (
            f"expected fresh proposal, got corrects_group_id="
            f"{group.get('corrects_group_id')!r}"
        )
    else:
        assert group.get("corrects_group_id") == expected_corrects_group_id, (
            f"expected corrects_group_id={expected_corrects_group_id!r}, "
            f"got {group.get('corrects_group_id')!r}"
        )

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
        """S1: add → propose → save. Single proposal, no correction."""
        groups = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(groups) == 1, f"expected 1 proposal, got {len(groups)}"
        _assert_full_payload(
            groups[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=None,
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
        groups = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "เงินเดือนเข้า 30000",
        )
        assert len(groups) == 1
        _assert_full_payload(
            groups[0],
            expected_amount=30000.0,
            expected_type="income",
            catalog=live_catalog,
            expected_corrects_group_id=None,
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
        groups = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว ที่ KFC 150 บาท",
        )
        assert len(groups) == 1
        _assert_full_payload(
            groups[0],
            expected_amount=150.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=None,
            note_contains="ข้าว",
            merchant_contains="KFC",
        )


# ── Pending-state behaviour (S2, S3) ─────────────────────────────────────────


class TestPendingProposals:
    """A still-unconfirmed proposal must NOT bleed into the next add turn,
    but it MUST remain visible to the LLM in case the user corrects it."""

    async def test_S2_two_adds_both_pending(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """S2: add coffee 100 → (no save) → add rice 20. Both must be fresh
        (corrects_group_id=None) — the second add is a new transaction,
        NOT a correction. Sum-state on mobile = 2 pending cards."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1
        _assert_full_payload(
            first[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=None,
            note_contains="กาแฟ",
        )

        # NOTE: deliberately DON'T call /chat/intent. The first card stays
        # pending — the LLM still sees it in history.
        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว 20",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=20.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=None,  # NEW, not correction
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
        """S3: add → save → add. Second proposal is also fresh; saving the
        first one must NOT cause the second to reference it."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        await _send_intent(
            http_client, studio_url, default_user_id, thread_id,
            "transaction_saved", first[0]["group_id"],
        )

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว 20 บาท",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=20.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=None,
            note_contains="ข้าว",
        )


# ── Edit / correction intent (S4 + E1-E6) ────────────────────────────────────


class TestEditLastTransaction:
    """User corrects the most-recent still-pending proposal. The new
    proposal must carry `corrects_group_id` so mobile auto-discards the
    stale card. Covers every field a user might change on the card."""

    async def test_S4_correct_amount(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """S4 / E1: 'ฉันบอกผิด ไม่ใช่ 100 แต่เป็น 200' → new proposal
        with corrects_group_id = first.group_id, amount=200."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "ฉันบอกผิด ไม่ใช่ 100 แต่เป็น 200",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=200.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=first[0]["group_id"],
            note_contains="กาแฟ",
        )

    async def test_E2_correct_category(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """E2: change category. The new proposal must use a different
        category_id (still valid in the catalog) and keep amount unchanged."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "ซื้ออาหาร 200 บาท",
        )
        assert len(first) == 1
        first_cat = first[0]["transactions"][0]["category_id"]

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "ขอแก้ category จากอาหารเป็น shopping",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=200.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=first[0]["group_id"],
        )
        new_cat = second[0]["transactions"][0]["category_id"]
        assert new_cat != first_cat, (
            f"category_id should change after correction; both are {new_cat}"
        )

    async def test_E3_correct_note(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """E3: change note/name only. Amount + category unchanged, note
        reflects the new value."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "แก้ note เป็น กาแฟลาเต้",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=first[0]["group_id"],
            note_contains="ลาเต้",
        )

    async def test_E4_add_merchant(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """E4: add a merchant the original message didn't have."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "ขอแก้ merchant เป็น Starbucks",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=first[0]["group_id"],
            merchant_contains="Starbucks",
        )

    async def test_E5_correct_wallet(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """E5: switch wallet. Requires ≥2 general wallets — auto-skip
        otherwise. The new wallet_id must be different and still belong to
        the catalog."""
        wallets = live_catalog["wallets"]
        if len(wallets) < 2:
            pytest.skip(
                f"test user has {len(wallets)} general wallet(s); need ≥2"
            )
        first_wallet = wallets[0]
        second_wallet = wallets[1]

        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            f"กินกาแฟ 100 จาก {first_wallet['name']}",
        )
        assert len(first) == 1
        assert first[0]["wallet_id"] == first_wallet["sync_id"], (
            f"expected first proposal on wallet {first_wallet['name']}, "
            f"got {first[0]['wallet_id']}"
        )

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            f"ขอแก้เป็นจาก {second_wallet['name']} แทน",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=first[0]["group_id"],
        )
        assert second[0]["wallet_id"] == second_wallet["sync_id"], (
            f"expected wallet switch to {second_wallet['name']}, "
            f"got {second[0]['wallet_id']}"
        )

    async def test_E6_correct_date(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """E6: back-date a transaction to yesterday."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1
        assert first[0]["transactions"][0]["date"].startswith(
            date.today().isoformat()
        ), "first proposal should be today by default"

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "เมื่อวานนะ ไม่ใช่วันนี้",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=100.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=first[0]["group_id"],
            expected_date=yesterday,
        )

    async def test_dismissed_proposal_is_not_corrected(
        self,
        http_client: httpx.AsyncClient,
        studio_url: str,
        default_user_id: str,
        thread_id: str,
        live_catalog: dict[str, Any],
    ):
        """Edge case: user dismisses the first card, then types a NEW
        entry. The new proposal must be fresh (corrects_group_id=None) —
        a dismissed proposal is closed, not editable."""
        first = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินกาแฟ 100 บาท",
        )
        assert len(first) == 1

        await _send_intent(
            http_client, studio_url, default_user_id, thread_id,
            "transaction_dismissed", first[0]["group_id"],
        )

        second = await _send_message(
            http_client, studio_url, default_user_id, thread_id,
            "กินข้าว 50 บาท",
        )
        assert len(second) == 1
        _assert_full_payload(
            second[0],
            expected_amount=50.0,
            expected_type="expense",
            catalog=live_catalog,
            expected_corrects_group_id=None,
            note_contains="ข้าว",
        )
