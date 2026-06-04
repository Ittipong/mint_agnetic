"""Onboarding stage machine + block augmentation — pure, side-effect-free.

A user becomes "ready" only after they (1) own a general wallet and
(2) have recorded at least one transaction. Until then the agent keeps
nudging them through setup. Stages:

    no_wallet       → no general wallet yet
    no_transaction  → has a general wallet but zero transactions
    done            → ready (no nudging)

`compute_stage` derives the stage from the already-loaded entity catalog
(no extra DB query: wallet types + per-wallet usage_count are enough).
`augment_blocks` returns the extra blocks to append per stage. The
conversational invite text itself is LLM-generated (stage is injected into
the CodeAct prompt); here we only build the fixed structured blocks.

Kept pure so it's trivially unit-testable and callable from finalize.
"""

from __future__ import annotations

from typing import Any

WALLET_CREATE_TARGET = "/wallets/create"
TXN_CREATE_TARGET = "/transactions/create"

STAGE_NO_WALLET = "no_wallet"
STAGE_NO_TRANSACTION = "no_transaction"
STAGE_DONE = "done"

# Hidden marker mobile sends (exact text, normal POST /chat/stream, same thread)
# right after the user creates their first wallet from the in-chat CTA. The
# agent recognises it WITHOUT an LLM call and replies with a templated ack.
WALLET_CREATED_MARKER = "[INTENT:wallet_created]"
# Templated Thai acknowledgement streamed as answer_token (NO LLM call). Tone
# mirrors the assistant's คะ/ครับ voice used across onboarding blocks.
WALLET_CREATED_ACK = (
    "🎉 สร้างกระเป๋าเงินเรียบร้อยแล้ว! "
    "พร้อมบันทึกรายการแรก หรือถามอะไรก็ได้เลยนะคะ"
)


def compute_stage(catalog, *, has_pool: bool) -> str:
    """Derive the onboarding stage from the loaded catalog.

    `has_pool` gates real detection: with no pool (tests / no grounding) we
    return `done` so onboarding never fires on an empty-because-untested
    catalog. With a pool, an empty catalog is a genuine new user.

    - no general wallet            → no_wallet  (credit-card/goal alone still counts)
    - general wallet, zero txns    → no_transaction
    - otherwise                    → done

    Transaction count is read from `usage_count` (the catalog's wallet query
    already COUNTs non-deleted transactions per wallet), summed across ALL
    wallet types.
    """
    if not has_pool:
        return STAGE_DONE
    wallets = getattr(catalog, "wallets", None) or []
    general = [w for w in wallets
               if getattr(w, "wallet_type", "general") == "general"]
    if not general:
        return STAGE_NO_WALLET
    total_txn = sum(int(getattr(w, "usage_count", 0) or 0) for w in wallets)
    if total_txn == 0:
        return STAGE_NO_TRANSACTION
    return STAGE_DONE


# ---------------------------------------------------------------------------
# Block builders (fixed structured content; conversational text is LLM-gen)
# ---------------------------------------------------------------------------
def _has_block(blocks: list[dict[str, Any]], btype: str) -> bool:
    return any(b.get("type") == btype for b in blocks)


def _wallet_cta() -> dict[str, Any]:
    return {
        "type": "wallet_required",
        "text": "ดูเหมือนคุณยังไม่มีกระเป๋าเงินครับ "
                "สร้างกระเป๋าใบแรกเพื่อเริ่มบันทึกรายรับรายจ่ายได้เลย",
        "action": {"label": "สร้างกระเป๋าเงิน", "target": WALLET_CREATE_TARGET},
    }


def _suggestions(text: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    # Each item carries `label` plus exactly one of `send` (chip → sends the
    # text as a message) or `deeplink` (chip → navigates the client).
    return {"type": "suggestions", "text": text, "items": items}


_CHIPS_NO_TRANSACTION = [
    {"label": "บันทึกค่ากาแฟ 50 บาท", "send": "บันทึกค่ากาแฟ 50 บาท"},
    {"label": "กรอกเอง", "deeplink": TXN_CREATE_TARGET},
]
_CHIPS_DONE = [
    {"label": "เดือนนี้ใช้ไปเท่าไหร่", "send": "เดือนนี้ใช้ไปเท่าไหร่"},
    {"label": "ยอดเงินคงเหลือ", "send": "ยอดเงินคงเหลือเท่าไหร่"},
    {"label": "ใช้หมวดไหนเยอะสุด", "send": "เดือนนี้ใช้หมวดไหนเยอะสุด"},
]


def augment_blocks(
    blocks: list[dict[str, Any]],
    *,
    stage: str,
    just_completed: bool = False,
) -> list[dict[str, Any]]:
    """Return EXTRA blocks to append for the given onboarding stage.

    - no_wallet      → wallet CTA only (skipped if a worker already emitted one);
                       no suggestion chips, so the single action is "create wallet"
    - no_transaction → starter chips (record-first-transaction + manual entry)
    - done           → first-question chips ONLY on the turn setup just completed
    """
    extra: list[dict[str, Any]] = []
    if stage == STAGE_NO_WALLET:
        if not _has_block(blocks, "wallet_required"):
            extra.append(_wallet_cta())
    elif stage == STAGE_NO_TRANSACTION:
        extra.append(_suggestions("มีกระเป๋าแล้ว ลองบันทึกรายการแรกดูครับ:",
                                  _CHIPS_NO_TRANSACTION))
    elif stage == STAGE_DONE and just_completed:
        extra.append(_suggestions("ตั้งค่าครบแล้ว พร้อมใช้งาน ลองถามได้เลย:",
                                  _CHIPS_DONE))
    return extra
