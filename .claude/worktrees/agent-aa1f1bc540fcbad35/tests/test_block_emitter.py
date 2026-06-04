"""Unit tests for `src.agent.streaming.block_emitter`.

Covers UT-BE01..BE05. The validator is the single gate that protects the
mobile decoder from malformed payloads; bugs here cascade into silent
card-drops on the app, so the test surface here is deliberately thorough.
"""

from __future__ import annotations

import json

import pytest

from src.agent.streaming.block_emitter import (
    EMITTABLE_BLOCK_TYPES,
    emit_block,
    validate_block,
)


# ---------------------------------------------------------------------------
# Test fixtures — minimal valid payloads per block type
# ---------------------------------------------------------------------------


def _txn_inner() -> dict:
    return {
        "sync_id": "sync-1",
        "type": "expense",
        "amount": 120.0,
        "wallet_sync_id": "wallet-1",
        "category_sync_id": "cat-1",
        "note": "ค่ากาแฟ",
        "date": "2026-05-29",
        "currency_code": "THB",
    }


VALID_BLOCKS: dict[str, dict] = {
    "answer": {"type": "answer", "text": "สวัสดีครับ"},
    "clarification": {"type": "clarification", "text": "เป็นรายจ่ายอะไรครับ"},
    "wallet_required": {
        "type": "wallet_required",
        "text": "ยังไม่มีบัญชี สร้างก่อนนะครับ",
        "action": {"label": "สร้างบัญชี", "target": "create_wallet"},
    },
    "suggestions": {
        "type": "suggestions",
        "items": [{"label": "ดูยอดเดือนนี้", "send": "ดูยอดเดือนนี้"}],
    },
    "transaction_proposal": {
        "type": "transaction_proposal",
        "proposal_id": "prop-1",
        "transaction": _txn_inner(),
        "low_confidence": False,
    },
    "transaction_proposal_group": {
        "type": "transaction_proposal_group",
        "proposal_id": "group-1",
        "group_id": "group-1",
        "total": 350.0,
        "wallet_sync_id": "wallet-1",
        "currency_code": "THB",
        "low_confidence": False,
        "transactions": [_txn_inner()],
    },
    "discard_proposal": {"type": "discard_proposal", "target": "prop-1"},
    "transcript": {"type": "transcript", "text": "กาแฟห้าสิบบาท"},
    "stt_error": {"type": "stt_error", "reason": "no_speech"},
    "category_breakdown": {
        "type": "category_breakdown",
        "title": "เปรียบเทียบรายหมวด",
        "period_label": "เดือนนี้ vs เดือนที่แล้ว",
        "total": 4870,
        "items": [
            {
                "name": "อาหาร",
                "amount": 850,
                "icon": {"type": "asset", "value": "food.png", "bg": "0xFFF"},
                "diff": 850,
                "pct": 17.5,
            },
            {"name": "เดินทาง", "amount": 4020, "icon": None},
        ],
    },
}


# ---------------------------------------------------------------------------
# UT-BE01 — all 9 emittable block types validate with a valid payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("btype", EMITTABLE_BLOCK_TYPES)
def test_UT_BE01_validates_all_block_types(btype: str) -> None:
    """UT-BE01: every type in EMITTABLE_BLOCK_TYPES has a valid example
    payload that passes validate_block. Catches regressions where a new
    type is added to the list but the validator forgets to require its
    keys."""
    assert btype in VALID_BLOCKS, (
        f"VALID_BLOCKS missing example for {btype}; add one to keep the "
        "test surface complete"
    )
    block = VALID_BLOCKS[btype]
    assert validate_block(block) is True, (
        f"validate_block rejected valid {btype} payload: {block}"
    )


# ---------------------------------------------------------------------------
# UT-BE02 — rejects unknown block type
# ---------------------------------------------------------------------------


def test_UT_BE02_rejects_unknown_type() -> None:
    """UT-BE02: a block whose `type` is not in EMITTABLE_BLOCK_TYPES is
    rejected. Mobile would log + drop these silently; we surface them at
    the server boundary so the trace shows the bug."""
    block = {"type": "hide_block", "target": "prop-1"}  # deprecated in v3
    assert validate_block(block) is False

    block = {"type": "split_proposal", "items": []}  # never supported
    assert validate_block(block) is False

    # Missing type altogether.
    assert validate_block({"text": "hello"}) is False
    # Non-dict input.
    assert validate_block("not a dict") is False
    assert validate_block(None) is False


# ---------------------------------------------------------------------------
# UT-BE03 — rejects missing required keys (per block type)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "btype,missing_key",
    [
        ("answer", "text"),
        ("clarification", "text"),
        ("wallet_required", "text"),
        ("wallet_required", "action"),
        ("suggestions", "items"),
        ("transaction_proposal", "proposal_id"),
        ("transaction_proposal", "transaction"),
        ("transaction_proposal_group", "group_id"),
        ("transaction_proposal_group", "transactions"),
        ("discard_proposal", "target"),
        ("transcript", "text"),
        ("stt_error", "reason"),
        ("category_breakdown", "title"),
        ("category_breakdown", "items"),
    ],
)
def test_UT_BE03_rejects_missing_required_key(
    btype: str, missing_key: str
) -> None:
    """UT-BE03: removing a required key from any block type causes
    rejection. Parametrized so every required-key entry in the validator
    map has at least one regression guard."""
    valid = VALID_BLOCKS[btype]
    broken = {k: v for k, v in valid.items() if k != missing_key}
    assert validate_block(broken) is False, (
        f"validate_block did not reject {btype} missing {missing_key!r}: "
        f"{broken}"
    )


# ---------------------------------------------------------------------------
# UT-BE04 — emit_block produces correct SSE wire format with Thai pass-through
# ---------------------------------------------------------------------------


def test_UT_BE04_emit_block_wire_format() -> None:
    """UT-BE04: emit_block returns the exact SSE wire string mobile
    decoder expects — `event: block\\ndata: <json>\\n\\n` — and
    `ensure_ascii=False` keeps Thai characters readable.

    Format matches v2 server.py line 541:
        yield {"event": "block", "data": json.dumps(block, ensure_ascii=False)}
    which sse_starlette serializes as `event: block\\r\\ndata: <payload>\\r\\n\\r\\n`.
    We emit `\\n\\n` here because the mobile decoder normalizes CRLF/LF
    differences on its end (memory project_sse_ngrok_framing).
    """
    block = {"type": "answer", "text": "ยอด 1,234 บาท"}
    wire = emit_block(block)

    assert wire.startswith("event: block\n")
    assert wire.endswith("\n\n")
    # Extract the data line and verify JSON + Thai pass-through.
    lines = wire.split("\n")
    assert lines[0] == "event: block"
    assert lines[1].startswith("data: ")
    payload_str = lines[1][len("data: "):]
    decoded = json.loads(payload_str)
    assert decoded == block
    # Thai characters must NOT be escape-encoded (\\uXXXX would break
    # mobile decoder's expectations for length-based truncation).
    assert "ยอด" in payload_str
    assert "\\u" not in payload_str

    # Rejects malformed block.
    with pytest.raises(ValueError):
        emit_block({"type": "bogus", "x": 1})


# ---------------------------------------------------------------------------
# UT-BE05 — A3 Hybrid clarification rules
# ---------------------------------------------------------------------------


def test_UT_BE05_A3_clarification_no_option_kind_passes() -> None:
    """UT-BE05a (A3): clarification with only `text` (no option_kind) is
    valid. Other option_kind values (none currently used) need only text
    per the A3 Hybrid spec."""
    block = {"type": "clarification", "text": "ยอดเงินเท่าไรครับ"}
    assert validate_block(block) is True


def test_UT_BE05_A3_wallet_clarification_without_options_rejected() -> None:
    """UT-BE05b (A3): clarification with `option_kind=="wallet"` BUT no
    `options` list is REJECTED. Mobile silently falls back to plain text
    when `options` is absent — that fallback is a UX bug we'd rather
    surface at the server boundary."""
    block = {
        "type": "clarification",
        "text": "เลือกบัญชีไหนดี",
        "option_kind": "wallet",
    }
    assert validate_block(block) is False

    # Empty list also fails — must be non-empty.
    block_empty = {**block, "options": []}
    assert validate_block(block_empty) is False

    # Options missing sync_id or name fail (mobile chip needs both).
    block_partial = {**block, "options": [{"sync_id": "w-1"}]}
    assert validate_block(block_partial) is False
    block_partial2 = {**block, "options": [{"name": "เงินสด"}]}
    assert validate_block(block_partial2) is False


def test_UT_BE05_A3_wallet_clarification_with_options_passes() -> None:
    """UT-BE05c (A3): clarification with `option_kind=="wallet"` AND
    well-formed `options` is valid. Icon is optional — only sync_id +
    name are required per the mobile decoder."""
    block = {
        "type": "clarification",
        "text": "เลือกบัญชีไหนดี",
        "option_kind": "wallet",
        "options": [
            {"sync_id": "w-1", "name": "เงินสด"},
            {
                "sync_id": "w-2",
                "name": "บัตรเครดิต",
                "icon": '{"type":"asset","value":"x.png"}',
            },
        ],
    }
    assert validate_block(block) is True


# ---------------------------------------------------------------------------
# UT-BE06 — category_breakdown deep validation
# ---------------------------------------------------------------------------


def test_UT_BE06_category_breakdown_valid_single_period() -> None:
    """UT-BE06a: a single-period breakdown (no diff/pct) is valid — those
    fields are optional. icon may be a raw JSONB dict or omitted/None."""
    block = {
        "type": "category_breakdown",
        "title": "หมวดที่ใช้จ่าย",
        "total": 1700,
        "items": [
            {"name": "อาหาร", "amount": 900,
             "icon": {"type": "asset", "value": "food.png"}},
            {"name": "เดินทาง", "amount": 800},  # icon omitted entirely
        ],
    }
    assert validate_block(block) is True


def test_UT_BE06_category_breakdown_valid_compare() -> None:
    """UT-BE06b: a 2-period breakdown with diff/pct on each item is valid."""
    block = VALID_BLOCKS["category_breakdown"]
    assert validate_block(block) is True


def test_UT_BE06_category_breakdown_rejects_empty_items() -> None:
    """UT-BE06c: empty/non-list items → hard drop. The emitter guards ≥2
    upstream, so an empty list reaching here is a bug to surface, not hide."""
    assert validate_block(
        {"type": "category_breakdown", "title": "x", "items": []}
    ) is False
    assert validate_block(
        {"type": "category_breakdown", "title": "x", "items": "อาหาร"}
    ) is False


def test_UT_BE06_category_breakdown_rejects_item_missing_name() -> None:
    """UT-BE06d: an item without a non-empty string `name` → reject. The
    mobile card renders the name as the row label; a missing/blank name is
    a malformed payload."""
    block = {
        "type": "category_breakdown",
        "title": "x",
        "items": [{"amount": 100}, {"name": "อาหาร", "amount": 200}],
    }
    assert validate_block(block) is False

    block_blank = {
        "type": "category_breakdown",
        "title": "x",
        "items": [{"name": "  ", "amount": 100}, {"name": "ก", "amount": 1}],
    }
    assert validate_block(block_blank) is False


def test_UT_BE06_category_breakdown_rejects_bad_amount_type() -> None:
    """UT-BE06e: `amount` must be a real number — string amounts (the
    classic Decimal-not-coerced bug) and bool are rejected so the mobile
    decoder never receives a non-numeric amount."""
    str_amount = {
        "type": "category_breakdown",
        "title": "x",
        "items": [{"name": "อาหาร", "amount": "900"},
                  {"name": "เดินทาง", "amount": 800}],
    }
    assert validate_block(str_amount) is False

    bool_amount = {
        "type": "category_breakdown",
        "title": "x",
        "items": [{"name": "อาหาร", "amount": True},
                  {"name": "เดินทาง", "amount": 800}],
    }
    assert validate_block(bool_amount) is False


def test_UT_BE06_category_breakdown_rejects_bad_diff_pct_type() -> None:
    """UT-BE06f: when present, diff/pct must be numbers — a string slips a
    non-numeric into a field the mobile card treats as a delta number."""
    bad_diff = {
        "type": "category_breakdown",
        "title": "x",
        "items": [{"name": "อาหาร", "amount": 900, "diff": "+100"},
                  {"name": "เดินทาง", "amount": 800}],
    }
    assert validate_block(bad_diff) is False

    bad_pct = {
        "type": "category_breakdown",
        "title": "x",
        "items": [{"name": "อาหาร", "amount": 900, "pct": "17%"},
                  {"name": "เดินทาง", "amount": 800}],
    }
    assert validate_block(bad_pct) is False


def test_UT_BE06_category_breakdown_emit_block_serializes() -> None:
    """UT-BE06g: a valid breakdown round-trips through emit_block to the SSE
    wire string (Thai preserved, `event: block` framing)."""
    wire = emit_block(VALID_BLOCKS["category_breakdown"])
    assert wire.startswith("event: block\ndata: ")
    assert "category_breakdown" in wire
    assert "อาหาร" in wire  # ensure_ascii=False keeps Thai readable
