"""Block validator + SSE wire serializer for mobile-facing chat blocks.

NEW in Wave 5. Sourced from `docs/v3/phase2_streaming_adapter.md` §6 and the
frozen mobile decoder at
`mobile/lib/data/datasources/remote/api/chat_api_datasource.dart::decodeBlockJson`.

Nine allowed block types — `hide_block` is DEPRECATED (the mobile decoder
still handles it for back-compat with old buffered payloads, but v3 emits
`discard_proposal` instead, per memory `project_chat_edit_repropose`).

Validation philosophy (A3 Hybrid, locked in Wave 5 kickoff):
  - `clarification`: `text` REQUIRED ALWAYS; when `option_kind == "wallet"`
    `options` ALSO REQUIRED (the wallet picker is the only typed option_kind
    the mobile decoder special-cases). Any other clarification — generic
    text-only prompts — needs only `text`.
  - All other types: required keys per the mobile decoder switch cases.

A failed validation is a HARD DROP (returns False) — never a silent rewrite.
Malformed blocks are a production bug, not a user-facing surprise: the mobile
decoder would either render gibberish or drop the block silently, masking the
bug. We log + drop here at the server boundary so the trace shows it.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# 1. Allowed block types — mirrors the mobile decoder switch cases.
# ---------------------------------------------------------------------------

# Order matches the documented mapping table in
# `docs/v3/phase2_streaming_adapter.md` §3 (re-ordered for readability,
# functionally a set — exposed as a list so callers can iterate predictably
# in tests / docs).
EMITTABLE_BLOCK_TYPES: list[str] = [
    "answer",
    "clarification",
    "wallet_required",
    "suggestions",
    "transaction_proposal",
    "transaction_proposal_group",
    "discard_proposal",
    "transcript",
    "stt_error",
    "category_breakdown",
]

# Membership lookup — convert to a frozenset once for O(1) `in`. Tests
# iterate the list; runtime uses the set.
_EMITTABLE_BLOCK_SET: frozenset[str] = frozenset(EMITTABLE_BLOCK_TYPES)


# ---------------------------------------------------------------------------
# 2. Required-key map per block type.
# ---------------------------------------------------------------------------

# Shallow required-key map. Type-specific deep checks live in `validate_block`
# (see `transaction_proposal.transaction`, `wallet_required.action`, etc.).
_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "answer": frozenset({"text"}),
    # `clarification` text-only check; the wallet-picker conditional rule is
    # implemented in `validate_block` directly (A3 Hybrid).
    "clarification": frozenset({"text"}),
    "wallet_required": frozenset({"text", "action"}),
    "suggestions": frozenset({"items"}),
    "transaction_proposal": frozenset({"proposal_id", "transaction"}),
    "transaction_proposal_group": frozenset(
        {"group_id", "total", "transactions"}
    ),
    "discard_proposal": frozenset({"target"}),
    "transcript": frozenset({"text"}),
    "stt_error": frozenset({"reason"}),
    # `category_breakdown`: the card needs a heading + at least the items
    # array. `total`/`period_label` and per-item `icon`/`diff`/`pct` are
    # optional (single-period breakdowns omit diff/pct). Deep per-item checks
    # live in `validate_block`.
    "category_breakdown": frozenset({"title", "items"}),
}


# Transaction inner-payload keys the mobile proposal card reads. Missing any
# of these and the card renders an "incomplete" placeholder; we reject up-
# front so the failure surfaces as a server-side drop, not a mobile crash.
_TRANSACTION_INNER_KEYS: frozenset[str] = frozenset(
    {"sync_id", "type", "amount", "wallet_sync_id"}
)


# ---------------------------------------------------------------------------
# 3. Validator
# ---------------------------------------------------------------------------


def validate_block(block: Any) -> bool:
    """Return True iff `block` matches the mobile-facing schema for its type.

    Conservative: missing required keys → False. Unknown type → False.
    Malformed blocks are dropped (caller logs); we never attempt to repair.

    A3 Hybrid rule:
      - `clarification.text` always required.
      - `clarification.option_kind == "wallet"` additionally requires
        `options` to be a non-empty list (the mobile decoder falls back to
        plain text otherwise — that fallback is a silent UX bug we'd rather
        surface as a server-side drop).
    """
    if not isinstance(block, dict):
        return False

    btype = block.get("type")
    if btype not in _EMITTABLE_BLOCK_SET:
        return False

    required = _REQUIRED_KEYS[btype]
    if not required.issubset(block.keys()):
        return False

    # ── Type-specific deep checks ─────────────────────────────────────────
    if btype == "answer":
        if not isinstance(block.get("text"), str):
            return False

    elif btype == "clarification":
        if not isinstance(block.get("text"), str):
            return False
        # A3 Hybrid — wallet picker MUST carry usable options.
        if block.get("option_kind") == "wallet":
            opts = block.get("options")
            if not isinstance(opts, list) or not opts:
                return False
            # Each option needs at least a sync_id + name (mobile chip needs
            # both; the icon is optional and falls back to a default glyph).
            for opt in opts:
                if not isinstance(opt, dict):
                    return False
                if not opt.get("sync_id") or not opt.get("name"):
                    return False

    elif btype == "wallet_required":
        action = block.get("action")
        if (
            not isinstance(action, dict)
            or "label" not in action
            or "target" not in action
        ):
            return False

    elif btype == "suggestions":
        items = block.get("items")
        if not isinstance(items, list) or not items:
            return False
        for item in items:
            # v2 mobile decoder accepts legacy plain strings OR dicts with
            # `send` (and optionally `label`). We accept both shapes.
            if isinstance(item, str):
                if not item.strip():
                    return False
                continue
            if not isinstance(item, dict):
                return False
            if not (item.get("send") or item.get("label")):
                return False

    elif btype == "transaction_proposal":
        txn = block.get("transaction")
        if not isinstance(txn, dict):
            return False
        if not _TRANSACTION_INNER_KEYS.issubset(txn.keys()):
            return False
        # `proposal_id` is the confirm-key; mobile drops the card without it.
        if not block.get("proposal_id"):
            return False

    elif btype == "transaction_proposal_group":
        txns = block.get("transactions")
        if not isinstance(txns, list) or not txns:
            return False
        # `group_id` doubles as the single proposal_id confirm key (per
        # memory `project_slip_vision_as_node`); the mobile group card
        # silently fails to render without it.
        if not block.get("group_id"):
            return False

    elif btype == "discard_proposal":
        target = block.get("target")
        if not isinstance(target, str) or not target.strip():
            return False

    elif btype == "stt_error":
        reason = block.get("reason")
        # Mobile decoder narrows to enum (`no_speech` | `too_short` |
        # `transcription_failed`); we accept any non-empty string here so a
        # future reason code doesn't require both ends to ship in lockstep.
        if not isinstance(reason, str) or not reason.strip():
            return False

    elif btype == "category_breakdown":
        items = block.get("items")
        # `items` must be a non-empty list of well-formed item dicts. An empty
        # or single-item breakdown should never reach here (the emitter guards
        # ≥2 upstream), but a malformed payload is still a HARD DROP so the bug
        # surfaces server-side instead of as a blank/crashing mobile card.
        if not isinstance(items, list) or not items:
            return False
        for it in items:
            if not isinstance(it, dict):
                return False
            name = it.get("name")
            if not isinstance(name, str) or not name.strip():
                return False
            amount = it.get("amount")
            # bool is an int subclass — reject it explicitly so a stray True
            # never passes as an amount. Numbers only (Decimal already coerced
            # to int/float at the wire edge); strings are rejected.
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                return False
            # icon/diff/pct are OPTIONAL. When present, diff/pct must be
            # numbers (icon is raw JSONB — any shape the mobile decoder reads).
            for opt_key in ("diff", "pct"):
                if opt_key in it:
                    v = it[opt_key]
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        return False

    # transcript / answer already covered by required-key + text type check.
    return True


# ---------------------------------------------------------------------------
# 4. SSE wire serializer
# ---------------------------------------------------------------------------


def emit_block(block: dict) -> str:
    """Serialize one block dict to the exact SSE wire string mobile expects.

    Format matches v2 `server.py` (see line ~541, ~620): `event: block` line,
    one `data:` line carrying the JSON payload, terminated by a blank line.
    `sse_starlette.EventSourceResponse` adds the outer CRLF framing when
    Wave 6's server yields the `{"event", "data"}` dict; this helper is for
    the slip / voice endpoints that build the wire string directly (they
    don't go through the dict-based generator path).

    Raises ValueError on a malformed block — slip/voice handlers MUST catch
    this and emit a `stt_error` / fallback `answer` block (silent drops mask
    production bugs per `docs/v3/phase2_streaming_adapter.md` §8).
    """
    if not validate_block(block):
        raise ValueError(
            f"emit_block: rejected malformed block (type={block.get('type')!r})"
        )
    # `ensure_ascii=False` keeps Thai characters readable in the wire trace;
    # the mobile decoder reads UTF-8 bytes.
    payload = json.dumps(block, ensure_ascii=False)
    # CRLF framing matches v2 server.py output via sse_starlette. We emit
    # the same `\n\n` delimiter v2 used; the mobile decoder normalizes
    # CRLF/LF differences on its end (memory `project_sse_ngrok_framing`).
    return f"event: block\ndata: {payload}\n\n"


__all__ = ["EMITTABLE_BLOCK_TYPES", "validate_block", "emit_block"]
