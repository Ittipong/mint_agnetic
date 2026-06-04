"""M3 server tests — Wave 7b adaptation (slim derive_thread_title slice).

v2 `tests/test_m3_server.py` was a 723-line server-integration suite hitting
the v2 lifespan + build_graph + confirm endpoints. v3 ships a different
server contract (Wave 6) — `test_server.py` already covers UT-SR01..10
(SSE event ordering, route inventory, lifespan, confirm, cancel, healthz).

Overlap with existing v3 unit tests:
- UT-M3001 SSE event order -> `test_sse_adapter.py` UT-S01..S07.
- UT-M3101..104 proposal/confirm flow -> `test_server.py` UT-SR04 confirm,
  UT-SR07 cancel.
- UT-M3201..217 transcript / history blocks -> `test_server.py` UT-SR05.

What's NOT covered in any existing v3 unit test is the pure-function thread-
title derivation (`_derive_thread_title`). That function survives byte-for-
byte from v2 (it's a small string-shaping helper) and the v2 UT-M3210..215
slice asserts behaviors mobile relies on for thread display naming.

This file lifts UT-M3210..215 — the only m3_server cases whose v3 surface
exists and isn't already exercised elsewhere.
"""

from __future__ import annotations

from src.agent.server import ChatStreamRequest, _derive_thread_title


# ---------------------------------------------------------------------------
# UT-M3210 — client-supplied title wins
# ---------------------------------------------------------------------------
def test_UT_M3210_derive_title_client_value_wins():
    """A client-supplied title always wins — the web UI passes it on first
    send and the server must not override it with a derived one."""
    body = ChatStreamRequest(
        thread_id="t", user_id="u",
        message="ค่ากาแฟ 60 บาท",
        title="งบเดือนนี้",
    )
    assert _derive_thread_title(body) == "งบเดือนนี้"


# ---------------------------------------------------------------------------
# UT-M3211 — no client title -> first user message
# ---------------------------------------------------------------------------
def test_UT_M3211_derive_title_text_from_first_message():
    """No client title -> a text thread is named from its first user message."""
    body = ChatStreamRequest(
        thread_id="t", user_id="u", message="ค่ากาแฟ 60 บาท",
    )
    assert _derive_thread_title(body) == "ค่ากาแฟ 60 บาท"


# ---------------------------------------------------------------------------
# UT-M3212 — slip single-line uses the note (merchant)
# ---------------------------------------------------------------------------
def test_UT_M3212_derive_title_slip_single_uses_note():
    """Scheme A: a one-line slip is named from the line's note (merchant)."""
    body = ChatStreamRequest(
        thread_id="t", user_id="u",
        message="[INTENT:parse_transaction_from_slip]",
    )
    rows = [{"note": "Starbucks", "category": "อาหาร", "type": "expense"}]
    assert _derive_thread_title(body, slip_rows=rows, is_slip=True) == "Starbucks"


# ---------------------------------------------------------------------------
# UT-M3213 — slip default note "สลิป" falls back to category
# ---------------------------------------------------------------------------
def test_UT_M3213_derive_title_slip_default_note_falls_back_to_category():
    """Scheme A: when vision found no merchant note (the default 'สลิป'),
    name the thread from the line's category instead of the useless default."""
    body = ChatStreamRequest(
        thread_id="t", user_id="u",
        message="[INTENT:parse_transaction_from_slip]",
    )
    rows = [{"note": "สลิป", "category": "อาหาร", "type": "expense"}]
    assert _derive_thread_title(body, slip_rows=rows, is_slip=True) == "อาหาร"


# ---------------------------------------------------------------------------
# UT-M3214 — multi-row slip appends "+N"
# ---------------------------------------------------------------------------
def test_UT_M3214_derive_title_slip_multi_row_appends_count():
    """Scheme A: a multi-line slip (payslip) shows the first line + '+N'."""
    body = ChatStreamRequest(
        thread_id="t", user_id="u",
        message="[INTENT:parse_transaction_from_slip]",
    )
    rows = [
        {"note": "เงินเดือน", "category": "เงินเดือน", "type": "income"},
        {"note": "ประกันสังคม", "category": "อื่นๆ", "type": "expense"},
        {"note": "ภาษี", "category": "อื่นๆ", "type": "expense"},
    ]
    assert _derive_thread_title(body, slip_rows=rows, is_slip=True) == "เงินเดือน +2"


# ---------------------------------------------------------------------------
# UT-M3215 — no-row slip never leaks the parse marker
# ---------------------------------------------------------------------------
def test_UT_M3215_derive_title_slip_no_rows_never_uses_marker():
    """An unreadable / no-wallet slip has no parsed lines. The title must
    NOT leak the parse-marker message — it falls back to None (so the
    upsert COALESCEs to the default 'แชตใหม่')."""
    body = ChatStreamRequest(
        thread_id="t", user_id="u",
        message="[INTENT:parse_transaction_from_slip]",
    )
    assert _derive_thread_title(body, slip_rows=None, is_slip=True) is None
    assert _derive_thread_title(body, slip_rows=[], is_slip=True) is None
