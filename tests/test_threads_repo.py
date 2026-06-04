"""ThreadsRepo serialization tests (Wave 7b — lifted verbatim from v2).

`_thread_out` is the single read-path mapper to the v1 ThreadOut wire shape,
so the slip-marker -> friendly-preview transform is asserted here (a pure
function — no DB needed).

Source: mint_agentic_v2/tests/test_threads_repo.py. Adjusted imports to
`src.agent.threads_repo` to match v3's import style.
"""

from __future__ import annotations

from src.agent.threads_repo import _thread_out


def _out(preview):
    return _thread_out(
        id="t1", user_id="u1", title="ค่ากาแฟ",
        created_at=None, last_active=None,
        message_count=2, last_message_preview=preview,
    )


def test_UT_TR01_slip_marker_preview_becomes_friendly():
    """A slip-only thread's latest text is the parse marker (the assistant turn
    is blocks-only with a null answer). The history list must not surface that
    raw code — it's replaced with a human-readable label."""
    out = _out("[INTENT:parse_transaction_from_slip]")
    assert out["last_message_preview"] == "📷 สลิป"


def test_UT_TR02_any_intent_marker_is_hidden():
    """Defensive: any `[INTENT:...]` marker (not just today's slip one) must
    never leak into the preview."""
    out = _out("[INTENT:something_else]")
    assert out["last_message_preview"] == "📷 สลิป"


def test_UT_TR03_normal_text_preview_passes_through():
    """A real user/assistant message is shown verbatim."""
    out = _out("ค่ากาแฟ 60 บาท")
    assert out["last_message_preview"] == "ค่ากาแฟ 60 บาท"


def test_UT_TR04_none_preview_stays_none():
    """No messages with text -> no preview (mobile renders its own empty state)."""
    out = _out(None)
    assert out["last_message_preview"] is None
