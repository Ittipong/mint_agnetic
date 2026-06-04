"""Unit tests for src.agent.utils.safe_json.

Covers UT-U01 — the safe_json tolerant extractor must round-trip:
- plain JSON
- fenced ```json blocks
- noisy LLM output with leading commentary
- pass-through Thai content
- explicit failure modes (unparseable, no JSON found)

Lifted from v2's contract for `_safe_json` (no dedicated v2 test existed;
behaviour is exercised via integration tests). v3 makes it a first-class
unit boundary because the validator + every endpoint handler reuses it.
"""

from __future__ import annotations

import pytest

from src.agent.utils.safe_json import safe_json


class TestSafeJsonHappyPath:
    def test_UT_U01_a_plain_json_round_trips_thai(self):
        """UT-U01a: bare JSON with Thai values parses untouched."""
        out = safe_json('{"category": "ค่าอาหาร", "amount": 250}')
        assert out == {"category": "ค่าอาหาร", "amount": 250}

    def test_UT_U01_b_strips_json_fence(self):
        """UT-U01b: triple-backtick `json` fence is stripped before parse."""
        raw = '```json\n{"amount": 250}\n```'
        assert safe_json(raw) == {"amount": 250}

    def test_UT_U01_c_strips_plain_triple_backtick(self):
        """UT-U01c: plain ``` fences (no language tag) also strip."""
        raw = '```\n{"amount": 99}\n```'
        assert safe_json(raw) == {"amount": 99}

    def test_UT_U01_d_extracts_object_from_leading_commentary(self):
        """UT-U01d: LLM preface like 'Here is the JSON:' before the object."""
        raw = 'Sure, here is the result: {"category": "ค่าน้ำ", "amount": 99}'
        assert safe_json(raw) == {"category": "ค่าน้ำ", "amount": 99}

    def test_UT_U01_e_handles_nested_braces_in_strings(self):
        """UT-U01e: balanced-brace scanner must skip braces inside strings."""
        raw = '{"note": "lunch {with} colleagues", "amount": 250}'
        assert safe_json(raw) == {"note": "lunch {with} colleagues", "amount": 250}


class TestSafeJsonFailureModes:
    def test_UT_U01_f_raises_on_no_object(self):
        """UT-U01f: no `{` anywhere -> ValueError (loud, not silent)."""
        with pytest.raises(ValueError):
            safe_json("just some prose, no JSON here")

    def test_UT_U01_g_raises_on_unbalanced_braces(self):
        """UT-U01g: opening `{` without closing -> ValueError."""
        with pytest.raises(ValueError):
            safe_json('{"unterminated": "value"')
