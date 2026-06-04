"""Unit tests for src.agent.tools.codeact.exceptions.

Covers UT-E01 — sandbox-friendly exceptions must carry the structured
fields the run_python wrapper relies on (query, kind, candidates,
question, options) for control-flow + tool-result surfacing.
"""

from __future__ import annotations

import pytest

from src.agent.tools.codeact.exceptions import (
    AmbiguousMatchError,
    ClarificationNeeded,
)


class TestAmbiguousMatchError:
    def test_UT_E01_a_constructor_captures_fields(self):
        """UT-E01a: AmbiguousMatchError exposes query/kind/candidates as
        attributes so the LLM error-feedback string can be rebuilt."""
        err = AmbiguousMatchError(
            query="กาแฟ",
            kind="category",
            candidates=["food", "drink"],
        )
        assert err.query == "กาแฟ"
        assert err.kind == "category"
        assert err.candidates == ["food", "drink"]

    def test_UT_E01_b_is_value_error_subclass(self):
        """UT-E01b: subclass of ValueError so generic except-clauses
        in resolver wrappers catch it."""
        with pytest.raises(ValueError):
            raise AmbiguousMatchError("q", "wallet", ["w1", "w2"])

    def test_UT_E01_c_message_lists_candidates(self):
        """UT-E01c: str(err) shows kind/query/candidates so the LLM
        observation gets the disambiguation context."""
        err = AmbiguousMatchError("กาแฟ", "category", ["food", "drink"])
        message = str(err)
        assert "category" in message
        assert "กาแฟ" in message
        assert "food" in message and "drink" in message


class TestClarificationNeeded:
    def test_UT_E01_d_default_options_empty_list(self):
        """UT-E01d: options defaults to [] when omitted (not None)."""
        err = ClarificationNeeded("คุณหมายถึง wallet ไหน?")
        assert err.question == "คุณหมายถึง wallet ไหน?"
        assert err.options == []

    def test_UT_E01_e_carries_options(self):
        """UT-E01e: options list is stored verbatim for the wallet_required
        block / clarification block emit step."""
        err = ClarificationNeeded(
            question="เลือก wallet:",
            options=[{"sync_id": "w1", "name": "เงินสด"}],
        )
        assert err.options == [{"sync_id": "w1", "name": "เงินสด"}]

    def test_UT_E01_f_is_plain_exception_subclass(self):
        """UT-E01f: not a ValueError — wrappers that re-raise
        AmbiguousMatchError must NOT swallow ClarificationNeeded."""
        with pytest.raises(ClarificationNeeded):
            raise ClarificationNeeded("?")
        # Ensure it's not under ValueError so the except branches stay clean.
        assert not issubclass(ClarificationNeeded, ValueError)
