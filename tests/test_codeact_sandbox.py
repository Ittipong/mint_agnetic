"""Unit tests for src.agent.tools.codeact.sandbox.

Covers UT-SB01..03 — lifted from v2 `test_codeact_sandbox.py` with import
paths adjusted for the v3 module layout.

The sandbox is a restricted Python executor used by `run_python`. It must
reject any construct that would let LLM-generated code escape the provided
namespace (imports, dunder access, file/network ops) while still allowing
ordinary arithmetic, comprehensions, function defs, and calls into the
supplied namespace.
"""

from __future__ import annotations

import pytest

from src.agent.tools.codeact.exceptions import ClarificationNeeded
from src.agent.tools.codeact.sandbox import execute


class TestSandboxRejects:
    def test_UT_SB01_rejects_import(self):
        """UT-SB01: `import os` must be rejected by the AST validator with
        a `sandbox rejected` error message, not silently allowed."""
        code = "import os\nresult = 1"
        result, _, err = execute(code, {})
        assert result is None
        assert err is not None
        assert "sandbox rejected" in err.lower()

    def test_UT_SB02_rejects_open_builtin(self):
        """UT-SB02: bare `open(...)` is a banned builtin name — must be
        rejected before the call even runs."""
        code = 'result = open("/etc/passwd")'
        result, _, err = execute(code, {})
        assert result is None
        assert err is not None
        assert "sandbox rejected" in err.lower()

    def test_UT_SB02b_rejects_dunder_access(self):
        """Additional guard (matches v2 UT-SBX-002): any attribute access on
        a dunder is banned, so the obj.__class__.__subclasses__() escape
        cannot be reached."""
        code = "result = (1).__class__"
        result, _, err = execute(code, {})
        assert result is None
        assert err is not None
        assert "sandbox rejected" in err.lower()


class TestSandboxAllows:
    def test_UT_SB03_executes_simple_expression(self):
        """UT-SB03: arithmetic via `sum` (a SAFE builtin) returns the right
        value with no error."""
        code = "result = sum([1, 2, 3])"
        result, _, err = execute(code, {"sum": sum})
        assert err is None
        assert result == 6

    def test_UT_SB03b_allows_list_comprehension(self):
        """UT-SB03b: list comprehensions over a namespace-supplied iterable
        return the expected list (no error)."""
        code = "result = [x * 2 for x in range(5)]"
        result, _, err = execute(code, {"range": range})
        assert err is None
        assert result == [0, 2, 4, 6, 8]

    def test_UT_SB03c_allows_function_def_and_call(self):
        """UT-SB03c: defining a function inside sandbox code works as long
        as it stays inside the safe builtins. Confirms the `__build_class__`
        + `__name__` shims are in place."""
        code = (
            "def double(x):\n"
            "    return x + x\n"
            "result = double(7)"
        )
        result, _, err = execute(code, {})
        assert err is None
        assert result == 14


class TestSandboxClarification:
    def test_UT_SB_clarification_propagates(self):
        """When a namespace function raises ClarificationNeeded, the
        sandbox MUST let it bubble up — the `run_python` tool wrapper
        catches it and emits a clarification block. Silencing it would
        strand the loop."""

        def ambiguous_lookup(_name: str):
            raise ClarificationNeeded("Which wallet do you mean?")

        code = "result = lookup('food')"
        with pytest.raises(ClarificationNeeded):
            execute(code, {"lookup": ambiguous_lookup})


class TestSandboxSignatureEnrichment:
    """E1 — a TypeError naming a namespace helper gets that helper's real
    signature + docstring summary appended, so the LLM's retry has the fix in
    hand instead of guessing (raises tool-error recovery rate)."""

    def test_UT_SB04_typeerror_appends_signature_and_doc(self):
        def compare_periods(*, period1_start, period1_end,
                            period2_start, period2_end, by="category"):
            """Compare TWO periods side-by-side."""
            return None

        # Call with a missing required kwarg → TypeError.
        code = "result = compare_periods(period1_start=1, period1_end=2)"
        result, _, err = execute(code, {"compare_periods": compare_periods})

        assert result is None
        assert err is not None
        assert err.startswith("TypeError:")
        # The enrichment names the real signature + docstring summary.
        assert "Correct signature: compare_periods(" in err
        assert "period2_start" in err
        assert "Compare TWO periods side-by-side." in err

    def test_UT_SB05_non_typeerror_not_enriched(self):
        def boom():
            raise ValueError("plain boom")

        code = "result = boom()"
        _, _, err = execute(code, {"boom": boom})
        assert err == "ValueError: plain boom"  # untouched
        assert "Correct signature" not in err
