"""Sandbox-friendly exceptions for the CodeAct loop.

Ported from v2 (`codeact_subgraph/exceptions.py`) in Wave 1 — relocated
under `src/agent/tools/codeact/` because v3 collapses the CodeAct
sub-graph into a single `run_python` tool (still uses these exceptions).

These are raised inside sandbox helpers (resolve_*, clarify) and caught
by the run_python tool wrapper to drive control flow:

- AmbiguousMatchError  -> fed back as a tool error so the LLM picks
  one candidate explicitly in the next iteration.
- ClarificationNeeded  -> terminates the loop and surfaces a question
  to the user.
"""

from __future__ import annotations


class AmbiguousMatchError(ValueError):
    """Multiple candidates tied — let the LLM disambiguate next iter."""

    def __init__(self, query: str, kind: str, candidates: list[str]):
        self.query = query
        self.kind = kind
        self.candidates = candidates
        super().__init__(
            f"{kind!r} query {query!r} could match multiple candidates: "
            f"{candidates}. Pick one explicitly and call again."
        )


class ClarificationNeeded(Exception):
    """Sandbox helper signals the user must answer a question first.

    Raised by `clarify(question, options)` inside sandbox code. The
    run_python tool catches it and ends the loop with the question
    surfaced as the tool result.
    """

    def __init__(self, question: str, options: list[str] | None = None):
        self.question = question
        self.options = options or []
        super().__init__(question)
