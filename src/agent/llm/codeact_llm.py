"""Thin shim that adapts the OpenRouter HTTP adapter to the two interfaces
the legacy codeact files were written against:

  llm.ainvoke([messages]) -> Response(.content: str)
  llm.with_structured_output(Schema) -> wrapper with ainvoke(prompt) -> Schema

The OpenRouter HTTP adapter (`src.agent.llm_openrouter.make_llm_call`) is
itself an `async (messages) -> str`. We wrap it once per role and expose
both `ainvoke(messages)` (raw text in a Response wrapper) and the
structured-output adapter the resolver uses.

Design constraints:
- No hardcoded model names — roles resolve via env at first call.
- Lazy construction — building the adapter requires OPENROUTER_API_KEY,
  which test envs do not set. Construction is deferred to first call so
  unit tests that mock at a higher layer never touch the adapter.
- Decimal-safe: nothing here computes amounts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Response wrapper — step.py reads `response.content`
# ---------------------------------------------------------------------------
@dataclass
class _LLMResponse:
    content: str


# ---------------------------------------------------------------------------
# Lazy adapter factory — built on first use, then cached
# ---------------------------------------------------------------------------
_cached: dict[str, Callable[[list[dict]], Awaitable[str]]] = {}


def _get_adapter(role: str) -> Callable[[list[dict]], Awaitable[str]]:
    if role not in _cached:
        # Import locally so missing env vars only blow up when actually called.
        from ..llm_openrouter import make_llm_call

        _cached[role] = make_llm_call(role)
    return _cached[role]


# ---------------------------------------------------------------------------
# Structured-output adapter — extracts a JSON object then validates it against
# the requested pydantic schema. Mirrors langchain's `with_structured_output`
# surface (`obj.ainvoke(prompt) -> BaseModel`).
# ---------------------------------------------------------------------------
class _StructuredAdapter:
    def __init__(self, role: str, schema: type[BaseModel]) -> None:
        self._role = role
        self._schema = schema

    async def ainvoke(self, prompt: str) -> BaseModel:
        call = _get_adapter(self._role)
        # Force JSON-mode via the system prompt — OpenRouter relays whatever
        # the underlying provider supports; if a provider strips response_format
        # we still get JSON because we asked for it explicitly.
        messages = [
            {
                "role": "system",
                "content": (
                    "Return ONLY a JSON object that matches the schema "
                    "the user describes. No markdown fences. No prose."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw = await call(messages)
        data = _extract_json_object(raw)
        return self._schema.model_validate(data)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply, tolerating fences."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    # Greedy span between the first '{' and the last '}'.
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"LLM response has no JSON object: {text!r}")
    return json.loads(s[i : j + 1])


# ---------------------------------------------------------------------------
# Public surface (what legacy modules import)
# ---------------------------------------------------------------------------
class _CodeActLLM:
    """`llm.ainvoke(messages)` for the codeact step loop. Uses role='codeact'."""

    async def ainvoke(self, messages: list[dict]) -> _LLMResponse:
        call = _get_adapter("codeact")
        text = await call(messages)
        return _LLMResponse(content=text)


class _ResolverLLM:
    """`llm.with_structured_output(Schema)` for the resolver rerank.

    Uses role="codeact" — the CODEACT_MODEL env is dedicated to this
    resolver (entity disambiguation); see .env section §2."""

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredAdapter:
        return _StructuredAdapter("codeact", schema)


codeact_llm = _CodeActLLM()
llm = _ResolverLLM()


__all__ = ["codeact_llm", "llm"]
