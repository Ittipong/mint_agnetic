"""LLM shim layer used by the codeact subgraph.

Provides two import surfaces expected by legacy code:
- ``codeact_llm`` — exposes ``ainvoke(messages) -> Response`` for the codeact
  step loop. Backed by the OpenRouter adapter, role='codeact'.
- ``llm`` — exposes ``with_structured_output(BaseModel)`` for the resolver's
  LLM rerank. Backed by the OpenRouter adapter, role='understand'.

The shim is intentionally lazy: the real OpenRouter clients are constructed
on first attribute access so unit tests that never touch the network never
trip on missing env vars.
"""

from __future__ import annotations

from .codeact_llm import codeact_llm, llm  # re-exports

__all__ = ["codeact_llm", "llm"]
