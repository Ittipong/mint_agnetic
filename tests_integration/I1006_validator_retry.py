"""I1006 — numerical validator retry, end-to-end through the real graph.

Wave 7c NEW test. Goal: drive ReAct end-to-end with a FAKE LLM that emits a
KNOWN hallucination (a money number not in any tool output) → assert the
real post_model_hook validator fires, injects a [VALIDATOR_RETRY], and the
SECOND pass corrects the answer. The graph plumbing (state slots, hook
wiring, retry counter) is exercised in real form — only the LLM is faked.

This sits between unit-tests (`tests/test_react_loop.py::UT_G04`, which is
the same idea but against `build_graph()` in isolation) and the eval gate
(`evals/run_eval.py` rows that exercise ADVISOR / EMOTIONAL hallucination
patterns). UT-G04 catches the wiring bug; the eval gate catches statistical
hallucination rates; I1006 sits between them as the canary that proves the
WHOLE graph (incl. pre/post turn hooks, store, repo) still routes correctly
under a deliberate retry.

Importantly we use `build_graph(model=fake)` directly — we do NOT go through
the TestClient SSE path because:
  1. The fake-LLM mechanism plugs at the model layer, not the SSE layer;
     adding /chat/stream just adds noise.
  2. The state slots we assert (`__validator_retries__`,
     `__validator_failed__`) are state-internal — not exposed over the wire.
The lifespan-bound real graph in `app.state.agent_graph` uses the OpenRouter
LLM and isn't replaceable without rebuilding — so we build a fresh test
graph with the same factory.

NOT hermetic in the sense of needing `BACKEND_DATABASE_URL` for the entity
catalog stub — we skip that by installing a fake catalog loader. So this
test runs OFFLINE — no env required, no network calls. It's an integration
test in shape (full graph through build_graph + checkpointer) but is
deterministic.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
    set_catalog_loader,
)
from src.agent.graph import build_graph


class _FakeToolingModel(BaseChatModel):
    """Replays a fixed list of AIMessages — one per `_generate` call.

    Sufficient for ReAct integration: the prebuilt agent only needs
    `bind_tools` + `_agenerate` to chain tool calls. We don't honour the
    bound tool schemas — every test ships AIMessages with hand-crafted
    `tool_calls` payloads that already reference the right tool name + args.
    """

    responses: list = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake_react"

    def _generate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        idx = self.cursor
        object.__setattr__(self, "cursor", idx + 1)
        if idx >= len(self.responses):
            raise RuntimeError(
                f"_FakeToolingModel ran out of responses at call #{idx + 1}"
            )
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

    async def _agenerate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        return self._generate(messages, stop=stop, **kwargs)

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        return self


def _seed_fake_catalog() -> None:
    """Install a tiny in-memory catalog so `get_user_context` doesn't try
    to open BACKEND_DATABASE_URL. Mirrors the UT-G04 setup verbatim."""

    async def loader(user_id: str) -> EntityCatalog:
        return EntityCatalog(
            wallets=[
                WalletEntry(
                    sync_id="w-cash", name="Cash", currency="THB",
                    wallet_type="general", is_default=True,
                ),
            ],
            categories=[
                CategoryEntry(sync_id="c-coffee", name="กาแฟ", type="expense"),
            ],
        )

    set_catalog_loader(loader)


@pytest.fixture(autouse=True)
def _reset_catalog():
    set_catalog_loader(None)
    yield
    set_catalog_loader(None)


@pytest.mark.id("I1006")
def test_I1006_validator_soft_warn_end_to_end():
    """I1006: drive a real graph (build_graph w/ MemorySaver) with a fake LLM
    script that emits a hallucinated final answer. The validator must NEVER
    reject or retry — it SOFT-WARNS by appending a footer to the same answer.
    Assertions:

      1. `__validator_retries__ == 0` — the retry path is gone; the validator
         does not loop the model back.
      2. `__validator_failed__` is False — no terminal rejection / silent error.
      3. The final AIMessage STILL carries the model's number (99,999) — the
         answer is delivered as-is, not dropped or corrected.
      4. The warning footer (`_WARNING_MARKER`) is appended to that same answer
         so the user is told the numbers may be off.

    Sandbox is patched so `run_python` returns a controlled (result, stdout,
    error) tuple — no real Python execution, no real DB.
    """
    from src.agent.validators.numerical import _WARNING_MARKER

    _seed_fake_catalog()

    import importlib
    codeact_mod = importlib.import_module("src.agent.tools.codeact")

    def fake_execute(code, namespace):
        # Mimic the canonical (result, stdout, error) tuple of `execute`.
        return ({"amount_thb": 250}, "", None)

    # Script — the model grounds 250 via run_python but then states an
    # ungrounded 99,999 in the final answer. No correction turn: with soft-warn
    # the graph ends on the first final answer (footer appended), so the model
    # is only ever asked for THREE responses.
    #   Step 1: get_user_context (catalog bootstrap)
    #   Step 2: run_python (returns 250)
    #   Step 3: HALLUCINATED final answer (99,999) → soft-warned, delivered
    responses = [
        AIMessage(content="", tool_calls=[{
            "name": "get_user_context", "args": {},
            "id": "tc-1", "type": "tool_call",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "run_python",
            "args": {"code": "result = sum_expense(...)"},
            "id": "tc-2", "type": "tool_call",
        }]),
        AIMessage(content="เดือนนี้ใช้ไป 99,999 บาทครับ"),    # HALLUCINATION
    ]
    model = _FakeToolingModel(responses=responses)

    async def _run():
        async def memsaver_factory():
            return MemorySaver()

        with patch.object(codeact_mod, "execute", fake_execute):
            graph = await build_graph(
                repo=None,
                store=None,
                model=model,
                checkpointer_factory=memsaver_factory,
            )
            out = await graph.ainvoke(
                {
                    "user_id": "u-i1006",
                    "thread_id": "t-i1006",
                    "messages": [HumanMessage("เดือนนี้ใช้ไปเท่าไหร่")],
                },
                config={
                    "configurable": {"thread_id": "t-i1006"},
                    "recursion_limit": 25,
                },
            )
        return out

    out = asyncio.run(_run())

    # 1. No retry loop — the validator never bounces the model back.
    assert out.get("__validator_retries__", 0) == 0, (
        f"expected validator_retries=0 (no retry), got "
        f"{out.get('__validator_retries__')}"
    )
    # 2. No terminal rejection / silent error.
    assert out.get("__validator_failed__", False) is False, (
        f"validator escalated to failure: {out.get('__validator_failure_detail__')}"
    )

    # 3. The model's answer is delivered as-is (not dropped, not corrected).
    ai_messages = [m for m in out["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "no AIMessage in final state"
    final_text = ai_messages[-1].content
    assert "99,999" in final_text, (
        f"original answer was dropped/altered instead of delivered: {final_text!r}"
    )

    # 4. The warning footer is appended to that same answer.
    assert _WARNING_MARKER in final_text, (
        f"soft-warn footer missing from delivered answer: {final_text!r}"
    )
