"""Unit tests for the Wave 8 classify-router shortcut.

Covers:
  - Router OFF (default): classify_intent_node skips the LLM and routes react.
  - Slot coercion + Completeness Gate B (_coerce_slots).
  - route_after_classify edge logic.
  - End-to-end graph routing with a mocked classifier:
      * complete ADD ("กาแฟ 50")     → direct_propose, proposal + answer block.
      * E9 bare amount after advisor → react (NOT direct_propose).
      * incomplete ADD               → react.
      * analyst/advisor question     → react.
  - direct_propose error path → routes react.
  - sse_adapter custom {answer_token} → emit + buffered into answer block.

Tests assert CORRECT behavior per the locked spec, not current code shape.
Test IDs: UT-CR01..UT-CR12.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from src.agent.entity_catalog import (
    CategoryEntry,
    EntityCatalog,
    WalletEntry,
    set_catalog_loader,
)
from src.agent.graph import build_graph
from src.agent.nodes import classify_intent as ci


# ─────────────────────────────────────────────────────────────────────────────
# Fakes / fixtures
# ─────────────────────────────────────────────────────────────────────────────


class _UnusedModel(BaseChatModel):
    """A model that raises if the LLM is ever called.

    The direct_propose path must NOT touch the ReAct model; the classify path
    mocks `_run_classify_llm` (httpx-direct), so neither uses this. If the
    graph wrongly routes to react in a direct_propose test, this raises loudly.
    """

    @property
    def _llm_type(self) -> str:
        return "unused"

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        return self

    def _generate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("ReAct model called — direct_propose should not reach react")

    async def _agenerate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        return self._generate(messages, stop=stop, **kwargs)


class _ReactStubModel(BaseChatModel):
    """A model that returns one fixed final AIMessage (no tool calls).

    Used for react-route tests so the graph terminates immediately — we only
    assert the route reached react, not what react did.
    """

    @property
    def _llm_type(self) -> str:
        return "react_stub"

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        return self

    def _generate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        msg = AIMessage(content="REACT_HANDLED")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        return self._generate(messages, stop=stop, **kwargs)


def _seed_catalog() -> None:
    async def loader(user_id: str) -> EntityCatalog:
        return EntityCatalog(
            wallets=[
                WalletEntry(sync_id="w-cash", name="Cash", currency="THB",
                            wallet_type="general", is_default=True),
            ],
            categories=[
                CategoryEntry(sync_id="c-coffee", name="กาแฟ", type="expense"),
            ],
        )

    set_catalog_loader(loader)


@pytest.fixture(autouse=True)
def _reset_catalog_loader():
    set_catalog_loader(None)
    yield
    set_catalog_loader(None)


@pytest.fixture(autouse=True)
def _router_on(monkeypatch):
    """Enable the router for every test here (default is OFF in prod).

    CLASSIFY_MODEL/FALLBACK are set so `make_llm_call("classify")` resolves
    cleanly even if a test forgets to patch it (we patch it per-test via
    `_patch_classifier`). The OFF test flips CLASSIFY_ROUTER_ENABLED back.
    """
    monkeypatch.setenv("CLASSIFY_ROUTER_ENABLED", "1")
    monkeypatch.setenv("CLASSIFY_MODEL", "fake/classify")
    monkeypatch.setenv(
        "CLASSIFY_FALLBACK_MODELS", '["fake/classify-fb1","fake/classify-fb2"]'
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    yield


async def _build(model):
    async def memsaver_factory():
        return MemorySaver()

    return await build_graph(
        repo=None, store=None, model=model, checkpointer_factory=memsaver_factory,
    )


def _patch_classifier(monkeypatch, result_dict):
    """Stub `make_llm_call("classify")` so the classify node never hits the
    network, while still exercising the real JSON-parse path (_extract_json_object).

    `result_dict` is serialized to a JSON string the way the LLM would return it.
    Pass a non-dict / non-JSON raw string OR set result_dict=None to simulate a
    failure: None makes the stubbed call raise OpenRouterError → fail-safe react.
    """
    from src.agent.llm_openrouter import OpenRouterError

    async def fake_call(messages):
        if result_dict is None:
            raise OpenRouterError("simulated classify failure")
        return json.dumps(result_dict)

    def fake_make_llm_call(role, *, timeout_s=30.0):
        assert role == "classify", f"unexpected role {role!r}"
        return fake_call

    monkeypatch.setattr(ci, "make_llm_call", fake_make_llm_call)


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR01 — router OFF → no LLM call, always react route
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR01_router_off_skips_llm_and_routes_react(monkeypatch):
    monkeypatch.setenv("CLASSIFY_ROUTER_ENABLED", "0")

    called = {"n": 0}

    def boom_make_llm_call(role, *, timeout_s=30.0):
        called["n"] += 1
        raise AssertionError("router OFF must not build a classify LLM call")

    monkeypatch.setattr(ci, "make_llm_call", boom_make_llm_call)

    state = {"messages": [HumanMessage("กาแฟ 50")]}
    out = asyncio.run(ci.classify_intent_node(state))

    assert called["n"] == 0, "router OFF must NOT call the classifier LLM"
    assert out["__classify_route__"] == "react"
    assert out["__classified_add__"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR02..CR05 — _coerce_slots + Gate B
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR02_complete_add_passes_gate_b():
    slots = ci._coerce_slots(
        {"intent": "ADD", "complete": True, "amount": 50,
         "type": "expense", "category_label": "กาแฟ"}
    )
    assert slots["complete"] is True
    assert slots["amount"] == 50.0
    assert slots["category_label"] == "กาแฟ"


def test_UT_CR03_label_only_no_amount_is_incomplete():
    slots = ci._coerce_slots(
        {"intent": "ADD", "complete": True, "amount": None,
         "category_label": "กาแฟ"}
    )
    # Re-derived: amount missing → NOT complete even if LLM said true.
    assert slots["complete"] is False


def test_UT_CR04_amount_only_no_label_is_incomplete():
    slots = ci._coerce_slots(
        {"intent": "ADD", "complete": True, "amount": 50,
         "category_label": None, "note": None}
    )
    assert slots["complete"] is False


def test_UT_CR05_note_satisfies_label_requirement():
    slots = ci._coerce_slots(
        {"intent": "ADD", "complete": True, "amount": 50,
         "category_label": None, "note": "ค่ากาแฟ"}
    )
    # Gate B: amount + (label OR note).
    assert slots["complete"] is True


def test_UT_CR06_nonpositive_amount_rejected():
    slots = ci._coerce_slots(
        {"intent": "ADD", "complete": True, "amount": 0, "category_label": "กาแฟ"}
    )
    assert slots["amount"] is None
    assert slots["complete"] is False


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR07 — route_after_classify edge
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR07_route_after_classify_reads_state():
    assert ci.route_after_classify({"__classify_route__": "direct_propose"}) == "direct_propose"
    assert ci.route_after_classify({"__classify_route__": "react"}) == "react"
    assert ci.route_after_classify({}) == "react"  # missing → safe default


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR08 — complete ADD → direct_propose, proposal + answer block, no react
# ─────────────────────────────────────────────────────────────────────────────


def _stub_pair_resolver(monkeypatch, *, wallet_sync_id="w-cash", category_sync_id="c-coffee"):
    """Bypass the PROPOSE_MODEL LLM — pick the seeded wallet+category.

    Same technique as test_propose_transaction_atomic: the joint resolver hits
    OpenRouter live, so unit tests must stub it. Without this, _propose_core
    returns joint_resolve_failed and direct_propose falls back to react.
    """
    import importlib

    # importlib (not `import ... as`) — the parent package re-exports the
    # @tool object under the same name, shadowing the submodule. See the note
    # in test_propose_transaction_atomic.py.
    pt_module = importlib.import_module("src.agent.tools.propose_transaction")

    async def fake_resolver(**kwargs):
        return pt_module._PairChoice(
            wallet_sync_id=wallet_sync_id,
            category_sync_id=category_sync_id,
            confidence=0.95,
            reason="stub",
        )

    monkeypatch.setattr(pt_module, "_resolve_pair_async", fake_resolver)


def test_UT_CR08_complete_add_routes_direct_propose_and_emits_blocks(monkeypatch):
    _seed_catalog()
    _stub_pair_resolver(monkeypatch)
    _patch_classifier(monkeypatch, {
        "intent": "ADD", "complete": True, "amount": 50,
        "type": "expense", "category_label": "กาแฟ",
        "wallet_label": None, "note": None, "date_iso": None,
    })

    async def run():
        graph = await _build(_UnusedModel())  # raises if react reached
        out = await graph.ainvoke(
            {"user_id": "u-1", "thread_id": "t-CR08",
             "messages": [HumanMessage("กาแฟ 50")]},
            config={"configurable": {"thread_id": "t-CR08"}, "recursion_limit": 25},
        )
        return out

    out = asyncio.run(run())
    block_types = [b.get("type") for b in out.get("emitted_blocks_this_turn", [])]
    assert "transaction_proposal" in block_types
    # A deterministic confirmation AIMessage landed in history.
    ai_texts = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    assert any("กดยืนยันด้านล่าง" in t for t in ai_texts)
    # pending proposal set + amount exact (Decimal-safe, no float drift).
    assert out["pending_proposal"]["payload"]["amount"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR09 — E9: bare amount answering an advisor question → react, NOT direct_propose
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR09_e9_bare_amount_after_advisor_routes_react(monkeypatch):
    _seed_catalog()
    # The classifier (seeing history) correctly labels this OTHER, not ADD.
    _patch_classifier(monkeypatch, {
        "intent": "OTHER", "complete": False, "amount": 2000000,
        "type": None, "category_label": None, "note": None, "date_iso": None,
    })

    history = [
        HumanMessage("อยากซื้อรถ"),
        AIMessage("งบประมาณรถที่อยากได้ราคาเท่าไหร่ครับ"),
        HumanMessage("รถ 2 ล้าน"),
    ]

    async def run():
        graph = await _build(_ReactStubModel())
        return await graph.ainvoke(
            {"user_id": "u-1", "thread_id": "t-CR09", "messages": history},
            config={"configurable": {"thread_id": "t-CR09"}, "recursion_limit": 25},
        )

    out = asyncio.run(run())
    # No proposal block — the turn stayed on the advisor flow via react.
    block_types = [b.get("type") for b in out.get("emitted_blocks_this_turn", [])]
    assert "transaction_proposal" not in block_types
    ai_texts = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    assert any(t == "REACT_HANDLED" for t in ai_texts)


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR10 — incomplete ADD → react
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR10_incomplete_add_routes_react(monkeypatch):
    _seed_catalog()
    _patch_classifier(monkeypatch, {
        "intent": "ADD", "complete": False, "amount": None,
        "type": "expense", "category_label": "กาแฟ",
        "wallet_label": None, "note": None, "date_iso": None,
    })

    async def run():
        graph = await _build(_ReactStubModel())
        return await graph.ainvoke(
            {"user_id": "u-1", "thread_id": "t-CR10",
             "messages": [HumanMessage("จ่ายค่ากาแฟ")]},
            config={"configurable": {"thread_id": "t-CR10"}, "recursion_limit": 25},
        )

    out = asyncio.run(run())
    block_types = [b.get("type") for b in out.get("emitted_blocks_this_turn", [])]
    assert "transaction_proposal" not in block_types


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR11 — analyst question → react
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR11_analyst_question_routes_react(monkeypatch):
    _seed_catalog()
    _patch_classifier(monkeypatch, {
        "intent": "OTHER", "complete": False, "amount": None,
        "type": None, "category_label": None, "note": None, "date_iso": None,
    })

    async def run():
        graph = await _build(_ReactStubModel())
        return await graph.ainvoke(
            {"user_id": "u-1", "thread_id": "t-CR11",
             "messages": [HumanMessage("เดือนนี้ใช้ไปเท่าไหร่")]},
            config={"configurable": {"thread_id": "t-CR11"}, "recursion_limit": 25},
        )

    out = asyncio.run(run())
    ai_texts = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    assert any(t == "REACT_HANDLED" for t in ai_texts)


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR12 — classifier failure (None) → react fail-safe
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR12_classifier_failure_routes_react(monkeypatch):
    _patch_classifier(monkeypatch, None)
    state = {"messages": [HumanMessage("กาแฟ 50")]}
    out = asyncio.run(ci.classify_intent_node(state))
    assert out["__classify_route__"] == "react"
    assert out["__classified_add__"] == {}


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR13 — direct_propose error path (no wallet) → routes react
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR13_direct_propose_no_wallet_routes_react(monkeypatch):
    # Catalog loader returns NO wallets → _propose_core onboarding (not ok).
    async def empty_loader(user_id: str) -> EntityCatalog:
        return EntityCatalog(wallets=[], categories=[])

    set_catalog_loader(empty_loader)

    state = {
        "user_id": "u-1",
        "__classified_add__": {
            "amount": 50, "type": "expense", "category_label": "กาแฟ",
            "wallet_label": None, "note": None, "date_iso": None,
        },
    }
    out = asyncio.run(ci.direct_propose_node(state))
    assert out["__classify_route__"] == "react"


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR14 — direct_propose confirmation is emitted EXACTLY ONCE (no double-emit)
#
# Regression for the double-confirmation bug (CLASSIFY_ROUTER_ENABLED=1, the
# direct_propose path produced answer_chars≈110 — the same ~55-char sentence twice).
# Root cause: direct_propose streamed the confirmation via BOTH a custom
# {answer_token} writer AND a node-authored AIMessage; the SSE adapter buffered
# both into answer_chunks. Fix: the AIMessage (messages-mode) is the SINGLE
# source. This test drives the SSE adapter over a fake graph that reproduces the
# production stream shape (a node-authored AIMessage on the messages stream) and
# asserts the confirmation appears exactly once — live AND in the answer block.
# ─────────────────────────────────────────────────────────────────────────────


_CR14_CONFIRM = "ขอยืนยันรายการ 50 บาท หมวดกาแฟ — กดยืนยันด้านล่างเพื่อบันทึกได้เลยครับ"


def test_UT_CR14_direct_propose_confirmation_emitted_exactly_once():
    from src.agent.streaming import sse_adapter

    class _FakeGraph:
        """Reproduce the PRE-FIX production stream shape so this test fails on
        the buggy adapter and passes on the fixed one.

        Pre-fix, direct_propose emitted the confirmation on BOTH streams:
          1. a custom {answer_token} writer push, AND
          2. a node-authored AIMessage on the `messages` stream.
        The buggy adapter buffered both into answer_chunks → the confirmation
        streamed/assembled TWICE (answer_chars≈110). The fixed adapter ignores
        custom answer_token entirely, so only the messages-mode AIMessage
        counts → exactly once. We replay both events here; the assertions below
        require single emission regardless."""

        async def astream(self, initial_state, config, stream_mode):
            # (1) the old custom-writer push — fixed adapter must IGNORE this.
            yield ("custom", {"answer_token": _CR14_CONFIRM})
            # (2) the node-authored AIMessage — the single legitimate source.
            yield ("messages", (AIMessage(content=_CR14_CONFIRM), {}))

    async def collect():
        events = []
        async for ev in sse_adapter.stream_chat(
            _FakeGraph(),
            {"thread_id": "t-CR14", "messages": [HumanMessage("กาแฟ 50")]},
            {"configurable": {"thread_id": "t-CR14"}},
        ):
            events.append(ev)
        return events

    events = asyncio.run(collect())

    # Live answer_token: the confirmation text streams EXACTLY ONCE — assembling
    # all answer_token data must equal the confirmation verbatim, not twice it.
    answer_tokens = [e["data"] for e in events if e["event"] == "answer_token"]
    assembled = "".join(answer_tokens)
    assert assembled == _CR14_CONFIRM, (
        f"answer must stream exactly once (len={len(assembled)}, "
        f"expected len={len(_CR14_CONFIRM)}); got {assembled!r}"
    )

    # Closing answer block: text == confirmation verbatim (not doubled).
    answer_blocks = [
        json.loads(e["data"])
        for e in events
        if e["event"] == "block" and json.loads(e["data"]).get("type") == "answer"
    ]
    assert len(answer_blocks) == 1, "exactly one answer block expected"
    assert answer_blocks[0]["text"] == _CR14_CONFIRM


# ─────────────────────────────────────────────────────────────────────────────
# UT-CR15 — direct_propose_node does NOT push a custom answer_token writer
#
# Guards the fix at the source: even if a future edit re-adds a stream writer,
# this asserts direct_propose_node emits the confirmation ONLY via its AIMessage and
# never calls get_stream_writer (which would re-introduce the double-emit).
# ─────────────────────────────────────────────────────────────────────────────


def test_UT_CR15_direct_propose_node_does_not_use_stream_writer(monkeypatch):
    _seed_catalog()
    _stub_pair_resolver(monkeypatch)

    # Trip-wire: if direct_propose_node ever calls get_stream_writer, fail loudly.
    import langgraph.config as lg_config

    def boom_get_stream_writer():  # pragma: no cover - must never be called
        raise AssertionError(
            "direct_propose must NOT push a custom answer_token (double-emit bug)"
        )

    monkeypatch.setattr(lg_config, "get_stream_writer", boom_get_stream_writer)

    state = {
        "user_id": "u-1",
        "__classified_add__": {
            "amount": 50, "type": "expense", "category_label": "กาแฟ",
            "wallet_label": None, "note": None, "date_iso": None,
        },
    }
    out = asyncio.run(ci.direct_propose_node(state))

    # The confirmation rides on a single AIMessage — the lone emission source.
    msgs = out.get("messages") or []
    ai_texts = [m.content for m in msgs if isinstance(m, AIMessage)]
    assert len(ai_texts) == 1
    assert "กดยืนยันด้านล่าง" in ai_texts[0]
