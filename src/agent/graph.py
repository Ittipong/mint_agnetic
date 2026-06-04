"""Compile the v3 ReAct + CodeAct graph.

NEW in Wave 4. Spec sources:
  - `docs/v3/phase2_state_and_graph.md` §4 (the create_react_agent skeleton)
  - `docs/v3/phase2_system_prompt.md` (rendered per-turn via prompts.py)
  - `docs/v3/phase3_decisions.md` Q1 (search_path=v3 schema isolation)
  - `docs/v3/phase3_decisions.md` Q4 (no sync `graph = asyncio.run(...)` workaround)

Composition:

    START → pre_turn → classify_intent
      classify_intent --(ADD complete)--> direct_propose → post_turn → END
      classify_intent --(else/OFF/error)--> react → gen_suggestions
                                                  → post_turn → END

`classify_intent` (Wave 8, src/agent/nodes/classify_intent.py) is the stateful
classify-router: with `CLASSIFY_ROUTER_ENABLED=0` (default) it SKIPS the LLM
and always routes to `react` (exact legacy behavior, zero added latency). When
ON, a complete unambiguous ADD shortcuts to `direct_propose` (a ReAct-free path
reusing `propose_transaction._propose_core`), which streams a deterministic
confirmation and goes STRAIGHT to `post_turn`. Everything else — Analyst,
Advisor, incomplete ADD, the E9 bare-amount-answers-advisor case, or any
classifier failure — falls through to `react` unchanged.

The inner `react_agent` is the prebuilt `create_react_agent(...)` graph —
agent ↔ tools loop with `post_model_hook` running the numerical validator
after every LLM step. The outer StateGraph wraps the agent with the
per-turn hooks `phase2_state_and_graph.md` §3 defined, so scratch state
(`tool_outputs_this_turn`, `emitted_blocks_this_turn`, etc.) is cleared at
the START of each turn and `last_txn` / `onboarding_stage` are finalized
at the END.

Issue 2 (`__repo__` injection):
  `propose_transaction` reads `state["__repo__"]` to persist proposals to
  the audit table. We attach the repo via the `repo` argument to
  `build_graph(repo=...)`. The pre-turn hook then writes it into state on
  every turn entry so even resumed threads see a fresh repo reference.
  Wave 6's server.py is expected to call `build_graph(repo=ProposalRepo(pool))`
  in the FastAPI lifespan. If `repo=None`, the pre-turn hook still runs but
  `state["__repo__"]` stays None; tools degrade gracefully (the persist call
  in propose_transaction already swallows None).

Q4 (no sync graph at import time):
  We do NOT expose `graph = asyncio.run(build_graph())` at module load —
  that would create a stale graph the FastAPI lifespan never overrides.
  LangGraph CLI compat is dropped per Q4; `langgraph.json` must point at
  this module's `build_graph` async factory.

Hard bounds:
  - `recursion_limit = 25` (per phase2_state_and_graph.md §6) bounds the
    ReAct loop. Validator retry + Advisor 4-step chain stays well under.
  - Per-tool timeout is handled inside individual tools (sandbox uses 15s;
    LLM calls use 30s default).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import render_system_prompt
from src.agent.user_preferences import (
    format_about_user_block,
    load_user_preferences,
)
from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS
from src.agent.validators.numerical import make_validator_post_model_hook


# ─────────────────────────────────────────────────────────────────────────────
# History windowing (Fix B — de-poison the ReAct context)
# ─────────────────────────────────────────────────────────────────────────────
#
# A long, analyst-dominated thread poisons the LLM: it sees dozens of prior
# turns and its own text-only ADD acknowledgements, then imitates the pattern
# and answers an ADD with "บันทึก…ยืนยันด้านบน" WITHOUT calling
# propose_transaction (no card reaches mobile). We feed the LLM only the last
# N turns so the always-present few-shot regains weight. This trims ONLY what
# `_make_prompt` hands the LLM — the checkpoint + messages_repo keep the full
# history, so follow-up data the user references is never lost from storage.

_HISTORY_WINDOW_TURNS_DEFAULT = 16


def _history_window_turns() -> int:
    """Read `REACT_HISTORY_WINDOW_TURNS` (positive int) or fall back to default."""
    raw = os.getenv("REACT_HISTORY_WINDOW_TURNS")
    if not raw:
        return _HISTORY_WINDOW_TURNS_DEFAULT
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _HISTORY_WINDOW_TURNS_DEFAULT
    return n if n > 0 else _HISTORY_WINDOW_TURNS_DEFAULT


def _window_messages(messages: list, max_turns: int) -> list:
    """Keep only the last `max_turns` conversation turns for the LLM prompt.

    A turn boundary is a HumanMessage. The window always begins AT a
    HumanMessage, so every `AIMessage(tool_calls)` → `ToolMessage` group stays
    intact — we never orphan a ToolMessage (which OpenRouter/OpenAI reject with
    a 400). Returns the list unchanged when there are ≤ max_turns turns.
    Pure — never mutates `messages`.
    """
    if not messages or max_turns <= 0:
        return list(messages or [])
    human_idxs = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idxs) <= max_turns:
        return list(messages)
    cut = human_idxs[-max_turns]
    return list(messages[cut:])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model factory — OpenRouter via langchain-openai (BaseChatModel adapter)
# ─────────────────────────────────────────────────────────────────────────────


def _make_model():
    """Return a `BaseChatModel` that `create_react_agent` can call.

    OpenRouter exposes an OpenAI-compatible chat completions API, so we
    point `ChatOpenAI` at the OpenRouter base URL.

    The model name resolves via `REACT_MODEL` (preferred) then
    `CODEACT_MODEL` — symmetric fallback so a single env covers the agent
    even if only one of the two is configured.

    Note: `create_react_agent` calls `model.bind_tools(...)` internally;
    ChatOpenAI supports OpenAI's tool-calling spec, which OpenRouter
    relays to providers that announce tool-calling capability. Provider
    routing prefs (`sort=latency`, `order=...`) live in OpenRouter env
    vars and apply to OpenRouter-backed requests by default.
    """
    # Lazy import — keeps unit tests that build the graph via a fake model
    # off the langchain-openai dependency path.
    from langchain_openai import ChatOpenAI

    # Absolute imports — langgraph studio loads this module outside the
    # `src.agent` package, where relative `from .X` raises
    # `ImportError: attempted relative import with no known parent package`.
    from src.agent.observability import ReActSessionLogCallback
    from src.agent.llm_openrouter import _resolve_fallbacks

    model_name = os.getenv("REACT_MODEL")
    if not model_name:
        raise RuntimeError(
            "build_graph: REACT_MODEL is required (no default — agent must "
            "declare its model). Validated at server lifespan startup; this "
            "branch only fires for non-lifespan paths (e.g. langgraph studio)."
        )
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "build_graph: OPENROUTER_API_KEY is required to construct the LLM"
        )
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    timeout_s = float(os.getenv("REACT_TIMEOUT_S", "60"))
    # max_tokens caps cost per turn — sensible upper bound for advisor turns.
    # Bumped default 2048 → 8192 so gemini-2.5-flash-lite's hidden reasoning
    # tokens (often 1.9K+ per turn) don't starve the visible-content budget
    # and trigger empty completions (memory: chat empty-answer root cause).
    max_tokens = int(os.getenv("REACT_MAX_TOKENS", "8192"))
    # OpenRouter server-side fallback chain — if the primary model errors or
    # returns an empty completion, OpenRouter automatically retries the next
    # entry in `models` within the SAME request (no extra round-trip). Reads
    # REACT_FALLBACK_MODELS (JSON array, ≥2 models required by the startup
    # validator `validate_llm_env`).
    fallbacks = [m for m in _resolve_fallbacks("react") if m != model_name]
    extra_body: dict = {}
    if fallbacks:
        extra_body["models"] = [model_name] + fallbacks
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_s,
        max_tokens=max_tokens,
        # Temperature 0 = deterministic for tool-calling + numerical answers.
        # Per-turn empathy comes from the prompt, not from sampling noise.
        temperature=0.0,
        extra_body=extra_body or None,
        # Bridge LangChain callbacks → SessionLogger so every ReAct LLM step +
        # every tool call (incl. CodeAct run_python) shows up in
        # mint_agentic_v3/logs/session_<thread>.log. Without this the main
        # ReAct loop is invisible in the log (the slog-instrumented openrouter
        # wrapper only sees resolver + structured-output calls).
        callbacks=[ReActSessionLogCallback()],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompt callable — injects {today} + {user_id} per turn
# ─────────────────────────────────────────────────────────────────────────────


def _make_prompt(state: AgentState) -> list:
    """Build the messages list for one ReAct invocation.

    Called by `create_react_agent` at each LLM step. Returns the rendered
    SystemMessage followed by the existing conversation messages.
    """
    today_iso = date.today().isoformat()
    user_id = state.get("user_id") or ""
    # `user_preferences` is loaded once per turn by `_pre_turn_hook` and merges
    # across the react boundary (scalar channel). Render it into [about_user];
    # absent/no-consent → "" → the placeholder collapses to nothing.
    about_user = format_about_user_block(state.get("user_preferences"))
    system_text = render_system_prompt(
        today=today_iso, user_id=user_id, about_user=about_user,
    )
    windowed = _window_messages(
        state.get("messages") or [], _history_window_turns()
    )
    return [SystemMessage(content=system_text)] + windowed


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pre/post turn hooks (outer StateGraph nodes)
# ─────────────────────────────────────────────────────────────────────────────


# Captured `repo` reference — closed over by `_pre_turn_hook` so the hook
# can write the audit-table repo into state on every turn entry without
# threading it through ReAct's internal nodes. Set by `build_graph(repo=…)`.
_REPO_FACTORY_STATE: dict[str, Any] = {"repo": None}


async def _pre_turn_hook(state: AgentState) -> dict:
    """Clear per-turn scratch + inject `__repo__` (Issue 2) + load preferences.

    Persisted keys (messages, proposals, last_txn, etc.) are untouched —
    we only reset what should NOT survive a turn boundary.

    `tool_outputs_this_turn` and `emitted_blocks_this_turn` use
    `append_reducer` (state.py). To CLEAR them at turn start we send the
    sentinel `["__RESET__"]` — the reducer detects the marker and resets
    the channel to `[]`. Any other write (including `[block]` from a tool)
    appends as usual.

    Scalar keys (`user_context`, validator scratch, `__repo__`) use the
    default replace-reducer; setting them here propagates verbatim.

    Async because it loads `user_preferences` from the backend DB once per
    turn (one round-trip, like the catalog). Degrades to None when the pool
    is unwired (tests) — the prompt simply renders without `[about_user]`.
    """
    user_preferences = await load_user_preferences(state.get("user_id") or "")
    return {
        # Durable user context for the [about_user] prompt block (scalar).
        "user_preferences": user_preferences,
        # Per-turn append channels — explicit reset via the sentinel.
        "tool_outputs_this_turn": ["__RESET__"],
        "emitted_blocks_this_turn": ["__RESET__"],
        # Scalar replace channels — direct value.
        "user_context": None,
        # Follow-up chips — cleared so a skip turn never inherits a stale block.
        "suggestions_block": None,
        "__validator_retries__": 0,
        "__validator_failed__": False,
        "__validator_failure_detail__": {},
        "__tool_error_retries__": 0,
        # Classify-router scratch (Wave 8) — reset so a resumed thread never
        # inherits a stale route/slots. classify_intent_node overwrites these.
        "__classify_route__": "",
        "__classified_add__": {},
        # Issue 2: audit repo. The graph factory's `repo` argument lives in
        # the module-scope closure, so this resolves on every turn without
        # the caller having to pass it via state on every invoke.
        "__repo__": _REPO_FACTORY_STATE["repo"],
    }


def _post_turn_hook(state: AgentState) -> dict:
    """Finalize last_txn + onboarding_stage.

    `last_txn` is also written by `propose_transaction` via Command(update=)
    — this hook is the safety net that keeps the field in sync with the
    newest proposal even if a future tool forgets to write it.
    """
    updates: dict = {}
    proposals = state.get("proposals") or []
    if proposals:
        newest = proposals[-1]
        payload = newest.get("payload") or {}
        updates["last_txn"] = {
            "id": newest.get("proposal_id"),
            "type": payload.get("type"),
            "amount": payload.get("amount"),
            "category": payload.get("category"),
            "pending": newest.get("status") == "pending",
        }
    user_context = state.get("user_context")
    if user_context is not None:
        wallets = user_context.get("wallets") or []
        updates["onboarding_stage"] = "no_wallet" if not wallets else "done"
    return updates


def _last_text(messages: list, predicate) -> str:
    """Return the `.content` (str) of the newest message matching `predicate`."""
    for m in reversed(messages):
        if predicate(m):
            c = getattr(m, "content", None)
            return c if isinstance(c, str) else ""
    return ""


def _add_turn(state: AgentState, messages: list) -> bool:
    """True when THIS turn recorded a transaction proposal (→ skip chips).

    Only the react path reaches this node (the direct_propose shortcut goes
    straight to post_turn), so we detect a react-loop ADD two ways, portable
    first so it holds on BOTH the flat and prebuilt paths:
      - a `propose_transaction` tool call in `messages` since this turn's last
        HumanMessage (`messages` merges across the subgraph boundary via
        `add_messages`, so this works on prebuilt too), or
      - an ADD block in `emitted_blocks_this_turn` (flat path — the append
        channel doesn't merge across the prebuilt subgraph boundary).
    """
    from src.agent.suggest_followups import is_add_block

    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == "propose_transaction":
                return True
    return any(is_add_block(blk) for blk in state.get("emitted_blocks_this_turn") or [])


async def _gen_suggestions(state: AgentState) -> dict:
    """Generate the follow-up `suggestions` block.

    Runs right after react/direct_propose (before `post_turn`), once the answer is
    settled in `messages` — it needs none of post_turn's writes. On the
    live flat path there is no subgraph boundary, so this node reads every
    signal the chip generator wants directly from state: the answer + user
    text (`messages`), the catalog (`user_context`), this-turn tool output
    (`tool_outputs_this_turn`), and the ADD signal (`emitted_blocks_this_turn`
    / classify route). It writes a ready-to-emit block (or None) to the
    `suggestions_block` channel; the SSE adapter emits it AFTER the answer
    block so chips render below the answer. Never raises — chips are
    nice-to-have (`build_suggestions_block` swallows its own failures).
    """
    from langchain_core.messages import AIMessage as _AIMessage
    from src.agent.suggest_followups import build_suggestions_block

    messages = state.get("messages") or []
    answer_text = _last_text(
        messages,
        lambda m: isinstance(m, _AIMessage) and not getattr(m, "tool_calls", None),
    )
    user_text = _last_text(messages, lambda m: isinstance(m, HumanMessage))

    tool_data = "\n".join(
        f"[{o.get('tool', '?')}] {o.get('stdout') or o.get('result') or ''}"
        for o in (state.get("tool_outputs_this_turn") or [])
        if isinstance(o, dict) and (o.get("stdout") or o.get("result"))
    ) or "(none)"

    block = await build_suggestions_block(
        user_text=user_text,
        answer_text=answer_text,
        tool_data=tool_data,
        user_context=state.get("user_context"),
        proposal_emitted=_add_turn(state, messages),
    )
    return {"suggestions_block": block}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph factory
# ─────────────────────────────────────────────────────────────────────────────


# Optional checkpointer factory override — tests inject MemorySaver via this
# hook so they don't need a live Postgres connection. Keep callers loosely
# coupled: a function that accepts no args and returns a checkpointer.
_CheckpointerFactory = Callable[[], Awaitable[Any]]


async def _default_checkpointer() -> Any:
    """Build the production Postgres checkpointer.

    Returns `None` when DATABASE_URL is unset — graph still compiles but
    threads aren't persistable (acceptable for one-shot eval runs).
    Per Q1, the DSN carries `?options=-c%20search_path%3Dv3` so v3 sits in
    its own schema separate from v2's checkpoint tables.

    Note: Production wiring (server.py lifespan) opens the checkpointer
    via `async with AsyncPostgresSaver.from_conn_string(...) as cp:` and
    passes the open instance into `build_graph(checkpointer=cp)`. The
    default returned here uses the same path but manages the lifecycle
    itself via `__aenter__` — caller MUST keep a reference for the
    process lifetime, or the connection will be garbage-collected.
    """
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        return None
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except Exception:  # noqa: BLE001
        return None
    cm = AsyncPostgresSaver.from_conn_string(dsn)
    saver = await cm.__aenter__()
    try:
        await saver.setup()
    except Exception:  # noqa: BLE001 — idempotent DDL, prior boot may have run it
        pass
    return saver


async def build_graph(
    *,
    repo: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
    model: Optional[Any] = None,
    checkpointer_factory: Optional[_CheckpointerFactory] = None,
):
    """Build and compile the v3 ReAct + CodeAct graph.

    Args:
      repo:                ProposalRepo (db.py) — injected into state on
                           every turn via the pre-turn hook so
                           `propose_transaction` can persist to the audit
                           table (Issue 2). None disables persistence
                           silently — the tool already swallows None.
      checkpointer:        Pre-built checkpointer. When provided, takes
                           precedence over `checkpointer_factory` and the
                           default Postgres factory.
      model:               Pre-built model. Tests pass a fake here. When
                           None, the OpenRouter ChatOpenAI is constructed
                           from env.
      checkpointer_factory: Async factory for the checkpointer (test hook
                           — pass a MemorySaver factory to skip Postgres).

    Returns:
      Compiled outer StateGraph:
        START → pre_turn → classify → (direct_propose | react) → gen_suggestions
              → post_turn → END.

    Side effects:
      Captures `repo` in module-level closure for the pre-turn hook to
      read. Calling `build_graph` twice with different `repo` values
      OVERRIDES the prior reference. This is intentional — only one graph
      should exist per FastAPI process.
    """
    # Capture repo in module-level slot — read by `_pre_turn_hook`.
    _REPO_FACTORY_STATE["repo"] = repo

    # Resolve dependencies (lazy / DI-friendly).
    chat_model = model if model is not None else _make_model()
    if checkpointer is None:
        factory = checkpointer_factory or _default_checkpointer
        checkpointer = await factory()

    # ── Inner agent: the ReAct loop ───────────────────────────────────────
    # Notes:
    # - `tools` includes all 7 from `tools/__init__.py::ALL_TOOLS` (Q6
    #   enforcement — clarify_wallet NOT included). UT-T00 guards count.
    # - `prompt` is a callable so we inject {today}/{user_id} per turn.
    # - `post_model_hook` runs after every LLM step (incl. intermediate
    #   tool-call turns); validator only acts on FINAL answers (no
    #   tool_calls). See validators/numerical.py.
    # - `checkpointer` is applied at the OUTER graph below — the inner
    #   agent inherits the parent's checkpointer when composed.
    # ── REACT_IMPL switch (rebuild Phase 1) ───────────────────────────────
    # Default `prebuilt` = the battle-tested create_react_agent subgraph below.
    # `flat` = the hand-rolled boundary-free loop (flat_react.wire_flat_react),
    # which removes the subgraph boundary so downstream nodes can read this-turn
    # channels directly. Dormant unless REACT_IMPL=flat. See
    # docs/react_rebuild_assessment.md.
    from src.agent.flat_react import is_flat_enabled, wire_flat_react

    use_flat = is_flat_enabled()
    react_agent = None
    if not use_flat:
        # Prebuilt keeps the numerical validator + tool-error guard via the
        # post_model_hook. The flat path intentionally omits both (removed by
        # request) — see flat_react._route_after_agent.
        react_agent = create_react_agent(
            model=chat_model,
            tools=ALL_TOOLS,
            state_schema=AgentState,
            prompt=_make_prompt,
            post_model_hook=make_validator_post_model_hook(),
        )

    # ── Outer graph: wrap with pre/post turn hooks ────────────────────────
    # Follow-up `suggestions` chips ARE a graph node now (`gen_suggestions`,
    # wired after react and before post_turn below). On the live flat path
    # there is no subgraph boundary,
    # so the node reads this-turn signals (tool output, user_context, ADD
    # blocks) directly. `build_suggestions_block` still owns the gating + LLM;
    # the node just feeds it state and writes the result to `suggestions_block`,
    # which the SSE adapter emits after the answer block. The adapter is now a
    # pure forwarder (no generation). On the prebuilt fallback path the
    # subgraph boundary withholds the per-turn APPEND channels
    # (`tool_outputs_this_turn`), so chips lose the raw tool-data hint there —
    # but `messages` + `user_context` still merge, so the answer, the catalog,
    # and ADD detection (via the `propose_transaction` tool call in messages)
    # all hold. prebuilt is dormant (REACT_IMPL=prebuilt) so this minor
    # degradation is acceptable.
    # Classify-router shortcut nodes (Wave 8). Imported here (not at module
    # top) so a build that never enables the router doesn't pay the import,
    # and to keep graph.py's import surface stable for existing tests.
    from src.agent.nodes import (
        classify_intent_node,
        direct_propose_node,
        route_after_classify,
    )

    builder = StateGraph(AgentState)
    builder.add_node("pre_turn", _pre_turn_hook)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("direct_propose", direct_propose_node)
    builder.add_node("post_turn", _post_turn_hook)
    builder.add_node("gen_suggestions", _gen_suggestions)

    # The react entry node differs by impl: prebuilt adds one "react" subgraph
    # node; flat wires agent⇄tools⇄validate and returns "agent" as the entry.
    if use_flat:
        react_entry = wire_flat_react(
            builder,
            chat_model=chat_model,
            prompt=_make_prompt,
            exit_node="gen_suggestions",
        )
    else:
        builder.add_node("react", react_agent)
        react_entry = "react"

    builder.add_edge(START, "pre_turn")
    # Classify-router (Wave 8): after pre_turn, classify the turn. When the
    # router is OFF (default) classify_intent_node SKIPS the LLM and always
    # routes to react — exact legacy behavior. When ON, a complete unambiguous
    # ADD shortcuts to direct_propose; everything else (incl. E9 bare-amount answers
    # to advisor questions, parse failures) routes to react.
    builder.add_edge("pre_turn", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"direct_propose": "direct_propose", "react": react_entry},
    )
    # direct_propose is ALWAYS an ADD turn → it never produces follow-up chips, so
    # it skips gen_suggestions entirely and goes straight to post_turn (the
    # terminal bookkeeping step). Only the react path can yield a chip-worthy
    # (non-ADD) answer, so only react flows through gen_suggestions.
    #   - prebuilt: the single "react" subgraph node feeds gen_suggestions.
    #   - flat: the loop exits to gen_suggestions via `_route_after_agent`'s
    #           `_EXIT` sentinel (wired through `exit_node` above — no static
    #           edge here).
    builder.add_edge("direct_propose", "post_turn")
    if not use_flat:
        builder.add_edge("react", "gen_suggestions")
    builder.add_edge("gen_suggestions", "post_turn")
    builder.add_edge("post_turn", END)

    return builder.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Q4 — no module-level sync graph
# ─────────────────────────────────────────────────────────────────────────────
#
# Per phase3_decisions.md Q4, we deliberately do NOT expose a module-level
# `graph = asyncio.run(build_graph())`. LangGraph CLI compat (langgraph.json
# pointing at `graph.py:graph`) is dropped — only the FastAPI lifespan calls
# `build_graph()`. If a future operator wants CLI access they can write a
# small wrapper script that awaits the factory; we don't pay the
# double-init risk for a workflow we rarely use.


# Module-level async factory for LangGraph CLI / Studio. The CLI accepts a
# callable that returns a compiled graph; Studio runs in a separate process
# from FastAPI so the Q4 "no double-init" concern (single-process re-entry)
# does not apply. Studio gets a minimal graph with NO repo/store/checkpointer
# injection — fine for visual flow inspection, not for prod traffic.
async def graph():
    return await build_graph()


__all__ = ["build_graph", "graph"]
