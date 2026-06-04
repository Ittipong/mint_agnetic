"""Flat (boundary-free) ReAct loop — Phase 1 of the rebuild assessment.

WHY THIS EXISTS
  `create_react_agent(...)` returns a COMPILED SUBGRAPH that `graph.py` adds as a
  single `react` node. LangGraph 1.x does not merge that subgraph's per-turn
  append-channel writes (`tool_outputs_this_turn`, `emitted_blocks_this_turn`)
  into the parent state for a DOWNSTREAM outer node to read — which forced the
  suggestion chips into the SSE adapter. See `docs/react_rebuild_assessment.md`.

  This module rebuilds the loop as PLAIN OUTER-GRAPH NODES (agent ⇄ tools), so
  there is no subgraph boundary and every channel merges normally. It is now the
  DEFAULT impl (set `REACT_IMPL=prebuilt` to fall back to create_react_agent).

WHAT FLAT DOES NOT DO (removed by request)
  - NO numerical validator: ungrounded numbers in the answer are NOT soft-warned
    with the ⚠️ footer. (The 99% numerical-accuracy net is OFF on the flat path.)
  - NO tool-error retry guard: if a tool errors and the model gives up, flat
    accepts the give-up instead of forcing a retry.
  Both still run on the PREBUILT path via `post_model_hook` (validators/numerical).
  Tests for that behavior (UT-G04/G05/G06) pin `react_impl="prebuilt"`.

PARITY NOTE (streaming)
  The agent model is built with `disable_streaming=True` so `stream_mode=
  "messages"` emits ONE complete AIMessage per step — matching the prebuilt
  subgraph boundary, so the SSE adapter's narration-vs-answer router behaves
  identically (no narration leaking into the answer bubble).

This module deliberately does NOT compile a graph. It exposes `wire_flat_react`
which adds its nodes/edges onto the caller's outer `StateGraph` builder, keeping
ONE flat scope.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.tools import ALL_TOOLS


# Node names — kept distinct from the prebuilt's single "react" node so a build
# can never wire both by accident.
AGENT_NODE = "agent"
TOOLS_NODE = "tools"


def is_flat_enabled() -> bool:
    """True unless REACT_IMPL=prebuilt. Default is now 'flat' — the flat loop is
    the verified default impl (≡ prebuilt: unit 395/8, live battery 11/11; see
    docs/react_rebuild_baseline.md). Set REACT_IMPL=prebuilt to fall back to the
    legacy create_react_agent subgraph."""
    return os.getenv("REACT_IMPL", "flat").strip().lower() != "prebuilt"


def _make_agent_node(chat_model: Any, prompt: Callable[[AgentState], list]) -> Callable:
    """Build the `agent` node: render the prompt, call the tool-bound model.

    Mirrors what the prebuilt's model node does — `bind_tools` + invoke with the
    rendered system prompt + windowed history (the SAME `_make_prompt` callable
    `graph.py` passes the prebuilt, so prompt behavior is identical).

    STREAMING PARITY (R1): the prebuilt subgraph delivered each AIMessage to
    `stream_mode="messages"` as a COMPLETE message at the boundary, so the SSE
    adapter's narration-vs-answer router (`_yield_message_event`: content+tool_call
    → status_token; tool-call-free content → answer_token) always saw the message
    whole. A live token stream breaks that — early content chunks of a
    narration-with-tool_call message arrive BEFORE the tool_call, so the adapter
    mis-routes the narration ("กำลังรวมยอด…") into the answer bubble. We disable
    token streaming on the agent model so messages-mode emits ONE complete
    message per step — exact prebuilt parity for routing.
    """
    # Disable token streaming so messages-mode emits a complete AIMessage per
    # step (see STREAMING PARITY above). model_copy keeps the original (used by
    # the prebuilt path) untouched.
    try:
        agent_model = chat_model.model_copy(update={"disable_streaming": True})
    except Exception:  # noqa: BLE001 — fall back to the as-is model
        agent_model = chat_model
    model_with_tools = agent_model.bind_tools(ALL_TOOLS)

    async def agent_node(state: AgentState) -> dict:
        messages = prompt(state)
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    return agent_node


# Logical exit sentinel — the router decides "run tools" vs "leave the loop"
# WITHOUT knowing which node comes after the loop. `wire_flat_react` maps this
# to the caller-chosen `exit_node`, keeping the flat loop decoupled from the
# outer graph's shape.
_EXIT = "__exit__"


def _route_after_agent(state: AgentState) -> str:
    """Route straight off the model's latest message — no validate step.

      - AIMessage WITH tool_calls    → run the tools (intermediate step)
      - AIMessage WITHOUT tool_calls → final answer → leave the loop (`_EXIT`)

    NOTE: the numerical validator + tool-error retry guard were REMOVED from the
    flat loop by request — flat no longer soft-warns ungrounded numbers nor
    forces a retry after a tool error. The prebuilt path (REACT_IMPL=prebuilt)
    still runs both via `post_model_hook`.
    """
    messages = state.get("messages") or []
    if not messages:
        return _EXIT
    tail = messages[-1]
    if isinstance(tail, AIMessage) and getattr(tail, "tool_calls", None):
        return TOOLS_NODE
    return _EXIT


def wire_flat_react(
    builder: StateGraph,
    *,
    chat_model: Any,
    prompt: Callable[[AgentState], list],
    exit_node: str,
) -> str:
    """Add the flat ReAct nodes + internal edges to `builder` (one flat scope).

    Returns the ENTRY node name (`agent`) so the caller wires
    `classify_intent → agent` to it. The agent loop exits to `exit_node` (the
    caller's next step after react) via `_route_after_agent`.

    The caller still owns: pre_turn, classify_intent, direct_propose, the post-react
    node(s), and the START/END edges. This only owns the agent⇄tools cycle.
    """
    builder.add_node(AGENT_NODE, _make_agent_node(chat_model, prompt))
    builder.add_node(TOOLS_NODE, ToolNode(ALL_TOOLS))

    builder.add_conditional_edges(
        AGENT_NODE,
        _route_after_agent,
        {TOOLS_NODE: TOOLS_NODE, _EXIT: exit_node},
    )
    builder.add_edge(TOOLS_NODE, AGENT_NODE)
    return AGENT_NODE


__all__ = ["wire_flat_react", "is_flat_enabled", "AGENT_NODE"]
