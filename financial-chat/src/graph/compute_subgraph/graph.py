"""Compile the analyze subgraph + provide the act_node bridge for the parent ReAct.

Smart CodeAct flow (Phase 3 — legacy plan/resolve/gate/execute removed):

    START
      ↓
    codeact_step ─┐  (loop until result is set or MAX_STEPS reached)
      ↑           │
      └───────────┘
      ↓
    respond
      ↓
     END

The codeact loop composes resolve_*() / parse_period() / SQL wrappers in
sandboxed Python — see codeact/namespace.py for the exposed surface.
"""

from __future__ import annotations

from datetime import date as date_fn, datetime as datetime_fn
from decimal import Decimal
from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from src.entity_catalog import EntityCatalog, fetch_user_catalog
from src.graph.compute_subgraph.codeact.step import codeact_step_node
from src.graph.compute_subgraph.nodes.responder import respond_node
from src.graph.compute_subgraph.state import ComputeSubState
from src.graph.state import AgentState

ANALYZE_TOOL_NAME = "analyze_user_finances"


# ── Routing ──────────────────────────────────────────────────────────────────


def _after_codeact_step(state: ComputeSubState) -> str:
    return "respond" if state.get("codeact_done") else "codeact_step"


# ── Compile ──────────────────────────────────────────────────────────────────


def _build() -> StateGraph:
    g = StateGraph(ComputeSubState)
    g.add_node("codeact_step", codeact_step_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "codeact_step")
    g.add_conditional_edges(
        "codeact_step",
        _after_codeact_step,
        {"respond": "respond", "codeact_step": "codeact_step"},
    )
    g.add_edge("respond", END)
    return g


compute_subgraph = _build().compile()


# ── Bridge: parent ReAct's `act_node` ────────────────────────────────────────


async def act_node(state: AgentState) -> dict:
    """Bridge from parent AgentState → compute_subgraph → ToolMessage.

    Reads the tool call placed by reason_node, invokes the subgraph, and wraps
    the result in a ToolMessage that ReAct can keep reasoning over. Diagnostic
    info goes into `additional_kwargs` so we don't pollute the message body.
    """
    t0 = datetime_fn.now()
    user_id = state.get("user_id") or ""
    last_msg = state["messages"][-1]
    tool_call = next(
        tc for tc in last_msg.tool_calls if tc["name"] == ANALYZE_TOOL_NAME
    )

    if not user_id:
        return {
            "messages": [
                ToolMessage(
                    content="Error: user_id is required but not provided.",
                    tool_call_id=tool_call["id"],
                    name=ANALYZE_TOOL_NAME,
                )
            ]
        }

    today_iso = state.get("current_date") or date_fn.today().isoformat()
    task = tool_call["args"]["task"]

    # Fetch the entity catalog once and thread it through the subgraph.
    # The parent reasoner already saw the same names; sharing the snapshot
    # keeps the resolvers consistent with what the LLM was just told.
    t1 = datetime_fn.now()
    catalog: EntityCatalog = await fetch_user_catalog(user_id)
    t2 = datetime_fn.now()
    catalog_ms = (t2 - t1).total_seconds() * 1000

    sub_input: ComputeSubState = {
        "task": task,
        "user_id": user_id,
        "today": today_iso,
        "catalog": catalog,
    }
    t3 = datetime_fn.now()
    sub_out: dict = await compute_subgraph.ainvoke(sub_input)
    t4 = datetime_fn.now()
    compute_ms = (t4 - t3).total_seconds() * 1000
    total_ms = (t4 - t0).total_seconds() * 1000
    print(f"[PERF] act_node: catalog={catalog_ms:.0f}ms, compute={compute_ms:.0f}ms, total={total_ms:.0f}ms, task={task!r}")

    answer: str = sub_out.get("answer", "(no result)")
    step_info = {
        "needs_clarification": bool(sub_out.get("needs_clarification")),
        "codeact_steps": len(sub_out.get("codeact_history") or []),
    }

    # Build a JSON-safe structured payload for the streaming UI. The
    # frontend reads `content` for the human-readable streaming text and
    # this payload for cards/tables — same turn, different surfaces.
    structured_data = _build_structured_data(sub_out)

    if structured_data is not None:
        await adispatch_custom_event("structured_data", structured_data)

    return {
        "messages": [
            ToolMessage(
                content=answer,
                tool_call_id=tool_call["id"],
                name=ANALYZE_TOOL_NAME,
                artifact=structured_data,
                additional_kwargs={
                    "step_info": step_info,
                    # Ground truth catalog — the reasoner reads this to keep
                    # entity names verbatim across the whole conversation.
                    "entity_catalog": catalog.to_dict(),
                },
            )
        ]
    }


# ── Structured payload for the UI ────────────────────────────────────────────


def _json_safe(value: Any) -> Any:
    """Recursively convert Decimal / date / datetime to JSON-serializable types."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date_fn, datetime_fn)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):  # pydantic
        return _json_safe(value.model_dump())
    return value


def _build_structured_data(sub_out: dict) -> dict | None:
    """Codeact-only payload — the rows shape is whatever the LLM-composed
    code returned, surfaced verbatim to the frontend."""
    needs_clar = bool(sub_out.get("needs_clarification"))
    history = sub_out.get("codeact_history") or []
    if not history and not needs_clar:
        return None

    return {
        "kind": "clarification" if needs_clar else "result",
        "metric": "freeform_codeact",
        "rows": _json_safe(sub_out.get("codeact_final")),
        "metadata": {
            "steps": len(history),
            "clarification_question": sub_out.get("clarification_question"),
            "clarification_options": sub_out.get("clarification_options"),
        },
    }
