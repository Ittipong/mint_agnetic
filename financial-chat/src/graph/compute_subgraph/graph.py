"""Compile the analyze subgraph + provide the act_node bridge for the parent ReAct.

Flow:
    START
      ↓
    plan
      ├──────────────┐
      ↓              ↓
    entity_resolve  time_resolve     (run in parallel)
      └──────────────┘
              ↓
            gate
              ↓
       ┌──────┴──────┐
       │             │
       ↓             ↓
     respond       sql_build → execute → respond
   (clarify)
              ↓
             END
"""

from __future__ import annotations

from datetime import date as date_fn, datetime as datetime_fn
from decimal import Decimal
from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from src.entity_catalog import EntityCatalog, fetch_user_catalog
from src.graph.analyze_subgraph.codeact.step import codeact_step_node
from src.graph.analyze_subgraph.nodes.entity_resolver import entity_resolve_node
from src.graph.analyze_subgraph.nodes.executor import execute_node
from src.graph.analyze_subgraph.nodes.gate import gate_node
from src.graph.analyze_subgraph.nodes.plan import plan_node
from src.graph.analyze_subgraph.nodes.responder import respond_node
from src.graph.analyze_subgraph.nodes.time_resolver import time_resolve_node
from src.graph.analyze_subgraph.state import AnalyzeSubState
from src.graph.state import AgentState

ANALYZE_TOOL_NAME = "analyze_user_finances"


# ── Routing ──────────────────────────────────────────────────────────────────


def _after_plan(state: AnalyzeSubState):
    """Choose whitelisted pipeline (fan out to entity + time) vs Templates-CodeAct.

    Returning a list fans out in parallel; returning a string routes to one node.
    """
    plan = state.get("plan")
    if plan is not None and plan.metric == "freeform_codeact":
        return "codeact_step"
    return ["entity_resolve", "time_resolve"]


def _after_gate(state: AnalyzeSubState) -> str:
    return "respond" if state.get("needs_clarification") else "execute"


def _after_codeact_step(state: AnalyzeSubState) -> str:
    return "respond" if state.get("codeact_done") else "codeact_step"


# ── Compile ──────────────────────────────────────────────────────────────────


def _build() -> StateGraph:
    g = StateGraph(AnalyzeSubState)
    g.add_node("plan", plan_node)
    g.add_node("entity_resolve", entity_resolve_node)
    g.add_node("time_resolve", time_resolve_node)
    g.add_node("gate", gate_node)
    g.add_node("execute", execute_node)
    g.add_node("codeact_step", codeact_step_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "plan")
    # After plan: either fan out to (entity + time) for structured pipeline,
    # or hop straight to codeact_step. Returning a list from _after_plan fans
    # out in parallel; both branches converge later.
    g.add_conditional_edges(
        "plan",
        _after_plan,
        ["entity_resolve", "time_resolve", "codeact_step"],
    )
    g.add_edge("entity_resolve", "gate")
    g.add_edge("time_resolve", "gate")
    g.add_conditional_edges(
        "gate",
        _after_gate,
        {"respond": "respond", "execute": "execute"},
    )
    g.add_edge("execute", "respond")
    # Templates-CodeAct loop
    g.add_conditional_edges(
        "codeact_step",
        _after_codeact_step,
        {"respond": "respond", "codeact_step": "codeact_step"},
    )
    g.add_edge("respond", END)
    return g


analyze_subgraph = _build().compile()


# ── Bridge: parent ReAct's `act_node` ────────────────────────────────────────


async def act_node(state: AgentState) -> dict:
    """Bridge from parent AgentState → analyze_subgraph → ToolMessage.

    Reads the tool call placed by reason_node, invokes the subgraph, and wraps
    the result in a ToolMessage that ReAct can keep reasoning over. Diagnostic
    info goes into `additional_kwargs` so we don't pollute the message body.
    """
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
    # keeps planner/resolver consistent with what the LLM was just told.
    catalog: EntityCatalog = await fetch_user_catalog(user_id)

    sub_input: AnalyzeSubState = {
        "task": task,
        "user_id": user_id,
        "today": today_iso,
        "catalog": catalog,
    }
    sub_out: dict = await analyze_subgraph.ainvoke(sub_input)

    answer: str = sub_out.get("answer", "(no result)")
    step_info = {
        "confidence": sub_out.get("confidence"),
        "needs_clarification": bool(sub_out.get("needs_clarification")),
        **(sub_out.get("step_info") or {}),
    }

    # Build a JSON-safe structured payload for the streaming UI. The
    # frontend reads `content` for the human-readable streaming text and
    # this payload for cards/tables — same turn, different surfaces.
    structured_data = _build_structured_data(sub_out)

    # Emit as a top-level custom event so it's visible in LangSmith
    # timeline (not buried inside additional_kwargs of a ToolMessage).
    # The server's astream_events listener catches `on_custom_event` and
    # forwards it as an SSE `data` event.
    if structured_data is not None:
        await adispatch_custom_event(
            "structured_data", structured_data
        )

    return {
        "messages": [
            ToolMessage(
                content=answer,
                tool_call_id=tool_call["id"],
                name=ANALYZE_TOOL_NAME,
                # `artifact` is the langchain-idiomatic place for non-text
                # tool output. LangGraph and LangSmith both surface it
                # alongside the message body.
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
    """Translate subgraph output into a UI-ready payload.

    Shape is deliberately stable across paths so the frontend can render the
    same component for any metric:
      {
        "kind": "result" | "clarification" | "no_data",
        "metric": str,
        "period": {"start": ISO, "end": ISO, "granularity": str},
        "currency": "THB" | "USD" | "ALL" | "THB_CONVERTED",
        "rows": [...],          # detail rows when applicable
        "metadata": {...},      # resolved entities, planner diagnostics
        "confidence": float,
      }
    """
    if sub_out.get("needs_clarification"):
        clarification = sub_out.get("clarification")
        return {
            "kind": "clarification",
            "payload": _json_safe(clarification),
        }

    spec = sub_out.get("spec")
    plan = sub_out.get("plan")

    # Codeact branch — final result lives in `codeact_final`. The shape is
    # whatever the LLM-composed code returned, so we surface it as-is and
    # tag the metric so the frontend knows it's freeform.
    if plan is not None and getattr(plan, "metric", None) == "freeform_codeact":
        return {
            "kind": "result",
            "metric": "freeform_codeact",
            "rows": _json_safe(sub_out.get("codeact_final")),
            "metadata": {
                "steps": len(sub_out.get("codeact_history") or []),
            },
            "confidence": sub_out.get("confidence"),
        }

    if spec is None:
        return None

    rows = sub_out.get("rows") or []
    period = {
        "start": spec.time_range.start.isoformat(),
        "end": spec.time_range.end.isoformat(),
        "granularity": spec.time_range.granularity,
    }
    currency = "THB_CONVERTED" if spec.convert_to_thb else spec.currency
    metadata = {
        "wallets": [
            {"sync_id": w.sync_id, "name": w.display_name, "score": w.score}
            for w in spec.wallets
        ],
        "categories": [c.display_name for c in spec.categories],
        "tags": [t.display_name for t in spec.tags],
        "transaction_type": spec.transaction_type,
        "budget_name_phrase": spec.budget_name_phrase,
    }
    return {
        "kind": "result" if rows else "no_data",
        "metric": spec.metric,
        "period": period,
        "currency": currency,
        "rows": _json_safe([_row_to_dict(r) for r in rows]),
        "metadata": metadata,
        "confidence": sub_out.get("confidence"),
    }


def _row_to_dict(row) -> dict:
    """ExecRow → dict, lifting `extra` keys to the top so the UI doesn't have
    to know about the internal split between named columns and extras."""
    base = {
        "bucket": row.bucket,
        "currency": row.currency,
        "amount": row.amount,  # already string-stringified Decimal
        "count": row.count,
    }
    base.update(row.extra or {})
    return base
