"""LangGraph workflow for FinancialCodeActAgent.

Requires: pip install langgraph

Usage:
    # Direct run
    python langgraph_simple.py "เดือนนี้ฉันจ่ายเงินไปเท่าไหร่" --user-id <UUID>

    # LangGraph Studio (requires langgraph CLI)
    langgraph dev
    # Then open http://localhost:5433 in browser
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from financial_agent.agent import FinancialCodeActAgent


# ─── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]  # conversation messages
    current_result: dict[str, Any] | None  # raw agent result
    user_id: str


# ─── Nodes ─────────────────────────────────────────────────────────────────────

async def run_financial_agent(state: AgentState) -> AgentState:
    """Run FinancialCodeActAgent on the last user message."""
    last_msg = state["messages"][-1]["content"] if state["messages"] else ""
    user_id = state.get("user_id", "")

    async with FinancialCodeActAgent() as agent:
        result = await agent.solve(last_msg, user_id=user_id)

    state["current_result"] = {
        "status": result.status,
        "result": result.result,
        "breakdown": result.breakdown,
        "metadata": result.metadata,
        "total_steps": result.total_steps,
    }
    return state


def format_response(state: AgentState) -> AgentState:
    """Format agent result into a user-friendly message."""
    current = state.get("current_result")
    if not current:
        return state

    r = current
    status = r.get("status", "")
    result = r.get("result", "")
    breakdown = r.get("breakdown", [])
    metadata = r.get("metadata", {})

    lines = [f"ผลลัพธ์: {result}"]
    if breakdown:
        lines.append("\nรายละเอียด:")
        for item in breakdown:
            label = item.get("label", "")
            value = item.get("value", "")
            unit = item.get("unit", "THB")
            lines.append(f"  • {label}: {value} {unit}")
    if metadata.get("caveat"):
        lines.append(f"\nหมายเหตุ: {metadata['caveat']}")
    lines.append(f"\nสถานะ: {status} | ขั้นตอน: {r.get('total_steps', 0)}")

    state["messages"].append({
        "role": "assistant",
        "content": "\n".join(lines),
    })
    return state


# ─── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    """Build the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", run_financial_agent)
    graph.add_node("formatter", format_response)

    # Edges: agent → formatter → END
    graph.add_edge("agent", "formatter")
    graph.add_edge("formatter", END)

    # Set entry point
    graph.set_entry_point("agent")

    return graph


# ─── Compiled graph (for LangGraph Studio) ────────────────────────────────────

# LangGraph API handles persistence automatically — no custom checkpointer needed
graph = build_graph().compile()


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_graph(user_query: str, user_id: str) -> dict[str, Any]:
    """Run the compiled graph."""
    config = {"configurable": {"thread_id": "1"}}

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_query}],
        "current_result": None,
        "user_id": user_id,
    }

    result = await graph.ainvoke(initial_state, config=config)

    return {
        "messages": result.get("messages", []),
        "current_result": result.get("current_result"),
        "response": result["messages"][-1]["content"] if result.get("messages") else "",
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="LangGraph Financial Agent")
    parser.add_argument("task", nargs="+", help="Task in natural language")
    parser.add_argument("--user-id", required=True, help="User ID (UUID)")
    args = parser.parse_args()

    task = " ".join(args.task)
    result = await run_graph(task, user_id=args.user_id)

    print("\n" + "=" * 60)
    print(f"  คำถาม: {task}")
    print("=" * 60)
    print(result["response"])
    print("=" * 60)
    print(f"\n[Debug] Raw result:")
    print(json.dumps(result["current_result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
