"""ReAct LangGraph with dedicated Reasoner and Actor.

Architecture (Standard ReAct):
  START → reason → [act|tool|respond] → reason (loop) → END

ReAct Loop:
  1. Reason: LLM decides action (call tool or respond)
  2. Act: Execute tool → return ToolMessage
  3. Loop: LLM sees ToolMessage in state → decides next step
  4. Repeat until final response → END

Key concept: After tool execution, state contains ToolMessage.
The next reason_node call automatically sees it — no separate "observe" needed.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from src.graph.nodes import (
    reason_node,
    REGULAR_TOOLS,
    ANALYZE_TOOL_NAMES,
)
from src.graph.analyze_subgraph import act_node


def _should_route(state: AgentState) -> str:
    """Route after reason_node: CodeAct, regular tools, or respond directly."""
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        # No tool call = LLM gave direct answer
        return "respond"

    tool_names = {tc["name"] for tc in last_msg.tool_calls}

    # CodeAct tool → run CodeAct subgraph
    if tool_names & ANALYZE_TOOL_NAMES:
        return "act"

    # Regular tools → run via ToolNode
    return "tool"


def _build_builder() -> StateGraph:
    builder = StateGraph(AgentState)

    # Nodes - ReAct loop
    builder.add_node("reason", reason_node)      # Think: LLM decides action
    builder.add_node("act", act_node)            # Act: CodeAct execution
    builder.add_node("tool", ToolNode(REGULAR_TOOLS))  # Act: regular tools

    # Edges
    builder.add_edge(START, "reason")

    # After reasoning: decide route
    builder.add_conditional_edges(
        "reason",
        _should_route,
        {
            "act": "act",        # CodeAct for computations
            "tool": "tool",       # Regular tools
            "respond": END,        # Direct answer → END
        },
    )

    # ReAct loop: Act/Tool → Reason (LLM sees ToolMessage automatically)
    builder.add_edge("act", "reason")
    builder.add_edge("tool", "reason")

    return builder


def build_graph(checkpointer=None):
    """Compile graph — pass a checkpointer for persistent chat history."""
    return _build_builder().compile(checkpointer=checkpointer)


def build_async_graph(checkpointer):
    """Compile graph with an AsyncPostgresSaver for server use."""
    return _build_builder().compile(checkpointer=checkpointer)


# Exported for LangGraph Studio (no checkpointer — Studio manages its own)
graph = build_graph()
