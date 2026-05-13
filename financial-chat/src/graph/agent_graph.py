"""ReAct LangGraph with dedicated Reasoner and Actor — plus a parallel
slip-to-transaction vision subgraph for image-attached turns.

Architecture:
  START
    ├─ images?  → slip → [slip_tool] → END         (vision flow)
    └─ otherwise → reason → [act|tool|respond] → ↶ (ReAct loop)

ReAct Loop:
  1. Reason: LLM decides action (call tool or respond)
  2. Act: Execute tool → return ToolMessage
  3. Loop: LLM sees ToolMessage in state → decides next step
  4. Repeat until final response → END

Slip flow is intentionally NOT a loop — vision LLM runs once, optionally
fires the propose_transaction tool, then we terminate. Failures throw
to the FastAPI stream layer (no silent fallback to text reply).
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from src.graph.nodes import (
    reason_node,
    REGULAR_TOOLS,
    ANALYZE_TOOL_NAMES,
)
from src.graph.compute_subgraph import act_node
from src.graph.slip_node import slip_node, propose_validation_node


def _route_by_input(state: AgentState) -> str:
    """Entry router: slip subgraph for image turns, ReAct for text-only."""
    if state.get("images"):
        return "slip"
    return "reason"


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


def _route_after_slip(state: AgentState) -> str:
    """After slip_node: execute tool if the LLM called one, else END."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "slip_tool"
    return "end"


def _build_builder() -> StateGraph:
    builder = StateGraph(AgentState)

    # Slip subgraph (vision LLM → optional propose_transaction tool)
    builder.add_node("slip", slip_node)
    builder.add_node("slip_tool", ToolNode([propose_transaction]))

    # ReAct loop nodes
    builder.add_node("reason", reason_node)
    builder.add_node("act", act_node)
    builder.add_node("tool", ToolNode(REGULAR_TOOLS))

    # Entry: pick lane based on whether images are attached
    builder.add_conditional_edges(
        START,
        _route_by_input,
        {"slip": "slip", "reason": "reason"},
    )

    # Slip lane is straight-line — no loop back to reason
    builder.add_conditional_edges(
        "slip",
        _route_after_slip,
        {"slip_tool": "slip_tool", "end": END},
    )
    builder.add_edge("slip_tool", END)

    # ReAct lane (unchanged)
    builder.add_conditional_edges(
        "reason",
        _should_route,
        {
            "act": "act",        # CodeAct for computations
            "tool": "tool",       # Regular tools
            "respond": END,        # Direct answer → END
        },
    )
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
