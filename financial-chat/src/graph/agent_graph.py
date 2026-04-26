"""Main graph — ReAct orchestrator with dedicated CodeAct subgraph node.

For LangGraph Studio: `graph` is exported without a checkpointer (Studio manages its own).
For server use: call `build_async_graph(checkpointer)` with an AsyncPostgresSaver.

Routing:
  reason → "codeact"  when ReAct calls analyze_user_finances
  reason → "tools"    when ReAct calls any regular tool
  reason → END        when ReAct has a final answer (no tool calls)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from src.graph.nodes import reason_node, REGULAR_TOOLS, CODEACT_TOOL_NAMES
from src.graph.codeact_subgraph import codeact_node

_MAX_HISTORY = 40  # keep last N messages to avoid context overflow


def _should_continue(state: AgentState) -> str:
    """Route after reason_node: codeact subgraph, regular tools, or done."""
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return END

    tool_names = {tc["name"] for tc in last_msg.tool_calls}

    # Any call to the financial analysis tool → dedicated CodeAct node
    if tool_names & CODEACT_TOOL_NAMES:
        return "codeact"

    return "tools"


def _trim_history(state: AgentState) -> dict:
    """Keep only the last _MAX_HISTORY messages to bound context size."""
    msgs = state["messages"]
    if len(msgs) > _MAX_HISTORY:
        return {"messages": msgs[-_MAX_HISTORY:]}
    return {}


def _build_builder() -> StateGraph:
    builder = StateGraph(AgentState)
    tool_node = ToolNode(REGULAR_TOOLS)

    builder.add_node("reason", reason_node)
    builder.add_node("tools", tool_node)
    builder.add_node("codeact", codeact_node)  # dedicated CodeAct subgraph node
    builder.add_node("trim", _trim_history)

    builder.add_edge(START, "reason")
    builder.add_conditional_edges(
        "reason",
        _should_continue,
        {"codeact": "codeact", "tools": "tools", END: END},
    )
    builder.add_edge("codeact", "trim")
    builder.add_edge("tools", "trim")
    builder.add_edge("trim", "reason")

    return builder


def build_graph(checkpointer=None):
    """Compile graph — pass a checkpointer for persistent chat history."""
    return _build_builder().compile(checkpointer=checkpointer)


def build_async_graph(checkpointer):
    """Compile graph with an AsyncPostgresSaver for server use."""
    return _build_builder().compile(checkpointer=checkpointer)


# Exported for LangGraph Studio (no checkpointer — Studio manages its own)
graph = build_graph()
