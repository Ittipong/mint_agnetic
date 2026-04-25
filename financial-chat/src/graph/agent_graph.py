"""Main ReAct agent graph — minimal, LangGraph Studio compatible.

For LangGraph Studio: config.json references this module.
For programmatic use: use get_graph() with PostgresSaver separately.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from src.graph.state import AgentState
from src.graph.nodes import reason_node, TOOLS


def build_graph():
    """Build and compile the ReAct agent graph (no checkpointer).

    Use get_graph() for Studio or get_async_graph() for programmatic use.
    """
    builder = StateGraph(AgentState)

    tool_node = ToolNode(TOOLS)

    builder.add_node("reason", reason_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "reason")

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    builder.add_conditional_edges(
        "reason", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "reason")

    return builder.compile()


# Exported for LangGraph Studio (references: ./src/graph/agent_graph.py:graph)
graph = build_graph()
