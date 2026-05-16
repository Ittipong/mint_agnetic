"""ReAct LangGraph with dedicated Reasoner and Actor — plus a parallel
slip-to-transaction vision subgraph for image-attached turns and a
quick-add lane for text journal entries.

Architecture:
  START
    ├─ images?  → slip → [slip_tool] → cleanup → END           (vision flow)
    └─ no image → classify_intent
                    ├─ add_transaction → quick_add → propose_validation → cleanup → END
                    └─ other           → reason → [act|tool|respond] → ↶   (ReAct loop)

ReAct Loop:
  1. Reason: LLM decides action (call tool or respond)
  2. Act: Execute tool → return ToolMessage
  3. Loop: LLM sees ToolMessage in state → decides next step
  4. Repeat until final response → END

Slip + quick_add are intentionally NOT loops — single LLM hop, optionally
fires propose_transaction, then we terminate. Failures throw to the
FastAPI stream layer (no silent fallback to text reply).
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
from src.graph.confirmation_node import (
    confirmation_node,
    is_confirmation_marker,
)
from src.graph.intent_classifier import classify_intent_node
from src.graph.quick_add_node import quick_add_node
from src.graph.slip_node import (
    slip_node,
    propose_validation_node,
    slip_cleanup_node,
)

from langchain_core.messages import HumanMessage as _HumanMessage


def _route_by_input(state: AgentState) -> str:
    """Entry router: image turn → slip; save/dismiss marker →
    confirmation; everything else → classifier.
    """
    if state.get("images"):
        return "slip"
    # Latest HumanMessage may be a system marker from mobile after
    # the user saved or dismissed a transaction card. Route those to
    # the dedicated confirmation node so they bypass intent
    # classification and quick_add entirely.
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, _HumanMessage):
            content = m.content
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text = str(blk.get("text", ""))
                        break
            if is_confirmation_marker(text):
                return "confirmation"
            break
    return "classify_intent"


def _route_by_intent(state: AgentState) -> str:
    """After classify_intent: quick-add lane or regular reason loop."""
    if state.get("intent") == "add_transaction":
        return "quick_add"
    return "reason"


def _route_after_quick_add(state: AgentState) -> str:
    """After quick_add: dispatch tool call OR end with ask-back text.

    Quick-add is now multi-turn — when the LLM is missing a required
    field (almost always `amount`) it returns an AIMessage with text
    only, no tool_call. We route that case straight to END so the
    text reply lands in chat history and the next user turn can
    follow up. When tool_calls are present we dispatch them and end —
    NO cleanup, because the messages of the add_transaction sequence
    are now valuable context (user may say "อันก่อนหน้านี้ ขอเปลี่ยน
    เป็น 200" etc.) and slip_cleanup's "remove from last HumanMessage
    onward" would only strip the final turn anyway.
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "quick_add_tool"
    return "end"


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
    """After slip_node: execute tool if the LLM called one, else clean up."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "slip_tool"
    return "cleanup"


def _build_builder() -> StateGraph:
    builder = StateGraph(AgentState)

    # Slip subgraph (vision LLM → validate + dispatch propose_transaction)
    builder.add_node("slip", slip_node)
    builder.add_node("slip_tool", propose_validation_node)
    # Terminal cleanup — strips the slip / quick-add turn from the
    # checkpoint so it doesn't pollute history reloads or bloat token
    # usage on future ReAct turns. The proposal payload was already
    # sent to mobile via the SSE custom event, so the messages have
    # no further purpose. Shared between slip + quick_add lanes.
    builder.add_node("slip_cleanup", slip_cleanup_node)

    # Intent classifier + quick-add lane (text journal entries)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("quick_add", quick_add_node)
    # Reuse propose_validation_node — same propose_transaction tool,
    # same id-validation + SSE dispatch contract as the slip lane.
    builder.add_node("quick_add_tool", propose_validation_node)

    # Confirmation node — reacts to mobile's save/dismiss marker by
    # generating a short Thai acknowledgement that references the
    # actual transaction the user just acted on.
    builder.add_node("confirmation", confirmation_node)

    # ReAct loop nodes
    builder.add_node("reason", reason_node)
    builder.add_node("act", act_node)
    builder.add_node("tool", ToolNode(REGULAR_TOOLS))

    # Entry: image turns → slip, save/dismiss markers → confirmation,
    # text turns → classifier.
    builder.add_conditional_edges(
        START,
        _route_by_input,
        {
            "slip": "slip",
            "confirmation": "confirmation",
            "classify_intent": "classify_intent",
        },
    )
    builder.add_edge("confirmation", END)

    # Classifier → quick-add lane or regular ReAct
    builder.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {"quick_add": "quick_add", "reason": "reason"},
    )

    # Slip lane is straight-line — no loop back to reason. Both
    # branches funnel into slip_cleanup so cleanup runs unconditionally.
    builder.add_conditional_edges(
        "slip",
        _route_after_slip,
        {"slip_tool": "slip_tool", "cleanup": "slip_cleanup"},
    )
    builder.add_edge("slip_tool", "slip_cleanup")
    builder.add_edge("slip_cleanup", END)

    # Quick-add lane is multi-turn — both branches end WITHOUT
    # cleanup so the conversation history persists for follow-ups.
    # Tool branch: validate + dispatch SSE card → END.
    # Text branch: ask-back AIMessage stays in history → END.
    builder.add_conditional_edges(
        "quick_add",
        _route_after_quick_add,
        {"quick_add_tool": "quick_add_tool", "end": END},
    )
    builder.add_edge("quick_add_tool", END)

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
