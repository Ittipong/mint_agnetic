"""ReAct agent state schema."""

from typing import Annotated, NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """ReAct agent state.

    ReAct Loop: Reason → Act → Reason (loop until final answer)
    After tool execution, ToolMessage appears in messages — LLM sees it on next reason call.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: NotRequired[str]  # optional so Studio chat can init with messages only
    current_date: NotRequired[str]  # ISO date string for the CodeAct agent

    # Slip-to-transaction input: list of data URLs (typically a single
    # `data:image/jpeg;base64,...`). Present → graph routes to slip_node
    # instead of the ReAct reason loop. Empty/missing → regular chat.
    images: NotRequired[list[str]]
