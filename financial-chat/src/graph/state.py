"""ReAct agent state schema."""

from typing import TypedDict, Annotated
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Minimal state for a ReAct agent with tool-calling capability."""

    messages: Annotated[list, add_messages]
    user_id: str
