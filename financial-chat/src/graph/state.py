"""ReAct agent state schema."""

from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Minimal state for a ReAct agent with tool-calling capability."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: NotRequired[str]  # optional so Studio chat can init with messages only
