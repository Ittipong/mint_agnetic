"""ReAct agent state schema."""

from typing import Annotated, Literal, NotRequired
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

    # Set by `intent_classifier_node` for text-only turns. `add_transaction`
    # routes to the quick_add lane (mirrors slip lane, no image);
    # `other` routes to the regular ReAct reason loop. Not persisted
    # across turns — re-classified on every new HumanMessage.
    intent: NotRequired[Literal["add_transaction", "other"]]

    # Voice-chat input: raw audio bytes from `POST /chat/voice` plus
    # the mime type detected from the upload (e.g. "audio/m4a"). When
    # present the entry router sends the turn through `stt_node` first
    # — the resulting transcript is stuffed into the latest
    # HumanMessage and the graph continues like a text turn.
    #
    # NotRequired so existing text-only callers (and Studio JSON) do
    # not need to know about these fields. The bytes never make it
    # into the checkpoint — `stt_node` strips them on the way out so
    # the AsyncPostgresSaver never persists a multi-megabyte blob.
    audio_data: NotRequired[bytes | None]
    audio_mime: NotRequired[str | None]

    # Set by `stt_node` to the produced transcript text. Currently
    # informational only — the same text is also injected into the
    # latest HumanMessage so downstream nodes see it as if the user
    # typed it. Useful when the SSE layer wants to emit a
    # `transcript` event without re-reading messages.
    transcript: NotRequired[str | None]

    # Failure mode marker set by `stt_node`:
    #   "no_speech"            — model returned empty text
    #   "too_short"            — audio bytes below the 0.5s heuristic
    #   "transcription_failed" — model raised or returned malformed output
    # When set, the entry router skips downstream nodes and lands on
    # `stt_error_node` which dispatches an SSE error event and ends.
    stt_error: NotRequired[Literal["no_speech", "too_short", "transcription_failed"] | None]
