"""ReAct + CodeAct session-log callback handler.

Bridges LangChain's callback API into the per-thread session logger so every
ReAct LLM step and every tool call (including `run_python` / CodeAct) writes a
human-readable entry to `mint_agentic_v3/logs/session_<thread>.log`.

Why: `create_react_agent` calls `ChatOpenAI` directly — it bypasses the
slog-instrumented `llm_openrouter` wrapper. Without this handler the session
log shows only resolver/openrouter-wrapper calls, not the main ReAct loop or
the CodeAct sandbox execution, so investigators see an empty turn even though
the model produced an answer.

Wired in `graph.py::_make_model()` via `callbacks=[ReActSessionLogCallback()]`.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from ..session_logger import slog, slog_block, slog_error

_TAG_REACT = "react.llm"
_TAG_TOOL = "react.tool"

# Bigger blobs go through slog_block (multi-line). Inline strings are clamped
# to keep the log readable without losing critical detail.
_INLINE_MAX = 400


def _clip(text: str, n: int = _INLINE_MAX) -> str:
    if len(text) <= n:
        return text
    return text[:n] + f"… (+{len(text) - n} chars)"


def _format_message(m: BaseMessage) -> str:
    """One-line summary of a LangChain message for the prompt log block."""
    role = m.type  # 'system' | 'human' | 'ai' | 'tool'
    content = m.content if isinstance(m.content, str) else json.dumps(
        m.content, ensure_ascii=False, default=str,
    )
    tool_calls = getattr(m, "tool_calls", None) or []
    extras = ""
    if tool_calls:
        names = [tc.get("name") for tc in tool_calls]
        extras = f"  tool_calls={names}"
    return f"[{role}] {_clip(content)}{extras}"


class ReActSessionLogCallback(BaseCallbackHandler):
    """LangChain callback → SessionLogger bridge.

    Logged events:
      • on_chat_model_start  → full prompt (messages + tools)
      • on_llm_end           → response content + tool_calls + token usage
      • on_llm_error         → error + traceback
      • on_tool_start        → tool name + input args
      • on_tool_end          → tool output (truncated to keep log readable)
      • on_tool_error        → tool error
    """

    # ── Chat model (ReAct main LLM) ───────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            msgs = messages[0] if messages else []
            model_name = (
                serialized.get("kwargs", {}).get("model")
                or serialized.get("name")
                or "?"
            )
            n = len(msgs)
            preview = "\n".join(_format_message(m) for m in msgs[-6:])
            slog_block(
                _TAG_REACT,
                f"PROMPT model={model_name} messages={n} run_id={run_id}",
                preview,
            )
        except Exception as e:  # noqa: BLE001 — never break the agent on logging
            slog_error(_TAG_REACT, f"on_chat_model_start logging failed: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            gens = response.generations or []
            if not gens or not gens[0]:
                return
            gen = gens[0][0]
            msg = getattr(gen, "message", None)
            text = (msg.content if msg is not None else gen.text) or ""
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False, default=str)
            tool_calls = getattr(msg, "tool_calls", None) or []
            usage = (response.llm_output or {}).get("token_usage") or {}
            usage_str = (
                f"prompt={usage.get('prompt_tokens', '?')} "
                f"completion={usage.get('completion_tokens', '?')}"
            )
            if tool_calls:
                tc_brief = [
                    {
                        "name": tc.get("name"),
                        "args": _clip(
                            json.dumps(tc.get("args") or {}, ensure_ascii=False, default=str),
                            300,
                        ),
                    }
                    for tc in tool_calls
                ]
                slog_block(
                    _TAG_REACT,
                    f"RESPONSE tool_calls={len(tool_calls)} {usage_str} run_id={run_id}",
                    json.dumps(tc_brief, ensure_ascii=False, indent=2),
                )
            if text.strip():
                slog_block(
                    _TAG_REACT,
                    f"RESPONSE text {usage_str} run_id={run_id}",
                    text,
                )
        except Exception as e:  # noqa: BLE001
            slog_error(_TAG_REACT, f"on_llm_end logging failed: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        slog_error(_TAG_REACT, error)

    # ── Tools (propose_transaction, run_python/CodeAct, get_user_context,
    #          set_user_preference) ──────────────────────────────────────────
    # Note: wallet_required_cta is no longer in ALL_TOOLS — both
    # propose_transaction and run_python emit that block in-place.
    # memory_recall / memory_write were retired with the LangGraph store.

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = serialized.get("name") or "?"
            args_blob = (
                json.dumps(inputs, ensure_ascii=False, default=str)
                if inputs is not None
                else input_str
            )
            slog_block(
                _TAG_TOOL,
                f"START name={name} run_id={run_id}",
                _clip(args_blob, 1200),
            )
        except Exception as e:  # noqa: BLE001
            slog_error(_TAG_TOOL, f"on_tool_start logging failed: {e}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if isinstance(output, str):
                blob = output
            else:
                blob = json.dumps(output, ensure_ascii=False, default=str)
            slog_block(
                _TAG_TOOL,
                f"END run_id={run_id}",
                _clip(blob, 1500),
            )
        except Exception as e:  # noqa: BLE001
            slog_error(_TAG_TOOL, f"on_tool_end logging failed: {e}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        slog_error(_TAG_TOOL, error)
