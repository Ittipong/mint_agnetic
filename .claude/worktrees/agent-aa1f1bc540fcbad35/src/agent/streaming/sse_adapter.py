"""SSE streaming adapter: LangGraph internals -> v2 mobile wire format.

NEW in Wave 5. Sourced from `docs/v3/phase2_streaming_adapter.md` §1-§5 and
the v2 wire reference in `mint_agentic_v2/src/agent/server.py` (yield calls
~line 736 for answer_token, ~line 783 for block, ~line 832 for done).

What this module does:
  - Drives `graph.astream(state, config, stream_mode=["messages", "custom",
    "updates"])` and translates each event source into one v2-shaped SSE
    event dict {"event": ..., "data": ...}.
  - Filters the LangChain `AIMessageChunk` stream: only chunks with empty
    `tool_calls` AND non-empty `content` reach the wire as `answer_token`
    events (intermediate reasoning + tool-call chunks are dropped).
  - Forwards `custom` events with a `status_token` key as `event:
    status_token`. The validator + tool entry points use `get_stream_writer()`
    to push these per `phase2_validator.md` §10.7 + `phase2_tools_design.md`.
  - Tracks `emitted_blocks_this_turn` across `updates` events and forwards
    each NEW entry as `event: block` via `block_emitter.emit_block`.
  - Buffers the final answer text from `answer_token` chunks; emits a closing
    `event: block` with `{"type": "answer", "text": ...}` AFTER the stream
    completes so history replay can render the answer the same way it
    renders proposals (mobile dedupes vs the live token stream via
    `fromAnswerStream:true`).
  - Terminal events: `event: done` carries the thread_id, `event: error`
    carries `{code, message}` WITHOUT a stack trace (PII risk + opaque to
    users; full trace lives in the session log).

What this module does NOT do:
  - Validator retries — those happen INSIDE the graph via the post-model
    hook (`validators/numerical.py`); the adapter just streams whatever
    the graph produces (the hook writes a new message + state delta that
    the next astream iteration picks up).
  - SSE framing CRLF — `sse_starlette.EventSourceResponse` adds the
    `event:`/`data:` lines + `\r\n\r\n` terminator when Wave 6's server
    yields our dicts. Per memory `project_sse_ngrok_framing`, ngrok mangles
    framing on chunk boundaries, but sse_starlette handles that correctly;
    we only need to yield ONE dict per logical event (no batching).
  - Channel-warning suppression — LangGraph occasionally logs a benign
    "remaining_steps" channel warning on update events; we filter it via
    a one-shot module-level log gate so we don't spam every turn.

Mobile decoder reference:
  `mobile/lib/data/datasources/remote/api/chat_api_datasource.dart`
  - `_decodeSegment` (~lines 758-799): event-name switch
    (`status_token` / `answer_token` / `block` / `done` / `error`).
  - `decodeBlockJson` (~lines 859-1018): per-block-type decoder.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, AIMessageChunk

from src.agent.session_logger import slog, slog_error
from src.agent.streaming.block_emitter import emit_block, validate_block
from src.agent.validators.numerical import _WARNING_MARKER


# ---------------------------------------------------------------------------
# 0. One-shot warning gate (LangGraph remaining_steps channel)
# ---------------------------------------------------------------------------

# LangGraph's prebuilt ReAct agent maintains a `remaining_steps` channel that
# occasionally surfaces in `updates` events under unusual circumstances. The
# adapter doesn't translate it (mobile has no use for it), but we log once
# so an operator can correlate if behavior changes.
_REMAINING_STEPS_WARNED: dict[str, bool] = {"emitted": False}


def _warn_remaining_steps_once() -> None:
    """Emit the remaining_steps observation exactly once per process."""
    if _REMAINING_STEPS_WARNED["emitted"]:
        return
    _REMAINING_STEPS_WARNED["emitted"] = True
    slog(
        "sse_adapter",
        "observed LangGraph `remaining_steps` channel in updates — "
        "filtering as benign (mobile has no consumer for it)",
    )


# ---------------------------------------------------------------------------
# 1. BlockBuffer — cross-update dedupe for emitted_blocks_this_turn
# ---------------------------------------------------------------------------


class BlockBuffer:
    """Track which entries of `emitted_blocks_this_turn` we have streamed.

    LangGraph `updates` mode delivers the FULL channel value on each update
    (since `append_reducer` returns a new list each time). Without a high-
    water mark we'd resend every prior block on every update. Per-turn
    instance — Wave 6's server constructs a fresh buffer per call.

    Why a class not a counter int: the buffer also identity-checks against
    the underlying list reference so a server-side append (e.g. validator
    inserting a `discard_proposal`) doesn't accidentally rewind our index
    after a `__RESET__` reducer cycle.
    """

    __slots__ = ("_count", "_last_id")

    def __init__(self) -> None:
        self._count: int = 0
        self._last_id: int = -1

    def take_new(self, blocks: list[dict]) -> list[dict]:
        """Return blocks since the last call; advance the high-water mark.

        Detects channel resets (a NEW list object whose length is smaller
        than our prior count) and replays from index 0 of the new list.
        """
        if blocks is None:
            return []
        # New list object detected → potential reset (`__RESET__` reducer
        # cycle). Recompute from scratch.
        current_id = id(blocks)
        if current_id != self._last_id:
            self._last_id = current_id
            if len(blocks) < self._count:
                # Channel shrank → it was reset. Start fresh.
                self._count = 0
        tail = blocks[self._count :]
        self._count = len(blocks)
        return list(tail)


# ---------------------------------------------------------------------------
# 2. Per-stream-mode handlers (pure — no I/O)
# ---------------------------------------------------------------------------


# Internal retry-control markers injected into rewritten ToolMessages by the
# numerical validator and the tool-error finalize guard (validators/numerical.py).
# They steer the ReAct loop and must NEVER surface in user-facing answer text —
# a weak model occasionally echoes them verbatim as narration.
_INTERNAL_MARKERS = ("[TOOL_ERROR_RETRY", "[VALIDATOR_RETRY]")


# User-facing message when the numerical validator drops an answer (its numbers
# couldn't be grounded in this turn's tool outputs). DISTINCT from server.py's
# generic model-error fallback so mobile + transcript can tell "we withheld a
# possibly-wrong number" apart from "the model crashed".
_VALIDATOR_REJECT_TEXT = (
    "ขอโทษครับ ผมยังยืนยันความถูกต้องของตัวเลขบางส่วนในคำตอบไม่ได้ "
    "เลยขอไม่แสดงข้อมูลที่อาจคลาดเคลื่อนนะครับ ลองถามใหม่อีกครั้งได้เลยครับ"
)


def _is_dev_env() -> bool:
    """True in dev/local/staging/test, False only in production.

    Mirrors session_logger's gate — `ENVIRONMENT != "production"`. Used to
    decide whether to append a developer-facing error code to the
    validator-reject message so bugs are visible during development and
    never leak to end users in prod.
    """
    return os.getenv("ENVIRONMENT", "development").lower() != "production"


def _validator_reject_message(detail: Optional[dict]) -> str:
    """Compose the validator-reject answer. In non-prod, append an error code
    (`__validator_failed__` + the offending number) so devs see the bug."""
    msg = _VALIDATOR_REJECT_TEXT
    if _is_dev_env():
        d = detail or {}
        msg += (
            "\n\n⚠️ [DEV] validator-reject (__validator_failed__) "
            f"offending={d.get('offending_number')} "
            f"reason={d.get('reason')} "
            f"tool_numbers={d.get('tool_numbers')}"
        )
    return msg


def _strip_internal_markers(text: str) -> str:
    """Drop any line carrying an internal retry-control marker, then trim.

    Line-granular so a legitimate sentence on a neighbouring line survives.
    Returns "" when nothing meaningful remains (caller drops the token).
    """
    if not any(marker in text for marker in _INTERNAL_MARKERS):
        return text
    kept = [
        line
        for line in text.splitlines()
        if not any(marker in line for marker in _INTERNAL_MARKERS)
    ]
    return "\n".join(kept).strip()


def _message_has_tool_calls(msg: Any) -> bool:
    """True when this AI message carries a tool call alongside its content.

    The LIVE path (outer StateGraph wraps `create_react_agent` as a subgraph)
    delivers the inner agent's "narrate + call a tool" step as ONE complete
    `AIMessage` whose `tool_calls` is populated WHILE `content` holds the Thai
    "thinking out loud" preamble — so `tool_calls` is a reliable discriminator
    between narration (ephemeral) and the final answer (persistent). For the
    inline-chunk path we also check `tool_call_chunks` (the streaming shape).
    """
    if getattr(msg, "tool_calls", None):
        return True
    if getattr(msg, "tool_call_chunks", None):
        return True
    return False


def _yield_message_event(chunk: Any) -> Optional[dict]:
    """Translate one `messages` stream entry into a token event dict.

    `chunk` is `(AIMessageChunk | AIMessage, metadata)`. We forward content
    whenever it is a non-empty string, EVEN IF the same chunk also carries
    `tool_calls` or `tool_call_chunks` — this enables "thinking out loud" UX:
    the LLM writes a short Thai preamble ("กำลังเช็คยอดบัตรให้นะ") as `content`
    in the SAME response that emits the tool_call.

    Routing — narration vs final answer:
      A message that ALSO carries a tool_call is the model NARRATING what it is
      about to do ("กำลังรวมยอดให้สักครู่นะ"); the real answer arrives later in
      a tool-call-FREE message. We route narration to `narration_token` (DYNAMIC
      LLM text — mobile renders it apart from STATIC `status_token` tool words)
      and ONLY the tool-call-free content to `answer_token` (the persistent
      bubble + history answer block). This keeps the warm preamble from sticking
      in the bubble in front of the answer.

    Why accept AIMessage (not just AIMessageChunk):
      When the outer StateGraph wraps `create_react_agent` as a subgraph,
      LangGraph's `stream_mode="messages"` delivers the inner agent's
      responses as COMPLETE `AIMessage` objects at the outer-graph boundary —
      not as streaming chunks. Accepting both types lets the SSE adapter work
      whether the graph runs the agent inline (chunks) or via a compiled
      subgraph node (complete messages).

    Returns None to drop (empty / non-string / non-AI message).
    """
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return None
    msg, _metadata = chunk
    if not isinstance(msg, (AIMessageChunk, AIMessage)):
        return None
    content = msg.content
    # OpenRouter usually returns string content; multimodal providers may
    # return a list[dict]. We only stream plain-string content.
    text = content if isinstance(content, str) else ""
    if not text:
        return None
    # Strip internal control markers a weak model may echo from a rewritten
    # ToolMessage (the tool-error guard / numerical validator inject
    # `[TOOL_ERROR_RETRY .../[VALIDATOR_RETRY]` to force a retry — gemini-flash
    # sometimes parrots them as narration). These MUST never reach the user.
    text = _strip_internal_markers(text)
    if not text:
        return None
    # Content bundled with a tool_call = narration → DYNAMIC narration_token
    # (distinct from STATIC status_token so mobile can style it differently).
    # Tool-call-free content = the real answer → persistent answer token.
    event = "narration_token" if _message_has_tool_calls(msg) else "answer_token"
    return {"event": event, "data": text}


def _yield_custom_event(chunk: Any) -> Optional[dict]:
    """Translate one `custom` event into a wire token dict.

    `chunk` is whatever `get_stream_writer()(payload)` pushed. Recognized keys
    (both map to `event: status_token`, the ephemeral typing label):
      - `status_token` → reserved generic typing label (no current producer).
      - `status` → pushed by every tool's `_emit_status` + the numerical
        validator. These are the per-step progress narrations.

    Custom `answer_token` is intentionally NOT recognized. The `direct_propose` node
    (classify-router shortcut) emits its deterministic confirmation as a
    node-authored AIMessage, which messages-mode streams as `answer_token` for
    us — see `_yield_message_event` + the messages branch in `stream_chat`.
    A previous version had direct_propose ALSO push a custom `answer_token`; that
    duplicated the confirmation on the wire (both sources fed `answer_chunks`),
    so the custom writer was removed. No producer emits custom `answer_token`
    anymore; recognizing it here would only be dead code that risks re-
    introducing the double-emit, so we drop it.

    Anything other than `status_token` / `status` is dropped silently — future
    custom event types should register here explicitly.
    """
    if not isinstance(chunk, dict):
        return None
    # Accept BOTH keys: `status_token` is a reserved generic label, but every
    # tool's `_emit_status` (get_user_context, codeact, propose_transaction,
    # advice_playbook, …) and the numerical validator push `status`. Reading
    # only `status_token` silently dropped all 9 tool-level progress emissions,
    # so the user saw a single preamble then 5-15s of silence. Coalescing both
    # surfaces the agent's real step-by-step progress as it works.
    status = chunk.get("status_token") or chunk.get("status")
    if not status:
        return None
    return {"event": "status_token", "data": str(status)}


# ---------------------------------------------------------------------------
# 3. Main streaming generator
# ---------------------------------------------------------------------------


async def stream_chat(
    graph: Any,
    initial_state: dict,
    config: dict,
    *,
    thread_id: Optional[str] = None,
) -> AsyncIterator[dict]:
    """Drive one ReAct turn; yield SSE event dicts that match the v2 wire.

    Args:
      graph: a compiled LangGraph instance (see `graph.build_graph`).
      initial_state: dict that satisfies `AgentState` TypedDict — at minimum
        `user_id`, `thread_id`, `messages`. Wave 6's server builds it.
      config: LangGraph runtime config (`{"configurable": {"thread_id": ...},
        "recursion_limit": 25}`).
      thread_id: explicit thread_id for the `done` event payload. Falls
        back to `initial_state["thread_id"]` then `config["configurable"]
        ["thread_id"]` so callers don't have to repeat themselves.

    Yields:
      dicts shaped `{"event": <event_name>, "data": <str>}` for
      `sse_starlette.EventSourceResponse` to frame as SSE on the wire.
      Event names mirror v2's contract, plus `narration_token`:
        - `status_token` (raw text) — STATIC tool/progress words only
        - `narration_token` (raw text) — DYNAMIC LLM text (preamble +
          tool-call "thinking out loud"); mobile renders it apart from
          static status so the two streams are visually distinguishable
        - `answer_token` (raw text) — the final answer
        - `block` (JSON payload)
        - `done` ({thread_id})
        - `error` ({code, message})
    """
    resolved_thread_id = (
        thread_id
        or initial_state.get("thread_id")
        or (config or {}).get("configurable", {}).get("thread_id", "")
    )

    buffer = BlockBuffer()
    answer_chunks: list[str] = []

    # ── Follow-up suggestion chips — generated by the `gen_suggestions` graph
    # node (graph.py), NOT here. The node writes a ready-to-emit block (or None)
    # to the `suggestions_block` channel; we capture it off the `updates` stream
    # and emit it AFTER the answer block (chips render below the answer). This
    # layer no longer gates or calls the LLM — it is a pure forwarder.
    pending_suggestions: Optional[dict] = None
    # Set when the numerical validator pushes a `validator_failed` custom event
    # (terminal rejection). Used at end-of-graph to emit a validator-specific
    # message instead of letting the empty answer fall through to server.py's
    # generic model-error fallback.
    validator_failure: Optional[dict] = None
    # Soft-warn footer reconciliation (REACT_IMPL parity). The numerical
    # validator appends a warning footer by REWRITING the final AIMessage. In the
    # prebuilt subgraph that rewrite happens BEFORE the message crosses the
    # boundary, so the footer arrives via `answer_token` and lands in
    # `answer_chunks`. In the FLAT impl the answer is streamed by the agent node
    # (no footer yet) and the `react_validate` node rewrites it AFTER — that
    # rewrite is NOT re-emitted on the messages stream, so the footer would be
    # lost from the live answer + the answer block. We capture it from the
    # `updates` stream (the validate node's message delta) and reconcile it into
    # `final_text` below. No-op for prebuilt (the marker is already in
    # `final_text`, so the guard skips).
    soft_warn_footer: Optional[str] = None

    try:
        async for stream_mode, chunk in graph.astream(
            initial_state,
            config=config,
            stream_mode=["messages", "custom", "updates"],
        ):
            # ── messages stream — answer tokens + narration status ─────────
            if stream_mode == "messages":
                ev = _yield_message_event(chunk)
                if ev is not None:
                    # Only the real answer (tool-call-free content) feeds the
                    # final answer block. Narration routed to `status_token`
                    # is ephemeral and must NEVER reach the history answer.
                    if ev["event"] == "answer_token":
                        answer_chunks.append(ev["data"])
                    yield ev
                continue

            # ── custom stream — status tokens from inside tools ───────────
            if stream_mode == "custom":
                # Validator terminal-rejection signal — captured, NOT forwarded
                # as a status token. Acted on at end-of-graph.
                if isinstance(chunk, dict) and "validator_failed" in chunk:
                    validator_failure = chunk.get("validator_failed") or {}
                    continue
                # Only `status_token` reaches the wire from the custom stream
                # (ephemeral typing label — never buffered into the answer).
                # direct_propose's confirmation flows via the messages stream, not
                # here; see _yield_custom_event for why custom answer_token was
                # removed (it double-emitted the confirmation).
                ev = _yield_custom_event(chunk)
                if ev is not None:
                    yield ev
                continue

            # ── updates stream — drain emitted_blocks_this_turn tail ──────
            if stream_mode == "updates":
                if not isinstance(chunk, dict):
                    continue
                # chunk shape: {node_name: state_delta_dict}
                for _node_name, delta in chunk.items():
                    if not isinstance(delta, dict):
                        continue
                    # Filter the LangGraph remaining_steps channel — benign,
                    # warn once for traceability.
                    if "remaining_steps" in delta and "emitted_blocks_this_turn" not in delta:
                        _warn_remaining_steps_once()
                    # Capture the chips the `gen_suggestions` node produced.
                    # `None` is a valid value (skip turn), so test key presence
                    # — not truthiness — and emit at end-of-graph after the
                    # answer block.
                    if "suggestions_block" in delta:
                        pending_suggestions = delta["suggestions_block"]
                    # Capture a soft-warn footer the validator wrote into a
                    # rewritten message this turn (FLAT parity — see the
                    # `soft_warn_footer` note above). Scan the node's message
                    # delta; the footer is everything from the "\n\n---\n⚠️…"
                    # separator onward.
                    for m in delta.get("messages") or []:
                        mc = getattr(m, "content", None)
                        if isinstance(mc, str) and _WARNING_MARKER in mc:
                            sep = mc.rfind("\n\n---\n")
                            soft_warn_footer = (
                                mc[sep:] if sep != -1
                                else mc[mc.find(_WARNING_MARKER):]
                            )
                    # `updates` stream surfaces RAW per-node writes BEFORE the
                    # append_reducer collapses `["__RESET__"]` sentinels into
                    # `[]`. Filter to dict blocks so the sentinel (a str)
                    # never reaches validate_block / mobile. See memory
                    # `project_sse_ngrok_framing` + Wave 7a R3 catch.
                    new_blocks = [
                        b
                        for b in (delta.get("emitted_blocks_this_turn") or [])
                        if isinstance(b, dict)
                    ]
                    if not new_blocks:
                        continue
                    for blk in buffer.take_new(new_blocks):
                        if not validate_block(blk):
                            # Malformed block — drop + log. Never propagate
                            # to mobile (would crash the decoder).
                            slog(
                                "sse_adapter",
                                f"dropped malformed block: type="
                                f"{(blk or {}).get('type')!r} keys="
                                f"{list((blk or {}).keys())}",
                            )
                            continue
                        yield {
                            "event": "block",
                            "data": json.dumps(blk, ensure_ascii=False),
                        }
                continue

            # Unknown stream mode — ignore silently. Adding modes later is
            # forward-compatible without an error path here.

        # ── End of graph — flush final answer + done ──────────────────────
        # Re-strip the assembled answer: per-token filtering catches the common
        # case (a marker echoed as one complete message), but if a provider
        # split the marker across streaming chunks, the seam only closes here.
        final_text = _strip_internal_markers("".join(answer_chunks)).strip()

        # Reconcile a soft-warn footer that streamed without it (FLAT impl).
        # Guarded by `_WARNING_MARKER not in final_text` → no-op for prebuilt,
        # where the footer already arrived via `answer_token`. Emit it as a
        # trailing answer_token so the live bubble matches the answer block.
        if final_text and soft_warn_footer and _WARNING_MARKER not in final_text:
            yield {"event": "answer_token", "data": soft_warn_footer}
            final_text = f"{final_text}{soft_warn_footer}"

        # Validator dropped the answer (numbers ungrounded) and nothing else
        # filled the gap → emit a validator-SPECIFIC message (+ dev error code
        # in non-prod) so it's distinguishable from server.py's generic
        # model-error fallback, which only fires when no block reaches the wire.
        emitted_reject = False
        if not final_text and validator_failure is not None:
            reject_text = _validator_reject_message(validator_failure)
            slog(
                "sse_adapter",
                f"validator-reject → distinct message (dev={_is_dev_env()}) "
                f"detail={validator_failure}",
            )
            yield {"event": "answer_token", "data": reject_text}
            reject_block = {"type": "answer", "text": reject_text}
            if validate_block(reject_block):
                yield {
                    "event": "block",
                    "data": json.dumps(reject_block, ensure_ascii=False),
                }
            final_text = reject_text
            emitted_reject = True

        if final_text and not emitted_reject:
            answer_block = {"type": "answer", "text": final_text}
            # Validate defensively — if the validator ever rejects our own
            # answer block we have a serious bug; surface it as a dropped
            # block instead of silently yielding malformed JSON.
            if validate_block(answer_block):
                yield {
                    "event": "block",
                    "data": json.dumps(answer_block, ensure_ascii=False),
                }
            else:
                slog(
                    "sse_adapter",
                    "self-emitted answer block failed validation — dropped",
                )

        # ── Follow-up suggestion chips ─────────────────────────────────────
        # Generated by the `gen_suggestions` graph node (gating + LLM), captured
        # off the `updates` stream above. Emitted HERE — after the answer block,
        # before `done` — so chips render below the answer. `None` means skip
        # (ADD / crisis / no-answer / LLM-skip / disabled).
        if pending_suggestions is not None and validate_block(pending_suggestions):
            yield {
                "event": "block",
                "data": json.dumps(pending_suggestions, ensure_ascii=False),
            }

        yield {
            "event": "done",
            "data": json.dumps({"thread_id": resolved_thread_id}),
        }

    except Exception as exc:  # noqa: BLE001 — forward all errors to client
        tb = traceback.format_exc()
        # PII: log the full traceback to the session file but NEVER send it
        # to mobile — error events expose code + a friendly Thai message
        # only. Per spec section 9 Streaming Flow + memory
        # `project_sse_ngrok_framing` (Pattern D contract).
        slog_error("sse_adapter", exc, tb)
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "code": type(exc).__name__,
                    "message": str(exc) or "เกิดข้อผิดพลาดในการประมวลผล",
                },
                ensure_ascii=False,
            ),
        }


__all__ = ["stream_chat", "BlockBuffer"]
