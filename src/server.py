"""FastAPI server with SSE streaming for the ReAct chat agent."""

# Load .env into os.environ BEFORE any import that reads tracing env
# vars (langchain/langsmith clients read LANGCHAIN_TRACING_V2 etc. at
# import time). pydantic-settings populates the Settings object but
# does not inject into os.environ, so LangSmith would otherwise see no
# tracing config and silently disable traces.
from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.debug_log import LogLevel, get_request_id, log as _debug_log_fn, set_request_id
from src.graph.agent_graph import build_async_graph
from src import threads_repo


def _debug_log(tag: str, msg: str, **kwargs):
    """Thin wrapper preserving the old call signature.

    Defaults to `MILESTONE` because every existing call site at this
    layer (HTTP entry, stream lifecycle, thread mutations, tool
    boundaries) is high-signal. Switch individual calls to
    `LogLevel.DETAIL` when adding verbose payload dumps.
    """
    level = kwargs.pop("_level", LogLevel.MILESTONE)
    _debug_log_fn(tag, msg, level=level, **kwargs)


# ── Lifespan: set up checkpointer + graph once at startup ────────────────────

_graph: Any = None
_pool: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _pool

    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _debug_log("SERVER", "Starting up...", database_url=settings.database_url[:50])

    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await _pool.open()

    checkpointer = AsyncPostgresSaver(_pool)
    await checkpointer.setup()  # creates checkpoint tables if not exist

    _graph = build_async_graph(checkpointer)
    _debug_log("SERVER", "Graph ready")

    yield

    await _pool.close()
    _debug_log("SERVER", "Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────

API_DESCRIPTION = """
**Mint Money Chat Agent** — LangGraph ReAct agent for Thai financial Q&A.

Combines a reasoning LLM (Typhoon) with a CodeAct compute subgraph that
runs deterministic SQL templates over the user's transaction data.

---

### Streaming chat (SSE)

`POST /chat/stream` returns **`text/event-stream`** (Server-Sent Events).
Each event is a single JSON object on a `data:` line. Possible event types:

| `type`        | Payload fields                | When |
|---------------|-------------------------------|------|
| `status`      | `phase`, `label`              | Agent lifecycle: `thinking` → `calculating` → `writing` |
| `tool_start`  | `tool`                        | A regular tool started |
| `tool_end`    | `tool`                        | A regular tool finished |
| `data`        | `payload`                     | Structured tool result (rows, metric, metadata) |
| `token`       | `content`                     | LLM streaming token — concatenate into a Markdown reply |
| `suggestions` | `items` (array of strings)    | Follow-up question chips, emitted once just before `done` |
| `done`        | —                             | Stream finished cleanly |
| `error`       | `message`                     | An exception was raised mid-stream |

For a **text-only** mobile UI, you only need to handle `token` (concat into
a Markdown buffer) and `done` (close the stream). `status` events power the
typing/thinking indicator.

### Thread lifecycle

The first call to `/chat/stream` with a new `thread_id` auto-creates a
row in `chat_threads` (title = first 40 chars of the user message).
List, rename, and delete threads via the `/threads` endpoints.
`DELETE /threads/{id}` hard-deletes the thread plus all LangGraph
checkpoints in one transaction.

### Proposal save / dismiss intents

When the user taps Save or Discard on a transaction proposal card,
mobile collapses the card in place — there is **no** separate chat
ack bubble. The mobile client fires **`POST /chat/intent`** which
appends ONLY a marker `HumanMessage` to the LangGraph checkpoint
silently (no LLM round-trip, no `AIMessage`). This used to flow
through `/chat/stream`'s in-graph `confirmation` lane and used to
also persist an ack; both behaviours have been removed.
"""

TAGS_METADATA = [
    {
        "name": "Chat",
        "description": "Streaming conversational endpoints (SSE) and conversation history.",
    },
    {
        "name": "Threads",
        "description": "Sidebar metadata: create / list / rename / delete chat sessions.",
    },
    {
        "name": "System",
        "description": "Health and diagnostic endpoints.",
    },
]

app = FastAPI(
    title="Mint Money Chat Agent",
    version="1.0.0",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    contact={"name": "Mint Money", "email": "ittipong.it@gmail.com"},
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """Attach a correlation id to every request.

    Honors a caller-supplied `X-Request-Id` (so mobile can stamp its own
    and join logs end-to-end) or generates a uuid4 when absent. The id
    lives in a ContextVar so any `debug_log.log()` call inside the
    request scope — including LangGraph nodes — picks it up without an
    explicit parameter.
    """
    rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    set_request_id(rid)
    try:
        response = await call_next(request)
    finally:
        # Clear so background tasks spawned after the response don't
        # leak the previous request's id.
        set_request_id(None)
    response.headers["X-Request-Id"] = rid
    return response


# ── Request schemas ───────────────────────────────────────────────────────────

_EXAMPLE_USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
_EXAMPLE_THREAD_ID = "thread-2026-05-12-001"


class ChatRequest(BaseModel):
    """Payload for `POST /chat/stream`."""
    user_id: str = Field(..., description="Owner user UUID")
    thread_id: str = Field(..., description="Client-generated thread id (UUID). Used for both routing and checkpoint key.")
    message: str = Field(..., description="User's natural-language input (Thai)", min_length=1)
    image_b64s: list[str] = Field(
        default_factory=list,
        description=(
            "Optional data URLs (e.g. 'data:image/jpeg;base64,...') for "
            "slip-to-transaction turns. When present, the graph routes "
            "to the vision subgraph instead of the regular ReAct loop. "
            "Images are forwarded to the vision LLM verbatim and never "
            "persisted server-side."
        ),
    )
    default_currency_code: str = Field(
        default="THB",
        description=(
            "User's preferred display currency, sourced from the app's "
            "settings screen. Quick-add proposals default to this code; "
            "slip parsing only uses it as a fallback when the slip itself "
            "doesn't carry a currency."
        ),
    )
    default_currency_symbol: str = Field(
        default="฿",
        description="Display symbol that pairs with default_currency_code.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": _EXAMPLE_USER_ID,
                "thread_id": _EXAMPLE_THREAD_ID,
                "message": "ใช้เงินไปเท่าไรเดือนนี้",
                "default_currency_code": "THB",
                "default_currency_symbol": "฿",
            }
        }
    )


class StudioChatRequest(BaseModel):
    """LangGraph Studio JSON format — mirrors `ChatRequest`."""
    user_id: str = Field(..., description="Owner user UUID")
    thread_id: str = Field(..., description="Thread id")
    message: str = Field(..., min_length=1, description="User message")
    image_b64s: list[str] = Field(
        default_factory=list,
        description="Optional data URLs for slip-to-transaction turns.",
    )
    default_currency_code: str = Field(default="THB")
    default_currency_symbol: str = Field(default="฿")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": _EXAMPLE_USER_ID,
                "thread_id": _EXAMPLE_THREAD_ID,
                "message": "งบเดือนนี้เหลือเท่าไหร่",
                "default_currency_code": "THB",
                "default_currency_symbol": "฿",
            }
        }
    )


class CreateThreadRequest(BaseModel):
    user_id: str = Field(..., description="Owner user UUID")
    title: str | None = Field(None, description="Optional initial title. If omitted, defaults to 'แชตใหม่' until the first user message overrides it.")

    model_config = ConfigDict(
        json_schema_extra={"example": {"user_id": _EXAMPLE_USER_ID, "title": None}}
    )


class RenameThreadRequest(BaseModel):
    title: str = Field(..., min_length=1, description="New display title")

    model_config = ConfigDict(
        json_schema_extra={"example": {"title": "งบประมาณเดือน พ.ค. 2026"}}
    )


class IntentRequest(BaseModel):
    """Payload for `POST /chat/intent`.

    Sent by mobile (fire-and-forget) when the user taps Save or Discard
    on a ProposedTransactionGroupCard. The server appends ONLY a marker
    `HumanMessage` to the checkpoint WITHOUT running the graph, calling
    the LLM, or persisting any AIMessage ack. Mobile's "collapsed card"
    UX is itself the acknowledgement — no separate chat bubble.
    """

    user_id: str = Field(..., description="Owner user UUID")
    thread_id: str = Field(..., description="Thread the proposal lives in")
    action: Literal["transaction_saved", "transaction_dismissed"] = Field(
        ...,
        description="Which button the user tapped on the proposal card.",
    )
    group_id: str = Field(
        ...,
        description="Identifier of the proposal group the user acted on.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": _EXAMPLE_USER_ID,
                "thread_id": _EXAMPLE_THREAD_ID,
                "action": "transaction_saved",
                "group_id": "group-2026-05-17-001",
            }
        }
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class ThreadOut(BaseModel):
    """A single chat thread row."""
    thread_id: str
    user_id: str
    title: str
    last_message_preview: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": _EXAMPLE_THREAD_ID,
                "user_id": _EXAMPLE_USER_ID,
                "title": "ใช้เงินไปเท่าไรเดือนนี้",
                "last_message_preview": "ระหว่างวันที่ 1 เม.ย. - 12 พ.ค. 2026 คุณใช้เงินไปทั้งหมด 51,762.80 บาท...",
                "message_count": 2,
                "created_at": "2026-05-12T14:55:35.678552Z",
                "updated_at": "2026-05-12T14:55:55.304049Z",
            }
        }
    )


class ListThreadsOut(BaseModel):
    threads: list[ThreadOut]
    next_cursor: str | None = Field(None, description="Reserved for future pagination — currently always null.")


class RenameThreadOut(BaseModel):
    thread_id: str
    title: str
    updated_at: datetime


class DeleteThreadOut(BaseModel):
    status: str = Field(..., examples=["deleted"])
    thread_id: str
    found: bool = Field(..., description="True if a row was deleted, False if the thread_id did not exist.")


class IntentOut(BaseModel):
    """Response for `POST /chat/intent` — confirms the marker landed."""

    thread_id: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thread_id": _EXAMPLE_THREAD_ID,
            }
        }
    )


class HistoryMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str
    suggestions: list[str] = Field(
        default_factory=list,
        description="Follow-up suggestion chips extracted from the assistant reply.",
    )


class HistoryOut(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class HealthOut(BaseModel):
    status: str = Field(..., examples=["ok"])
    react_model: str
    codeact_model: str


class ErrorOut(BaseModel):
    detail: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_STATUS_LABELS = {
    "thinking": "กำลังคิด...",
    "calculating": "กำลังคำนวณ...",
    "writing": "กำลังเรียบเรียงคำตอบ...",
    "reading_slip": "กำลังอ่านสลิป...",
    "saving_slip": "กำลังจัดรายการ...",
}

# Nodes that mean the agent is doing heavy compute (CodeAct / analyze
# subgraph). Entering any of these emits the `calculating` status.
_CALCULATING_NODES = {"act", "analyze"}

# Slip subgraph nodes — image turns bypass the ReAct loop entirely, so
# they need their own phase labels to drive the typing indicator. The
# `slip` node runs the vision LLM (reading the receipt) and `slip_tool`
# validates + dispatches the propose_transaction event.
_SLIP_READING_NODES = {"slip"}
_SLIP_SAVING_NODES = {"slip_tool"}


# ── Stream pacing ─────────────────────────────────────────────────────────────
# How often to emit an SSE keep-alive when the graph is silent. Mobile
# clients read this to detect dead connections; intermediaries (nginx,
# Cloudflare, ngrok) also reset idle TCP after ~30–60s, so a 15s tick
# keeps the pipe warm.
_HEARTBEAT_SEC = 15.0
# Hard ceiling: if no real event arrives within this window the stream
# is presumed wedged (LLM timeout will normally trip first at 60s per
# call; this is the safety net for tool/subgraph hangs).
_HARD_TIMEOUT_SEC = 120.0


# ── <suggestions> tag streaming filter ────────────────────────────────────────

_SUGGEST_OPEN = "<suggestions>"
_SUGGEST_CLOSE = "</suggestions>"

# Some models (notably Amazon Nova Lite) emit `<thinking>...</thinking>`
# blocks alongside the user-facing reply. The block is the model's
# internal reasoning and must NOT reach the chat UI — strip it in the
# token stream before suggestions parsing runs.
_THINK_OPEN = "<thinking>"
_THINK_CLOSE = "</thinking>"
# Extracts and strips <suggestions>…</suggestions> from stored message content.
_HISTORY_SUGGEST_RE = re.compile(
    r"<suggestions>(.*?)</suggestions>", re.DOTALL
)
# Holdback enough chars so a tag split across token chunks isn't leaked.
_SUGGEST_HOLD = max(len(_SUGGEST_OPEN), len(_SUGGEST_CLOSE))


class _SuggestionsFilter:
    """Streaming filter that strips <suggestions>...</suggestions> blocks
    from the visible token stream and accumulates their JSON payload.

    Tokens arrive split arbitrarily — a single tag may straddle multiple
    chunks. The filter holds back up to len(close_tag) chars on the trailing
    edge to prevent leaking a partial `<suggest` to the client. On `flush`
    (stream end) it drains the holdback and returns any parsed payloads.
    """

    def __init__(self) -> None:
        self._in_tag = False
        self._hold = ""           # trailing buffer (potential tag prefix)
        self._payload_buf = ""    # accumulated content between open and close
        self._payloads: list[list[str]] = []

    @staticmethod
    def _trailing_partial(text: str, target: str) -> int:
        """Length of the suffix of `text` that is a prefix of `target`."""
        max_len = min(len(text), len(target) - 1)
        for n in range(max_len, 0, -1):
            if target.startswith(text[-n:]):
                return n
        return 0

    def feed(self, token: str) -> str:
        """Process a streaming token. Returns the (possibly empty) substring
        that is safe to emit to the client right now."""
        buf = self._hold + token
        self._hold = ""
        out = ""

        while buf:
            if not self._in_tag:
                idx = buf.find(_SUGGEST_OPEN)
                if idx != -1:
                    out += buf[:idx]
                    buf = buf[idx + len(_SUGGEST_OPEN):]
                    self._in_tag = True
                    continue
                # No open tag found. Hold back any suffix that could be a
                # partial open tag so the client never sees "<suggest".
                hold_len = self._trailing_partial(buf, _SUGGEST_OPEN)
                if hold_len:
                    out += buf[:-hold_len]
                    self._hold = buf[-hold_len:]
                else:
                    out += buf
                buf = ""
            else:
                idx = buf.find(_SUGGEST_CLOSE)
                if idx != -1:
                    self._payload_buf += buf[:idx]
                    self._commit_payload()
                    buf = buf[idx + len(_SUGGEST_CLOSE):]
                    self._in_tag = False
                    continue
                hold_len = self._trailing_partial(buf, _SUGGEST_CLOSE)
                if hold_len:
                    self._payload_buf += buf[:-hold_len]
                    self._hold = buf[-hold_len:]
                else:
                    self._payload_buf += buf
                buf = ""

        return out

    def flush(self) -> str:
        """Stream finished — drain remaining holdback as visible text.
        If we're still mid-tag (malformed output) the payload is discarded."""
        if self._in_tag:
            # The LLM never closed the tag — drop the partial payload.
            self._payload_buf = ""
            self._hold = ""
            return ""
        tail = self._hold
        self._hold = ""
        return tail

    def _commit_payload(self) -> None:
        raw = self._payload_buf.strip()
        self._payload_buf = ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            _debug_log("SUGGEST", "Invalid JSON payload", raw=raw[:120])
            return
        if not isinstance(data, list):
            _debug_log("SUGGEST", "Payload is not a list", raw=raw[:120])
            return
        items = [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
        if items:
            self._payloads.append(items)

    @property
    def payloads(self) -> list[list[str]]:
        return self._payloads


class _ThinkingFilter:
    """Streaming filter that drops `<thinking>...</thinking>` blocks.

    Simpler than [_SuggestionsFilter] because we have no payload to
    capture — the content between the tags is the model's reasoning
    and gets thrown away. The trailing holdback prevents leaking a
    partial open tag (e.g. "<think") to the client when the chunk
    boundary lands mid-tag.
    """

    def __init__(self) -> None:
        self._in_tag = False
        self._hold = ""

    @staticmethod
    def _trailing_partial(text: str, target: str) -> int:
        max_len = min(len(text), len(target) - 1)
        for n in range(max_len, 0, -1):
            if target.startswith(text[-n:]):
                return n
        return 0

    def feed(self, token: str) -> str:
        buf = self._hold + token
        self._hold = ""
        out = ""

        while buf:
            if not self._in_tag:
                idx = buf.find(_THINK_OPEN)
                if idx != -1:
                    out += buf[:idx]
                    buf = buf[idx + len(_THINK_OPEN):]
                    self._in_tag = True
                    continue
                hold_len = self._trailing_partial(buf, _THINK_OPEN)
                if hold_len:
                    out += buf[:-hold_len]
                    self._hold = buf[-hold_len:]
                else:
                    out += buf
                buf = ""
            else:
                idx = buf.find(_THINK_CLOSE)
                if idx != -1:
                    buf = buf[idx + len(_THINK_CLOSE):]
                    self._in_tag = False
                    continue
                hold_len = self._trailing_partial(buf, _THINK_CLOSE)
                if hold_len:
                    self._hold = buf[-hold_len:]
                # Else discard buf entirely — it's reasoning text.
                buf = ""

        return out

    def flush(self) -> str:
        """Stream finished — emit any remaining holdback if we're outside
        a tag; otherwise discard (malformed unterminated reasoning)."""
        if self._in_tag:
            self._hold = ""
            return ""
        tail = self._hold
        self._hold = ""
        return tail


async def _stream_graph(
    user_id: str,
    thread_id: str,
    message: str,
    image_b64s: list[str] | None = None,
    audio_data: bytes | None = None,
    audio_mime: str | None = None,
    default_currency_code: str = "THB",
    default_currency_symbol: str = "฿",
) -> AsyncGenerator[str, None]:
    image_b64s = image_b64s or []
    _debug_log(
        "STREAM",
        "Starting",
        user_id=user_id,
        thread_id=thread_id,
        message=message[:50],
        image_count=len(image_b64s),
        has_audio=bool(audio_data),
        audio_bytes=(len(audio_data) if audio_data else 0),
        audio_mime=audio_mime or "",
        default_currency_code=default_currency_code,
    )

    # Upsert thread metadata before the run — first message becomes the title.
    # Voice turns start with an empty placeholder; the transcript only
    # exists after stt_node runs, so we skip the thread upsert here for
    # voice and let the post-run preview update carry the load.
    if _pool is not None and message:
        try:
            inserted = await threads_repo.upsert_on_first_message(
                _pool, thread_id, user_id, message,
            )
            if inserted:
                _debug_log("THREAD", "Created", thread_id=thread_id)
        except Exception as exc:
            _debug_log("THREAD", "Upsert failed", error=str(exc), thread_id=thread_id)

    # `user_id` is forwarded into `configurable` so tools executed by
    # LangGraph's ToolNode (e.g. `lookup_entity`) can access it via
    # injected `RunnableConfig` — state isn't reachable from there.
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    # For voice turns the message starts empty — stt_node replaces the
    # latest HumanMessage with the transcript before any downstream
    # node looks at it. Always include a placeholder so LangGraph has
    # something to anchor the turn on (the message-add reducer drops
    # empty content, but the HumanMessage object itself must exist for
    # the entry router to find it).
    placeholder_human = HumanMessage(content=message or "[INTENT:voice_pending]")
    input_data: dict[str, Any] = {
        "messages": [placeholder_human],
        "user_id": user_id,
        # Always reset `images` per turn. The checkpointer persists every
        # state field across turns; if we omit `images` on a text-only
        # follow-up the previous slip's data URLs survive in state and
        # `_route_by_input` re-routes the turn through the vision
        # subgraph — producing a stray proposal for a question like "hi".
        # Explicit empty list forces the router back onto the ReAct lane.
        "images": image_b64s or [],
        # Same reset story for the voice fields — without these a stale
        # `audio_data` from a previous turn would survive in state and
        # mis-route the next plain-text turn into stt_node.
        "audio_data": audio_data,
        "audio_mime": audio_mime,
        "stt_error": None,
        "transcript": None,
        # Forward the user's app currency setting so quick_add and
        # slip's never-null fallbacks default to it instead of THB.
        "default_currency_code": default_currency_code,
        "default_currency_symbol": default_currency_symbol,
    }

    # Track status emission so we never repeat a phase inside one run.
    emitted_phases: set[str] = set()
    assistant_reply_chunks: list[str] = []
    suggestions_filter = _SuggestionsFilter()
    # Strip <thinking>...</thinking> blocks BEFORE suggestions parsing
    # so the inner content (model reasoning) never reaches the chat UI.
    thinking_filter = _ThinkingFilter()
    # `on_tool_start` → start ts; popped on `on_tool_end` to compute
    # duration. Keyed by LangChain run_id so concurrent tool calls
    # (e.g., parallel `get_financial_advice` invocations) don't collide.
    tool_starts: dict[str, tuple[str, float]] = {}

    def _status_event(phase: str) -> str:
        return _sse({
            "type": "status",
            "phase": phase,
            "label": _STATUS_LABELS.get(phase, phase),
        })

    # First SSE event — gives the client a stable id to echo back when
    # reporting a bug. Mobile parsers should ignore unknown `type`
    # values, so this is forward-compatible.
    yield _sse({
        "type": "meta",
        "request_id": get_request_id() or "",
        "thread_id": thread_id,
        "model": settings.react_model,
    })

    # Manual iteration so we can interleave SSE heartbeats during quiet
    # stretches and bail out hard if the graph wedges. The previous
    # `async for` had no way to inject keep-alives or to observe how
    # long we'd been idle.
    #
    # Why `asyncio.wait` and not `asyncio.wait_for`: `wait_for` cancels
    # the awaited coroutine on timeout, which would propagate
    # CancelledError into the LangGraph async generator and close it —
    # so the first heartbeat would silently kill the stream. We instead
    # keep the same pending Task across heartbeat ticks and only cancel
    # it on the hard-timeout escape hatch.
    events_iter = _graph.astream_events(
        input_data, config, version="v2"
    ).__aiter__()
    last_real_event = time.monotonic()
    stream_aborted = False
    pending: asyncio.Task | None = None

    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(events_iter.__anext__())

            done, _pending_set = await asyncio.wait(
                {pending}, timeout=_HEARTBEAT_SEC
            )

            if pending not in done:
                # Heartbeat tick — the iterator is still working.
                idle_sec = time.monotonic() - last_real_event
                if idle_sec >= _HARD_TIMEOUT_SEC:
                    _debug_log(
                        "STREAM",
                        "Hard timeout — aborting",
                        user_id=user_id,
                        thread_id=thread_id,
                        idle_sec=int(idle_sec),
                    )
                    pending.cancel()
                    yield _sse({
                        "type": "error",
                        "message": (
                            f"LLM unresponsive for {int(idle_sec)}s — "
                            "request aborted"
                        ),
                    })
                    stream_aborted = True
                    break
                _debug_log(
                    "STREAM",
                    "Heartbeat",
                    idle_sec=int(idle_sec),
                    _level=LogLevel.DETAIL,
                )
                # SSE comment line — clients (Dio / EventSource) ignore
                # it but the framing keeps proxies from idling the TCP.
                yield ": keep-alive\n\n"
                continue

            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            finally:
                pending = None

            last_real_event = time.monotonic()
            kind = event["event"]
            node = (event.get("metadata") or {}).get("langgraph_node")

            # === Status: thinking (on first reason entry) ===
            # === Status: calculating (on first act/analyze entry) ===
            # === Status: reading_slip / saving_slip (slip subgraph) ===
            if kind == "on_chain_start":
                if node == "reason" and "thinking" not in emitted_phases:
                    emitted_phases.add("thinking")
                    yield _status_event("thinking")
                elif node in _CALCULATING_NODES and "calculating" not in emitted_phases:
                    emitted_phases.add("calculating")
                    yield _status_event("calculating")
                elif node in _SLIP_READING_NODES and "reading_slip" not in emitted_phases:
                    emitted_phases.add("reading_slip")
                    yield _status_event("reading_slip")
                elif node in _SLIP_SAVING_NODES and "saving_slip" not in emitted_phases:
                    emitted_phases.add("saving_slip")
                    yield _status_event("saving_slip")

            # === Status: writing (when calculating finishes — clean handoff) ===
            elif kind == "on_chain_end" and node in _CALCULATING_NODES:
                if "writing" not in emitted_phases:
                    emitted_phases.add("writing")
                    yield _status_event("writing")

            elif kind == "on_chat_model_stream":
                # Stream tokens from user-facing nodes only. `reason`
                # streams the ReAct final answer; `quick_add` streams
                # its ask-back text. Proposal save / dismiss acks are
                # NOT streamed here — they land via `POST /chat/intent`
                # and never run through the graph.
                if node not in ("reason", "quick_add"):
                    continue
                chunk = event["data"]["chunk"]
                if not chunk.content:
                    continue
                # Note: `writing` is emitted via on_chain_end for the
                # calculating nodes (clean handoff). We deliberately
                # don't fire writing here — if no calculation happens
                # (pure conversational), `thinking` stays visible until
                # `done`, which is acceptable UX given tokens are
                # streaming visibly.
                # Two-stage filter: strip <thinking> first (model reasoning
                # that must never reach the UI), then strip <suggestions>
                # (extracted into a separate SSE event after the run).
                no_thinking = thinking_filter.feed(chunk.content)
                if not no_thinking:
                    continue
                visible = suggestions_filter.feed(no_thinking)
                if visible:
                    assistant_reply_chunks.append(visible)
                    yield _sse({"type": "token", "content": visible})

            elif kind == "on_tool_start":
                run_id = str(event.get("run_id") or "")
                tool_starts[run_id] = (event["name"], time.monotonic())
                _debug_log(
                    "STREAM",
                    "Tool started",
                    tool=event["name"],
                    tool_run_id=run_id[:8] or "-",
                )
                yield _sse({"type": "tool_start", "tool": event["name"]})

            elif kind == "on_tool_end":
                run_id = str(event.get("run_id") or "")
                started = tool_starts.pop(run_id, None)
                duration_ms = (
                    int((time.monotonic() - started[1]) * 1000)
                    if started
                    else -1
                )
                _debug_log(
                    "STREAM",
                    "Tool ended",
                    tool=event["name"],
                    tool_run_id=run_id[:8] or "-",
                    duration_ms=duration_ms,
                )
                yield _sse({"type": "tool_end", "tool": event["name"]})

            elif kind == "on_custom_event" and event.get("name") == "structured_data":
                payload = event.get("data")
                if payload is not None:
                    _debug_log(
                        "STREAM",
                        "data event",
                        payload_type=payload.get("type"),
                        kind=payload.get("kind"),
                        metric=payload.get("metric"),
                        rows=len(payload.get("rows") or []) if isinstance(payload.get("rows"), list) else None,
                    )
                    yield _sse({"type": "data", "payload": payload})
                else:
                    _debug_log("STREAM", "data event with null payload — skipped")

            elif kind == "on_custom_event" and event.get("name") == "stt_transcript":
                # Voice lane — emit BEFORE any token events so the
                # mobile client can render the transcript in the user
                # bubble while the assistant's reply is still streaming.
                payload = event.get("data") or {}
                transcript = (payload.get("text") or "").strip()
                _debug_log(
                    "STREAM",
                    "transcript event",
                    transcript_len=len(transcript),
                    transcript_preview=transcript[:80].replace("\n", " "),
                )
                # Persist now so threads_repo has the real first message
                # (the placeholder we sent in was "[INTENT:voice_pending]").
                # The thread is auto-created on first text turn; voice
                # turns hit this branch instead.
                if _pool is not None and transcript:
                    try:
                        inserted = await threads_repo.upsert_on_first_message(
                            _pool, thread_id, user_id, transcript,
                        )
                        if inserted:
                            _debug_log("THREAD", "Created (voice)", thread_id=thread_id)
                    except Exception as exc:
                        _debug_log(
                            "THREAD",
                            "Upsert failed (voice)",
                            error=str(exc),
                            thread_id=thread_id,
                        )
                yield _sse({"type": "transcript", "text": transcript})

            elif kind == "on_custom_event" and event.get("name") == "stt_error_event":
                payload = event.get("data") or {}
                reason = payload.get("reason") or "transcription_failed"
                _debug_log(
                    "STREAM",
                    "stt_error event",
                    reason=reason,
                )
                yield _sse({"type": "stt_error", "reason": reason})

        if stream_aborted:
            # The hard-timeout branch already emitted an `error` event.
            # Skip the success-path tail/suggestions/persist — there's no
            # full reply to save, and emitting `done` after `error` would
            # confuse clients.
            return

        _debug_log("STREAM", "Done", user_id=user_id)

        # Drain holdbacks from both filters in the same order they run
        # during streaming (thinking → suggestions). The thinking tail
        # may be empty if the model never opened a `<think` prefix.
        thinking_tail = thinking_filter.flush()
        if thinking_tail:
            visible_tail = suggestions_filter.feed(thinking_tail)
            if visible_tail:
                assistant_reply_chunks.append(visible_tail)
                yield _sse({"type": "token", "content": visible_tail})
        tail = suggestions_filter.flush()
        if tail:
            assistant_reply_chunks.append(tail)
            yield _sse({"type": "token", "content": tail})

        # Emit follow-up suggestion chips (if the LLM included a tag).
        for items in suggestions_filter.payloads:
            _debug_log("SUGGEST", "Emit", count=len(items))
            yield _sse({"type": "suggestions", "items": items})

        # Persist preview + counters after the AI turn finished cleanly.
        if _pool is not None and assistant_reply_chunks:
            try:
                await threads_repo.update_after_message(
                    _pool, thread_id, "".join(assistant_reply_chunks),
                )
            except Exception as exc:
                _debug_log("THREAD", "Update preview failed", error=str(exc), thread_id=thread_id)

        yield _sse({"type": "done"})

    except Exception as exc:
        _debug_log("STREAM", "Error", error=str(exc), user_id=user_id)
        yield _sse({"type": "error", "message": str(exc)})




# ── Endpoints ─────────────────────────────────────────────────────────────────

# Shared OpenAPI definition for the streaming SSE response.
_SSE_RESPONSES: dict[int | str, dict] = {
    200: {
        "description": "Server-Sent Event stream. Each line is `data: <json>` with `type` in {status, tool_start, tool_end, data, token, done, error}.",
        "content": {
            "text/event-stream": {
                "example": (
                    'data: {"type":"status","phase":"thinking","label":"🧠 AI กำลังคิด..."}\n\n'
                    'data: {"type":"status","phase":"calculating","label":"🧮 AI คำนวณข้อมูล..."}\n\n'
                    'data: {"type":"status","phase":"writing","label":"✍️ AI เรียบเรียงคำตอบ..."}\n\n'
                    'data: {"type":"token","content":"สวั"}\n\n'
                    'data: {"type":"token","content":"สดี"}\n\n'
                    'data: {"type":"suggestions","items":["ค่าใช้จ่ายหมวดไหนเยอะสุด","งบประมาณเดือนนี้เหลือเท่าไร","เปรียบเทียบกับเดือนที่แล้ว"]}\n\n'
                    'data: {"type":"done"}\n\n'
                )
            }
        },
    },
    503: {"model": ErrorOut, "description": "Graph or database not yet initialized."},
}


@app.post(
    "/chat/stream",
    tags=["Chat"],
    summary="Stream a chat reply (SSE)",
    description=(
        "Send a user message and receive a `text/event-stream` reply. "
        "The first call with a previously-unseen `thread_id` auto-creates "
        "the thread row (title = first 40 chars of the message). After the "
        "`done` event, the thread's `last_message_preview` and "
        "`message_count` are updated."
    ),
    responses=_SSE_RESPONSES,
)
async def chat_stream(req: ChatRequest, request: Request):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    client = request.client
    _debug_log(
        "HTTP",
        "POST /chat/stream",
        client=f"{client.host}:{client.port}" if client else "unknown",
        thread_id=req.thread_id,
        user_id=req.user_id,
        message=req.message[:80],
    )

    return StreamingResponse(
        _stream_graph(
            req.user_id,
            req.thread_id,
            req.message,
            req.image_b64s,
            default_currency_code=req.default_currency_code,
            default_currency_symbol=req.default_currency_symbol,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Voice upload — capped well below the 6 MB hard cap inside stt_node.
# Mobile clips are typically <250 KB at 60s (AAC 32 kbps), so a 5 MB
# ceiling leaves headroom for an outlier without inviting abuse.
_MAX_VOICE_BYTES = 5 * 1024 * 1024

# Suffix → mime fallback when the multipart envelope lies (some HTTP
# clients send `application/octet-stream` for everything).
_VOICE_SUFFIX_MIME = {
    ".m4a": "audio/m4a",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


def _detect_audio_mime(upload: UploadFile) -> str:
    """Decide on the mime to feed stt_node.

    Priority: declared `content_type` (when not the generic
    octet-stream placeholder) → filename extension → "audio/m4a".
    """
    declared = (upload.content_type or "").lower()
    if declared and declared not in {"application/octet-stream", ""}:
        return declared
    filename = (upload.filename or "").lower()
    for suffix, mime in _VOICE_SUFFIX_MIME.items():
        if filename.endswith(suffix):
            return mime
    return "audio/m4a"


@app.post(
    "/chat/voice",
    tags=["Chat"],
    summary="Stream a chat reply for an uploaded voice clip (SSE)",
    description=(
        "Multipart upload (`audio` file + `user_id` + `thread_id`). The "
        "server runs speech-to-text on the clip, then routes the "
        "transcript through the same graph as `/chat/stream`. Emits two "
        "voice-specific SSE event types **in addition to** the standard "
        "set documented on `/chat/stream`:\n\n"
        "| `type`        | Payload fields | When |\n"
        "|---------------|----------------|------|\n"
        "| `transcript`  | `text`          | Emitted once after STT, before any token. |\n"
        "| `stt_error`   | `reason` (`no_speech`/`too_short`/`transcription_failed`) | STT failed — token stream will not appear. |\n\n"
        "Accepted audio formats: m4a / aac / mp3 / wav / ogg / webm / "
        "flac, up to 5 MB. The mobile client uploads ~60s AAC clips."
    ),
    responses=_SSE_RESPONSES
    | {
        413: {"model": ErrorOut, "description": "Audio exceeds 5 MB limit."},
        400: {"model": ErrorOut, "description": "Missing or empty audio file."},
    },
)
async def chat_voice(
    request: Request,
    audio: UploadFile = File(..., description="Audio clip (m4a/aac/mp3/wav, ≤60s, ≤5 MB)"),
    user_id: str = Form(..., description="Owner user UUID"),
    thread_id: str = Form(..., description="Client-generated thread id (UUID)."),
):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    client = request.client
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio file is empty")
    if len(audio_bytes) > _MAX_VOICE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio exceeds {_MAX_VOICE_BYTES // (1024 * 1024)} MB limit",
        )

    mime = _detect_audio_mime(audio)
    _debug_log(
        "HTTP",
        "POST /chat/voice",
        client=f"{client.host}:{client.port}" if client else "unknown",
        thread_id=thread_id,
        user_id=user_id,
        filename=audio.filename or "",
        content_type=audio.content_type or "",
        resolved_mime=mime,
        bytes=len(audio_bytes),
    )

    return StreamingResponse(
        _stream_graph(
            user_id=user_id,
            thread_id=thread_id,
            # Empty message — stt_node replaces the latest HumanMessage
            # with the produced transcript before any downstream node
            # reads it.
            message="",
            image_b64s=None,
            audio_data=audio_bytes,
            audio_mime=mime,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


_VALID_INTENT_ACTIONS = {"transaction_saved", "transaction_dismissed"}


@app.post(
    "/chat/intent",
    tags=["Chat"],
    summary="Persist a save/dismiss proposal intent (silent — no LLM)",
    description=(
        "Fire-and-forget endpoint for the mobile client. When the user "
        "taps Save or Discard on a `ProposedTransactionGroupCard`, mobile "
        "collapses the card in place — there is no separate chat ack "
        "bubble. The server appends ONLY a marker `HumanMessage` "
        "(`[INTENT:<action>] group_id=<id>`) to the LangGraph checkpoint "
        "via `aupdate_state` — the graph is NEVER invoked, NO LLM call "
        "is made, and NO `AIMessage` ack is persisted. This replaces the "
        "previous in-graph `confirmation` lane that routed the same "
        "markers through `/chat/stream` and burned a full LLM round-trip "
        "on a UI action that doesn't need one."
    ),
    response_model=IntentOut,
    responses={
        400: {"model": ErrorOut, "description": "Unknown action (non-production only)."},
        503: {"model": ErrorOut, "description": "Graph not ready."},
    },
)
async def chat_intent(req: IntentRequest, request: Request):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    # `action` is already constrained by the `Literal[...]` on the
    # request model, so pydantic rejects unknown values before we ever
    # get here in any environment. The explicit re-check below is a
    # belt-and-braces guard for the (rare) case where a future change
    # widens the field — production should soak the bad value into a
    # safe fallback per Postel's Law; dev/staging/test must fail fast
    # so the bug is caught before shipping.
    if req.action not in _VALID_INTENT_ACTIONS:
        if settings.is_production:
            _debug_log(
                "HTTP",
                "POST /chat/intent — unknown action, falling back to dismissed",
                thread_id=req.thread_id,
                action=req.action,
                group_id=req.group_id,
            )
            req = req.model_copy(update={"action": "transaction_dismissed"})
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown action '{req.action}' "
                    "(strict in non-production)"
                ),
            )

    client = request.client
    marker = f"[INTENT:{req.action}] group_id={req.group_id}"

    _debug_log(
        "HTTP",
        "POST /chat/intent",
        client=f"{client.host}:{client.port}" if client else "unknown",
        thread_id=req.thread_id,
        user_id=req.user_id,
        action=req.action,
        group_id=req.group_id,
        marker_preview=marker[:80],
    )

    config: dict[str, Any] = {"configurable": {"thread_id": req.thread_id}}

    # Single-message append. We no longer persist an ack AIMessage —
    # mobile's "collapsed card" UX is itself the acknowledgement, so a
    # chat-bubble ack would be visual duplication. Surface failures —
    # do NOT fall back to running the graph, the whole point is to
    # skip the LLM.
    await _graph.aupdate_state(
        config,
        {"messages": [HumanMessage(content=marker)]},
    )

    # Bump preview + counters mirroring the post-run update in
    # `/chat/stream`. We assume the thread row already exists (mobile
    # only fires this intent after a proposal turn, which itself ran
    # through `/chat/stream` and called `upsert_on_first_message`).
    # `update_after_marker` bumps `message_count` by exactly 1 — only
    # the marker HumanMessage was appended.
    if _pool is not None:
        try:
            await threads_repo.update_after_marker(
                _pool, req.thread_id, marker[:80],
            )
        except Exception as exc:
            _debug_log(
                "THREAD",
                "Intent preview update failed",
                error=str(exc),
                thread_id=req.thread_id,
            )

    return IntentOut(thread_id=req.thread_id)


@app.post(
    "/studio/chat",
    tags=["Chat"],
    summary="Stream a chat reply (LangGraph Studio compatible)",
    description=(
        "Identical behavior to `POST /chat/stream` — only the route differs. "
        "Provided so LangGraph Studio's default Send-Message JSON shape works "
        "without rewriting client code."
    ),
    responses=_SSE_RESPONSES,
)
async def studio_chat(req: StudioChatRequest, request: Request):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    client = request.client
    _debug_log(
        "HTTP",
        "POST /studio/chat",
        client=f"{client.host}:{client.port}" if client else "unknown",
        thread_id=req.thread_id,
        user_id=req.user_id,
        message=req.message[:80],
    )

    return StreamingResponse(
        _stream_graph(
            req.user_id,
            req.thread_id,
            req.message,
            req.image_b64s,
            default_currency_code=req.default_currency_code,
            default_currency_symbol=req.default_currency_symbol,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get(
    "/chat/{thread_id}",
    tags=["Chat"],
    summary="Get conversation history",
    description=(
        "Reads the LangGraph checkpoint state and returns the ordered list "
        "of user/assistant messages for the given thread. Returns an empty "
        "`messages` list if the thread has no checkpoints yet (e.g., the "
        "user created it but never sent a message)."
    ),
    response_model=HistoryOut,
    responses={503: {"model": ErrorOut, "description": "Graph not ready."}},
)
async def get_chat_history(thread_id: str):
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await _graph.aget_state(config)

    if not snapshot or not snapshot.values:
        return {"thread_id": thread_id, "messages": []}

    messages = []
    for msg in snapshot.values.get("messages", []):
        role = "user" if msg.type == "human" else "assistant"
        content = msg.content
        if not content:
            continue
        suggestions: list[str] = []
        if role == "assistant":
            # LangGraph stores the raw LLM output including <suggestions> tags.
            # Strip the tag from visible content and surface its payload as
            # structured suggestion chips, mirroring the streaming SSE event.
            match = _HISTORY_SUGGEST_RE.search(content)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, list):
                        suggestions = [
                            str(x).strip()
                            for x in data
                            if isinstance(x, (str, int, float)) and str(x).strip()
                        ]
                except (json.JSONDecodeError, ValueError):
                    pass
                content = _HISTORY_SUGGEST_RE.sub("", content).strip()
        if content:
            messages.append(
                {"role": role, "content": content, "suggestions": suggestions}
            )

    return {"thread_id": thread_id, "messages": messages}


@app.delete(
    "/chat/{thread_id}",
    tags=["Chat"],
    summary="Delete a thread (legacy alias)",
    description="Deprecated alias of `DELETE /threads/{thread_id}`. Kept for old clients.",
    response_model=DeleteThreadOut,
    responses={503: {"model": ErrorOut, "description": "Database not ready."}},
    deprecated=True,
)
async def delete_chat(thread_id: str):
    return await delete_thread(thread_id)


# ── Thread management ─────────────────────────────────────────────────────────

@app.post(
    "/threads",
    tags=["Threads"],
    status_code=201,
    summary="Create a new chat thread",
    description=(
        "Allocates a new thread_id (UUIDv4) and inserts a metadata row with "
        "the default title (`แชตใหม่`). The title is replaced automatically "
        "on the first `/chat/stream` call. You can also call `/chat/stream` "
        "directly with a fresh `thread_id` — this endpoint is only needed "
        "when the UI wants to open an empty session before the user types."
    ),
    response_model=ThreadOut,
    responses={503: {"model": ErrorOut, "description": "Database not ready."}},
)
async def create_thread(req: CreateThreadRequest):
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    row = await threads_repo.create_thread(_pool, req.user_id, req.title)
    _debug_log("THREAD", "Created", thread_id=row["thread_id"], user_id=req.user_id)
    return row


@app.get(
    "/threads",
    tags=["Threads"],
    summary="List a user's chat threads",
    description=(
        "Returns the user's threads sorted by `updated_at DESC` (most recent "
        "activity first). `next_cursor` is reserved for future pagination "
        "and is currently always `null`."
    ),
    response_model=ListThreadsOut,
    responses={503: {"model": ErrorOut, "description": "Database not ready."}},
)
async def list_threads(
    user_id: str = Query(..., description="Owner user UUID", examples=[_EXAMPLE_USER_ID]),
    limit: int = Query(50, ge=1, le=200, description="Max rows to return"),
):
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    rows = await threads_repo.list_threads(_pool, user_id, limit)
    return {"threads": rows, "next_cursor": None}


@app.patch(
    "/threads/{thread_id}",
    tags=["Threads"],
    summary="Rename a thread",
    description="Updates the thread's display title shown in the sidebar.",
    response_model=RenameThreadOut,
    responses={
        400: {"model": ErrorOut, "description": "Empty title."},
        404: {"model": ErrorOut, "description": "Thread not found."},
        503: {"model": ErrorOut, "description": "Database not ready."},
    },
)
async def rename_thread(thread_id: str, req: RenameThreadRequest):
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="title cannot be empty")
    row = await threads_repo.rename_thread(_pool, thread_id, req.title)
    if row is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return row


@app.delete(
    "/threads/{thread_id}",
    tags=["Threads"],
    summary="Delete a thread (hard, cascade)",
    description=(
        "Hard-deletes the thread metadata row **and** all LangGraph "
        "checkpoint rows for this thread (`checkpoints`, `checkpoint_blobs`, "
        "`checkpoint_writes`) in a single transaction. Idempotent — calling "
        "twice returns `found: false` on the second call."
    ),
    response_model=DeleteThreadOut,
    responses={503: {"model": ErrorOut, "description": "Database not ready."}},
)
async def delete_thread(thread_id: str):
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    deleted = await threads_repo.delete_thread_cascade(_pool, thread_id)
    _debug_log("THREAD", "Deleted", thread_id=thread_id, found=deleted)
    return {"status": "deleted", "thread_id": thread_id, "found": deleted}


@app.get(
    "/health",
    tags=["System"],
    summary="Liveness probe",
    description="Reports the currently configured ReAct and CodeAct model identifiers.",
    response_model=HealthOut,
)
async def health():
    return {
        "status": "ok",
        "react_model": settings.react_model,
        "codeact_model": settings.codeact_model,
    }
