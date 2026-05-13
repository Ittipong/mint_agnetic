"""FastAPI server with SSE streaming for the ReAct chat agent."""

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.graph.agent_graph import build_async_graph
from src import threads_repo

# ── Debug Logger Setup ─────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_DEBUG_LOG = _LOG_DIR / f"chat_debug_{datetime.now().strftime('%Y-%m-%d')}.log"

def _debug_log(tag: str, msg: str, **kwargs):
    """Write structured debug log to file."""
    parts = [f"[{datetime.now().isoformat()}] [{tag}] {msg}"]
    for k, v in kwargs.items():
        parts.append(f" {k}={v}")
    log_line = "".join(parts) + "\n"
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)
    logging.info(log_line.strip())


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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": _EXAMPLE_USER_ID,
                "thread_id": _EXAMPLE_THREAD_ID,
                "message": "ใช้เงินไปเท่าไรเดือนนี้",
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": _EXAMPLE_USER_ID,
                "thread_id": _EXAMPLE_THREAD_ID,
                "message": "งบเดือนนี้เหลือเท่าไหร่",
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
    "thinking": "🧠 AI กำลังคิด...",
    "calculating": "🧮 AI คำนวณข้อมูล...",
    "writing": "✍️ AI เรียบเรียงคำตอบ...",
}

# Nodes that mean the agent is doing heavy compute (CodeAct / analyze
# subgraph). Entering any of these emits the `calculating` status.
_CALCULATING_NODES = {"act", "analyze"}


# ── <suggestions> tag streaming filter ────────────────────────────────────────

_SUGGEST_OPEN = "<suggestions>"
_SUGGEST_CLOSE = "</suggestions>"
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


async def _stream_graph(
    user_id: str,
    thread_id: str,
    message: str,
    image_b64s: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    image_b64s = image_b64s or []
    _debug_log(
        "STREAM",
        "Starting",
        user_id=user_id,
        thread_id=thread_id,
        message=message[:50],
        image_count=len(image_b64s),
    )

    # Upsert thread metadata before the run — first message becomes the title.
    if _pool is not None:
        try:
            inserted = await threads_repo.upsert_on_first_message(
                _pool, thread_id, user_id, message,
            )
            if inserted:
                _debug_log("THREAD", "Created", thread_id=thread_id)
        except Exception as exc:
            _debug_log("THREAD", "Upsert failed", error=str(exc), thread_id=thread_id)

    config = {"configurable": {"thread_id": thread_id}}
    input_data: dict[str, Any] = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
    }
    if image_b64s:
        # Only set when present so text-only turns don't trip the
        # slip router (checks truthiness, not presence).
        input_data["images"] = image_b64s

    # Track status emission so we never repeat a phase inside one run.
    emitted_phases: set[str] = set()
    assistant_reply_chunks: list[str] = []
    suggestions_filter = _SuggestionsFilter()

    def _status_event(phase: str) -> str:
        return _sse({
            "type": "status",
            "phase": phase,
            "label": _STATUS_LABELS.get(phase, phase),
        })

    try:
        async for event in _graph.astream_events(input_data, config, version="v2"):
            kind = event["event"]
            node = (event.get("metadata") or {}).get("langgraph_node")

            # === Status: thinking (on first reason entry) ===
            # === Status: calculating (on first act/analyze entry) ===
            if kind == "on_chain_start":
                if node == "reason" and "thinking" not in emitted_phases:
                    emitted_phases.add("thinking")
                    yield _status_event("thinking")
                elif node in _CALCULATING_NODES and "calculating" not in emitted_phases:
                    emitted_phases.add("calculating")
                    yield _status_event("calculating")

            # === Status: writing (when calculating finishes — clean handoff) ===
            elif kind == "on_chain_end" and node in _CALCULATING_NODES:
                if "writing" not in emitted_phases:
                    emitted_phases.add("writing")
                    yield _status_event("writing")

            elif kind == "on_chat_model_stream":
                # Only stream tokens from the user-facing reasoner.
                if node != "reason":
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
                # Strip <suggestions> tag from visible stream — surfaces as
                # a separate `suggestions` SSE event after the run finishes.
                visible = suggestions_filter.feed(chunk.content)
                if visible:
                    assistant_reply_chunks.append(visible)
                    yield _sse({"type": "token", "content": visible})

            elif kind == "on_tool_start":
                _debug_log("STREAM", "Tool started", tool=event["name"])
                yield _sse({"type": "tool_start", "tool": event["name"]})

            elif kind == "on_tool_end":
                _debug_log("STREAM", "Tool ended", tool=event["name"])
                yield _sse({"type": "tool_end", "tool": event["name"]})

            elif kind == "on_custom_event" and event.get("name") == "structured_data":
                payload = event.get("data")
                if payload is not None:
                    _debug_log(
                        "STREAM",
                        "data event",
                        kind=payload.get("kind"),
                        metric=payload.get("metric"),
                        rows=len(payload.get("rows") or []) if isinstance(payload.get("rows"), list) else None,
                    )
                    yield _sse({"type": "data", "payload": payload})

        _debug_log("STREAM", "Done", user_id=user_id)

        # Drain any holdback that didn't form a tag.
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
        _stream_graph(req.user_id, req.thread_id, req.message, req.image_b64s),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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
        _stream_graph(req.user_id, req.thread_id, req.message, req.image_b64s),
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
