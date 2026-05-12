"""FastAPI server with SSE streaming for the ReAct chat agent."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

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

app = FastAPI(title="Mint Money Chat Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class StudioChatRequest(BaseModel):
    """LangGraph Studio JSON format — mirrors the user chat payload."""
    user_id: str
    thread_id: str
    message: str


class CreateThreadRequest(BaseModel):
    user_id: str
    title: str | None = None


class RenameThreadRequest(BaseModel):
    title: str


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


async def _stream_graph(
    user_id: str,
    thread_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    _debug_log("STREAM", "Starting", user_id=user_id, thread_id=thread_id, message=message[:50])

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
    input_data = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
    }

    # Track status emission so we never repeat a phase inside one run.
    emitted_phases: set[str] = set()
    assistant_reply_chunks: list[str] = []

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
                assistant_reply_chunks.append(chunk.content)
                yield _sse({"type": "token", "content": chunk.content})

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

@app.post("/studio/chat")
async def studio_chat(req: StudioChatRequest, request: Request):
    """LangGraph Studio calls this endpoint with the user's JSON payload.

    JSON format:
    {
        "user_id": "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9",
        "thread_id": "test-thread",
        "message": "ใช้เงินไปเท่าไร"
    }
    """
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
        _stream_graph(req.user_id, req.thread_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Stream chat response as Server-Sent Events."""
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
        _stream_graph(req.user_id, req.thread_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/chat/{thread_id}")
async def get_chat_history(thread_id: str):
    """Get conversation history for a thread."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await _graph.aget_state(config)

    if not snapshot or not snapshot.values:
        return {"thread_id": thread_id, "messages": []}

    messages = []
    for msg in snapshot.values.get("messages", []):
        role = "user" if msg.type == "human" else "assistant"
        if msg.content:
            messages.append({"role": role, "content": msg.content})

    return {"thread_id": thread_id, "messages": messages}


@app.delete("/chat/{thread_id}")
async def delete_chat(thread_id: str):
    """Legacy alias — forwards to hard-delete cascade."""
    return await delete_thread(thread_id)


# ── Thread management ─────────────────────────────────────────────────────────

@app.post("/threads", status_code=201)
async def create_thread(req: CreateThreadRequest):
    """Create a new chat thread. Optional `title` overrides the default."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    row = await threads_repo.create_thread(_pool, req.user_id, req.title)
    _debug_log("THREAD", "Created", thread_id=row["thread_id"], user_id=req.user_id)
    return row


@app.get("/threads")
async def list_threads(
    user_id: str = Query(..., description="Owner user_id"),
    limit: int = Query(50, ge=1, le=200),
):
    """List a user's chat threads, newest activity first."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    rows = await threads_repo.list_threads(_pool, user_id, limit)
    return {"threads": rows, "next_cursor": None}


@app.patch("/threads/{thread_id}")
async def rename_thread(thread_id: str, req: RenameThreadRequest):
    """Rename a thread's display title."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="title cannot be empty")
    row = await threads_repo.rename_thread(_pool, thread_id, req.title)
    if row is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return row


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Hard-delete a thread and all its LangGraph checkpoints."""
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    deleted = await threads_repo.delete_thread_cascade(_pool, thread_id)
    _debug_log("THREAD", "Deleted", thread_id=thread_id, found=deleted)
    return {"status": "deleted", "thread_id": thread_id, "found": deleted}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "react_model": settings.react_model,
        "codeact_model": settings.codeact_model,
    }
