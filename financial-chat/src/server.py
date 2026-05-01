"""FastAPI server with SSE streaming for the ReAct chat agent."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.config import settings
from src.graph.agent_graph import build_async_graph

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_graph(
    user_id: str,
    thread_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    _debug_log("STREAM", "Starting", user_id=user_id, thread_id=thread_id, message=message[:50])
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
    }
    _debug_log("STREAM", "Input data prepared", user_id=input_data["user_id"])

    try:
        async for event in _graph.astream_events(input_data, config, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})

            elif kind == "on_tool_start":
                _debug_log("STREAM", "Tool started", tool=event["name"])
                yield _sse({"type": "tool_start", "tool": event["name"]})

            elif kind == "on_tool_end":
                _debug_log("STREAM", "Tool ended", tool=event["name"])
                yield _sse({"type": "tool_end", "tool": event["name"]})

        _debug_log("STREAM", "Done", user_id=user_id)
        yield _sse({"type": "done"})

    except Exception as exc:
        _debug_log("STREAM", "Error", error=str(exc), user_id=user_id)
        yield _sse({"type": "error", "message": str(exc)})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/studio/chat")
async def studio_chat(req: StudioChatRequest):
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
    """Clear conversation — client should use a new thread_id to start fresh."""
    return {
        "status": "ok",
        "thread_id": thread_id,
        "message": "Create a new thread_id to start a fresh conversation",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model}
