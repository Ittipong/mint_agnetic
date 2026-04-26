"""FastAPI server with SSE streaming for the ReAct chat agent."""

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.config import settings
from src.graph.agent_graph import build_async_graph


# ── Lifespan: set up checkpointer + graph once at startup ────────────────────

_graph: Any = None
_pool: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _pool

    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

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

    yield

    await _pool.close()


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_graph(
    user_id: str,
    thread_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}
    input_data = {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
    }

    try:
        async for event in _graph.astream_events(input_data, config, version="v2"):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})

            elif kind == "on_tool_start":
                yield _sse({"type": "tool_start", "tool": event["name"]})

            elif kind == "on_tool_end":
                yield _sse({"type": "tool_end", "tool": event["name"]})

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
