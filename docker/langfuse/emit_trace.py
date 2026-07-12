"""Emit ONE trace into the local Langfuse via the SAME handler server.py uses.

Proves the Phase-1 wiring end-to-end: attach_langfuse() -> LangChain callback ->
Langfuse ingestion, stamped with a known user_id + session_id (thread_id) so the
Playwright check can find it. Uses a FakeListChatModel so the run is deterministic
and needs no OpenRouter balance — the tracing path is model-agnostic (a real chat
turn traces identically; only the model differs).

Run from the mint_agentic repo root:  .venv/bin/python docker/langfuse/emit_trace.py
"""

from __future__ import annotations

import os

# Point at the local self-hosted Langfuse with the headless-init fixed keys.
os.environ["LANGFUSE_ENABLED"] = "on"
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-mint-poc-public")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-mint-poc-secret")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3100")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from src.agent.observability.langfuse_tracing import attach_langfuse

USER = "poc-user-verify"
SESSION = "poc-thread-verify"
MARKER = "MINT-POC-TRACE-MARKER"

# Exactly what server.py builds for a chat turn.
config = {"configurable": {"thread_id": SESSION}, "recursion_limit": 25}
config = attach_langfuse(config, user_id=USER, session_id=SESSION, tags=["chat", "poc"])
assert config.get("callbacks"), "handler not attached — check LANGFUSE_ENABLED / keys"
print(f"handler attached · metadata={config['metadata']}")

model = FakeListChatModel(responses=[f"สวัสดีครับ นี่คือ trace ทดสอบ {MARKER}"])
resp = model.invoke(
    [HumanMessage(content=f"ทดสอบ trace {MARKER}")],
    config={
        "callbacks": config["callbacks"],
        "metadata": config["metadata"],
        "run_name": "mint-chat-poc-turn",
    },
)
print("LLM resp:", resp.content)

# CRITICAL: the handler batches events — flush before exit or nothing is sent.
from langfuse import get_client  # noqa: E402

get_client().flush()
print(f"FLUSHED → user={USER} session={SESSION} marker={MARKER}")
