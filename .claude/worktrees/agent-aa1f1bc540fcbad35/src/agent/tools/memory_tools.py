"""`memory_recall` + `memory_write` — cross-thread durable user memory.

NEW in Wave 3. Phase 2 spec: docs/v3/phase2_tools_design.md Tools 7 + 8.

Backed by a LangGraph `BaseStore` (InMemoryStore in tests, PostgresStore
in prod), namespaced as `("user", user_id, "facts")`. The store is
injected at compile time via `create_react_agent(..., store=store)`; the
tool reads it via `InjectedStore`.

Phase 3 starter is plain key-value (no embeddings) — recency-ordered
recall is good enough for ~20 facts/user. Phase 4 may swap in semantic
search; the tool signatures stay stable.

Caps & invariants:
- `memory_write`: note trimmed to 200 chars; tags capped at 3 (after
  individual whitespace strip); empty note → structured error.
- `memory_recall`: `k` clamped to 1..20 (defends against pathological
  prompts asking for k=1000).
- Both tools NEVER raise out — store errors degrade to {"error": ...}.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.session_logger import slog, slog_error


_RECALL_STATUS = "กำลังนึกย้อน..."
_WRITE_STATUS = "กำลังจดจำ..."

_MAX_NOTE_LEN = 200
_MAX_TAGS = 3
_RECALL_K_MIN = 1
_RECALL_K_MAX = 20


@tool
async def memory_recall(
    topic: str,
    k: int = 3,
    *,
    state: Annotated[dict, InjectedState],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Recall past advice / user preferences / decisions from long-term memory.

    Args:
      topic: Short search query (e.g. "debt", "savings goal", "kids").
      k: Max items to return (default 3, clamped to 1..20).

    Returns:
      {"memories": [{"note": str, "tags": [str], "ts": str, "id": str}, ...]}
      OR {"memories": [], "error": str} when the store call fails.
    """
    _emit_status(_RECALL_STATUS)

    user_id = state.get("user_id")
    if not user_id:
        return {"memories": [], "error": "state.user_id missing"}

    k_clamped = max(_RECALL_K_MIN, min(_RECALL_K_MAX, int(k or _RECALL_K_MIN)))
    namespace = ("user", user_id, "facts")

    try:
        items = await _store_asearch(store, namespace, query=topic, limit=k_clamped)
    except Exception as exc:  # noqa: BLE001 — memory must not crash a turn
        slog_error("memory_recall", exc)
        return {"memories": [], "error": str(exc)}

    memories = []
    for item in items:
        value = _extract_value(item)
        if not isinstance(value, dict):
            continue
        note = value.get("note") or value.get("content") or ""
        if not note:
            continue
        memories.append({
            "note": str(note),
            "tags": list(value.get("tags") or []),
            "ts": value.get("ts") or value.get("updated_at") or "",
            "id": _extract_key(item),
        })

    slog(
        "memory_recall",
        f"topic={topic!r} k={k_clamped} found={len(memories)}",
    )
    return {"memories": memories}


@tool
async def memory_write(
    note: str,
    tags: list[str],
    *,
    state: Annotated[dict, InjectedState],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Persist a fact about the user to long-term memory.

    Use after giving meaningful advice or learning a user-level fact
    (goals, debt level, family situation, risk tolerance). Keep notes
    short (≤200 chars) and tag with 1-3 keywords.

    Args:
      note: ≤200 chars Thai/English fact.
      tags: 1-3 keywords ("debt", "goal", "family", "preference").

    Returns:
      {"memory_id": str}
      OR {"error": str, "kind": str} on validation / store failure.
    """
    _emit_status(_WRITE_STATUS)

    user_id = state.get("user_id")
    if not user_id:
        return {"error": "state.user_id missing", "kind": "invalid_state"}

    note_clean = (note or "").strip()
    if not note_clean:
        return {"error": "note is empty", "kind": "invalid_note"}
    note_clean = note_clean[:_MAX_NOTE_LEN]

    tags_clean: list[str] = []
    for t in (tags or []):
        if t is None:
            continue
        s = str(t).strip()
        if not s:
            continue
        tags_clean.append(s)
        if len(tags_clean) >= _MAX_TAGS:
            break

    memory_id = f"mem_{uuid.uuid4().hex[:12]}"
    namespace = ("user", user_id, "facts")
    value = {
        "note": note_clean,
        "tags": tags_clean,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await _store_aput(store, namespace, key=memory_id, value=value)
    except Exception as exc:  # noqa: BLE001
        slog_error("memory_write", exc)
        return {"error": str(exc), "kind": "store_write_failed"}

    slog(
        "memory_write",
        f"wrote {memory_id} note_len={len(note_clean)} tags={tags_clean}",
    )
    return {"memory_id": memory_id}


# ── Helpers ────────────────────────────────────────────────────────────────


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


async def _store_asearch(
    store: BaseStore, namespace: tuple[str, ...], *, query: str, limit: int,
):
    """Cross-implementation store search.

    LangGraph's `BaseStore` has `asearch` (async) and `search` (sync). Some
    in-memory test stores only implement the sync variant; PostgresStore is
    async-only. Prefer async; fall back to sync; final fallback is `alist`
    (no query support — recency-ordered).

    Some store backends accept `query=` (keyword) and some don't. We try
    with the keyword first, then without — covers both InMemoryStore (no
    embeddings, ignores `query`) and a future semantic store.
    """
    asearch = getattr(store, "asearch", None)
    if callable(asearch):
        try:
            return await asearch(namespace, query=query, limit=limit)
        except TypeError:
            return await asearch(namespace, limit=limit)
    search = getattr(store, "search", None)
    if callable(search):
        try:
            return search(namespace, query=query, limit=limit)
        except TypeError:
            return search(namespace, limit=limit)
    # Final fallback: list (recency-ordered, no query support)
    alist = getattr(store, "alist", None)
    if callable(alist):
        return await alist(namespace, limit=limit)
    return []


async def _store_aput(
    store: BaseStore, namespace: tuple[str, ...], *, key: str, value: dict,
):
    """Cross-implementation store put — `aput` first, sync `put` fallback."""
    aput = getattr(store, "aput", None)
    if callable(aput):
        return await aput(namespace, key=key, value=value)
    put = getattr(store, "put", None)
    if callable(put):
        return put(namespace, key=key, value=value)
    raise RuntimeError("store has neither aput nor put")


def _extract_value(item: Any) -> Any:
    """Pull `.value` off a store Item, tolerating dict / object shapes."""
    if item is None:
        return None
    val = getattr(item, "value", None)
    if val is not None:
        return val
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def _extract_key(item: Any) -> str:
    key = getattr(item, "key", None)
    if key:
        return str(key)
    if isinstance(item, dict):
        return str(item.get("key") or "")
    return ""


__all__ = ["memory_recall", "memory_write"]
