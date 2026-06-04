"""`get_user_context` — fetch + cache the user's entity catalog.

NEW in Wave 3. Phase 2 spec: docs/v3/phase2_tools_design.md Tool 6.

This tool is the FIRST data tool the ReAct loop should call on any turn
that touches wallets, categories, budgets, or goals. The result lands in
`state.user_context` (per-turn cache, cleared by the pre-turn hook) and
all downstream tools (`propose_transaction`, `run_python`) read it from
there instead of re-querying Postgres.

Why retry-once on empty wallets:
A transient pool blip can return zero rows for an existing user, which
would falsely route a returning user into `wallet_required_cta` onboarding.
v2 saw this flake at the integration boundary (memory
`project_wallet_index0_ordering` — `_load_catalog` retries once). We
reproduce that pattern here so a single bad read self-corrects; a genuinely
wallet-less user still gets onboarding (the second attempt is empty too).

Contract:
- Status word "กำลังเช็คข้อมูล..." emitted at entry.
- Idempotent within a turn — repeated calls return the cached dict.
- Never raises — load failures degrade to {"error": ..., "kind": ...}.
- Output is plain dicts so the value survives the LangGraph checkpointer
  (matches `UserContext` TypedDict in state.py).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langsmith import traceable as _ls_traceable

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover — older LangGraph fallback
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.entity_catalog import EntityCatalog, load_catalog_for_user
from src.agent.session_logger import slog


_STATUS_WORD = "กำลังเช็คข้อมูล..."

# Wallet ordering — must match the SQL `ORDER BY` in entity_catalog._WALLETS_SQL
# so the index-0 fallback in `propose_transaction` lands deterministically on a
# general wallet (per memory `project_wallet_index0_ordering`).
_WALLET_ORDER = {"general": 0, "creditcard": 1, "goal": 2}


@tool
async def get_user_context(
    *,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Fetch the user's wallets, categories, budgets, goals.

    Call ONCE per turn before propose_transaction or run_python. Result is
    cached in state.user_context and reused by other tools — repeated calls
    return the cached value.

    Returns:
      Command(update={
        "user_context": <context dict>,
        "messages": [ToolMessage(JSON of the context | error payload)],
      })

      The ToolMessage body mirrors the context dict (or {"error": ..., "kind":
      ...} on failure) so the LLM can read the structure directly without
      another state hop.
    """
    _emit_status(_STATUS_WORD)

    cached = state.get("user_context")
    if cached is not None:
        slog("get_user_context", "cache hit — returning cached user_context")
        return _ok_command(tool_call_id, cached)

    user_id = state.get("user_id")
    if not user_id:
        return _error_command(
            tool_call_id, error="state.user_id missing", kind="invalid_state",
        )

    context, _catalog, load_error = await load_user_context_dict(user_id)
    if context is None:
        return _error_command(
            tool_call_id,
            error=f"failed to load catalog: {load_error}",
            kind="context_load_failed",
        )

    # Defensive in-place write — only visible to direct-invocation tests that
    # share the `state` dict reference. The Command(update=) below is the
    # canonical write that LangGraph propagates across the tool boundary.
    state["user_context"] = context

    return _ok_command(tool_call_id, context)


@_ls_traceable(
    name="load_user_context",
    run_type="tool",
    tags=["classify_router", "catalog"],
)
async def load_user_context_dict(
    user_id: str,
) -> tuple[Optional[dict], Optional[EntityCatalog], Optional[Exception]]:
    """Load the user's catalog; return BOTH a slim wire dict AND the raw
    EntityCatalog so `propose_transaction` can scope per-wallet categories
    without losing entries to the dedup-race serialization bug.

    Why two values (wave 5.1):
      * The wire dict is shipped to ReAct via `state.user_context`. It now
        OMITS the `categories` field entirely — see TODO inline — to keep
        ReAct's prompt small (saves ~200 tokens / turn on a user with ~40
        deduped cats). ReAct doesn't need category NAMES inline for the
        ADD / chat hot path; the codeact sandbox and `propose_transaction`
        load their own EntityCatalog directly.
      * `EntityCatalog` carries the FULL per-wallet `categories_all` view
        in memory. `propose_transaction` reads this directly for the joint
        resolver scoping (skips serialization → no dedup-race loss).

    Retry-once-on-empty semantics preserved — a transient pool blip self-
    heals before falling through to onboarding (memory
    `project_wallet_index0_ordering`).

    Returns:
      (context_dict, catalog, None) on success.
      (None, None, last_exception) when both load attempts raised.
    """
    catalog: Optional[EntityCatalog] = None
    last_error: Optional[Exception] = None
    for attempt in (1, 2):
        try:
            catalog = await load_catalog_for_user(user_id)
        except Exception as exc:
            last_error = exc
            catalog = None
            slog(
                "load_user_context",
                f"catalog load attempt {attempt} raised: "
                f"{type(exc).__name__}: {exc}",
            )
        if catalog is not None and getattr(catalog, "wallets", None):
            break
        if attempt == 1:
            slog(
                "load_user_context",
                "catalog empty/failed on first load — retrying once",
            )

    if catalog is None:
        return None, None, last_error

    # Marshal to plain dicts so the value survives LangGraph's MsgPack serde.
    # Wallets are pre-ordered general → creditcard → goal by the SQL itself,
    # but we sort again here defensively in case `catalog.wallets` was hand-
    # built (tests) without honoring the SQL order.
    wallets = sorted(
        [_wallet_to_dict(w) for w in (catalog.wallets or [])],
        key=lambda w: (_WALLET_ORDER.get(w.get("wallet_type") or "general", 9),),
    )
    tags = [{"sync_id": t.sync_id, "name": t.name} for t in (catalog.tags or [])]

    context = {
        "wallets": wallets,
        # TODO: re-add a `categories` field here when an Analyst-path ReAct
        # flow legitimately needs catalog NAMES inline in the prompt. As of
        # wave 5.1 there are no such consumers — `propose_transaction` and
        # the codeact sandbox both load their own EntityCatalog directly,
        # which sidesteps the dedup-race serialization bug AND keeps the
        # ReAct prompt small. When re-adding, serialize
        # `catalog.categories_all` (the per-wallet view), NOT just
        # `catalog.categories`, or per-wallet scoping will silently drop
        # entries again.
        # Budgets + goals are not yet sourced from EntityCatalog in v3; surface
        # empty lists so downstream consumers can rely on the keys existing.
        "budgets": [],
        "goals": [],
        "tags": tags,
        # Catalog default wallet (computed). Lives under user_context["wallet_id"]
        # — the propose_transaction cascade falls back to it when the client
        # didn't pick a wallet (state["wallet_id"] empty).
        "wallet_id": _pick_default_wallet(wallets),
        "default_currency_code": _pick_default_currency(wallets),
        "fetched_at": _now_iso(),
    }
    slog(
        "load_user_context",
        f"loaded: wallets={len(wallets)} "
        f"categories_all={len(catalog.categories_all or [])} "
        f"tags={len(tags)} default_wallet={context['wallet_id']!r}",
    )
    return context, catalog, None


# ── Helpers ────────────────────────────────────────────────────────────────


def _ok_command(tool_call_id: str, context: dict) -> Command:
    """Build the success Command — propagates `user_context` AND ships the
    same dict as the ToolMessage payload so the LLM sees the catalog without
    a second tool call."""
    return Command(update={
        "user_context": context,
        "messages": [
            ToolMessage(
                content=json.dumps(context, ensure_ascii=False, default=str),
                tool_call_id=tool_call_id,
            ),
        ],
    })


def _error_command(tool_call_id: str, *, error: str, kind: str) -> Command:
    """Build the error Command — no `user_context` update so any stale value
    from a prior turn stays cleared (the pre-turn hook resets it to None)."""
    payload = {"error": error, "kind": kind}
    return Command(update={
        "messages": [
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=tool_call_id,
            ),
        ],
    })


def _emit_status(word: str) -> None:
    """Stream a status word (best-effort, never raises)."""
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
        # Status updates are UX-only — never break the tool over a writer hiccup.
        pass


def _wallet_to_dict(w: Any) -> dict:
    """Marshal a WalletEntry (or already-dict wallet) to the wire schema."""
    if isinstance(w, dict):
        return dict(w)
    return {
        "sync_id": w.sync_id,
        "name": w.name,
        "currency": w.currency,
        "wallet_type": w.wallet_type,
        "is_default": bool(w.is_default),
        "usage_count": int(w.usage_count or 0),
        "icon": w.icon,
    }


def _category_to_dict(c: Any) -> dict:
    if isinstance(c, dict):
        return dict(c)
    return {
        "sync_id": c.sync_id,
        "name": c.name,
        "type": c.type,
        "parent_id": c.parent_id,
        "usage_count": int(c.usage_count or 0),
        "wallet_sync_id": getattr(c, "wallet_sync_id", None),
    }


def _pick_default_wallet(wallets: list[dict]) -> Optional[str]:
    """Catalog-flagged default → first general → first any."""
    for w in wallets:
        if w.get("is_default"):
            return w.get("sync_id")
    for w in wallets:
        if (w.get("wallet_type") or "general") == "general":
            return w.get("sync_id")
    return wallets[0].get("sync_id") if wallets else None


def _pick_default_currency(wallets: list[dict]) -> str:
    """Currency of the chosen default wallet, falling back to THB."""
    if not wallets:
        return "THB"
    default_sid = _pick_default_wallet(wallets)
    for w in wallets:
        if w.get("sync_id") == default_sid:
            return w.get("currency") or "THB"
    return wallets[0].get("currency") or "THB"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["get_user_context", "load_user_context_dict"]
