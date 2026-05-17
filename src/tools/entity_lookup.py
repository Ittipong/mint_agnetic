"""Long-tail entity lookup for the ReAct agent.

The system prompt only renders the user's top-N most-used wallets and
categories (see `EntityCatalog.render_for_prompt_truncated`). When the
user mentions a name that isn't in that list, the LLM has no `sync_id`
to plug into a tool call — calling `lookup_entity` is the explicit
escape hatch so it never guesses or paraphrases.

Returns a short markdown block of matches; empty string when nothing
matches so the LLM can apologize naturally instead of hallucinating.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


def _matches(name: str, query: str) -> bool:
    """Case-insensitive substring + token-prefix match.

    Token-prefix catches abbreviations like "true" → "TrueMonney" even
    when the substring check would also pass, but it's cheap to keep
    both for the rare case where the user types space-separated tokens.
    """
    n = (name or "").lower()
    q = (query or "").lower().strip()
    if not q:
        return False
    if q in n:
        return True
    return any(part.startswith(q) for part in n.split())


@tool
async def lookup_entity(
    entity_type: Literal["wallet", "category", "tag"],
    name_query: str,
    config: RunnableConfig,
) -> str:
    """Look up a wallet / category / tag by name when it's NOT in the
    top-N catalog shown in your context. Use this BEFORE writing any
    `analyze_user_finances` task that names a long-tail entity — guessing
    a `sync_id` produces wrong query results.

    Args:
        entity_type: 'wallet' | 'category' | 'tag'.
        name_query: full or partial name (case-insensitive). Thai or
            English both work. Examples: "TrueMonney", "true", "kfc",
            "ค่าเดิน".

    Returns:
        Markdown list of up to 5 matches with `sync_id` and `name`, or
        "(no matches)" when nothing fits. Pick the entity whose name is
        the closest exact match and use its `sync_id` verbatim.
    """
    cfg = (config or {}).get("configurable") or {}
    user_id = cfg.get("user_id") or cfg.get("thread_user_id")
    if not user_id:
        return "(lookup_entity: missing user_id in config — cannot search)"

    from src.entity_catalog import fetch_user_catalog  # noqa: PLC0415

    catalog = await fetch_user_catalog(user_id)

    if entity_type == "wallet":
        candidates = [
            (w.sync_id, w.name, f"type={w.wallet_type} currency={w.currency}")
            for w in catalog.wallets
            if _matches(w.name, name_query)
        ]
    elif entity_type == "category":
        candidates = [
            (c.sync_id, c.name, f"type={c.type}")
            for c in catalog.categories
            if _matches(c.name, name_query)
        ]
    elif entity_type == "tag":
        candidates = [
            (t.sync_id, t.name, "")
            for t in catalog.tags
            if _matches(t.name, name_query)
        ]
    else:
        return f"(lookup_entity: unsupported entity_type '{entity_type}')"

    if not candidates:
        return "(no matches)"

    lines = [f"Matches for `{name_query}` ({entity_type}):"]
    for sync_id, name, extra in candidates[:5]:
        suffix = f" — {extra}" if extra else ""
        lines.append(f"- sync_id=`{sync_id}` name=`{name}`{suffix}")
    if len(candidates) > 5:
        lines.append(f"_…and {len(candidates) - 5} more — refine name_query._")
    return "\n".join(lines)
