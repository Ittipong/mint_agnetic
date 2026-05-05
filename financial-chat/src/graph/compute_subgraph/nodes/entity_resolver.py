"""entity_resolver — map free-text mentions → ResolvedEntity[].

MVP strategy (no pg_trgm, no embedding):
1. Fetch all candidates of the kind for this user (small N — wallets/categories
   are usually <50 per user).
2. Score with deterministic Python heuristic (exact > prefix > contains > token).
3. If top score is high enough and gap to runner-up is large, accept directly.
4. Otherwise call LLM to rerank with confidence.

When pg_trgm becomes available, swap step 1 for `ORDER BY similarity(name, $term) DESC LIMIT 20`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.entity_catalog import EntityCatalog
from src.graph.compute_subgraph.schemas import (
    EntityMention,
    QueryPlan,
    ResolvedEntity,
    SlotKind,
)
from src.graph.compute_subgraph.state import ComputeSubState


# ── Candidate fetch ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Candidate:
    sync_id: str
    name: str
    wallet_type: str = "general"  # 'general' | 'creditcard' | 'goal'
    wallet_category: str | None = None  # e.g. 'savings', 'cash', 'eWallet'
    # Category-specific fields for semantic hierarchy matching
    parent_id: str | None = None
    keywords: list[str] | None = None


def _candidates(kind: SlotKind, catalog: EntityCatalog) -> list[_Candidate]:
    """Pull candidates from the in-state catalog. The catalog is fetched
    once per turn at the bridge (`act_node`) so we never hit the DB here."""
    if kind == "wallet":
        return [_Candidate(sync_id=w.sync_id, name=w.name, wallet_type=w.wallet_type,
                          wallet_category=w.wallet_category)
                for w in catalog.wallets]
    if kind == "category":
        return [_Candidate(sync_id=c.sync_id, name=c.name, parent_id=c.parent_id, keywords=c.keywords)
                for c in catalog.categories]
    if kind == "tag":
        return [_Candidate(sync_id=t.sync_id, name=t.name) for t in catalog.tags]
    return []


# ── Heuristic scoring ────────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _score_pair(query: str, candidate: str) -> float:
    """Cheap deterministic similarity in [0,1]."""
    q = _normalize(query)
    c = _normalize(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if c.startswith(q) or q.startswith(c):
        return 0.85
    if q in c or c in q:
        return 0.7

    q_tokens = set(q.split())
    c_tokens = set(c.split())
    if q_tokens and c_tokens:
        overlap = len(q_tokens & c_tokens) / max(len(q_tokens), len(c_tokens))
        if overlap > 0:
            return 0.4 + 0.3 * overlap

    # last-ditch: char overlap
    common = len(set(q) & set(c)) / max(len(set(q) | set(c)), 1)
    return 0.2 * common


def _rank(query: str, candidates: list[_Candidate]) -> list[tuple[_Candidate, float]]:
    """Score candidates by name. LLM rerank handles semantic matching including wallet_category."""
    scored = [(c, _score_pair(query, c.name)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── LLM rerank (only when needed) ────────────────────────────────────────────


class _LLMChoice(BaseModel):
    sync_id: str = Field(description="The sync_id of the chosen candidate")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


async def _llm_rerank(
    kind: SlotKind, query: str, top: list[tuple[_Candidate, float]]
) -> tuple[_Candidate, float] | None:
    if not top:
        return None
    try:
        from src.llm import llm
    except Exception:
        return None
    # Build bullet with relevant fields per kind
    if kind == "wallet":
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r} | type={c.wallet_type} | category={c.wallet_category!r} | heuristic_score={s:.2f}"
            if c.wallet_category else
            f"- sync_id={c.sync_id} | name={c.name!r} | type={c.wallet_type} | heuristic_score={s:.2f}"
            for c, s in top
        )
        prompt = (
            f"Pick the {kind} that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider Thai/English semantics and typos (e.g., 'pat show' ~ 'pet shop'). "
            f"For wallets, also consider wallet_type (creditcard/general/goal) and wallet_category. "
            f"If none plausibly match, pick the closest but report low confidence."
        )
    elif kind == "category":
        # Include parent_id and keywords for hierarchy-aware matching
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r} | parent={c.parent_id!r} | keywords={c.keywords!r} | heuristic_score={s:.2f}"
            for c, s in top
        )
        prompt = (
            f"Pick the {kind}(s) that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"IMPORTANT: Consider parent-child relationships for category expansion:\n"
            f"- If user asks about 'เดินทาง' (travel), include 'แท็กซี่', 'BTS/MRT', 'น้ำมัน' "
            f"because they are sub-categories of travel.\n"
            f"- If user asks about 'อาหาร' (food), include 'ร้านอาหาร', 'กาแฟ', 'ของทานเล่น'.\n"
            f"- A category 'เดินทาง' has parent=null; 'แท็กซี่' has parent pointing to 'เดินทาง'.\n"
            f"- keywords field contains related terms (e.g., 'แท็กซี่' keywords=['เดินทาง','รถ','สัญจร']).\n"
            f"If user says 'ดูรายการเดินทาง', return ALL categories whose name OR parent_id OR keywords match 'เดินทาง'.\n"
            f"You may return MULTIPLE sync_ids - include the parent and all children.\n"
            f"Return the sync_id of the PRIMARY match (user's exact target)."
        )
    else:
        bullet = "\n".join(
            f"- sync_id={c.sync_id} | name={c.name!r} | heuristic_score={s:.2f}"
            for c, s in top
        )
        prompt = (
            f"Pick the {kind} that the user most likely means by {query!r}.\n"
            f"Candidates:\n{bullet}\n\n"
            f"Consider Thai/English semantics and typos. "
            f"If none plausibly match, pick the closest but report low confidence."
        )
    try:
        structured = llm.with_structured_output(_LLMChoice)
        out: _LLMChoice = await structured.ainvoke(prompt)
        chosen = next((c for c, _ in top if c.sync_id == out.sync_id), None)
        if chosen is None:
            return None
        return chosen, max(0.0, min(1.0, out.confidence))
    except Exception:
        return None


# ── Resolve a single mention ─────────────────────────────────────────────────


def _expand_subcategories(
    primary: _Candidate, all_candidates: list[_Candidate]
) -> list[str]:
    """Find all sub-category names to expand when user queries a parent category.

    Returns category NAMES (not sync_ids) because the SQL filter matches on
    category_name text field.

    Expansion logic:
    - If primary has NO parent (is a root/parent like "เดินทาง"), expand to include
      all its direct children (e.g., "แท็กซี่", "BTS/MRT", "น้ำมัน").
    - If primary HAS a parent (is already a child like "น้ำมัน"), do NOT expand —
      only return transactions for that specific category.
    """
    if primary.parent_id is None:
        # Primary is a parent (root category), expand to include all children
        return [c.name for c in all_candidates if c.parent_id == primary.sync_id]
    else:
        # Primary is a child category, don't expand — only show this specific category
        return []


async def _resolve_mention(
    mention: EntityMention,
    catalog: EntityCatalog,
) -> ResolvedEntity | None:
    candidates = _candidates(mention.kind, catalog)
    if not candidates:
        return None

    ranked = _rank(mention.text, candidates)
    top = ranked[: min(5, len(ranked))]
    best, best_score = top[0]

    # Always let LLM choose among the top-N using semantic matching
    rerank = await _llm_rerank(mention.kind, mention.text, top)
    if rerank is None:
        # fall back to heuristic best
        return ResolvedEntity(
            sync_id=best.sync_id,
            display_name=best.name,
            kind=mention.kind,
            score=best_score * 0.8,
            alternatives=[
                {"sync_id": c.sync_id, "name": c.name, "score": s}
                for c, s in top[1:]
            ],
        )
    chosen, conf = rerank

    # For categories, expand to include sub-categories (same parent)
    expand_ids: list[str] = []
    if mention.kind == "category":
        expand_ids = _expand_subcategories(chosen, candidates)

    return ResolvedEntity(
        sync_id=chosen.sync_id,
        display_name=chosen.name,
        kind=mention.kind,
        score=conf,
        alternatives=[
            {"sync_id": c.sync_id, "name": c.name, "score": s}
            for c, s in top
            if c.sync_id != chosen.sync_id
        ],
        expand_ids=expand_ids,
    )


# ── Node ─────────────────────────────────────────────────────────────────────


async def entity_resolve_node(state: ComputeSubState) -> dict:
    plan: QueryPlan | None = state.get("plan")
    if plan is None or not plan.entity_mentions:
        return {
            "resolved_wallets": [],
            "resolved_categories": [],
            "resolved_tags": [],
        }

    catalog: EntityCatalog = state.get("catalog") or EntityCatalog()  # type: ignore[assignment]
    wallets: list[ResolvedEntity] = []
    categories: list[ResolvedEntity] = []
    tags: list[ResolvedEntity] = []

    for mention in plan.entity_mentions:
        resolved = await _resolve_mention(mention, catalog)
        if resolved is None:
            continue
        if mention.kind == "wallet":
            wallets.append(resolved)
        elif mention.kind == "category":
            categories.append(resolved)
        elif mention.kind == "tag":
            tags.append(resolved)

    return {
        "resolved_wallets": wallets,
        "resolved_categories": categories,
        "resolved_tags": tags,
    }
