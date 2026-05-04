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
from src.graph.analyze_subgraph.schemas import (
    EntityMention,
    QueryPlan,
    ResolvedEntity,
    SlotKind,
)
from src.graph.analyze_subgraph.state import AnalyzeSubState


# ── Candidate fetch ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Candidate:
    sync_id: str
    name: str


def _candidates(kind: SlotKind, catalog: EntityCatalog) -> list[_Candidate]:
    """Pull candidates from the in-state catalog. The catalog is fetched
    once per turn at the bridge (`act_node`) so we never hit the DB here."""
    if kind == "wallet":
        return [_Candidate(sync_id=w.sync_id, name=w.name) for w in catalog.wallets]
    if kind == "category":
        return [_Candidate(sync_id=c.sync_id, name=c.name) for c in catalog.categories]
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
    bullet = "\n".join(
        f"- sync_id={c.sync_id} | name={c.name!r} | heuristic_score={s:.2f}"
        for c, s in top
    )
    prompt = (
        f"Pick the {kind} that the user most likely means by {query!r}.\n"
        f"Candidates:\n{bullet}\n\n"
        f"Consider Thai/English semantics and typos (e.g., 'pat show' ~ 'pet shop'). "
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
    runner = top[1][1] if len(top) > 1 else 0.0

    # High confidence + clear gap → accept without LLM
    if best_score >= 0.85 and (best_score - runner) >= 0.15:
        return ResolvedEntity(
            sync_id=best.sync_id,
            display_name=best.name,
            kind=mention.kind,
            score=best_score,
            alternatives=[
                {"sync_id": c.sync_id, "name": c.name, "score": s}
                for c, s in top[1:]
            ],
        )

    # Otherwise let the LLM choose among the top-N
    rerank = await _llm_rerank(mention.kind, mention.text, top)
    if rerank is None:
        # fall back to heuristic best with degraded confidence
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
    )


# ── Node ─────────────────────────────────────────────────────────────────────


async def entity_resolve_node(state: AnalyzeSubState) -> dict:
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
