"""gate node — confidence aggregation + QuerySpec assembly.

Aggregates resolver confidences. Below the floor, emit a clarification payload
that the parent ReAct can surface to the user. Above the floor, freeze the
results into a QuerySpec for the SQL builder.

This MVP does NOT call `interrupt()` — instead, low confidence flows back to
ReAct as a clarifying message. Upgrading to LangGraph `interrupt()` is an
additive change once the mobile client supports the resume protocol.
"""

from __future__ import annotations

from src.graph.compute_subgraph.schemas import (
    ClarificationPayload,
    QueryPlan,
    QuerySpec,
    ResolvedEntity,
    TimeRange,
)
from src.graph.compute_subgraph.state import ComputeSubState

# Below this floor we ask the user instead of guessing.
_CONFIDENCE_FLOOR = 0.6


def _aggregate_confidence(
    time_range: TimeRange,
    entities: list[ResolvedEntity],
) -> float:
    """Worst-of-all-stages — one weak link drags the whole pipeline down.

    Why min instead of mean: a wrong wallet completely invalidates the answer,
    so we surface the weakest signal rather than averaging it away.
    """
    scores = [time_range.confidence] + [e.score for e in entities]
    return min(scores) if scores else 1.0


def _first_low_confidence(
    plan: QueryPlan,
    time_range: TimeRange,
    entities: list[ResolvedEntity],
) -> ClarificationPayload | None:
    if time_range.confidence < _CONFIDENCE_FLOOR:
        return ClarificationPayload(
            kind="time",
            question=(
                f"ไม่แน่ใจช่วงเวลา {time_range.raw_phrase!r} — "
                f"หมายถึง {time_range.start.isoformat()} ถึง "
                f"{time_range.end.isoformat()} ใช่หรือไม่?"
            ),
            candidates=[
                {
                    "start": time_range.start.isoformat(),
                    "end": time_range.end.isoformat(),
                }
            ],
        )

    weak = [e for e in entities if e.score < _CONFIDENCE_FLOOR]
    if weak:
        target = weak[0]
        return ClarificationPayload(
            kind="entity",
            question=(
                f"ไม่แน่ใจว่าหมายถึง {target.kind} ตัวไหน "
                f"(ตอนนี้เลือก {target.display_name!r}) — เลือกอันไหนดี?"
            ),
            candidates=[
                {
                    "sync_id": target.sync_id,
                    "name": target.display_name,
                    "score": target.score,
                },
                *target.alternatives,
            ],
        )
    return None


async def gate_node(state: ComputeSubState) -> dict:
    plan: QueryPlan = state["plan"]
    time_range: TimeRange = state["time_range"]
    wallets = state.get("resolved_wallets") or []
    categories = state.get("resolved_categories") or []
    tags = state.get("resolved_tags") or []
    entities = [*wallets, *categories, *tags]

    confidence = _aggregate_confidence(time_range, entities)
    clarification = _first_low_confidence(plan, time_range, entities)

    spec = QuerySpec(
        metric=plan.metric,
        wallets=wallets,
        categories=categories,
        tags=tags,
        time_range=time_range,
        group_by=plan.group_by,
        currency=plan.currency,
        order_by=plan.order_by,
        limit=plan.limit,
        budget_name_phrase=plan.budget_name_phrase,
        goal_name_phrase=plan.goal_name_phrase,
        convert_to_thb=plan.convert_to_thb,
        transaction_type=plan.transaction_type,
    )

    return {
        "spec": spec,
        "confidence": confidence,
        "needs_clarification": clarification is not None,
        "clarification": clarification,
    }
