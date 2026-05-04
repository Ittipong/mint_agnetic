"""State for the analyze subgraph.

Internal state is isolated from parent AgentState — only the bridge
(`act_node`) reads/writes the parent's `messages`.
"""

from __future__ import annotations

from typing import NotRequired
from typing_extensions import TypedDict

from src.graph.compute_subgraph.schemas import (
    ClarificationPayload,
    ExecRow,
    QueryPlan,
    QuerySpec,
    TimeRange,
)


class ComputeSubState(TypedDict):
    # Input
    task: str
    user_id: str
    today: str  # ISO date
    # Entity catalog passed in from the parent — planner/resolver/responder
    # all read it, but no node mutates it (frozen dataclass).
    catalog: NotRequired[object]  # EntityCatalog (kept as `object` to avoid circular import)

    # Stage 1 — planner
    plan: NotRequired[QueryPlan]

    # Stage 2 — resolvers (run in parallel)
    spec: NotRequired[QuerySpec]
    time_range: NotRequired[TimeRange]
    resolved_wallets: NotRequired[list]  # ResolvedEntity[]
    resolved_categories: NotRequired[list]
    resolved_tags: NotRequired[list]

    # Stage 3 — gate
    confidence: NotRequired[float]
    needs_clarification: NotRequired[bool]
    clarification: NotRequired[ClarificationPayload]

    # Stage 4 — execution
    rows: NotRequired[list[ExecRow]]
    sql_debug: NotRequired[str]  # rendered SQL for log/debug

    # Stage 5 — output
    answer: NotRequired[str]

    # Diagnostics — surfaced back to ToolMessage.additional_kwargs
    step_info: NotRequired[dict]

    # Templates-CodeAct branch (only used when plan.metric == 'freeform_codeact')
    codeact_history: NotRequired[list]   # list[{step, code, stdout, error, result}]
    codeact_done: NotRequired[bool]
    codeact_final: NotRequired[object]   # last non-null `result` from the loop
