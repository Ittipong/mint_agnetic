"""execute node — builds SQL from the resolved spec and runs it.

Combined the previous sql_build + execute steps so build_query is called
once. step_info (planner/resolver diagnostics + a one-line SQL render for
logs) is emitted alongside the rows.
"""

from __future__ import annotations

from decimal import Decimal

from src.graph.analyze_subgraph.db import get_pool
from src.graph.analyze_subgraph.schemas import ExecRow, QuerySpec
from src.graph.analyze_subgraph.sql_templates import build_query
from src.graph.analyze_subgraph.state import AnalyzeSubState


async def execute_node(state: AnalyzeSubState) -> dict:
    spec: QuerySpec = state["spec"]
    sql, params = build_query(spec, state["user_id"])
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = await conn.fetch(sql, *params)

    rows: list[ExecRow] = [_record_to_row(spec, r) for r in records]

    step_info = {
        **(state.get("step_info") or {}),
        "metric": spec.metric,
        "wallets": [w.sync_id for w in spec.wallets],
        "categories": [c.sync_id for c in spec.categories],
        "tags": [t.sync_id for t in spec.tags],
        "time_range": [
            spec.time_range.start.isoformat(),
            spec.time_range.end.isoformat(),
        ],
        "currency": spec.currency,
    }
    return {
        "rows": rows,
        "sql_debug": _render_for_log(sql, params),
        "step_info": step_info,
    }


def _render_for_log(sql: str, params: list) -> str:
    """Compact SQL+params for logs — never sent back to the LLM."""
    return f"{' '.join(sql.split())}  -- params={params}"


def _record_to_row(spec: QuerySpec, record) -> ExecRow:
    """Map a Postgres record to ExecRow — Decimal values stringified.

    LLM responder never sees raw numbers, only pre-formatted strings.
    """
    d = dict(record)

    bucket = (
        d.get("bucket")
        or d.get("wallet_name")
        or d.get("note")
        or d.get("category_name")
    )
    if bucket is not None:
        bucket = str(bucket)

    amount_raw = d.get("amount")
    amount_str = _decimal_str(amount_raw)

    count = d.get("cnt") or d.get("count")
    if count is not None:
        count = int(count)

    extra = {
        k: _serialize(v)
        for k, v in d.items()
        if k not in {"bucket", "amount", "cnt", "count"}
    }
    return ExecRow(
        bucket=bucket,
        currency=d.get("currency"),
        amount=amount_str,
        count=count,
        extra=extra,
    )


def _decimal_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return str(v)
    # double precision lands here — coerce via Decimal(str(v)) to dodge float repr.
    if isinstance(v, (int, float)):
        return str(Decimal(str(v)))
    return str(v)


def _serialize(v):
    if isinstance(v, Decimal):
        return str(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v
