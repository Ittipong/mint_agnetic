"""responder — code-only deterministic formatter.

We deliberately skip the LLM here. The parent ReAct (reason_node) already does
the final Thai prose; this node just produces a structured, readable tool
output with all numbers pre-formatted as strings (eliminating any math
hallucination risk).

If the gate flagged low confidence, we surface that as a clarification
question instead of the data — letting ReAct ask the user.
"""

from __future__ import annotations

from decimal import Decimal

from src.graph.compute_subgraph.schemas import (
    ClarificationPayload,
    ExecRow,
    QuerySpec,
    decimal_to_display,
)
from src.graph.compute_subgraph.state import ComputeSubState


# ── Public node ──────────────────────────────────────────────────────────────


async def respond_node(state: ComputeSubState) -> dict:
    from datetime import datetime as dt
    t0 = dt.now()
    if state.get("needs_clarification") and state.get("clarification") is not None:
        return {"answer": _format_clarification(state["clarification"])}

    # Templates-CodeAct branch — final result lives in `codeact_final`,
    # not `rows`. Render it as a structured tool result the parent ReAct
    # can read.
    plan = state.get("plan")
    if plan is not None and plan.metric == "freeform_codeact":
        return {"answer": _format_codeact(state)}

    spec: QuerySpec = state["spec"]
    rows: list[ExecRow] = state.get("rows") or []
    answer = _format_result(spec, rows)
    t1 = dt.now()
    ms = (t1 - t0).total_seconds() * 1000
    print(f"[PERF] respond_node: {ms:.0f}ms rows={len(rows)}")
    return {"answer": answer}


def _format_codeact(state: ComputeSubState) -> str:
    """Render the codeact loop result as a tool-readable string.

    The header tags it as `metric=freeform_codeact` so the reasoner
    knows the data came from a composed query, not a single template.
    """
    import json
    from decimal import Decimal
    from datetime import date as _date

    final = state.get("codeact_final")
    history = state.get("codeact_history") or []
    n_steps = len(history)

    def _default(o):
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, _date):
            return o.isoformat()
        return str(o)

    if final is None:
        # Loop bailed out without setting `result` — surface the last error
        # so the LLM/user can see why instead of presenting empty success.
        last_err = next(
            (h.get("error") for h in reversed(history) if h.get("error")),
            None,
        )
        body = (
            f"NO_RESULT after {n_steps} step(s)"
            + (f"; last error: {last_err}" if last_err else "")
        )
        return f"[metric=freeform_codeact | steps={n_steps}]\n{body}"

    try:
        rendered = json.dumps(final, default=_default, ensure_ascii=False, indent=2)
    except Exception:
        rendered = str(final)
    if len(rendered) > 4000:
        rendered = rendered[:4000] + "\n… (truncated)"
    return f"[metric=freeform_codeact | steps={n_steps}]\n{rendered}"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_clarification(c: ClarificationPayload) -> str:
    lines = [f"NEEDS_CLARIFICATION ({c.kind}): {c.question}"]
    if c.candidates:
        lines.append("Candidates:")
        for cand in c.candidates[:5]:
            label = cand.get("name") or cand.get("start") or str(cand)
            score = cand.get("score")
            extra = f" (score={score:.2f})" if isinstance(score, (int, float)) else ""
            lines.append(f"  - {label}{extra}")
    return "\n".join(lines)


_LIST_METRICS = {"list", "budget_transactions"}


def _format_result(spec: QuerySpec, rows: list[ExecRow]) -> str:
    # Always render the header (with resolved entity names) so the downstream
    # LLM has the canonical wallet/category names to copy — even when the
    # query returned zero rows.
    header = _header(spec)
    if not rows:
        return f"{header}\n{_empty_message(spec)}"

    body = _body(spec, rows)
    parts = [header, body]
    # `list` and `budget_transactions` share the row-level breakdown format —
    # _body() already emitted the summary line; we append (a) a code-computed
    # per-category / per-wallet aggregate so the downstream LLM never has to
    # group rows by hand, and (b) the per-tx detail for transparency.
    if spec.metric in _LIST_METRICS:
        summaries = _row_aggregates(rows)
        if summaries:
            parts.append(summaries)
        parts.append(_breakdown(rows))
    return "\n".join(p for p in parts if p)


def _row_aggregates(rows: list[ExecRow]) -> str:
    """Pre-compute per-category and per-wallet totals from list-style rows.

    The LLM gets these as plain text so a question like 'ใช้ไปกับหมวดอะไรเยอะ
    ที่สุด' or 'wallet ไหนสูงสุด' can be answered by copying numbers verbatim
    — never by summing the rows itself. Multi-currency stays separate (we
    bucket by (key, currency) so we never mix THB and USD into one figure).
    """
    by_cat: dict[tuple[str, str], Decimal] = {}
    by_wallet: dict[tuple[str, str], Decimal] = {}
    for r in rows:
        amount = _to_decimal(r.amount) or Decimal(0)
        ccy = r.currency or "THB"
        cat = r.extra.get("category_name") or "(uncategorized)"
        wallet = r.extra.get("wallet_name") or "(unknown wallet)"
        by_cat[(cat, ccy)] = by_cat.get((cat, ccy), Decimal(0)) + amount
        by_wallet[(wallet, ccy)] = by_wallet.get((wallet, ccy), Decimal(0)) + amount

    sections: list[str] = []
    if len(by_cat) > 1 or (len(by_cat) == 1 and len(rows) > 1):
        lines = ["Summary by category:"]
        for (cat, ccy), total in sorted(
            by_cat.items(), key=lambda kv: (-kv[1], kv[0][0])
        ):
            lines.append(f"  - {cat}: {decimal_to_display(total)} {ccy}".rstrip())
        sections.append("\n".join(lines))

    if len(by_wallet) > 1 or (len(by_wallet) == 1 and len(rows) > 1):
        lines = ["Summary by wallet:"]
        for (wallet, ccy), total in sorted(
            by_wallet.items(), key=lambda kv: (-kv[1], kv[0][0])
        ):
            lines.append(f"  - {wallet}: {decimal_to_display(total)} {ccy}".rstrip())
        sections.append("\n".join(lines))

    return "\n".join(sections)


def _header(spec: QuerySpec) -> str:
    if spec.time_range.granularity == "all":
        period = f"all time (… → {spec.time_range.end.isoformat()})"
    else:
        period = (
            f"{spec.time_range.start.isoformat()} → {spec.time_range.end.isoformat()}"
        )
    bits = [f"metric={spec.metric}", f"period={period}"]
    if spec.wallets:
        bits.append("wallets=" + ", ".join(w.display_name for w in spec.wallets))
    if spec.categories:
        bits.append("categories=" + ", ".join(c.display_name for c in spec.categories))
    if spec.tags:
        bits.append("tags=" + ", ".join(t.display_name for t in spec.tags))
    if spec.currency != "ALL":
        bits.append(f"currency={spec.currency}")
    if spec.convert_to_thb:
        # Mark the result as FX-converted so the AI cannot mistake it for
        # a native-currency total. The note is intentionally explicit so the
        # downstream LLM relays the caveat to the user.
        bits.append("currency=THB(converted at today's rate)")
    return "[" + " | ".join(bits) + "]"


def _body(spec: QuerySpec, rows: list[ExecRow]) -> str:
    if spec.metric == "balance":
        # balance has one row per wallet — include the wallet name so the
        # downstream LLM doesn't have to invent it.
        return _format_balance(rows)
    if spec.metric in {"sum_income", "sum_expense"}:
        return _format_totals(rows)
    if spec.metric == "count":
        return _format_counts(rows)
    if spec.metric in {"sum_by_category", "sum_by_wallet"}:
        return _format_breakdown(rows)
    if spec.metric in _LIST_METRICS:
        return _format_list_summary(rows)
    if spec.metric == "budget_list":
        return _format_budget_list(rows)
    if spec.metric == "budget_remaining":
        return _format_budget_remaining(rows)
    if spec.metric == "goal_list":
        return _format_goal_list(rows)
    if spec.metric == "goal_progress":
        return _format_goal_progress(rows)
    if spec.metric == "goal_transactions":
        return _format_list_summary(rows)
    if spec.metric == "creditcard_list":
        return _format_creditcard_list(rows)
    return ""


def _format_goal_list(rows: list[ExecRow]) -> str:
    lines = ["Goals:"]
    for r in rows:
        d = r.extra
        amount = decimal_to_display(_to_decimal(r.amount))
        ccy = r.currency or d.get("currency", "")
        target_date = d.get("target_date") or "(no deadline)"
        closed = " [closed]" if d.get("is_closed") else ""
        lines.append(
            f"  - {d.get('name','(unnamed)')}: target={amount} {ccy} | "
            f"target_date={target_date}{closed}".rstrip()
        )
    return "\n".join(lines)


def _format_goal_progress(rows: list[ExecRow]) -> str:
    """One line per goal: name | target | balance | remaining | pct | status |
    days_left | daily_required. Status derived in code (`achieved` / `expired` /
    `on_track` / `behind`) so the LLM never has to classify it."""
    lines = ["Goals progress:"]
    for r in rows:
        d = r.extra
        target = _to_decimal(r.amount)
        balance = _to_decimal(d.get("balance"))
        remaining = _to_decimal(d.get("remaining_to_target"))
        pct = _to_decimal(d.get("pct_completed"))
        ccy = r.currency or d.get("currency", "")
        days_left = d.get("days_left")
        daily_required = d.get("daily_required")
        status = _goal_status(d, pct)
        if daily_required is None:
            daily_str = "expired" if status == "expired" else "n/a"
        else:
            daily_str = (
                f"{decimal_to_display(_to_decimal(daily_required))} {ccy}/day"
            )
        lines.append(
            f"  - {d.get('name','(unnamed)')}: "
            f"target={decimal_to_display(target)} {ccy} | "
            f"balance={decimal_to_display(balance)} {ccy} | "
            f"remaining={decimal_to_display(remaining)} {ccy} | "
            f"pct_completed={pct or 0}% | status={status} | "
            f"days_left={days_left if days_left is not None else 'n/a'} | "
            f"daily_required={daily_str} | "
            f"target_date={d.get('target_date') or '(no deadline)'}"
        )
    return "\n".join(lines)


def _goal_status(extra: dict, pct) -> str:
    if extra.get("is_closed"):
        return "closed"
    if extra.get("achieved_at"):
        return "achieved"
    target_date = extra.get("target_date")
    days_left = extra.get("days_left")
    try:
        p = float(pct) if pct is not None else 0.0
    except (TypeError, ValueError):
        p = 0.0
    if p >= 100:
        return "achieved"
    if target_date and (days_left == 0 or days_left is None):
        return "expired"
    return "on_track" if p >= 50 else "behind"


def _format_creditcard_list(rows: list[ExecRow]) -> str:
    """Format credit card list: name | credit_limit | used | available.

    The used/available values come from the backend-maintained cache and may
    be NULL if never written — we explicitly say "unknown" rather than 0
    so the LLM never presents stale data as fresh.
    """
    lines = ["Credit cards:"]
    for r in rows:
        d = r.extra
        name = d.get("name", "(unnamed)")
        # credit_limit lives in extra (amount column is NULL for cc queries)
        credit_limit = decimal_to_display(_to_decimal(d.get("credit_limit")))
        used = _to_decimal(d.get("used"))
        available = _to_decimal(d.get("available"))
        ccy = r.currency or d.get("currency", "THB")
        cache_updated = d.get("cache_updated_at") or ""

        if used is None:
            used_str = "unknown"
        else:
            used_str = decimal_to_display(used)

        if available is None:
            available_str = "unknown"
        else:
            available_str = decimal_to_display(available)

        cache_note = f" (cache: {cache_updated})" if cache_updated else ""
        lines.append(
            f"  - {name}: credit_limit={credit_limit} {ccy} | "
            f"used={used_str} | available={available_str} {ccy}{cache_note}".rstrip()
        )
    return "\n".join(lines)


def _format_budget_list(rows: list[ExecRow]) -> str:
    lines = ["Budgets (active):"]
    for r in rows:
        d = r.extra
        amount = decimal_to_display(_to_decimal(r.amount))
        ccy = r.currency or d.get("currency", "")
        lines.append(
            f"  - {d.get('name','(unnamed)')}: {amount} {ccy} "
            f"| {d.get('period','')} | {d.get('start_date','')}–{d.get('end_date','')} "
            f"| mode={d.get('mode','')}".rstrip()
        )
    return "\n".join(lines)


def _format_budget_remaining(rows: list[ExecRow]) -> str:
    """One line per budget: name | amount | spent | remaining | pct used | status.

    Status is derived in code (over / near / ok) so the LLM never has to
    classify it from raw numbers.

    Note: `amount` is taken from `r.amount` (the executor lifts the column
    named `amount` to that field), the rest from `r.extra`.
    """
    lines = ["Budgets remaining:"]
    for r in rows:
        d = r.extra
        amount = _to_decimal(r.amount)
        spent = _to_decimal(d.get("spent"))
        remaining = _to_decimal(d.get("remaining"))
        pct = _to_decimal(d.get("pct_used"))
        ccy = r.currency or d.get("currency", "")
        status = _budget_status(pct)
        days_remaining = d.get("days_remaining")
        daily_allowance = d.get("daily_allowance")
        # When the budget has already ended, present "expired" so the LLM
        # cannot make up a daily figure (we explicitly forbid LLM math).
        if daily_allowance is None:
            daily_str = "expired"
        else:
            daily_str = f"{decimal_to_display(_to_decimal(daily_allowance))} {ccy}/day"
        lines.append(
            f"  - {d.get('name','(unnamed)')}: "
            f"amount={decimal_to_display(amount)} {ccy} | "
            f"spent={decimal_to_display(spent)} {ccy} | "
            f"remaining={decimal_to_display(remaining)} {ccy} | "
            f"pct_used={pct or 0}% | status={status} | "
            f"days_remaining={days_remaining} | "
            f"daily_allowance={daily_str} | "
            f"period={d.get('start_date','')}–{d.get('end_date','')}"
        )
    return "\n".join(lines)


def _budget_status(pct) -> str:
    if pct is None:
        return "unknown"
    try:
        p = float(pct)
    except (TypeError, ValueError):
        return "unknown"
    if p >= 100:
        return "over"
    if p >= 80:
        return "near"
    return "ok"


def _format_balance(rows: list[ExecRow]) -> str:
    lines = ["Balances:"]
    for r in rows:
        amount = decimal_to_display(_to_decimal(r.amount))
        wallet = r.extra.get("wallet_name") or "(unknown wallet)"
        ccy = r.currency or ""
        lines.append(f"  - {wallet}: {amount} {ccy}".rstrip())
    return "\n".join(lines)


def _format_totals(rows: list[ExecRow]) -> str:
    """One line per currency — never sum across currencies."""
    parts = []
    for r in rows:
        amount = _to_decimal(r.amount)
        ccy = r.currency or ""
        cnt = f" ({r.count} รายการ)" if r.count is not None else ""
        parts.append(f"{decimal_to_display(amount)} {ccy}{cnt}".strip())
    return "Total: " + " | ".join(parts)


def _format_counts(rows: list[ExecRow]) -> str:
    parts = [f"{r.count or 0} ({r.currency or 'ALL'})" for r in rows]
    return "Count: " + " | ".join(parts)


def _format_breakdown(rows: list[ExecRow]) -> str:
    lines = ["Breakdown:"]
    for r in rows[:20]:
        amount = _to_decimal(r.amount)
        bucket = r.bucket or "(unknown)"
        ccy = r.currency or ""
        lines.append(f"  - {bucket}: {decimal_to_display(amount)} {ccy}".rstrip())
    if len(rows) > 20:
        lines.append(f"  ... +{len(rows) - 20} more")
    return "\n".join(lines)


def _format_list_summary(rows: list[ExecRow]) -> str:
    return f"Found {len(rows)} transactions (showing up to 50)."


def _breakdown(rows: list[ExecRow]) -> str:
    """Detailed list — separate section so ReAct can present each row."""
    lines = ["breakdown:"]
    for r in rows:
        d = r.extra
        date = d.get("date", "")
        type_ = d.get("type", "")
        wallet = d.get("wallet_name", "")
        category = d.get("category_name") or "(uncategorized)"
        note = d.get("note") or ""
        amount = decimal_to_display(_to_decimal(r.amount))
        ccy = r.currency or ""
        lines.append(
            f"  - {date} | {type_} | {amount} {ccy} | {wallet} | {category} | {note}".rstrip(" |")
        )
    return "\n".join(lines)


def _empty_message(spec: QuerySpec) -> str:
    return f"NO_DATA in {spec.time_range.start.isoformat()} → {spec.time_range.end.isoformat()}"


def _to_decimal(s: str | None) -> Decimal | None:
    if s is None or s == "":
        return None
    try:
        return Decimal(s)
    except Exception:
        return None
