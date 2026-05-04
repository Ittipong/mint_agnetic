"""One iteration of the CodeAct loop: gen → execute → record.

The LLM sees the task, the catalog, the tool reference, and any prior steps
(code + stdout + error). It must either set `result = ...` to terminate or
emit more code that gets fed back into the next iteration.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from datetime import date
from decimal import Decimal
from typing import Any

from src.entity_catalog import EntityCatalog
from src.graph.compute_subgraph.codeact.namespace import build_namespace
from src.graph.compute_subgraph.codeact.sandbox import execute
from src.graph.compute_subgraph.state import ComputeSubState

# Loop bound — picked to cover compose / diff / multi-step but reject runaway.
MAX_STEPS = 5


# ── Prompt scaffolding ───────────────────────────────────────────────────────


_TOOL_REFERENCE = """\
You have ONLY these functions available. They each query the database and
return a list[dict]. Decimal values are real Decimal objects — arithmetic
on them is exact. Dates are ISO strings or date objects. Use the catalog
names verbatim.

  sum_income(start, end, *, wallet_names=None, category_names=None,
             tag_names=None, currency='ALL',
             convert_to_thb=False) -> list[dict]
      # rows: [{"currency": ..., "amount": Decimal, "cnt": int}, ...]
      # With convert_to_thb=True you get ONE row in THB instead of one per
      # currency. PREFER this over manually summing across currency rows.

  sum_expense(start, end, *, wallet_names=None, category_names=None,
              tag_names=None, currency='ALL',
              convert_to_thb=False) -> list[dict]

  sum_by_category(start, end, *, wallet_names=None, currency='ALL',
                  convert_to_thb=False) -> list[dict]
      # rows: [{"bucket": <category_name>, "currency": ..., "amount": Decimal,
      #         "cnt": int}, ...]

  sum_by_wallet(start, end, *, currency='ALL',
                convert_to_thb=False) -> list[dict]

  list_transactions(start, end, *, wallet_names=None, category_names=None,
                    tag_names=None, currency='ALL',
                    order_by='date_desc' | 'amount_desc', limit=50,
                    transaction_type=None) -> list[dict]
      # rows: [{"date": ..., "type": ..., "amount": Decimal, "currency": ...,
      #         "note": ..., "category_name": ..., "wallet_name": ...}, ...]
      # `transaction_type` ∈ {'income','expense','transfer','creditCardPay'}
      # narrows to one type. Use this for 'รายการรายรับ' / 'income only'.

  balance(*, as_of=None, wallet_names=None, currency='ALL',
          convert_to_thb=False) -> list[dict]
      # rows: [{"wallet_name": ..., "currency": ..., "amount": Decimal}, ...]
      # With convert_to_thb=True each wallet's balance is rolled up in THB.

  budget_remaining(*, start=None, end=None, budget_name_phrase=None) -> list[dict]
      # rows: [{"name": ..., "amount": ..., "spent": ..., "remaining": ...,
      #         "pct_used": ..., "days_remaining": ..., "daily_allowance": ...,
      #         "currency": ..., "period": ..., "start_date": ...,
      #         "end_date": ...}, ...]

  budget_transactions(*, budget_name_phrase, order_by='amount_desc',
                      limit=20) -> list[dict]

  budget_list() -> list[dict]

Helpers:
  Decimal(s)        # safe number, never use float() for money
  date(y, m, d)     # construct a date
  timedelta(days=N)
  today()           # returns today's date

Rules:
  - Set `result = <whatever you want returned>` when finished (a dict / list /
    Decimal / str — keep it small, under 5 KB).
  - NEVER use float() for money — keep everything Decimal.
  - NEVER sum across currency rows yourself. If you need a single THB total
    across mixed-currency data, pass convert_to_thb=True to the tool —
    the SQL applies a vetted FX chain (per-tx converted_amount → exchange_rate
    → today's rate from the currencies table). Trying to do FX in Python
    is forbidden.
  - NEVER call functions not listed above.
  - NEVER use import, open, exec, eval, getattr, dunder access.
  - You may print() to log intermediate values; the runtime captures stdout
    and shows it back to you next step.
  - Multi-step is OK: leave `result = None` to keep going. You have at most
    MAX_STEPS_PLACEHOLDER steps.
"""


_INSTRUCTION = """\
You are writing Python in a sandbox. Your job is to answer the user's
financial question by calling the provided tools and composing the results.

Write ONLY a Python code block — no explanation, no markdown fences.

Example — compare two months:

    march = sum_by_category(start='2026-03-01', end='2026-03-31')
    april = sum_by_category(start='2026-04-01', end='2026-04-30')
    by_cat = {}
    for r in march:
        by_cat.setdefault(r['bucket'], {})['march'] = r['amount']
    for r in april:
        by_cat.setdefault(r['bucket'], {})['april'] = r['amount']
    diff = []
    for cat, vals in by_cat.items():
        m = vals.get('march', Decimal('0'))
        a = vals.get('april', Decimal('0'))
        diff.append({'category': cat, 'march': m, 'april': a, 'change': a - m})
    diff.sort(key=lambda x: abs(x['change']), reverse=True)
    result = diff[:10]
"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior steps)"
    out = []
    for i, h in enumerate(history, 1):
        out.append(f"--- step {i} ---")
        out.append("CODE:")
        out.append(h.get("code", ""))
        if h.get("stdout"):
            out.append("STDOUT:")
            out.append(h["stdout"][:1000])
        if h.get("error"):
            out.append(f"ERROR: {h['error']}")
        elif h.get("result") is not None:
            out.append(f"RESULT (preview): {_preview(h['result'])}")
    return "\n".join(out)


def _preview(value: Any, limit: int = 800) -> str:
    """Stringify a result for showing back to the LLM — bounded length."""
    try:
        s = json.dumps(value, default=_json_default, ensure_ascii=False)
    except Exception:
        s = str(value)
    if len(s) > limit:
        s = s[:limit] + "… (truncated)"
    return s


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, (date,)):
        return o.isoformat()
    return str(o)


# ── Node ─────────────────────────────────────────────────────────────────────


async def codeact_step_node(state: ComputeSubState) -> dict:
    """Run one iteration. The graph loops until the LLM sets `result`
    or `MAX_STEPS` is hit."""
    from src.llm import llm  # imported lazily to avoid eager API client init

    history: list[dict] = state.get("codeact_history") or []
    if len(history) >= MAX_STEPS:
        return {
            "codeact_history": history,
            "codeact_done": True,
            "codeact_final": _last_non_null_result(history),
        }

    catalog: EntityCatalog | None = state.get("catalog")  # type: ignore[assignment]
    today = date.fromisoformat(state["today"])

    catalog_section = (
        catalog.render_for_prompt() if catalog else "(catalog unavailable)"
    )

    system = (
        _INSTRUCTION
        + "\n\n## Tool reference\n\n"
        + _TOOL_REFERENCE.replace("MAX_STEPS_PLACEHOLDER", str(MAX_STEPS))
        + "\n\n## Entity Catalog (use these names verbatim)\n\n"
        + catalog_section
    )
    user = (
        f"Today: {today.isoformat()}\n"
        f"Question: {state['task']}\n\n"
        f"## Prior steps\n{_format_history(history)}\n\n"
        f"Write the next Python block. If you have the answer, set `result`."
    )

    response = await llm.ainvoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    code = _extract_code(response.content if hasattr(response, "content") else str(response))

    main_loop = asyncio.get_running_loop()
    namespace = build_namespace(
        user_id=state["user_id"],
        catalog=catalog or EntityCatalog(),
        today=today,
        main_loop=main_loop,
    )

    # Sandbox runs in a worker thread so it can call back into the main loop
    # via run_coroutine_threadsafe (DB queries) without blocking it.
    result, stdout, error = await asyncio.to_thread(execute, code, namespace)

    history = [
        *history,
        {
            "step": len(history) + 1,
            "code": code,
            "stdout": stdout,
            "error": error,
            "result": result,
        },
    ]

    done = result is not None or len(history) >= MAX_STEPS
    final = result if result is not None else _last_non_null_result(history)
    return {
        "codeact_history": history,
        "codeact_done": done,
        "codeact_final": final,
    }


def _last_non_null_result(history: list[dict]) -> Any:
    for h in reversed(history):
        if h.get("result") is not None:
            return h["result"]
    return None


def _extract_code(text: str) -> str:
    """Strip markdown fences if the LLM ignored the instruction."""
    s = text.strip()
    if s.startswith("```"):
        # remove first fence + optional language
        s = s[3:]
        if s.startswith("python"):
            s = s[6:]
        s = s.lstrip("\n")
        if s.endswith("```"):
            s = s[:-3]
    return textwrap.dedent(s).strip()
