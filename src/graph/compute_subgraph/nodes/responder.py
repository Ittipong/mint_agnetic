"""responder — render the codeact loop's final result as a tool-readable string.

Smart CodeAct only — the legacy plan/execute path was removed in Phase 3.
We deliberately skip the LLM here. The parent ReAct (reason_node) writes
the final Thai prose; this node just produces a structured payload with
all numbers pre-formatted as strings so there's no math hallucination
risk between the SQL result and the user-visible answer.
"""

from __future__ import annotations

import json
from datetime import date as _date, datetime as dt
from decimal import Decimal

from src.graph.compute_subgraph.state import ComputeSubState


# ── Public node ──────────────────────────────────────────────────────────────


async def respond_node(state: ComputeSubState) -> dict:
    t0 = dt.now()

    # Sandbox-raised clarification — surface the question to the user.
    if state.get("needs_clarification") and state.get("clarification_question"):
        answer = _format_clarification(
            state["clarification_question"],
            state.get("clarification_options") or [],
        )
    else:
        answer = _format_codeact(state)

    ms = (dt.now() - t0).total_seconds() * 1000
    print(f"[PERF] respond_node: {ms:.0f}ms steps={len(state.get('codeact_history') or [])}")
    return {"answer": answer}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_clarification(question: str, options: list[str]) -> str:
    """Render sandbox-raised clarification as a tool-readable string."""
    lines = [f"NEEDS_CLARIFICATION (codeact): {question}"]
    if options:
        lines.append("Options:")
        for opt in options[:8]:
            lines.append(f"  - {opt}")
    return "\n".join(lines)


def _format_codeact(state: ComputeSubState) -> str:
    """Render the codeact loop result as a structured tool-readable string."""
    final = state.get("codeact_final")
    history = state.get("codeact_history") or []
    n_steps = len(history)

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
        rendered = json.dumps(final, default=_json_default, ensure_ascii=False, indent=2)
    except Exception:
        rendered = str(final)
    if len(rendered) > 4000:
        rendered = rendered[:4000] + "\n… (truncated)"
    return f"[metric=freeform_codeact | steps={n_steps}]\n{rendered}"


def _json_default(o):
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, _date):
        return o.isoformat()
    return str(o)
