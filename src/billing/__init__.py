"""Billing pipeline: extract OpenRouter cost → report to Go backend.

Two public entry points:

- `cost_extractor.extract_cost(response_or_meta, model)` parses the
  OpenRouter usage block (or falls through to two safety nets) and
  returns `CostBreakdown` with `cost_usd_micro` + token counts.

- `usage_reporter.report_usage(...)` fires a fire-and-forget callback
  to the Go `/api/v1/internal/ai-usage` endpoint with a 3s timeout.
  Failures are logged but never raised — the user-facing chat must
  continue even if a billing report is dropped (we'd rather eat the
  occasional missed charge than fail an otherwise-good turn).

The LangChain callback in `callbacks.py` wires both into every
OpenRouter LLM call site automatically by attaching itself via
`callbacks=[…]` when each LLM is constructed in `llm.py`. OpenRouter
calls run through `ChatOpenAI` pointed at the OpenRouter REST endpoint
with `extra_body={"usage": {"include": True}}` so cost surfaces inline.
"""

from src.billing.cost_extractor import CostBreakdown, extract_cost
from src.billing.usage_reporter import report_usage
from src.billing.user_context import (
    current_user_id,
    set_user_id,
    user_context,
)

__all__ = [
    "CostBreakdown",
    "current_user_id",
    "extract_cost",
    "report_usage",
    "set_user_id",
    "user_context",
]
