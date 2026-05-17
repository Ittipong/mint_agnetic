"""Convert an OpenRouter LLM response into a billable cost.

Three layers, applied in priority order per the project plan:

1. **Main path** — read `response.usage.cost` (in USD). Only present
   when the request was made with `usage:{include: true}` and the
   model has settled pricing. This is the only path that will fire
   for our default OpenRouter routes once `llm.py` is patched.

2. **Safety net 1** — call OpenRouter `/api/v1/generation?id=<gen_id>`.
   Some routes (cached prompt completions, very fast cached responses)
   return a `cost` of None on the inline usage block. The settled cost
   is available via the generation endpoint a few seconds later. We
   call synchronously here because the billing callback is already
   running off the SSE stream's critical path.

3. **Safety net 2** — multiply token counts by a local pricing table.
   Last-ditch fallback when both upstream signals are missing (network
   error fetching the generation, OpenRouter outage, untracked model).
   The table values are intentionally conservative — better to
   slightly over-charge than to silently leak revenue.

A "free model" (extracted cost == 0 micro USD) flows through cleanly
— the caller skips the usage report when both cost AND
token counts indicate a no-op call, but a real-token zero-cost call
still produces an audit log row server-side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import settings

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Outcome of cost extraction. All amounts are in micro USD
    (1 USD = 1_000_000) so they convert losslessly to the Go ledger's
    integer column."""

    cost_usd_micro: int
    prompt_tokens: int
    completion_tokens: int
    source: str  # 'usage_cost' | 'generation_api' | 'fallback_table' | 'none'

    @property
    def is_billable(self) -> bool:
        """A breakdown is billable when EITHER cost is positive OR
        tokens are present. A pure zero/zero result means the call
        never reached an LLM (cached intent classifier short-circuit,
        etc.) — skip the report to avoid log noise."""
        return self.cost_usd_micro > 0 or self.prompt_tokens > 0 or self.completion_tokens > 0


# Pricing fallback table. Numbers are per 1k tokens in USD. Sourced
# from openrouter.ai/models at the time of writing — refresh quarterly
# or whenever we add a new default model. Always biased upward — this
# table only kicks in when upstream signals failed, so over-counting
# protects the business at the cost of a slightly unfair charge to
# the user in an edge case.
_FALLBACK_PRICING_USD_PER_1K: dict[str, dict[str, float]] = {
    # Reasoning / chat
    "deepseek/deepseek-v4-flash":          {"prompt": 0.00015, "completion": 0.00060},
    "google/gemini-2.5-flash":             {"prompt": 0.00010, "completion": 0.00040},
    "google/gemini-2.5-flash-lite":        {"prompt": 0.00007, "completion": 0.00028},
    "google/gemini-2.0-flash-lite-001":    {"prompt": 0.00007, "completion": 0.00028},
    "qwen/qwen3.6-35b-a3b":                {"prompt": 0.00009, "completion": 0.00009},
    "amazon/nova-lite-v1":                 {"prompt": 0.00006, "completion": 0.00024},
    "anthropic/claude-haiku-5":            {"prompt": 0.00100, "completion": 0.00500},
    # Add more as we expand the default model set.
}

# How long to wait for the generation-detail API before falling
# through to the local pricing table. The endpoint is cheap and
# usually answers in <500ms; 2s is generous without holding the
# graph thread.
_GENERATION_TIMEOUT_SEC = 2.0


def _to_micro(usd: float | int | None) -> int:
    """Convert a USD float to integer micro USD, clamped at 0."""
    if usd is None or usd < 0:
        return 0
    return int(round(float(usd) * 1_000_000))


def extract_cost(usage_meta: dict[str, Any] | None, model: str, generation_id: str | None = None) -> CostBreakdown:
    """Compute a `CostBreakdown` for one LLM call.

    Parameters mirror what we can pull out of a langchain `AIMessage`
    `response_metadata` block (and what we cache in the OpenRouter
    SDK response):

    - `usage_meta` — typically `response.usage` dict with `cost`,
      `prompt_tokens`, `completion_tokens`. May be None when the
      callback fires before the response settled.
    - `model` — model identifier (`provider/name` for OpenRouter).
    - `generation_id` — OpenRouter's `id` from the response envelope.
      Required for safety net 1; when absent we skip straight to net 2.
    """
    usage = usage_meta or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cost_raw = usage.get("cost")

    if cost_raw is not None and cost_raw >= 0:
        return CostBreakdown(
            cost_usd_micro=_to_micro(cost_raw),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source="usage_cost",
        )

    # Safety net 1: query OpenRouter for the settled cost.
    if generation_id:
        settled = _fetch_generation_cost(generation_id)
        if settled is not None:
            return CostBreakdown(
                cost_usd_micro=_to_micro(settled),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source="generation_api",
            )

    # Safety net 2: local pricing table.
    pricing = _FALLBACK_PRICING_USD_PER_1K.get(model)
    if pricing and (prompt_tokens or completion_tokens):
        approx_usd = (
            prompt_tokens / 1000.0 * pricing.get("prompt", 0.0)
            + completion_tokens / 1000.0 * pricing.get("completion", 0.0)
        )
        return CostBreakdown(
            cost_usd_micro=_to_micro(approx_usd),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source="fallback_table",
        )

    return CostBreakdown(
        cost_usd_micro=0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        source="none",
    )


def _fetch_generation_cost(generation_id: str) -> float | None:
    """Call OpenRouter's `/api/v1/generation?id=` for the settled cost.

    Returns USD as a float, or None on any failure (network, non-2xx,
    malformed body). We deliberately swallow exceptions here — the
    caller has another fallback layer and a billing failure must not
    break chat for the user.
    """
    api_key = settings.openrouter_api_key
    if not api_key:
        return None
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/generation",
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_GENERATION_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            _LOG.warning(
                "billing.generation_api_non_200 status=%s gen_id=%s",
                resp.status_code,
                generation_id,
            )
            return None
        body = resp.json()
        data = body.get("data") or {}
        # OpenRouter returns `total_cost` (float, USD) on the settled
        # generation. Some routes also expose `native_tokens_*` which
        # we ignore — token counts come from the inline usage block.
        total_cost = data.get("total_cost")
        if total_cost is None:
            return None
        return float(total_cost)
    except (httpx.RequestError, ValueError, KeyError) as exc:
        _LOG.warning(
            "billing.generation_api_failed gen_id=%s err=%s",
            generation_id,
            exc,
        )
        return None
