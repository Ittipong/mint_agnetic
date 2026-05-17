"""LangChain callback that wires `cost_extractor` + `usage_reporter` to
every OpenRouter LLM call.

Usage from `llm.py`:

    from src.billing.callbacks import BillingCallback

    return ChatOpenRouter(
        ...,
        callbacks=[BillingCallback(feature='chat')],
        model_kwargs={'usage': {'include': True}, ...},
    )

What it does:

- On `on_llm_end`, extract `response.llm_output.usage` (or the
  generation-level metadata when usage is empty) and call
  `extract_cost(...)`. Then schedule `report_usage` as a background
  task via `usage_reporter.fire_and_forget`.

- Reads the active user from the request-scoped ContextVar
  (`user_context.current_user_id`). When None, the callback skips the
  report and warns once — better than guessing the user.

The callback is INERT for non-OpenRouter calls: ChatOpenAI/Typhoon
also fire `on_llm_end` but their response.usage block doesn't carry a
`cost` field, and we don't have a fallback price for the Typhoon
endpoint, so we'd just log a "missing cost" warning. Today we attach
this only on OpenRouter constructors (`_make_openrouter_llm`).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from src.billing import cost_extractor, usage_reporter
from src.billing.user_context import current_user_id

_LOG = logging.getLogger(__name__)


class BillingCallback(AsyncCallbackHandler):
    """Async callback used by every OpenRouter LLM. Stateless besides
    the `feature` tag so it can be shared across concurrent requests."""

    # AsyncCallbackHandler doesn't define __slots__ on its base; we
    # add ours to keep memory tight when the callback is attached to
    # every LLM call (one instance per ChatOpenRouter object).
    __slots__ = ("_feature", "_model_hint")

    def __init__(self, feature: str = "chat", *, model_hint: str | None = None) -> None:
        super().__init__()
        self._feature = feature
        # Some LLM impls don't include the model name in the response
        # metadata even though it's known at construction time. We
        # store it as a hint to fall back on.
        self._model_hint = model_hint

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        user_id = current_user_id()
        if not user_id:
            _LOG.debug(
                "billing.callback_skipped_no_user feature=%s",
                self._feature,
            )
            return

        usage_meta, generation_id, model = _drill_for_meta(response, self._model_hint)
        breakdown = cost_extractor.extract_cost(usage_meta, model, generation_id)
        if not breakdown.is_billable:
            _LOG.debug(
                "billing.callback_not_billable feature=%s model=%s",
                self._feature, model,
            )
            return
        if not generation_id:
            # No id → Go can't dedupe. Skip rather than ship a
            # non-idempotent charge that Python's retry could double.
            _LOG.warning(
                "billing.callback_missing_generation_id feature=%s model=%s cost_micro=%s",
                self._feature, model, breakdown.cost_usd_micro,
            )
            return

        usage_reporter.fire_and_forget(
            user_id=user_id,
            request_id=generation_id,
            feature=self._feature,
            model=model,
            cost_usd_micro=breakdown.cost_usd_micro,
            prompt_tokens=breakdown.prompt_tokens,
            completion_tokens=breakdown.completion_tokens,
        )


def _drill_for_meta(
    response: LLMResult,
    model_hint: str | None,
) -> tuple[dict[str, Any] | None, str | None, str]:
    """Pull (usage_dict, generation_id, model_name) out of an LLMResult.

    LangChain stuffs OpenAI-compatible usage in two places:
    `response.llm_output['token_usage']` (top-level summary) and the
    individual `Generation.generation_info` blocks. We prefer the
    per-generation block because it carries the response `id` we
    need for idempotency.
    """
    llm_output = getattr(response, "llm_output", None) or {}
    model = (
        (llm_output.get("model_name") if isinstance(llm_output, dict) else None)
        or model_hint
        or ""
    )

    if response.generations:
        # `generations` is a list-of-list because LangChain allows batch
        # calls; we only ever issue a single prompt so [0][0] is safe.
        try:
            gen = response.generations[0][0]
        except (IndexError, TypeError):
            gen = None
        if gen is not None:
            gen_info = getattr(gen, "generation_info", None) or {}
            usage = gen_info.get("usage") or gen_info.get("token_usage")
            generation_id = gen_info.get("id") or gen_info.get("generation_id")
            # Some routes nest the id one level deeper under `response_metadata`.
            if generation_id is None:
                message = getattr(gen, "message", None)
                meta = getattr(message, "response_metadata", None) or {}
                generation_id = meta.get("id") or meta.get("generation_id")
                if usage is None:
                    usage = meta.get("usage") or meta.get("token_usage")
                if not model:
                    model = meta.get("model") or meta.get("model_name") or ""
            if usage is not None or generation_id is not None:
                return usage, generation_id, model

    # Fall back to the top-level llm_output block.
    if isinstance(llm_output, dict):
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        generation_id = llm_output.get("id")
        return usage, generation_id, model

    return None, None, model
