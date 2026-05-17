"""Fire-and-forget reporter that posts an OpenRouter cost report to the
Go backend's `POST /api/v1/internal/ai-usage` endpoint.

Design constraints:

- **Must not block the SSE stream.** The reporter is invoked from a
  LangChain callback that runs on the same async task as the chat
  response. Use `asyncio.create_task` plus a short HTTP timeout so a
  wedged backend can never stall the user's reply.

- **Must tolerate transient backend failure.** Retry once with a tiny
  backoff (200ms) to absorb a momentary network blip. Beyond that we
  log loudly and accept the lost charge — chasing the chase is the
  job of an out-of-band reconciler (Phase 10).

- **Must never log the shared secret.** The header is set on the
  request but the value never enters the logging pipeline. The Go
  receiver does the constant-time compare and tells us about a
  mismatch via a 401, not in the response body.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from src.config import settings

_LOG = logging.getLogger(__name__)

# Hard per-call timeout. The Go callback is a single UPDATE + INSERT
# wrapped in a tx — should complete in <50ms locally and <200ms with
# a network hop. A 3s ceiling forgives a one-off GC pause without
# letting a wedged backend stall the chat.
_HTTP_TIMEOUT_SEC = 3.0
# One retry to absorb a transient network blip; further retries chase
# diminishing returns. Anything that fails twice is escalated to a
# warning log and dropped.
_MAX_ATTEMPTS = 2
_BACKOFF_SEC = 0.2


def _backend_url() -> str:
    """Resolve the Go backend's base URL. Env-driven so the same image
    runs against dev / staging / prod without code changes."""
    return (os.environ.get("GO_BACKEND_URL") or "").rstrip("/")


def _shared_secret() -> str:
    """The header value Go uses to verify this call originated from us."""
    return os.environ.get("GO_INTERNAL_API_KEY") or ""


def _is_production() -> bool:
    """Match the project-wide environment gate used in src.config."""
    return getattr(settings, "is_production", False)


async def _post_once(client: httpx.AsyncClient, url: str, secret: str, payload: dict[str, Any]) -> bool:
    """Single POST attempt. Returns True on a 2xx response."""
    try:
        resp = await client.post(
            url,
            json=payload,
            headers={"X-Internal-API-Key": secret},
            timeout=_HTTP_TIMEOUT_SEC,
        )
    except httpx.RequestError as exc:
        _LOG.warning("billing.usage_post_network_error err=%s", exc)
        return False

    if 200 <= resp.status_code < 300:
        return True

    # Authentication failure is operator-actionable — surface it loud
    # so a missing/rotated secret stops being silent.
    if resp.status_code == 401:
        _LOG.error(
            "billing.usage_post_unauthorized — Go rejected the shared secret"
        )
        return False

    _LOG.warning(
        "billing.usage_post_non_2xx status=%s req_id=%s",
        resp.status_code,
        payload.get("request_id"),
    )
    return False


async def report_usage(
    *,
    user_id: str,
    request_id: str,
    feature: str,
    model: str,
    cost_usd_micro: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Post a single usage row to Go. Fire-and-forget — exceptions are
    caught and logged, never re-raised.

    `request_id` is the OpenRouter generation id. Same id reposted
    twice is a no-op on the Go side (the chat_proxy ChargeIdempotent
    short-circuits) — safe to retry.
    """
    backend_url = _backend_url()
    secret = _shared_secret()
    if not backend_url or not secret:
        # Mis-configuration. In prod we choose the lossy path
        # (warn + drop) so chat keeps working; in dev/staging we want
        # the developer to notice immediately.
        if _is_production():
            _LOG.warning(
                "billing.usage_report_skipped_unconfigured "
                "backend_url_present=%s secret_present=%s",
                bool(backend_url), bool(secret),
            )
            return
        raise RuntimeError(
            "billing.usage_report misconfigured: set GO_BACKEND_URL and "
            "GO_INTERNAL_API_KEY (strict in non-production)."
        )

    if not request_id:
        _LOG.warning(
            "billing.usage_report_missing_request_id user_id=%s model=%s",
            user_id, model,
        )
        return

    payload = {
        "user_id": user_id,
        "request_id": request_id,
        "feature": feature,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd_micro": cost_usd_micro,
    }
    url = backend_url + "/api/v1/internal/ai-usage"

    async with httpx.AsyncClient() as client:
        for attempt in range(_MAX_ATTEMPTS):
            ok = await _post_once(client, url, secret, payload)
            if ok:
                _LOG.debug(
                    "billing.usage_reported feature=%s model=%s cost_micro=%s req_id=%s attempt=%s",
                    feature, model, cost_usd_micro, request_id, attempt + 1,
                )
                return
            if attempt + 1 < _MAX_ATTEMPTS:
                await asyncio.sleep(_BACKOFF_SEC)

    _LOG.warning(
        "billing.usage_report_dropped_after_retry user_id=%s req_id=%s model=%s cost_micro=%s",
        user_id, request_id, model, cost_usd_micro,
    )


def fire_and_forget(
    *,
    user_id: str,
    request_id: str,
    feature: str,
    model: str,
    cost_usd_micro: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Schedule a usage report as a background task.

    Used by the LangChain callback which runs inside an async loop but
    can't await the post itself without blocking the chat stream.
    Errors raised inside the task are swallowed by `report_usage`
    itself — they will never propagate to the chat handler.
    """
    coro = report_usage(
        user_id=user_id,
        request_id=request_id,
        feature=feature,
        model=model,
        cost_usd_micro=cost_usd_micro,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread (sync caller). Spin up a dedicated
        # one for the single post. Rare path — only happens when the
        # billing callback is triggered outside the FastAPI request
        # cycle (e.g. a CLI tool exercising the graph).
        asyncio.run(coro)
        return
    loop.create_task(coro)
