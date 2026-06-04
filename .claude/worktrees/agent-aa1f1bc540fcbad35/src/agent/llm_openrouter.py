"""OpenRouter LLM adapter (OpenAI-compatible).

We deliberately keep this thin: a `make_llm_call(role)` factory that
returns an `async (messages) -> str` callable. The role determines which
model env var is picked. No hardcoded model names.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx

from .session_logger import slog, slog_block, slog_error


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    pass


# Per-role base-URL override env vars. Vision/STT may route to a different
# provider than text (e.g. a direct Gemini endpoint); they fall back to the
# shared OpenRouter base when their env var is unset.
_BASE_URL_ENV = {
    "vision":  "VISION_BASE_URL",
    "stt":     "STT_BASE_URL",
    "propose": "PROPOSE_BASE_URL",
}

# Per-role primary model env vars. Each role MUST have a matching
# `<env>_FALLBACK_MODELS` entry below; validated at startup by
# `llm_env_validator.validate_llm_env()`. No cross-role implicit fallback
# chains — every role stands on its own contract (`_MODEL` + `_FALLBACK_MODELS`).
_MODEL_ENV = {
    "react":            "REACT_MODEL",
    "codeact":          "CODEACT_MODEL",
    "classify":         "CLASSIFY_MODEL",
    "propose":          "PROPOSE_MODEL",
    "vision":           "VISION_MODEL",
    "stt":              "STT_MODEL",
    "preamble":         "PREAMBLE_MODEL",
    "judge":            "JUDGE_MODEL",
}

_FALLBACK_ENV = {
    "react":            "REACT_FALLBACK_MODELS",
    "codeact":          "CODEACT_FALLBACK_MODELS",
    "classify":         "CLASSIFY_FALLBACK_MODELS",
    "propose":          "PROPOSE_FALLBACK_MODELS",
    "vision":           "VISION_FALLBACK_MODELS",
    "stt":              "STT_FALLBACK_MODELS",
    "preamble":         "PREAMBLE_FALLBACK_MODELS",
    "judge":            "JUDGE_FALLBACK_MODELS",
}


def _resolve_model(role: str) -> str:
    """Return env-driven model name for a role. Each role REQUIRES its own
    `<ROLE>_MODEL` env to be set — no cross-role implicit fallback. The
    startup validator (`validate_llm_env`) catches missing vars before any
    request is served; this check is the runtime safety net."""
    env_name = _MODEL_ENV.get(role)
    if env_name is None:
        raise OpenRouterError(f"unknown llm role: {role!r}")
    m = os.getenv(env_name)
    if not m:
        raise OpenRouterError(f"{env_name} is not set")
    return m


def _resolve_base(role: str) -> str:
    """Base URL for a role — per-role override env (vision/stt) else the
    shared OpenRouter base."""
    env_name = _BASE_URL_ENV.get(role)
    if env_name:
        override = os.getenv(env_name)
        if override:
            return override
    return os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)


def _resolve_fallbacks(role: str) -> list[str]:
    """Ordered fallback model names for a role (empty when unset).

    Accepts BOTH formats so config can't silently break:
      - JSON array:   ["deepseek/deepseek-v4-flash","google/gemini-2.5-flash"]
      - comma list:   deepseek/deepseek-v4-flash, google/gemini-2.5-flash
    The .env.example ships the JSON-array form; the old comma-only split turned
    it into garbage model names like '["deepseek/deepseek-v4-flash"'.
    """
    env_name = _FALLBACK_ENV.get(role)
    raw = (os.getenv(env_name, "") if env_name else "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(m).strip() for m in parsed if str(m).strip()]
        except json.JSONDecodeError:
            pass  # fall through to comma parsing
    return [m.strip() for m in raw.split(",") if m.strip()]


def _resolve_provider() -> dict | None:
    """OpenRouter `provider` routing prefs from env, or None when unset.

    Tail latency on the codeact loop is dominated by which upstream provider
    OpenRouter happens to route to (we observed one flash-lite call at 8.8s vs
    its ~1.5s siblings). `sort=latency` asks OpenRouter to prefer the
    lowest-latency provider; an explicit `order` pins providers outright.

      OPENROUTER_PROVIDER_SORT  = latency | throughput | price
      OPENROUTER_PROVIDER_ORDER = "deepinfra,google-vertex"  (or JSON array)
    """
    prefs: dict = {}
    sort = (os.getenv("OPENROUTER_PROVIDER_SORT") or "").strip()
    if sort:
        prefs["sort"] = sort
    order_raw = (os.getenv("OPENROUTER_PROVIDER_ORDER") or "").strip()
    if order_raw:
        order: list[str] = []
        if order_raw.startswith("["):
            try:
                parsed = json.loads(order_raw)
                if isinstance(parsed, list):
                    order = [str(p).strip() for p in parsed if str(p).strip()]
            except json.JSONDecodeError:
                order = []
        if not order:
            order = [p.strip() for p in order_raw.split(",") if p.strip()]
        if order:
            prefs["order"] = order
    return prefs or None


def _build_body(
    model: str,
    messages: list[dict],
    *,
    stream: bool,
    temperature: float = 0.2,
    with_provider: bool = True,
) -> dict:
    """Assemble the chat-completions body with optional provider routing.

    `with_provider=False` omits the OpenRouter-specific `provider` block — used
    when a role's base URL is overridden to a non-OpenRouter endpoint (a direct
    provider would reject the unknown field)."""
    body: dict = {"model": model, "messages": messages, "temperature": temperature}
    if with_provider:
        provider = _resolve_provider()
        if provider is not None:
            body["provider"] = provider
    if stream:
        body["stream"] = True
        # Ask for a trailing usage chunk so streamed calls can be cost-traced.
        body["stream_options"] = {"include_usage": True}
    return body


def _resolve_request(role: str) -> tuple[str, dict, str]:
    """Resolve (model, headers, url) for a role. Shared by the non-streaming
    and streaming factories so model/header/role logic lives in one place."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY env var is not set")
    base = _resolve_base(role)
    model = _resolve_model(role)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    url = base.rstrip("/") + "/chat/completions"
    return model, headers, url


def _log_usage(tag: str, model: str, usage: dict | None) -> None:
    """Record token usage so cache hits are verifiable in the session log.

    Gemini does implicit prefix caching on OpenRouter; `cached` > 0 confirms the
    codeact loop's stable system prefix is actually being reused across steps.
    Best-effort — never raises (usage shape varies by provider)."""
    if not usage:
        return
    try:
        details = usage.get("prompt_tokens_details") or {}
        slog(
            tag,
            f"usage model={model} prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')} "
            f"cached={details.get('cached_tokens')} cost={usage.get('cost')}",
        )
    except Exception:  # noqa: BLE001 — telemetry must never break the call
        pass


def make_llm_call(role: str, *, timeout_s: float = 30.0) -> Callable[[list[dict]], Awaitable[str]]:
    """Build an async callable that calls OpenRouter chat completions and
    returns just the assistant content string.

    Resilience (text roles): tries the primary model then each configured
    fallback (CODEACT_FALLBACK_MODELS / REACT_FALLBACK_MODELS) on transport
    error / empty content. With no fallback configured it still retries the
    sole model once so a transient blip doesn't fail the turn. A tighter
    default timeout (30s, was 60s) bounds the tail when a provider hangs;
    `provider.sort=latency` (env) steers routing away from slow providers."""
    primary, headers, url = _resolve_request(role)
    models = [primary] + [m for m in _resolve_fallbacks(role) if m != primary]
    # Guarantee ≥1 retry even when no fallback is configured.
    attempts = models if len(models) > 1 else [primary, primary]

    async def call(messages: list[dict]) -> str:
        # Central LLM I/O logging — every node's prompt + response funnels
        # through here, so the per-session debug file captures all of them in
        # one place. No-op when no session logger is bound (prod / tests).
        tag = f"llm.{role}"
        slog_block(tag, f"PROMPT models={models}", _format_messages(messages))
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for i, model in enumerate(attempts):
                body = _build_body(model, messages, stream=False)
                try:
                    r = await client.post(url, headers=headers, json=body)
                    r.raise_for_status()
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                except (httpx.HTTPError, KeyError, IndexError) as e:
                    last_err = e
                    slog_error(tag, OpenRouterError(f"attempt {i + 1} ({model}) failed: {e}"))
                    continue
                _log_usage(tag, model, data.get("usage"))
                if content and content.strip():
                    slog_block(tag, f"RESPONSE model={model}", content)
                    return content
                last_err = OpenRouterError(f"model {model} returned empty content")
                slog_error(tag, last_err)
        raise OpenRouterError(f"all {role} attempts failed; last error: {last_err}")

    return call


def make_multimodal_call(
    role: str, *, timeout_s: float = 90.0, temperature: float = 0.1
) -> Callable[[list[dict]], Awaitable[str]]:
    """Build an async callable for a multimodal role ('vision' | 'stt').

    Differs from `make_llm_call`:
      - tries the primary model then each fallback in order (transport error
        or empty content rolls to the next),
      - uses the role's own base URL (VISION_BASE_URL / STT_BASE_URL),
      - a longer default timeout (image/audio inference is slower),
      - a low temperature (deterministic extraction/transcription).

    `messages` already carry the multimodal content parts (image_url /
    input_audio) — the transport forwards them unchanged. Image/audio data is
    redacted from the debug log so the per-session file stays small.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY env var is not set")
    models: list[str] = [_resolve_model(role)]
    for m in _resolve_fallbacks(role):
        if m not in models:
            models.append(m)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    base = _resolve_base(role)
    url = base.rstrip("/") + "/chat/completions"
    # `provider` routing is OpenRouter-specific; vision/stt may be pointed at a
    # direct provider endpoint via *_BASE_URL, which would reject it.
    is_openrouter = "openrouter.ai" in base

    async def call(messages: list[dict]) -> str:
        tag = f"llm.{role}"
        slog_block(tag, f"PROMPT models={models}", _format_messages(_redact_media(messages)))
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for model in models:
                # Route through the shared body builder so multimodal calls
                # honor the same `provider` routing prefs (sort=latency / order)
                # as the text roles — no call path should silently bypass it.
                body = _build_body(
                    model, messages, stream=False,
                    temperature=temperature, with_provider=is_openrouter,
                )
                try:
                    r = await client.post(url, headers=headers, json=body)
                    r.raise_for_status()
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                except (httpx.HTTPError, KeyError, IndexError) as e:
                    last_err = e
                    slog_error(tag, OpenRouterError(f"model {model} failed: {e}"))
                    continue
                if content and content.strip():
                    slog_block(tag, f"RESPONSE model={model}", content)
                    return content
                last_err = OpenRouterError(f"model {model} returned empty content")
                slog_error(tag, last_err)
        raise OpenRouterError(f"all {role} models failed; last error: {last_err}")

    return call


def make_llm_stream(
    role: str, *, timeout_s: float = 60.0
) -> Callable[[list[dict]], AsyncIterator[str]]:
    """Build an async callable that streams OpenRouter chat completions and
    yields assistant content deltas as they arrive.

    Used by the CodeAct outer node's "writer" pass (A2a): the writer renders
    the loop's already-verified `final_answer` into a natural Thai message,
    streamed token-by-token so the mobile client shows the answer immediately.
    Streaming raw text is coalesce-immune over ngrok (unlike one big JSON
    block), which is the whole reason we restored v1-style token streaming.
    """
    primary, headers, url = _resolve_request(role)
    models = [primary] + [m for m in _resolve_fallbacks(role) if m != primary]
    attempts = models if len(models) > 1 else [primary, primary]

    def stream(messages: list[dict]) -> AsyncIterator[str]:
        tag = f"llm_stream.{role}"

        async def _gen() -> AsyncIterator[str]:
            slog_block(tag, f"PROMPT (stream) models={models}", _format_messages(messages))
            last_err: Exception | None = None
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                for i, model in enumerate(attempts):
                    body = _build_body(model, messages, stream=True)
                    acc: list[str] = []
                    yielded = False
                    try:
                        async with client.stream(
                            "POST", url, headers=headers, json=body
                        ) as r:
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                payload = line[len("data:"):].strip()
                                if payload == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(payload)
                                except json.JSONDecodeError:
                                    # Tolerate keep-alive / comment lines; a single
                                    # malformed delta must not abort the whole stream.
                                    continue
                                # Trailing usage-only chunk (stream_options) has
                                # empty choices — capture it for cache telemetry.
                                if chunk.get("usage"):
                                    _log_usage(tag, model, chunk["usage"])
                                try:
                                    delta = chunk["choices"][0]["delta"]
                                except (KeyError, IndexError):
                                    continue
                                content = delta.get("content")
                                if content:
                                    acc.append(content)
                                    yielded = True
                                    yield content
                        slog(tag, f"RESPONSE (stream) {len(''.join(acc))} chars model={model}")
                        return
                    except httpx.HTTPError as e:
                        last_err = e
                        slog_error(tag, OpenRouterError(f"attempt {i + 1} ({model}) stream error: {e}"))
                        # Tokens already sent downstream → can't cleanly retry on
                        # another model; surface the error. Only fall back when the
                        # failure happened before the first token.
                        if yielded:
                            raise OpenRouterError(f"stream HTTP error mid-stream: {e}") from e
                        continue
            raise OpenRouterError(f"all {role} stream attempts failed; last error: {last_err}")

        return _gen()

    return stream


def _format_messages(messages: list[dict]) -> str:
    """Render chat messages as `role: content` lines for the debug log."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"[{role}]\n{content}")
    return "\n".join(lines)


def _redact_media(messages: list[dict]) -> list[dict]:
    """Return a shallow copy of `messages` with inline image/audio base64
    payloads replaced by a `<image …>` / `<audio …>` placeholder, so the
    debug log records the prompt structure without the multi-megabyte blob.
    """
    redacted: list[dict] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            redacted.append(m)
            continue
        parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(part)
                continue
            ptype = part.get("type")
            if ptype == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                parts.append({"type": "image_url", "image_url": {"url": f"<image {len(url)} chars>"}})
            elif ptype == "input_audio":
                audio = part.get("input_audio") or {}
                fmt = audio.get("format", "?")
                data = audio.get("data", "")
                parts.append({"type": "input_audio",
                              "input_audio": {"format": fmt, "data": f"<audio {len(data)} chars>"}})
            else:
                parts.append(part)
        redacted.append({**m, "content": parts})
    return redacted
