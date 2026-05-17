"""Direct REST client for OpenRouter `/chat/completions`.

Why bypass `langchain_openai.ChatOpenAI`:
- Full control over request body — `models` array, `usage.include`,
  `cache_control` blocks, and any future OpenRouter-only field forwards
  verbatim without fighting an SDK that doesn't recognise them.
- One fewer indirection layer — debugging 4xx/5xx happens here, not in
  vendored `openai` package code.

Why still a `BaseChatModel` subclass:
- LangGraph and every node in `src/graph/*` builds against this
  interface, so keeping it means `.invoke()` / `.bind_tools()` /
  `.astream()` callers stay unchanged.
- Callbacks, LangSmith tracing, message typing, retry hooks come free.

Scope (Phase 1 + 2): sync + async `_generate`, sync + async `_stream`,
tool calling, structured content (cache_control / multimodal blocks
forwarded verbatim). Audio-input live testing is deferred — the shape
itself passes through verbatim since OpenRouter accepts the OpenAI
`input_audio` block one-for-one.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Iterator, Sequence

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_SSE_DATA_PREFIX = "data: "
_SSE_DONE_TOKEN = "[DONE]"


class ChatOpenRouterREST(BaseChatModel):
    """Chat model that POSTs directly to OpenRouter `/chat/completions`.

    Mirrors enough of ChatOpenAI's surface for LangGraph and tool-using
    nodes to swap drop-in. Fields not declared here (e.g. `top_p`,
    `presence_penalty`) are intentionally absent — the project doesn't
    use them today and adding them speculatively just hides bugs."""

    model: str
    api_key: SecretStr
    base_url: str = "https://openrouter.ai/api/v1"
    fallback_models: list[str] = Field(default_factory=list)
    # OpenRouter provider routing preferences. Both default to "absent"
    # so the emitted payload omits the `provider` block entirely when
    # the caller hasn't opted in — keeps default routing identical to
    # before this feature landed. See:
    # https://openrouter.ai/docs/features/provider-routing
    provider_order: list[str] = Field(default_factory=list)
    # Tri-state: None = don't emit (let OpenRouter default to true);
    # True/False = explicit override. We can't fold this into a bool
    # default because `False` is a meaningful pin-providers signal
    # that must not silently default to True.
    provider_allow_fallbacks: bool | None = None
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2

    @property
    def _llm_type(self) -> str:
        return "openrouter-rest"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseTool],
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        # `convert_to_openai_tool` normalises BaseTool / Pydantic / dict
        # to the {"type": "function", "function": {...}} shape that
        # OpenAI-compatible endpoints (including OpenRouter) accept.
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    # ── BaseChatModel overrides ──────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, stream=False, **kwargs)
        body = self._post_sync(payload)
        return _body_to_chat_result(body, self.model)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, stream=False, **kwargs)
        body = await self._post_async(payload)
        return _body_to_chat_result(body, self.model)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, stream=True, **kwargs)
        for raw_chunk in self._stream_sync(payload):
            chunk = _raw_chunk_to_generation_chunk(raw_chunk, self.model)
            if chunk is None:
                continue
            if run_manager:
                # Hand the text fragment to the callback so token-level
                # listeners (LangSmith, console streaming) see it.
                run_manager.on_llm_new_token(
                    chunk.message.content if isinstance(chunk.message.content, str) else "",
                    chunk=chunk,
                )
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stop, stream=True, **kwargs)
        async for raw_chunk in self._stream_async(payload):
            chunk = _raw_chunk_to_generation_chunk(raw_chunk, self.model)
            if chunk is None:
                continue
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.message.content if isinstance(chunk.message.content, str) else "",
                    chunk=chunk,
                )
            yield chunk

    # ── Payload build ────────────────────────────────────────────────

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            # Inline cost — only meaningful for OpenRouter. Harmless on
            # other OpenAI-compatible endpoints (they'd ignore it).
            "usage": {"include": True},
        }
        if stream:
            payload["stream"] = True
            # OpenRouter only emits the usage block in the final chunk
            # when explicitly asked for during streaming.
            payload["stream_options"] = {"include_usage": True}
        if self.fallback_models:
            # OpenRouter fallback queue. Per the docs, primary lives in
            # `model` and `models` is the ordered fallback list.
            payload["models"] = list(self.fallback_models)
        # OpenRouter provider routing — only emit the block when at
        # least one preference is configured. Sending an empty
        # `provider: {}` would still register as caller intent and
        # could subtly change routing behavior on OpenRouter's side.
        provider: dict[str, Any] = {}
        if self.provider_order:
            provider["order"] = list(self.provider_order)
        if self.provider_allow_fallbacks is not None:
            provider["allow_fallbacks"] = self.provider_allow_fallbacks
        if provider:
            payload["provider"] = provider
        if stop:
            payload["stop"] = stop
        if "tools" in kwargs and kwargs["tools"]:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        return payload

    # ── HTTP — non-streaming ─────────────────────────────────────────

    @property
    def _url(self) -> str:
        return f"{self.base_url.rstrip('/')}{_CHAT_COMPLETIONS_PATH}"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _post_sync(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self._url, headers=self._headers, json=payload)
                last_exc = _check_response_or_get_retry_exc(resp)
                if last_exc is None:
                    return _parse_body_or_raise(resp.json())
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc

            if attempt < self.max_retries:
                time.sleep(_backoff(attempt))

        raise last_exc or RuntimeError("OpenRouter request failed without exception")

    async def _post_async(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self._url, headers=self._headers, json=payload
                    )
                last_exc = _check_response_or_get_retry_exc(resp)
                if last_exc is None:
                    return _parse_body_or_raise(resp.json())
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc

            if attempt < self.max_retries:
                await asyncio.sleep(_backoff(attempt))

        raise last_exc or RuntimeError("OpenRouter request failed without exception")

    # ── HTTP — streaming (SSE) ───────────────────────────────────────

    def _stream_sync(self, payload: dict) -> Iterator[dict]:
        # No retry on streaming — by the time the connection is open and
        # tokens are flowing, a transient blip is the user's problem to
        # see (LangChain emits the partial as a chunk-level error).
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST", self._url, headers=self._headers, json=payload
            ) as resp:
                _raise_for_non_ok_stream(resp)
                for line in resp.iter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    if chunk is _SSE_DONE:
                        return
                    yield chunk

    async def _stream_async(self, payload: dict) -> AsyncIterator[dict]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self._url, headers=self._headers, json=payload
            ) as resp:
                await _araise_for_non_ok_stream(resp)
                async for line in resp.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    if chunk is _SSE_DONE:
                        return
                    yield chunk


# ── Sentinels ──────────────────────────────────────────────────────────


# Singleton sentinel returned by `_parse_sse_line` for the `[DONE]`
# marker. We compare with `is` so it can't collide with any real chunk.
_SSE_DONE = object()


# ── Helpers ────────────────────────────────────────────────────────────


def _backoff(attempt: int) -> float:
    """Exponential backoff between retries: 0.5s, 1s, 2s..."""
    return 0.5 * (2**attempt)


def _check_response_or_get_retry_exc(resp: httpx.Response) -> Exception | None:
    """Return None if response is 2xx (caller should parse body).
    Return a retryable exception for 5xx. Raise immediately for 4xx —
    client errors never get a retry pass."""
    if resp.status_code < 400:
        return None
    body_preview = resp.text[:500]
    if resp.status_code >= 500:
        return httpx.HTTPStatusError(
            f"OpenRouter HTTP {resp.status_code}: {body_preview}",
            request=resp.request,
            response=resp,
        )
    raise httpx.HTTPStatusError(
        f"OpenRouter HTTP {resp.status_code}: {body_preview}",
        request=resp.request,
        response=resp,
    )


def _parse_body_or_raise(body: dict) -> dict:
    """OpenRouter sometimes returns 200 with an `{"error": {...}}` body
    instead of choices — surface as an exception so callers don't try
    to read None and crash deep in the pipeline."""
    err = body.get("error") if isinstance(body, dict) else None
    if err:
        raise RuntimeError(f"OpenRouter error in 200 body: {err}")
    return body


def _raise_for_non_ok_stream(resp: httpx.Response) -> None:
    """Sync version. Read the body once to surface a useful message."""
    if resp.status_code < 400:
        return
    resp.read()
    raise httpx.HTTPStatusError(
        f"OpenRouter HTTP {resp.status_code} (stream): {resp.text[:500]}",
        request=resp.request,
        response=resp,
    )


async def _araise_for_non_ok_stream(resp: httpx.Response) -> None:
    """Async version."""
    if resp.status_code < 400:
        return
    await resp.aread()
    raise httpx.HTTPStatusError(
        f"OpenRouter HTTP {resp.status_code} (stream): {resp.text[:500]}",
        request=resp.request,
        response=resp,
    )


def _parse_sse_line(line: str) -> dict | object | None:
    """Parse one SSE line into a chunk dict, `_SSE_DONE`, or None.

    Returns:
        - `None` for empty lines, comments, or non-`data:` lines.
        - `_SSE_DONE` for the terminating `[DONE]` token.
        - `dict` for a real chunk (parsed JSON)."""
    if not line:
        return None
    if not line.startswith(_SSE_DATA_PREFIX):
        return None
    payload = line[len(_SSE_DATA_PREFIX):].strip()
    if not payload:
        return None
    if payload == _SSE_DONE_TOKEN:
        return _SSE_DONE
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Malformed chunk — skip rather than break the whole stream.
        return None


def _message_to_openai(m: BaseMessage) -> dict:
    """LangChain message → OpenAI chat-completions message dict.

    Content is forwarded VERBATIM. When the caller built a list of
    structured blocks (`cache_control`, `image_url`, `input_audio`), we
    pass them through untouched — OpenRouter accepts the OpenAI shape
    one-for-one for those features."""
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content}
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": m.content}
    if isinstance(m, AIMessage):
        out: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                }
                for tc in m.tool_calls
            ]
        return out
    if isinstance(m, ToolMessage):
        return {
            "role": "tool",
            "content": m.content,
            "tool_call_id": m.tool_call_id,
        }
    role = getattr(m, "role", "user")
    return {"role": role, "content": getattr(m, "content", "")}


def _parse_tool_calls(raw: list[dict]) -> list[dict]:
    """OpenAI-style tool_calls → LangChain `ToolCall` dicts.

    A LangChain `AIMessage.tool_calls` entry is just a dict with `name`,
    `args` (parsed), `id`, `type='tool_call'`. We tolerate malformed JSON
    arguments by emitting an empty dict — surfaces the issue in the
    downstream tool node rather than crashing the whole turn here."""
    parsed: list[dict] = []
    for tc in raw:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        parsed.append(
            {
                "name": fn.get("name", ""),
                "args": args,
                "id": tc.get("id") or "",
                "type": "tool_call",
            }
        )
    return parsed


def _body_to_chat_result(body: dict, requested_model: str) -> ChatResult:
    """Parse a non-streaming `/chat/completions` body into ChatResult.

    `requested_model` is used as the fallback when OpenRouter doesn't
    echo the model name back (rare but happens on some routes)."""
    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"OpenRouter returned no choices: {body}")
    choice = choices[0]
    message = choice.get("message") or {}

    content = message.get("content") or ""
    tool_calls = _parse_tool_calls(message.get("tool_calls") or [])

    usage = body.get("usage") or {}
    generation_id = body.get("id") or ""
    # `model` in the response body is the model that actually answered —
    # when OpenRouter falls back, this differs from the primary we
    # requested. BillingCallback charges the actual model.
    model_name = body.get("model") or requested_model
    finish_reason = choice.get("finish_reason")

    ai_message = AIMessage(
        content=content,
        tool_calls=tool_calls,
        response_metadata={
            "id": generation_id,
            "model_name": model_name,
            "finish_reason": finish_reason,
            "token_usage": usage,
        },
        usage_metadata={
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
    )

    generation = ChatGeneration(
        message=ai_message,
        generation_info={
            "id": generation_id,
            "finish_reason": finish_reason,
            "usage": usage,
        },
    )
    llm_output = {
        "id": generation_id,
        "model_name": model_name,
        "token_usage": usage,
    }
    return ChatResult(generations=[generation], llm_output=llm_output)


def _raw_chunk_to_generation_chunk(
    raw: dict,
    requested_model: str,
) -> ChatGenerationChunk | None:
    """Convert one OpenAI-style streaming chunk → `ChatGenerationChunk`.

    Returns None if the chunk is purely structural (no choices AND no
    usage block), so callers can simply skip it."""
    choices = raw.get("choices") or []
    usage = raw.get("usage")
    generation_id = raw.get("id")
    model_name = raw.get("model")

    if not choices and not usage:
        return None

    delta = (choices[0].get("delta") if choices else {}) or {}
    finish_reason = choices[0].get("finish_reason") if choices else None
    content_delta = delta.get("content") or ""
    role = delta.get("role")  # only present on the first chunk usually

    # Tool-call deltas arrive incrementally; each carries an `index`
    # plus *partial* `function.arguments` JSON. Forward as
    # `tool_call_chunks` and let LangChain's AIMessageChunk merger glue
    # them together when the consumer aggregates chunks.
    tc_chunks: list[dict] = []
    for tc in delta.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tc_chunks.append(
            {
                "name": fn.get("name"),
                "args": fn.get("arguments"),
                "id": tc.get("id"),
                "index": tc.get("index"),
            }
        )

    # `usage_metadata` only ships on the terminal chunk (when
    # `stream_options.include_usage=True`). Build it only when present
    # so partial accumulation doesn't confuse downstream readers.
    usage_metadata: dict[str, int] | None = None
    if usage:
        usage_metadata = {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    # `model_name` and `id` arrive on EVERY chunk. LangChain's chunk
    # merger string-concatenates duplicated metadata values, so emitting
    # them per-chunk yields garbage like "modelmodelmodel…". Restrict
    # the whole metadata bundle to the chunk that carries the `usage`
    # block — OpenRouter emits exactly one such chunk per stream when
    # `stream_options.include_usage=True`, so the aggregated message
    # ends up with exactly one copy of each field. The earlier
    # finish_reason-only chunk (some routes emit one) is treated as
    # plain content so the merger doesn't double-count.
    is_terminal = bool(usage)
    response_metadata: dict[str, Any] = {}
    if is_terminal:
        if generation_id:
            response_metadata["id"] = generation_id
        if model_name:
            response_metadata["model_name"] = model_name
        if finish_reason:
            response_metadata["finish_reason"] = finish_reason
        if usage:
            # Carry the full upstream usage dict (with `cost`) so the
            # terminal chunk surfaces the same billing fields the
            # non-streaming path puts in response_metadata.token_usage.
            response_metadata["token_usage"] = usage

    chunk_message = AIMessageChunk(
        content=content_delta,
        tool_call_chunks=tc_chunks,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        additional_kwargs={"role": role} if role else {},
    )
    # Same reasoning as response_metadata above — limit `generation_info`
    # to the terminal chunk so BillingCallback's `_drill_for_meta` sees
    # the right id/usage exactly once when LangChain aggregates.
    generation_info: dict[str, Any] | None = None
    if is_terminal:
        generation_info = {}
        if generation_id:
            generation_info["id"] = generation_id
        if finish_reason:
            generation_info["finish_reason"] = finish_reason
        if usage:
            generation_info["usage"] = usage

    return ChatGenerationChunk(
        message=chunk_message,
        generation_info=generation_info,
    )
