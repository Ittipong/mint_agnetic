"""A2a streaming OpenRouter adapter unit tests (Wave 7b adaptation).

Lifted from v2 `tests/test_llm_stream.py` — import path adjusted from
`agent.llm_openrouter` to `src.agent.llm_openrouter`. The function body is
unchanged; this is a verbatim lift with v3 paths.

The adapter must:
- yield assistant content deltas in order (concatenation == full message),
- stop at `data: [DONE]`,
- skip deltas with missing/None content and tolerate non-data / malformed lines,
- raise OpenRouterError on an HTTP error,
- set stream=True in the request body.

We mock httpx's streaming client so no network is touched.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.agent.llm_openrouter import OpenRouterError, make_llm_stream


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes for httpx.AsyncClient(...).stream("POST", ...) async-context manager.
# ---------------------------------------------------------------------------
def _sse(delta_content):
    """Render one OpenRouter SSE `data:` line carrying a content delta."""
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": delta_content}}]}
    )


class _FakeStreamResponse:
    def __init__(self, lines, *, raise_status=False):
        self._lines = lines
        self._raise = raise_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._raise:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("POST", "http://x"),
                response=httpx.Response(500),
            )

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln


class _FakeAsyncClient:
    def __init__(self, lines, *, raise_status=False, **kw):
        self._lines = lines
        self._raise = raise_status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, **kw):
        return _FakeStreamResponse(self._lines, raise_status=self._raise)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # _resolve_request needs a key + model; values are irrelevant (no network).
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CODEACT_MODEL", "test/model")


async def _collect(stream_callable, messages):
    return [chunk async for chunk in stream_callable(messages)]


# ---------------------------------------------------------------------------
# UT-STR-101: deltas yielded in order; concatenation == full message
# ---------------------------------------------------------------------------
async def test_UT_STR101_yields_concatenated_deltas_in_order(monkeypatch):
    lines = [
        _sse("ยอด"),
        _sse("คงเหลือ "),
        _sse("1,234.50"),
        _sse(" บาท"),
        "data: [DONE]",
        _sse("SHOULD NOT APPEAR"),  # after [DONE] -> never reached
    ]
    monkeypatch.setattr(
        "src.agent.llm_openrouter.httpx.AsyncClient",
        lambda **kw: _FakeAsyncClient(lines),
    )
    out = await _collect(
        make_llm_stream("codeact"), [{"role": "user", "content": "x"}],
    )
    assert out == ["ยอด", "คงเหลือ ", "1,234.50", " บาท"]
    assert "".join(out) == "ยอดคงเหลือ 1,234.50 บาท"


# ---------------------------------------------------------------------------
# UT-STR-102: missing/None content + non-data + malformed lines are skipped
# ---------------------------------------------------------------------------
async def test_UT_STR102_skips_empty_and_malformed_lines(monkeypatch):
    lines = [
        ": keep-alive comment",                       # non-data -> skip
        json.dumps({"choices": [{"delta": {}}]}),     # not a data: line -> skip
        _sse(None),                                   # content None -> skip
        _sse(""),                                     # content empty -> skip
        "data: {not json",                            # malformed -> skip, not crash
        _sse("A"),
        "data: " + json.dumps({"choices": []}),       # IndexError shape -> skip
        _sse("B"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        "src.agent.llm_openrouter.httpx.AsyncClient",
        lambda **kw: _FakeAsyncClient(lines),
    )
    out = await _collect(
        make_llm_stream("codeact"), [{"role": "user", "content": "x"}],
    )
    assert out == ["A", "B"]


# ---------------------------------------------------------------------------
# UT-STR-103: HTTP error -> OpenRouterError
# ---------------------------------------------------------------------------
async def test_UT_STR103_http_error_raises_openrouter_error(monkeypatch):
    monkeypatch.setattr(
        "src.agent.llm_openrouter.httpx.AsyncClient",
        lambda **kw: _FakeAsyncClient([], raise_status=True),
    )
    with pytest.raises(OpenRouterError):
        await _collect(
            make_llm_stream("codeact"), [{"role": "user", "content": "x"}],
        )


# ---------------------------------------------------------------------------
# UT-STR-104: request body sets stream=True (no accidental non-streaming call)
# ---------------------------------------------------------------------------
async def test_UT_STR104_request_body_has_stream_true(monkeypatch):
    captured = {}

    class _CapturingClient(_FakeAsyncClient):
        def stream(self, method, url, **kw):
            captured["method"] = method
            captured["json"] = kw.get("json")
            return _FakeStreamResponse([_sse("ok"), "data: [DONE]"])

    monkeypatch.setattr(
        "src.agent.llm_openrouter.httpx.AsyncClient",
        lambda **kw: _CapturingClient([]),
    )
    await _collect(
        make_llm_stream("codeact"), [{"role": "user", "content": "hi"}],
    )
    assert captured["method"] == "POST"
    assert captured["json"]["stream"] is True
    assert captured["json"]["model"] == "test/model"
