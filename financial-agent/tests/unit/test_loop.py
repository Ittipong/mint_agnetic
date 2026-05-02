"""Unit tests for CodeActLoop, LoopResult, and ToolResult."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from financial_agent.executor import ExecutionResult
from financial_agent.loop import CodeActLoop, LoopResult, ToolResult
from financial_agent.reasoner import ReasonerResult


# --- helpers ---

def _reasoner(code: str = "# code") -> MagicMock:
    r = MagicMock()
    r.generate_code = AsyncMock(return_value=ReasonerResult(code=code, raw=code))
    return r


def _executor(results: list[ExecutionResult]) -> MagicMock:
    e = MagicMock()
    e.run = AsyncMock(side_effect=results)
    return e


def _completed(result="42000", breakdown=None, metadata=None) -> ExecutionResult:
    return ExecutionResult(
        status="completed",
        result=result,
        breakdown=breakdown or [],
        metadata=metadata or {},
    )


def _error(msg="some error") -> ExecutionResult:
    return ExecutionResult(status="error", error=msg)


def _partial(result="partial") -> ExecutionResult:
    return ExecutionResult(status="partial", result=result)


# --- CodeActLoop ---

async def test_loop_completes_on_first_step():
    loop = CodeActLoop(_reasoner(), _executor([_completed("100")]), max_steps=5)
    result = await loop.run("task", {})
    assert result.status == "completed"
    assert result.result == "100"
    assert result.total_steps == 1


async def test_loop_retries_after_error_then_completes():
    loop = CodeActLoop(_reasoner(), _executor([_error(), _completed("200")]), max_steps=5)
    result = await loop.run("task", {})
    assert result.status == "completed"
    assert result.total_steps == 2


async def test_loop_error_on_last_step():
    loop = CodeActLoop(_reasoner(), _executor([_error()] * 3), max_steps=3)
    result = await loop.run("task", {})
    assert result.status == "error"
    assert result.total_steps == 3


async def test_loop_partial_returns_immediately():
    # partial is terminal — loop exits on first partial regardless of max_steps
    loop = CodeActLoop(_reasoner(), _executor([_partial()]), max_steps=5)
    result = await loop.run("task", {})
    assert result.status == "partial"
    assert result.total_steps == 1


async def test_loop_max_steps_exceeded():
    inprogress = ExecutionResult(status="inprogress")
    loop = CodeActLoop(_reasoner(), _executor([inprogress] * 2), max_steps=2)
    result = await loop.run("task", {})
    assert result.status == "error"
    assert result.metadata["error"] == "max_steps_exceeded"


async def test_loop_passes_history_to_reasoner():
    # Capture snapshots at call-time — mock holds a reference so the list mutates after the call
    call_histories: list[list] = []

    async def _capture(task, history, step, complex=False):
        call_histories.append(list(history))
        return ReasonerResult(code="# code", raw="# code")

    reasoner = MagicMock()
    reasoner.generate_code = _capture
    loop = CodeActLoop(reasoner, _executor([_error(), _completed()]), max_steps=5)
    await loop.run("task", {})

    assert len(call_histories[0]) == 0        # first call: empty history
    assert len(call_histories[1]) == 1        # second call: step 1's error
    assert call_histories[1][0]["step"] == 1


async def test_loop_passes_namespace_to_executor():
    executor = _executor([_completed()])
    loop = CodeActLoop(_reasoner(), executor, max_steps=5)
    ns = {"key": "value"}
    await loop.run("task", ns)
    assert executor.run.call_args[0][1] is ns


# --- LoopResult.to_tool_result() ---

def test_to_tool_result_completed_full():
    lr = LoopResult(
        status="completed",
        result="50000",
        breakdown=[{"label": "Food", "value": "10000"}, {"label": "Travel", "value": "40000"}],
        metadata={"confidence": "high", "data_range": "2026-04", "caveat": "estimate"},
    )
    tr = lr.to_tool_result()
    assert tr.status == "completed"
    assert tr.result == "50000"
    assert len(tr.breakdown) == 2
    assert tr.breakdown[0].label == "Food"
    assert tr.breakdown[0].value == "10000"
    assert tr.confidence == "high"
    assert tr.data_range == "2026-04"
    assert tr.caveat == "estimate"


def test_to_tool_result_error_with_metadata():
    lr = LoopResult(status="error", metadata={"error": "db_timeout"})
    tr = lr.to_tool_result()
    assert tr.status == "error"
    assert tr.error == "db_timeout"
    assert tr.result is None


def test_to_tool_result_error_no_metadata():
    lr = LoopResult(status="error")
    tr = lr.to_tool_result()
    assert tr.error == "unknown error"


def test_to_tool_result_partial():
    lr = LoopResult(status="partial", result="partial_data")
    tr = lr.to_tool_result()
    assert tr.status == "partial"
    assert tr.result == "partial_data"


def test_to_tool_result_result_none():
    lr = LoopResult(status="completed", result=None)
    tr = lr.to_tool_result()
    assert tr.result is None


def test_to_tool_result_result_coerced_to_str():
    lr = LoopResult(status="completed", result=12345)
    tr = lr.to_tool_result()
    assert tr.result == "12345"


# --- ToolResult.to_tool_content() ---

def test_tool_content_error():
    content = ToolResult(status="error", error="timeout").to_tool_content()
    assert "status: error" in content
    assert "error: timeout" in content


def test_tool_content_completed_minimal():
    content = ToolResult(status="completed", result="12345").to_tool_content()
    assert "status: completed" in content
    assert "result: 12345 THB" in content


def test_tool_content_completed_full():
    tr = ToolResult(
        status="completed",
        result="50000",
        breakdown=[{"label": "Food", "value": "10000"}, {"label": "Travel", "value": "40000"}],
        confidence="high",
        data_range="2026-04",
        caveat="estimate only",
    )
    content = tr.to_tool_content()
    assert "confidence: high" in content
    assert "data_range: 2026-04" in content
    assert "breakdown:" in content
    assert "Food: 10000 THB" in content
    assert "Travel: 40000 THB" in content
    assert "caveat: estimate only" in content
    assert "note:" not in content


def test_tool_content_partial_includes_note():
    content = ToolResult(status="partial", result="999").to_tool_content()
    assert "note: result may be incomplete" in content


def test_tool_content_no_result_field_when_none():
    content = ToolResult(status="completed").to_tool_content()
    assert "result:" not in content


def test_tool_result_to_dict_completed():
    tr = ToolResult(
        status="completed",
        result="5000",
        breakdown=[{"label": "A", "value": "3000"}, {"label": "B", "value": "2000"}],
        confidence="medium",
    )
    d = tr.to_dict()
    assert d["status"] == "completed"
    assert d["result"] == "5000"
    assert d["currency"] == "THB"
    assert len(d["breakdown"]) == 2
    assert d["confidence"] == "medium"
    assert "data_range" not in d


def test_tool_result_to_dict_error():
    d = ToolResult(status="error", error="failed").to_dict()
    assert d["status"] == "error"
    assert d["error"] == "failed"
    assert "result" not in d
