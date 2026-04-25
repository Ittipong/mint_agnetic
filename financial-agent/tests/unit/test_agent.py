"""Unit tests for FinancialCodeActAgent — mocked, no real DB or LLM."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from financial_agent.agent import FinancialCodeActAgent, _SOLVE_TIMEOUT_EXCEEDED
from financial_agent.config import Settings
from financial_agent.loop import LoopResult


def _settings(**overrides) -> Settings:
    base = dict(
        openrouter_api_key="test-key",
        database_url="postgresql+asyncpg://x:x@localhost/x",
        solve_timeout=5,
        max_steps=3,
    )
    base.update(overrides)
    return Settings(**base)


_PATCHES = [
    patch("financial_agent.agent.create_engine"),
    patch("financial_agent.agent.get_session_factory"),
    patch("financial_agent.agent.dispose_engine", new_callable=AsyncMock),
    patch("financial_agent.agent.Reasoner"),
    patch("financial_agent.agent.Executor"),
    patch("financial_agent.agent.CodeActLoop"),
    patch("financial_agent.agent.build_namespace", return_value={}),
]


def _apply_patches(patches):
    """Context manager that applies a list of patches and returns their mocks."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _ctx():
        with contextlib.ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            yield mocks

    return _ctx()


# --- constructor ---

def test_constructor_does_not_accept_user_id():
    import inspect
    sig = inspect.signature(FinancialCodeActAgent.__init__)
    assert "user_id" not in sig.parameters


def test_constructor_stores_settings():
    s = _settings()
    agent = FinancialCodeActAgent(settings=s)
    assert agent._settings is s


def test_constructor_uses_get_settings_when_none():
    with patch("financial_agent.agent.get_settings") as mock_get:
        mock_get.return_value = _settings()
        agent = FinancialCodeActAgent()
    mock_get.assert_called_once()


# --- context manager not entered ---

async def test_solve_outside_context_manager_raises():
    agent = FinancialCodeActAgent()
    with pytest.raises(RuntimeError, match="async context manager"):
        await agent.solve("task", user_id="user-1")


async def test_solve_as_tool_outside_context_manager_raises():
    agent = FinancialCodeActAgent()
    with pytest.raises(RuntimeError, match="async context manager"):
        await agent.solve_as_tool("task", user_id="user-1")


# --- solve() timeout ---

async def test_solve_timeout_returns_error_loop_result():
    settings = _settings(solve_timeout=1)

    async def _slow(*args, **kwargs):
        await asyncio.sleep(10)

    async with _apply_patches(_PATCHES) as mocks:
        _, _, _, MockReasoner, _, MockLoop, _ = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock()
        MockLoop.return_value.run = _slow

        async with FinancialCodeActAgent(settings=settings) as agent:
            agent._build_system_prompt = MagicMock(return_value="prompt")
            result = await agent.solve("task", user_id="user-1")

    assert result.status == "error"
    assert result.metadata["error"] == _SOLVE_TIMEOUT_EXCEEDED
    assert result.metadata["timeout_seconds"] == 1


# --- system prompt refresh ---

async def test_solve_refreshes_system_prompt_per_call():
    settings = _settings()
    call_count = 0
    prompts_seen: list[str] = []

    def _build():
        nonlocal call_count
        call_count += 1
        return f"prompt-{call_count}"

    completed = LoopResult(status="completed", result="ok")

    async with _apply_patches(_PATCHES) as mocks:
        _, _, _, MockReasoner, _, MockLoop, _ = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock(
            side_effect=lambda p: prompts_seen.append(p)
        )
        MockLoop.return_value.run = AsyncMock(return_value=completed)

        async with FinancialCodeActAgent(settings=settings) as agent:
            agent._build_system_prompt = _build
            await agent.solve("task1", user_id="user-1")
            await agent.solve("task2", user_id="user-2")

    assert call_count == 2
    assert prompts_seen == ["prompt-1", "prompt-2"]


# --- user_id passed to namespace ---

async def test_solve_passes_user_id_to_build_namespace():
    settings = _settings()
    completed = LoopResult(status="completed", result="ok")

    async with _apply_patches(_PATCHES) as mocks:
        _, _, _, MockReasoner, _, MockLoop, mock_build_ns = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock()
        MockLoop.return_value.run = AsyncMock(return_value=completed)
        mock_build_ns.return_value = {}

        async with FinancialCodeActAgent(settings=settings) as agent:
            agent._build_system_prompt = MagicMock(return_value="prompt")
            await agent.solve("task", user_id="abc-123")

    assert mock_build_ns.call_args[0][1] == "abc-123"


async def test_solve_different_user_ids_per_call():
    settings = _settings()
    completed = LoopResult(status="completed", result="ok")

    async with _apply_patches(_PATCHES) as mocks:
        _, _, _, MockReasoner, _, MockLoop, mock_build_ns = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock()
        MockLoop.return_value.run = AsyncMock(return_value=completed)
        mock_build_ns.return_value = {}

        async with FinancialCodeActAgent(settings=settings) as agent:
            agent._build_system_prompt = MagicMock(return_value="prompt")
            await agent.solve("task1", user_id="user-A")
            await agent.solve("task2", user_id="user-B")

    user_ids = [c[0][1] for c in mock_build_ns.call_args_list]
    assert user_ids == ["user-A", "user-B"]


# --- solve_as_tool() ---

async def test_solve_as_tool_returns_tool_result():
    settings = _settings()
    loop_result = LoopResult(
        status="completed",
        result="75000",
        metadata={"confidence": "high"},
    )

    async with _apply_patches(_PATCHES) as mocks:
        _, _, _, MockReasoner, _, MockLoop, _ = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock()
        MockLoop.return_value.run = AsyncMock(return_value=loop_result)

        async with FinancialCodeActAgent(settings=settings) as agent:
            agent._build_system_prompt = MagicMock(return_value="prompt")
            tool_result = await agent.solve_as_tool("task", user_id="user-1")

    assert tool_result.status == "completed"
    assert tool_result.result == "75000"
    assert tool_result.confidence == "high"


async def test_solve_as_tool_timeout_converts_to_error_tool_result():
    settings = _settings(solve_timeout=1)

    async def _slow(*args, **kwargs):
        await asyncio.sleep(10)

    async with _apply_patches(_PATCHES) as mocks:
        _, _, _, MockReasoner, _, MockLoop, _ = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock()
        MockLoop.return_value.run = _slow

        async with FinancialCodeActAgent(settings=settings) as agent:
            agent._build_system_prompt = MagicMock(return_value="prompt")
            tool_result = await agent.solve_as_tool("task", user_id="user-1")

    assert tool_result.status == "error"
    assert _SOLVE_TIMEOUT_EXCEEDED in tool_result.error


# --- __aexit__ disposes engine ---

async def test_aexit_disposes_engine():
    async with _apply_patches(_PATCHES) as mocks:
        _, _, mock_dispose, MockReasoner, _, _, _ = mocks
        MockReasoner.return_value.set_system_prompt = MagicMock()

        async with FinancialCodeActAgent(settings=_settings()):
            pass

    mock_dispose.assert_called_once()
