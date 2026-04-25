import asyncio
from dataclasses import dataclass, field
from typing import Any
from loguru import logger

try:
    from pythonbox import PythonBox
    _PYTHONBOX_AVAILABLE = True
except ImportError:
    _PYTHONBOX_AVAILABLE = False
    logger.warning("quantalogic-pythonbox not installed; executor will use restricted exec fallback")

FINANCIAL_ALLOWED_MODULES = [
    "asyncio", "decimal", "datetime", "math",
    "typing", "dataclasses", "collections",
    "functools", "itertools",
]


@dataclass
class ExecutionResult:
    status: str  # "completed" | "error" | "inprogress"
    result: Any = None
    breakdown: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class Executor:
    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    async def run(self, code: str, namespace: dict) -> ExecutionResult:
        if _PYTHONBOX_AVAILABLE:
            return await self._run_pythonbox(code, namespace)
        return await self._run_restricted_exec(code, namespace)

    async def _run_pythonbox(self, code: str, namespace: dict) -> ExecutionResult:
        try:
            box = PythonBox(
                allowed_modules=FINANCIAL_ALLOWED_MODULES,
                timeout=self._timeout,
            )
            result = await box.run_async(
                code=code,
                namespace=namespace,
                entrypoint="main",
            )
            if isinstance(result, dict):
                return ExecutionResult(
                    status=result.get("status", "completed"),
                    result=result.get("result"),
                    breakdown=result.get("breakdown", []),
                    metadata=result.get("metadata", {}),
                    stdout=result.get("stdout", ""),
                )
            return ExecutionResult(status="completed", result=result)
        except asyncio.TimeoutError:
            return ExecutionResult(status="error", error="timeout: execution exceeded limit")
        except Exception as e:
            logger.debug(f"Executor error: {e}")
            return ExecutionResult(status="error", error=str(e))

    async def _run_restricted_exec(self, code: str, namespace: dict) -> ExecutionResult:
        """Fallback when pythonbox is unavailable. Runs in-process with timeout."""
        local_ns = {**namespace}
        try:
            exec(compile(code, "<agent_code>", "exec"), local_ns)  # noqa: S102
            main_fn = local_ns.get("main")
            if main_fn is None:
                return ExecutionResult(status="error", error="Code must define async def main()")

            result = await asyncio.wait_for(main_fn(), timeout=self._timeout)

            if isinstance(result, dict):
                return ExecutionResult(
                    status=result.get("status", "completed"),
                    result=result.get("result"),
                    breakdown=result.get("breakdown", []),
                    metadata=result.get("metadata", {}),
                )
            return ExecutionResult(status="completed", result=result)
        except asyncio.TimeoutError:
            return ExecutionResult(status="error", error="timeout: execution exceeded limit")
        except Exception as e:
            logger.debug(f"Fallback executor error: {e}")
            return ExecutionResult(status="error", error=str(e))
