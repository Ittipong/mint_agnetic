import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal
from loguru import logger
from financial_agent.reasoner import Reasoner
from financial_agent.executor import Executor, ExecutionResult

# Optional LangSmith tracing — graceful no-op when langsmith is not installed
try:
    from langsmith import trace as _ls_trace
    _HAS_LANGSMITH = True
except ImportError:
    _HAS_LANGSMITH = False


@contextmanager
def _step_trace(name: str, inputs: dict):
    """Create a LangSmith span if available, otherwise no-op."""
    if _HAS_LANGSMITH:
        with _ls_trace(name=name, run_type="tool", inputs=inputs) as run:
            yield run
    else:
        yield None


def _end_trace(run: Any, outputs: dict) -> None:
    """Finalise a LangSmith span with outputs (no-op when tracing is off)."""
    if run is not None:
        run.end(outputs=outputs)


@dataclass
class StepResult:
    step: int
    code: str
    execution: ExecutionResult
    elapsed_ms: float = 0.0
    raw_llm_response: str = ""


@dataclass
class BreakdownItem:
    label: str
    value: str
    unit: str = ""


@dataclass
class ToolResult:
    """Clean response format for LangGraph ReAct agent tool calls.

    Strips execution internals (steps, total_steps) and normalises
    metadata into flat, consistently-typed fields.
    """

    status: Literal["completed", "partial", "error"]
    result: str | None = None
    currency: str = "THB"
    breakdown: list[BreakdownItem] = field(default_factory=list)
    confidence: str | None = None
    data_range: str | None = None
    caveat: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"status": self.status}
        if self.result is not None:
            d["result"] = self.result
            d["currency"] = self.currency
        if self.breakdown:
            d["breakdown"] = [{"label": b.label, "value": b.value, "unit": b.unit} for b in self.breakdown]
        if self.confidence:
            d["confidence"] = self.confidence
        if self.data_range:
            d["data_range"] = self.data_range
        if self.caveat:
            d["caveat"] = self.caveat
        if self.error:
            d["error"] = self.error
        return d

    def to_tool_content(self) -> str:
        """Plain-text representation suitable for LangGraph ToolMessage.content."""
        if self.status == "error":
            return f"status: error\nerror: {self.error}"

        lines = [f"status: {self.status}"]
        if self.result is not None:
            lines.append(f"result: {self.result}")
        if self.confidence:
            lines.append(f"confidence: {self.confidence}")
        if self.data_range:
            lines.append(f"data_range: {self.data_range}")
        if self.breakdown:
            lines.append("breakdown:")
            for item in self.breakdown:
                unit = item.unit or self.currency
                lines.append(f"  - {item.label}: {item.value} {unit}")
        if self.caveat:
            lines.append(f"caveat: {self.caveat}")
        if self.status == "partial":
            lines.append("note: result may be incomplete due to insufficient data")
        return "\n".join(lines)


@dataclass
class LoopResult:
    status: str
    result: Any = None
    breakdown: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    steps: list[StepResult] = field(default_factory=list)
    total_steps: int = 0

    def to_tool_result(self) -> ToolResult:
        """Convert to ToolResult for LangGraph ReAct agent consumption."""
        if self.status == "error":
            return ToolResult(
                status="error",
                error=str(self.metadata.get("error", "unknown error")),
            )

        breakdown = [
            BreakdownItem(
                label=item.get("label", ""),
                value=str(item.get("value", "")),
                unit=str(item.get("unit", "")),
            )
            for item in self.breakdown
        ]
        return ToolResult(
            status=self.status,  # type: ignore[arg-type]
            result=str(self.result) if self.result is not None else None,
            breakdown=breakdown,
            confidence=self.metadata.get("confidence"),
            data_range=self.metadata.get("data_range"),
            caveat=self.metadata.get("caveat"),
        )


def _emit_step_jsonl(s: StepResult) -> None:
    """Write one JSONL line per step to stderr so `2>steps.jsonl` stays structured."""
    line = json.dumps({
        "type": "step",
        "step": s.step,
        "elapsed_ms": s.elapsed_ms,
        "status": s.execution.status,
        "code": s.code,
        "raw_llm_response": s.raw_llm_response,
        "result": s.execution.result,
        "error": s.execution.error,
        "stdout": s.execution.stdout,
        "stderr": s.execution.stderr,
    }, ensure_ascii=False)
    print(line, file=sys.stderr, flush=True)


class CodeActLoop:
    def __init__(
        self,
        reasoner: Reasoner,
        executor: Executor,
        max_steps: int = 10,
        debug: bool = False,
    ):
        self._reasoner = reasoner
        self._executor = executor
        self._max_steps = max_steps
        self._debug = debug

    async def run(
        self,
        task: str,
        namespace: dict,
        complex: bool = False,
    ) -> LoopResult:
        history: list[dict] = []
        steps: list[StepResult] = []

        for step in range(1, self._max_steps + 1):
            logger.info("CodeAct step {}/{}", step, self._max_steps)

            t0 = time.perf_counter()

            # ── Code generation span ─────────────────────────────────────────
            with _step_trace(
                name="codeact_generate_code",
                inputs={"step": step, "task": task},
            ) as gen_run:
                reasoner_result = await self._reasoner.generate_code(task, history, step, complex)
                _end_trace(gen_run, outputs={"code": reasoner_result.code})

            # ── Code execution span ──────────────────────────────────────────
            with _step_trace(
                name="codeact_execute_code",
                inputs={"step": step, "code": reasoner_result.code},
            ) as exec_run:
                exec_result = await self._executor.run(reasoner_result.code, namespace)
                _end_trace(exec_run, outputs={
                    "status": exec_result.status,
                    "result": str(exec_result.result) if exec_result.result is not None else None,
                    "error": exec_result.error,
                    "stdout": exec_result.stdout or "",
                })

            elapsed_ms = (time.perf_counter() - t0) * 1000

            step_result = StepResult(
                step=step,
                code=reasoner_result.code,
                execution=exec_result,
                elapsed_ms=round(elapsed_ms, 1),
                raw_llm_response=reasoner_result.raw,
            )
            steps.append(step_result)

            logger.debug(
                "Step {} | {:.0f}ms | status={}\n--- code ---\n{}\n--- result: {} | error: {}",
                step, elapsed_ms, exec_result.status,
                reasoner_result.code,
                exec_result.result, exec_result.error,
            )

            if self._debug:
                _emit_step_jsonl(step_result)

            history.append({"step": step, "code": reasoner_result.code, "result": exec_result})

            if exec_result.status == "completed":
                logger.info("Task completed at step {}", step)
                return LoopResult(
                    status="completed",
                    result=exec_result.result,
                    breakdown=exec_result.breakdown,
                    metadata=exec_result.metadata,
                    steps=steps,
                    total_steps=step,
                )

            if exec_result.status == "partial":
                # terminal: agent signals insufficient data, return best-effort now
                logger.info("Task returned partial result at step {}", step)
                return LoopResult(
                    status="partial",
                    result=exec_result.result,
                    breakdown=exec_result.breakdown,
                    metadata=exec_result.metadata,
                    steps=steps,
                    total_steps=step,
                )

            if exec_result.status == "error":
                logger.warning("Step {} error: {}", step, exec_result.error)
                if step == self._max_steps:
                    return LoopResult(
                        status="error",
                        metadata={"error": exec_result.error, "steps": step},
                        steps=steps,
                        total_steps=step,
                    )
                # Continue loop — next step will see the error in history

            else:
                # inprogress or any unexpected status: continue to next step
                logger.debug("Step {} status={} — continuing", step, exec_result.status)

        return LoopResult(
            status="error",
            metadata={"error": "max_steps_exceeded", "steps": self._max_steps},
            steps=steps,
            total_steps=self._max_steps,
        )
