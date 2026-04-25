from __future__ import annotations
import argparse
import asyncio
import json
import sys
from datetime import date
from decimal import Decimal
from typing import Any

from loguru import logger

from financial_agent.config import Settings, get_settings
from financial_agent.db.connection import create_engine, get_session_factory, dispose_engine
from financial_agent.executor import Executor
from financial_agent.loop import CodeActLoop, LoopResult, ToolResult
from financial_agent.reasoner import Reasoner
from financial_agent.tools import build_namespace

_SOLVE_TIMEOUT_EXCEEDED = "solve_timeout_exceeded"


_DB_SCHEMA_SUMMARY = """\
Tables (key columns only):
- transactions: amount(float→Decimal!), type(income|expense|transfer), wallet_sync_id(text), category_sync_id(text), effect_on_wallet(+1/-1/0), date, is_deleted
- general_wallets: sync_id(uuid), initial_balance(NUMERIC15,2), cached_balance, user_id
- creditcard_wallets: sync_id, credit_limit(NUMERIC15,2), initial_used, cached_used_amount, billing_cycle_day, payment_due_day
- goal_wallets: sync_id, target_amount(NUMERIC15,2), cached_balance, target_date, is_closed, is_deleted
- obligation_wallets: sync_id, status(active|closed), obligation_mode, monthly_payment(NUMERIC15,2), due_day
- obligation_plan_versions: obligation_id(FK), principal(NUMERIC15,2), outstanding_principal, annual_rate(NUMERIC6,2 = % e.g.5.50), term_months, effective_to(NULL=latest)
- obligation_transactions: obligation_id(FK), event_type, amount(NUMERIC15,2), principal_paid, interest_paid, remaining_balance
- budgets: amount(NUMERIC15,2), spent_amount, period, start_date, end_date, is_deleted
- categories: sync_id(text), type(income|expense), wallet_sync_id(text), is_deleted
FK: transactions.wallet_sync_id → general_wallets.sync_id::text (not UUID join!)
"""


class _DecimalEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


class FinancialCodeActAgent:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._engine = None
        self._session_factory = None
        self._loop: CodeActLoop | None = None
        self._reasoner: Reasoner | None = None

    async def __aenter__(self) -> "FinancialCodeActAgent":
        log_level = "DEBUG" if self._settings.debug else self._settings.log_level
        logger.configure(handlers=[{"sink": sys.stderr, "level": log_level}])
        self._engine = create_engine(self._settings.database_url)
        self._session_factory = get_session_factory(self._engine)

        self._reasoner = Reasoner(self._settings)
        executor = Executor(timeout=self._settings.executor_timeout)
        self._loop = CodeActLoop(self._reasoner, executor, self._settings.max_steps, debug=self._settings.debug)

        logger.info("FinancialCodeActAgent ready")
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._engine:
            await dispose_engine(self._engine)

    def _build_system_prompt(self) -> str:
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        jinja = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "prompts")),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        tmpl = jinja.get_template("system_prompt.j2")
        return tmpl.render(
            db_schema_summary=_DB_SCHEMA_SUMMARY,
            today=date.today().isoformat(),
        )

    async def solve(self, task: str, user_id: str, complex: bool = False) -> LoopResult:
        if self._loop is None or self._reasoner is None:
            raise RuntimeError("Use agent as async context manager: async with FinancialCodeActAgent() as agent:")

        self._reasoner.set_system_prompt(self._build_system_prompt())

        namespace = build_namespace(self._session_factory, user_id)
        try:
            return await asyncio.wait_for(
                self._loop.run(task, namespace, complex=complex),
                timeout=self._settings.solve_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("solve() timed out after {}s for task={!r}", self._settings.solve_timeout, task)
            return LoopResult(
                status="error",
                metadata={"error": _SOLVE_TIMEOUT_EXCEEDED, "timeout_seconds": self._settings.solve_timeout},
            )

    async def solve_as_tool(self, task: str, user_id: str, complex: bool = False) -> ToolResult:
        """Run task and return a ToolResult ready for LangGraph ReAct tool_calls."""
        loop_result = await self.solve(task, user_id=user_id, complex=complex)
        return loop_result.to_tool_result()

    def result_to_json(self, result: LoopResult) -> str:
        data: dict = {
            "status": result.status,
            "result": result.result,
            "breakdown": result.breakdown,
            "metadata": result.metadata,
            "total_steps": result.total_steps,
        }
        if self._settings.debug:
            data["steps"] = [
                {
                    "step": s.step,
                    "elapsed_ms": s.elapsed_ms,
                    "code": s.code,
                    "raw_llm_response": s.raw_llm_response,
                    "execution": {
                        "status": s.execution.status,
                        "result": s.execution.result,
                        "error": s.execution.error,
                        "stdout": s.execution.stdout,
                        "stderr": s.execution.stderr,
                        "breakdown": s.execution.breakdown,
                        "metadata": s.execution.metadata,
                    },
                }
                for s in result.steps
            ]
        return json.dumps(data, indent=2, cls=_DecimalEncoder)


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Financial CodeAct Agent CLI")
    parser.add_argument("task", nargs="+", help="Task in natural language")
    parser.add_argument("--user-id", required=True, help="User ID (UUID)")
    parser.add_argument("--complex", action="store_true", help="Enable extended thinking")
    parser.add_argument("--dev", action="store_true", help="Dev mode: DEBUG logging + full step trace in JSON output")
    args = parser.parse_args()

    async def _run() -> None:
        settings = get_settings()
        if args.dev:
            settings.debug = True
        async with FinancialCodeActAgent(settings=settings) as agent:
            result = await agent.solve(" ".join(args.task), user_id=args.user_id, complex=args.complex)
            print(agent.result_to_json(result))

    asyncio.run(_run())
