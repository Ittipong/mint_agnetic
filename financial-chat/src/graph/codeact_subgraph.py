"""Dedicated CodeAct subgraph — wraps FinancialCodeActAgent as a LangGraph node.

Studio sees this as a distinct 'codeact' node (not a hidden tool execution).
ReAct routes here when it decides to analyze financial data.

Per-step LangSmith tracing (code generation + execution) happens inside
financial_agent/loop.py so each step is visible as a child span in real time.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

# Resolve financial-agent package path
_FA_PATH = Path(__file__).parent.parent.parent.parent.parent / "financial-agent"
if str(_FA_PATH) not in sys.path:
    sys.path.insert(0, str(_FA_PATH))

from financial_agent.agent import FinancialCodeActAgent  # noqa: E402
from financial_agent.config import Settings as FinancialSettings  # noqa: E402
from src.config import settings as _chat_settings  # noqa: E402
from src.graph.state import AgentState  # noqa: E402

# ── Debug Logger Setup ─────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_DEBUG_LOG = _LOG_DIR / f"codeact_debug_{datetime.now().strftime('%Y-%m-%d')}.log"

def _debug_log(tag: str, msg: str, **kwargs):
    """Write structured debug log to file."""
    parts = [f"[{datetime.now().isoformat()}] [{tag}] {msg}"]
    for k, v in kwargs.items():
        parts.append(f" {k}={v}")
    log_line = "".join(parts) + "\n"
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(log_line)
    logging.info(log_line.strip())

# Tool name that triggers routing to this subgraph
CODEACT_TOOL_NAME = "analyze_user_finances"


# ── CodeAct subgraph state ───────────────────────────────────────────────────

class CodeActSubState(TypedDict):
    """Internal state for the CodeAct subgraph — isolated from parent AgentState."""

    task: str
    user_id: str
    result: str  # plain-text result written by _codeact_run, read by the bridge
    total_steps: int  # number of CodeAct loop iterations
    loop_status: str  # completed/partial/error
    steps_detail: list  # detailed step info for evaluation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_financial_settings() -> FinancialSettings:
    """Build FinancialSettings from the chat agent config."""
    db_url = _chat_settings.backend_database_url
    if not db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return FinancialSettings(
        openrouter_api_key=_chat_settings.openrouter_api_key,
        database_url=db_url,
        openrouter_base_url=_chat_settings.codeact_base_url,
        model=_chat_settings.codeact_model,
        complex_model=_chat_settings.codeact_model,
    )


# ── Subgraph node ────────────────────────────────────────────────────────────

async def _codeact_run(state: CodeActSubState) -> dict:
    """Run the full CodeAct agent loop and return detailed result.

    Returns dict with:
    - result: plain text for ToolMessage
    - total_steps: number of loop iterations
    - loop_status: completed/partial/error
    - steps_detail: list of step info for evaluation

    Real-time per-step tracing (code + execution) is handled by
    financial_agent/loop.py using langsmith.trace context managers.
    """
    _debug_log("CODEACT", "_codeact_run called", user_id=state.get("user_id", "MISSING"), task=state.get("task", "")[:50])
    fin_settings = _make_financial_settings()
    async with FinancialCodeActAgent(settings=fin_settings) as agent:
        loop_result = await agent.solve(state["task"], user_id=state["user_id"])
        tool_result = loop_result.to_tool_result()

    # Build step details for evaluation
    steps_detail = []
    for s in loop_result.steps:
        steps_detail.append({
            "step": s.step,
            "code": s.code,
            "elapsed_ms": s.elapsed_ms,
            "status": s.execution.status,
            "result": str(s.execution.result) if s.execution.result is not None else None,
            "error": s.execution.error,
        })

    return {
        "result": tool_result.to_tool_content(),
        "loop_status": loop_result.status,
        "total_steps": loop_result.total_steps,
        "steps_detail": steps_detail,
    }


# ── Compiled subgraph ────────────────────────────────────────────────────────

_codeact_builder = StateGraph(CodeActSubState)
_codeact_builder.add_node("run", _codeact_run)
_codeact_builder.add_edge(START, "run")
_codeact_builder.add_edge("run", END)

# This compiled subgraph is what Studio shows as a drill-down node
codeact_subgraph = _codeact_builder.compile()


# ── Bridge: AgentState → subgraph → ToolMessage ──────────────────────────────

async def act_node(state: AgentState) -> dict:
    """Bridge node in the parent graph - the Act phase of ReAct.

    Extracts the tool call from Reasoner's last message, invokes the CodeAct subgraph,
    and returns a ToolMessage so ReAct can continue its reasoning loop.

    Also embeds step info (total_steps, steps_detail) in the ToolMessage content
    as a JSON block for evaluation purposes.
    """
    import json

    _debug_log("ACT", "act_node called", state_keys=list(state.keys()), user_id=state.get("user_id", "MISSING"))
    # Log full state for debugging intermittent issues
    _debug_log("ACT", "FULL_STATE", full_state=str(dict(state)))

    last_msg = state["messages"][-1]

    # Extract the analyze_user_finances tool call placed by ReAct
    tool_call = next(
        tc for tc in last_msg.tool_calls
        if tc["name"] == CODEACT_TOOL_NAME
    )

    # Validate user_id - reject if missing or empty
    user_id = state.get("user_id", "")
    _debug_log("ACT", "user_id extracted", user_id=user_id or "EMPTY/MISSING")

    if not user_id:
        error_msg = "User ID is required but not provided. Cannot access financial data."
        # Log full state for debugging intermittent issues
        _debug_log("ACT", "ERROR - user_id empty", user_id="EMPTY", full_state=str(dict(state)))
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {error_msg}",
                    tool_call_id=tool_call["id"],
                    name=CODEACT_TOOL_NAME,
                )
            ]
        }

    _debug_log("ACT", "Calling codeact_subgraph", user_id=user_id, task=tool_call["args"]["task"][:50])

    sub_result = await codeact_subgraph.ainvoke({
        "task": tool_call["args"]["task"],
        "user_id": user_id,
        "result": "",
    })

    # Build step info JSON for evaluation
    step_info = {
        "codeact_total_steps": sub_result.get("total_steps", 0),
        "codeact_steps": sub_result.get("steps_detail", []),
        "codeact_status": sub_result.get("loop_status", "unknown"),
    }

    # ToolMessage content = result + step info as special marker
    # Use unique delimiters that won't conflict with JSON in result
    step_info_json = json.dumps(step_info)
    content = sub_result["result"] + f"\n\n__CODEACT_INFO_START__{step_info_json}__CODEACT_INFO_END__"

    _debug_log("ACT", "Returning ToolMessage", user_id=user_id, result_preview=sub_result["result"][:100])

    # Wrap result as ToolMessage — ReAct sees this as the tool's response
    return {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call["id"],
                name=CODEACT_TOOL_NAME,
            )
        ]
    }
