"""End-to-end accuracy benchmark for codeact_step_node.

Runs 5 financial questions through the real LangGraph agent for 3
candidate CODEACT models, keeping the ReAct (reason_node) llm pinned
to openai/gpt-5-nano. Captures final reply + codeact_history per run
and dumps everything to JSON for offline scoring.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to sys.path so `src.*` imports work when running from scripts/.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env first so settings pick up OPENROUTER_API_KEY etc.
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from langchain_core.messages import HumanMessage  # noqa: E402

from src import llm as llm_module  # noqa: E402
from src.config import settings  # noqa: E402
from src.llm_openrouter import ChatOpenRouterREST  # noqa: E402


USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"

CASES = [
    {
        "id": 1,
        "label": "easy_this_month_expense",
        "question": "เดือนนี้ใช้เงินไปเท่าไหร่",
    },
    {
        "id": 2,
        "label": "medium_top3_april_categories",
        "question": "เดือนเมษายนใช้หมวดไหนเยอะสุด top 3",
    },
    {
        "id": 3,
        "label": "medium_starbucks_alltime",
        "question": "จ่ายให้ Starbucks ทั้งหมดเท่าไหร่",
    },
    {
        "id": 4,
        "label": "hard_compare_mar_apr",
        "question": "เปรียบเทียบรายจ่ายเดือนมีนากับเมษา แยกหมวด",
    },
    {
        "id": 5,
        "label": "hardest_be_year_food_thb",
        "question": "กุมภาพันธ์ 2569 ใช้เงินกับอาหารทั้งหมดคิดเป็นบาทเท่าไร",
    },
]

CODEACT_MODELS = [
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash",
]

REACT_MODEL = "openai/gpt-5-nano"  # pinned


def _make_llm(model: str, *, temperature: float) -> ChatOpenRouterREST:
    return ChatOpenRouterREST(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        fallback_models=[],
        temperature=temperature,
        timeout=120,
        max_retries=1,
    )


def _swap_llms(codeact_model: str) -> None:
    """Replace module-level singletons so codeact_step_node (lazy import)
    picks up the new model. reason_node also re-imports llm each call.
    """
    llm_module.codeact_llm = _make_llm(codeact_model, temperature=0.3)
    llm_module.llm = _make_llm(REACT_MODEL, temperature=0.3)


def _extract_codeact_history(state: dict) -> list[dict]:
    """Pull codeact_history from the compute_subgraph state if present."""
    history = state.get("codeact_history") or []
    # Strip non-serializable items (Decimal, date) for JSON dump.
    cleaned = []
    for step in history:
        cleaned.append({
            "step": step.get("step"),
            "code": step.get("code", ""),
            "stdout": (step.get("stdout") or "")[:2000],
            "error": step.get("error"),
            "result_preview": _safe_str(step.get("result"))[:2000],
        })
    return cleaned


def _safe_str(value) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return str(value)


def _extract_final_reply(messages: list) -> str:
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "ai" and getattr(m, "content", None):
            content = m.content
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return content
    return ""


def _extract_tool_message(messages: list) -> str:
    """The analyze_user_finances ToolMessage carries the codeact `answer`."""
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "tool":
            return getattr(m, "content", "") or ""
    return ""


async def run_one(graph, model: str, case: dict) -> dict:
    """Run one (model, case) and return a serializable result."""
    thread_id = f"bench-{model.replace('/', '_')}-{case['id']}"
    t0 = time.monotonic()
    err = None
    final_messages = []
    tool_msg = ""
    codeact_history: list[dict] = []
    steps_used = 0

    try:
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=case["question"])],
                "user_id": USER_ID,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        final_messages = result.get("messages", [])
        tool_msg = _extract_tool_message(final_messages)
        # codeact_history lives in the compute subgraph state; surface via tool message
        # by grepping STDOUT or by reading the act_node return — but easier to expose
        # via a side channel. For now, count from the agent state if present.
        codeact_history = []  # graph.ainvoke returns parent state only
        steps_used = result.get("codeact_steps") or 0
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        "model": model,
        "case_id": case["id"],
        "case_label": case["label"],
        "question": case["question"],
        "elapsed_ms": elapsed_ms,
        "error": err,
        "final_reply": _extract_final_reply(final_messages),
        "tool_message": tool_msg[:4000],
        "codeact_steps": steps_used,
        "codeact_history": codeact_history,
    }


async def main() -> None:
    # Pin "today" before graph construction so codeact + reason both see 2026-05-17
    os.environ.setdefault("MM_BENCH_TODAY", "2026-05-17")

    from src.graph.agent_graph import build_graph  # noqa: E402
    graph = build_graph()

    out_path = Path(__file__).parent.parent / "logs" / "codeact_bench_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for model in CODEACT_MODELS:
        print(f"\n========== CODEACT_MODEL = {model} ==========", flush=True)
        _swap_llms(model)
        for case in CASES:
            print(f"  case#{case['id']} {case['label']} ...", flush=True)
            res = await run_one(graph, model, case)
            print(f"    elapsed={res['elapsed_ms']}ms err={res['error']!s:.80s}", flush=True)
            print(f"    tool_msg[:300]={res['tool_message'][:300]!r}", flush=True)
            all_results.append(res)
            # checkpoint after every run
            out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    print(f"\nDone — wrote {len(all_results)} runs to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
