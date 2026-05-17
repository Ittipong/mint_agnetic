"""Pair benchmark: same model for ReAct + CodeAct end-to-end.

Tests each candidate model in both roles (reason_node + codeact_step_node)
across 5 financial Q&A cases. Captures final reply + tool message + latency.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from langchain_core.messages import HumanMessage  # noqa: E402

from src import llm as llm_module  # noqa: E402
from src.config import settings  # noqa: E402
from src.llm_openrouter import ChatOpenRouterREST  # noqa: E402


USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"

CASES = [
    {"id": 1, "label": "easy_this_month_expense", "question": "เดือนนี้ใช้เงินไปเท่าไหร่"},
    {"id": 2, "label": "medium_top3_april_categories", "question": "เดือนเมษายนใช้หมวดไหนเยอะสุด top 3"},
    {"id": 3, "label": "medium_starbucks_alltime", "question": "จ่ายให้ Starbucks ทั้งหมดเท่าไหร่"},
    {"id": 4, "label": "hard_compare_mar_apr", "question": "เปรียบเทียบรายจ่ายเดือนมีนากับเมษา แยกหมวด"},
    {"id": 5, "label": "hardest_be_year_food_thb", "question": "กุมภาพันธ์ 2569 ใช้เงินกับอาหารทั้งหมดคิดเป็นบาทเท่าไร"},
]

# Same model used for BOTH ReAct (reason_node) AND CodeAct (codeact_step_node).
PAIR_MODELS = [
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4-flash",
]


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


def _swap_to_pair(model: str) -> None:
    """Replace BOTH module-level llm singletons with the same model."""
    llm_module.codeact_llm = _make_llm(model, temperature=0.3)
    llm_module.llm = _make_llm(model, temperature=0.3)
    # Also swap intent_classifier_llm so the whole graph uses one model;
    # if we leave it on whatever was loaded at import, intent classification
    # variability would contaminate the pair test.
    llm_module.intent_classifier_llm = _make_llm(model, temperature=0.0)


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
    for m in reversed(messages or []):
        if getattr(m, "type", None) == "tool":
            return getattr(m, "content", "") or ""
    return ""


async def run_one(graph, model: str, case: dict) -> dict:
    thread_id = f"pair-{model.replace('/', '_')}-{case['id']}"
    t0 = time.monotonic()
    err = None
    final_messages = []
    tool_msg = ""

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
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        "pair_model": model,
        "case_id": case["id"],
        "case_label": case["label"],
        "question": case["question"],
        "elapsed_ms": elapsed_ms,
        "error": err,
        "final_reply": _extract_final_reply(final_messages),
        "tool_message": tool_msg[:4000],
        "n_messages": len(final_messages),
    }


async def main() -> None:
    from src.graph.agent_graph import build_graph  # noqa: E402
    graph = build_graph()

    out_path = _ROOT / "logs" / "pair_bench_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for model in PAIR_MODELS:
        print(f"\n========== PAIR (ReAct + CodeAct) = {model} ==========", flush=True)
        _swap_to_pair(model)
        for case in CASES:
            print(f"  case#{case['id']} {case['label']} ...", flush=True)
            res = await run_one(graph, model, case)
            print(f"    elapsed={res['elapsed_ms']}ms err={res['error']!s:.80s}", flush=True)
            print(f"    tool_msg[:300]={res['tool_message'][:300]!r}", flush=True)
            print(f"    reply[:200]={res['final_reply'][:200]!r}", flush=True)
            all_results.append(res)
            out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    print(f"\nDone — wrote {len(all_results)} runs to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
