"""Inspect the empty-result runs to find out why the graph terminated silently."""
from __future__ import annotations
import asyncio, sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from langchain_core.messages import HumanMessage  # noqa: E402
from src import llm as llm_module  # noqa: E402
from src.config import settings  # noqa: E402
from src.llm_openrouter import ChatOpenRouterREST  # noqa: E402

def _make(m):
    return ChatOpenRouterREST(model=m, api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1", fallback_models=[], temperature=0.3,
        timeout=120, max_retries=1)

CASES = {
    ("google/gemini-2.5-flash-lite", 1): "เดือนนี้ใช้เงินไปเท่าไหร่",
    ("deepseek/deepseek-v4-flash", 1): "เดือนนี้ใช้เงินไปเท่าไหร่",
    ("openai/gpt-5-nano", 4): "เปรียบเทียบรายจ่ายเดือนมีนากับเมษา แยกหมวด",
    ("deepseek/deepseek-v4-flash", 4): "เปรียบเทียบรายจ่ายเดือนมีนากับเมษา แยกหมวด",
}

USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"

async def main():
    from src.graph.agent_graph import build_graph
    graph = build_graph()
    for (model, cid), q in CASES.items():
        llm_module.codeact_llm = _make(model)
        llm_module.llm = _make(model)
        llm_module.intent_classifier_llm = _make(model)
        result = await graph.ainvoke(
            {"messages":[HumanMessage(content=q)], "user_id": USER_ID},
            config={"configurable":{"thread_id":f"inspect-{model.replace('/','_')}-{cid}"}})
        print(f"\n=== {model} case#{cid} ===")
        print(f"intent: {result.get('intent','?')}")
        print(f"keys: {list(result.keys())}")
        for i, m in enumerate(result.get("messages", [])):
            mtype = getattr(m,'type','?')
            content = getattr(m,'content','')
            tc = getattr(m,'tool_calls', None)
            print(f"  msg[{i}] type={mtype} content={str(content)[:160]!r} tool_calls={tc}")

if __name__ == "__main__":
    asyncio.run(main())
