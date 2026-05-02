#!/usr/bin/env python3
"""Test LangGraph Studio with financial queries.

Usage:
    python test_langgraph.py "สรุปการเงินของฉัน"
    python test_langgraph.py "เดือนนี้ฉันจ่ายเงินไปเท่าไหร่"
    python test_langgraph.py "net worth ของฉันเท่าไหร่"
"""
import argparse
import asyncio
import json
import httpx

STUDIO_URL = "http://127.0.0.1:2024"
GRAPH_ID = "financial_agent"


async def run_query(task: str, user_id: str, thread_id: str = "test-001") -> dict:
    """Send query to LangGraph Studio and stream results."""
    payload = {
        "assistant_id": GRAPH_ID,
        "graph_id": GRAPH_ID,
        "input": {
            "user_id": user_id,
            "messages": [{"role": "user", "content": task}],
        },
        "config": {"configurable": {"thread_id": thread_id}},
        "stream_mode": "values",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{STUDIO_URL}/runs/stream", json=payload) as resp:
            result = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line.removeprefix("event:").strip()
                    continue
                if line.startswith("data:"):
                    data = json.loads(line.removeprefix("data:").strip())
                    if event == "values":
                        result = data
            return result


async def main():
    parser = argparse.ArgumentParser(description="Test LangGraph Studio queries")
    parser.add_argument("task", help="Question in natural language")
    parser.add_argument("--user-id", required=True, help="User ID (UUID)")
    parser.add_argument("--thread-id", default="test-001", help="Thread ID for session")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  คำถาม: {args.task}")
    print(f"  User:  {args.user_id}")
    print(f"{'='*60}")

    result = await run_query(args.task, args.user_id, args.thread_id)

    if not result:
        print("❌ No result received")
        return

    # Extract final state
    messages = result.get("messages", [])
    current = result.get("current_result", {})

    print(f"\n📊 Raw Result:")
    print(json.dumps(current, indent=2, ensure_ascii=False))

    print(f"\n💬 Response:")
    if messages:
        for msg in messages:
            if msg.get("role") == "assistant":
                print(msg.get("content", ""))


if __name__ == "__main__":
    asyncio.run(main())
