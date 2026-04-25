"""Run LangSmith evaluation for AI Friend Phase 1.

Workflow:
  1. Pulls examples from LangSmith dataset 'ai_friend_phase1_eval'
  2. Runs each example through the agent graph directly (no HTTP, no Studio)
  3. Scores with 3 evaluators: tool_selection, no_shame, response_quality
  4. Reports results to LangSmith UI as an experiment

Usage:
    # Create dataset first (one-time)
    python -m tests.eva.dataset

    # Run evaluation
    cd agentic
    python -m tests.eva.run_eval

    # Custom experiment name
    python -m tests.eva.run_eval --experiment v2-gemma-tuned

    # Filter single intent
    python -m tests.eva.run_eval --intent SPENT_BUDGET
"""

import argparse
import asyncio
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langsmith import Client  # noqa: E402

from src.graph.agent_graph import build_graph  # noqa: E402
from tests.eva.evaluators import (  # noqa: E402
    evaluate_no_shame,
    evaluate_response_quality,
    evaluate_tool_selection,
)

DATASET_NAME = "ai_friend_phase1_eval"

# Persistent event loop in a background thread — prevents httpx/asyncpg from
# seeing "Event loop is closed" when asyncio.run() tears down the loop between calls.
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

# Graph and DB connections reuse the same loop for their lifetime.
_graph = build_graph()


async def _ainvoke(inputs: dict) -> dict:
    """Invoke agent graph and extract final response + tool calls."""
    result = await _graph.ainvoke({
        "messages": inputs["messages"],
        "user_id": inputs["user_id"],
    })

    messages = result.get("messages", [])

    tool_calls: list[str] = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend(tc["name"] for tc in msg.tool_calls)

    # Last AI message without pending tool_calls is the final reply
    response = ""
    for msg in reversed(messages):
        is_ai = getattr(msg, "type", None) == "ai"
        has_pending = bool(getattr(msg, "tool_calls", None))
        if is_ai and not has_pending:
            response = msg.content or ""
            break

    return {"response": response, "tool_calls": tool_calls}


def target(inputs: dict) -> dict:
    """Sync wrapper — submits work to the persistent loop; safe for repeated calls."""
    future = asyncio.run_coroutine_threadsafe(_ainvoke(inputs), _loop)
    return future.result(timeout=120)


def _filter_dataset(client: Client, dataset_name: str, intent: str | None) -> str:
    """If intent filter given, create a temporary in-memory slice via metadata filter."""
    if not intent:
        return dataset_name

    # For filtered runs, collect matching example IDs and pass subset
    # LangSmith SDK supports passing example list directly to client.evaluate(data=...)
    examples = [
        ex for ex in client.list_examples(dataset_name=dataset_name)
        if (ex.outputs or {}).get("intent") == intent
    ]
    if not examples:
        raise ValueError(f"No examples found for intent '{intent}' in dataset '{dataset_name}'")

    print(f"Filtered to {len(examples)} examples with intent={intent}")
    return examples  # client.evaluate accepts list[Example]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Friend LangSmith evaluation")
    parser.add_argument("--experiment", default="ai-friend-phase1", help="Experiment name prefix")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel runs (default: 1)")
    parser.add_argument(
        "--intent",
        choices=["SPENT_BUDGET", "WEEKLY_SUMMARY", "DEBT_BALANCE", "GOAL_PROGRESS", "PAYDAY", "FALLBACK"],
        help="Evaluate only this intent",
    )
    args = parser.parse_args()

    client = Client()

    if not list(client.list_datasets(dataset_name=DATASET_NAME)):
        print(f"Dataset '{DATASET_NAME}' not found.")
        print("Create it first: python -m tests.eva.dataset")
        return

    data = _filter_dataset(client, DATASET_NAME, args.intent)

    print(f"Dataset  : {DATASET_NAME}")
    print(f"Experiment: {args.experiment}")
    print(f"Concurrency: {args.concurrency}")
    print()

    results = client.evaluate(
        target,
        data=data,
        evaluators=[
            evaluate_tool_selection,
            evaluate_no_shame,
            evaluate_response_quality,
        ],
        experiment_prefix=args.experiment,
        max_concurrency=args.concurrency,
        metadata={
            "phase": "phase1",
            "model": "gemma-4-26b",
            "intent_filter": args.intent or "all",
        },
    )

    print(results)


if __name__ == "__main__":
    main()
