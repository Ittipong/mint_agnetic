"""Run evaluation against AI Friend agent.

This module provides CLI for running evaluations against the agent
using CSV datasets from tests/evaluation/datasets/.

Usage:
    # Run all tests against default dataset
    python -m tests.evaluation.run_eval

    # Run specific intent
    python -m tests.evaluation.run_eval --intent SPENT_BUDGET

    # Run with LangSmith logging
    python -m tests.evaluation.run_eval --langsmith

    # Dry run (show what would run)
    python -m tests.evaluation.run_eval --dry-run

    # Export results to JSON
    python -m tests.evaluation.run_eval --export results.json
"""

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from tests.evaluation.evaluators import (
    Intent,
    EvalMetrics,
    TOOL_TO_INTENT,
    calculate_accuracy,
    calculate_latency_stats,
    print_evaluation_report,
    extract_intent_from_tools,
    detect_shame,
)


# Configuration
STUDIO_URL = "http://localhost:8080"
DEFAULT_USER_ID = "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9"
CSV_PATH = Path(__file__).parent / "datasets" / "phase1_dataset.csv"


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    question: str
    intent: str
    edge_case: str
    predicted_intent: Optional[str]
    tool_calls: list[str]
    response_text: Optional[str]
    latency_ms: float
    success: bool
    error: Optional[str]


def load_test_cases(csv_path: Path, intent_filter: Optional[str] = None) -> list[dict]:
    """Load test cases from CSV."""
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if intent_filter and row["intent"] != intent_filter:
                continue
            cases.append(row)
    return cases


def extract_tool_calls(data: dict) -> list[str]:
    """Extract tool calls from agent response."""
    tools = []
    messages = data.get("messages", [])
    for msg in messages:
        if tool_calls := msg.get("tool_calls"):
            for tc in tool_calls:
                tools.append(tc.get("name"))
    return tools


def extract_response(data: dict) -> Optional[str]:
    """Extract AI response from agent data."""
    messages = data.get("messages", [])
    for msg in reversed(messages):
        if msg.get("type") == "ai" and not msg.get("tool_calls"):
            return msg.get("content")
    return None


async def run_single_test(
    question: str,
    user_id: str,
    studio_url: str,
    timeout: int = 60,
) -> EvalResult:
    """Run a single test case."""
    start_time = datetime.now()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{studio_url}/runs/wait",
                json={
                    "assistant_id": "agent",
                    "graph_name": "agent",
                    "input": {
                        "user_id": user_id,
                        "messages": [{"role": "user", "content": question}],
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            tool_calls = extract_tool_calls(data)
            predicted_intent = extract_intent_from_tools(tool_calls)
            response_text = extract_response(data)

            return EvalResult(
                question=question,
                intent="",  # Will be set by caller
                edge_case="",
                predicted_intent=predicted_intent,
                tool_calls=tool_calls,
                response_text=response_text,
                latency_ms=latency_ms,
                success=True,
                error=None,
            )

    except Exception as e:
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        return EvalResult(
            question=question,
            intent="",
            edge_case="",
            predicted_intent=None,
            tool_calls=[],
            response_text=None,
            latency_ms=latency_ms,
            success=False,
            error=str(e),
        )


async def run_evaluation(
    test_cases: list[dict],
    user_id: str,
    studio_url: str,
) -> list[EvalMetrics]:
    """Run evaluation for all test cases."""
    results = []

    for i, tc in enumerate(test_cases, 1):
        question = tc["question"]
        expected_intent = tc["intent"]
        edge_case = tc["edge_case"]

        print(f"[{i}/{len(test_cases)}] {expected_intent}: {question[:40]}...", end=" ")

        row_user_id = tc.get("user_id") or user_id
        result = await run_single_test(question, row_user_id, studio_url)
        result.intent = expected_intent
        result.edge_case = edge_case

        predicted_intent = result.predicted_intent or ""
        intent_correct = (predicted_intent == expected_intent) or (
            expected_intent == "FALLBACK" and not predicted_intent
        )

        metrics = EvalMetrics(
            test_id=f"{expected_intent}-{i:03d}",
            intent=Intent(expected_intent),
            question=question,
            edge_case=edge_case,
            predicted_intent=predicted_intent,
            tool_calls=result.tool_calls,
            intent_correct=intent_correct,
            latency_ms=result.latency_ms,
            has_response=bool(result.response_text),
            has_chips="chips" in (result.response_text or "").lower(),
            error=result.error,
        )

        status = "✅" if intent_correct else "❌"
        print(f"{status} ({result.latency_ms:.0f}ms)")

        if result.error:
            print(f"   Error: {result.error[:100]}")

        results.append(metrics)

    return results


def log_to_langsmith(
    results: list[EvalMetrics],
    dataset_name: str = "ai_friend_phase1",
):
    """Log evaluation results to LangSmith."""
    try:
        from langsmith import Client
        client = Client()

        # Get dataset
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if not datasets:
            print(f"⚠️ Dataset '{dataset_name}' not found in LangSmith")
            return

        dataset = datasets[0]

        # Log each result as a run
        for r in results:
            try:
                client.create_run(
                    dataset_name=dataset_name,
                    name=f"eval_{r.test_id}",
                    run_type="chain",
                    inputs={
                        "question": r.question,
                        "intent": r.intent.value,
                        "edge_case": r.edge_case,
                    },
                    outputs={
                        "predicted_intent": r.predicted_intent,
                        "tool_calls": r.tool_calls,
                        "intent_correct": r.intent_correct,
                        "latency_ms": r.latency_ms,
                    },
                    metadata={
                        "edge_case": r.edge_case,
                        "has_response": r.has_response,
                    },
                    error=r.error,
                )
            except Exception as e:
                print(f"⚠️ Failed to log {r.test_id}: {e}")

        print(f"\n✅ Logged {len(results)} runs to LangSmith dataset '{dataset_name}'")

    except ImportError:
        print("⚠️ langsmith not installed")
    except Exception as e:
        print(f"⚠️ Failed to log to LangSmith: {e}")


def export_results_json(results: list[EvalMetrics], filename: str):
    """Export results to JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation_time": datetime.now().isoformat(),
                "results": [asdict(r) for r in results],
                "accuracy": calculate_accuracy(results),
                "latency": calculate_latency_stats(results),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run AI Friend evaluation")
    parser.add_argument(
        "--intent",
        type=str,
        choices=[i.value for i in Intent],
        help="Filter by specific intent",
    )
    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help=f"User ID (default: {DEFAULT_USER_ID})",
    )
    parser.add_argument(
        "--studio-url",
        default=STUDIO_URL,
        help=f"LangGraph Studio URL (default: {STUDIO_URL})",
    )
    parser.add_argument(
        "--dataset",
        default="ai_friend_phase1",
        help="Dataset name for LangSmith logging",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help=f"CSV path (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without running",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Log results to LangSmith",
    )

    args = parser.parse_args()

    # Load test cases
    if not args.csv.exists():
        print(f"Error: CSV not found at {args.csv}")
        print("Run: python -m tests.evaluation.langsmith_sync --upload")
        return

    test_cases = load_test_cases(args.csv, args.intent)
    print(f"\nLoaded {len(test_cases)} test cases from {args.csv}")
    if args.intent:
        print(f"Filtered to intent: {args.intent}")

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        for tc in test_cases[:5]:
            print(f"  - {tc['intent']}: {tc['question'][:40]}...")
        if len(test_cases) > 5:
            print(f"  ... and {len(test_cases) - 5} more")
        return

    print(f"\n{'='*60}")
    print(f"Running {len(test_cases)} test cases...")
    print(f"Studio URL: {args.studio_url}")
    print(f"{'='*60}\n")

    # Run evaluation
    results = asyncio.run(run_evaluation(
        test_cases,
        args.user_id,
        args.studio_url,
    ))

    # Print report
    print_evaluation_report(results)

    # Export
    if args.export:
        export_results_json(results, args.export)

    # Log to LangSmith
    if args.langsmith:
        log_to_langsmith(results, args.dataset)


if __name__ == "__main__":
    main()
