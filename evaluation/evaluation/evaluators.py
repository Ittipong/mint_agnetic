"""Custom evaluators for AI Friend agent.

This module defines evaluation metrics for Phase 1 intents:
- Intent accuracy
- Hallucination detection
- Response latency
- Friend tone adherence
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    """5 intents from Phase 1 spec."""
    SPENT_BUDGET = "SPENT_BUDGET"
    WEEKLY_SUMMARY = "WEEKLY_SUMMARY"
    DEBT_BALANCE = "DEBT_BALANCE"
    GOAL_PROGRESS = "GOAL_PROGRESS"
    PAYDAY = "PAYDAY"
    FALLBACK = "FALLBACK"


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single test case."""
    test_id: str
    intent: Intent
    question: str
    edge_case: str

    # Predicted
    predicted_intent: Optional[str] = None
    tool_calls: list[str] = None

    # Metrics
    intent_correct: bool = False
    latency_ms: float = 0.0
    has_response: bool = False
    has_numbers: bool = False
    has_chips: bool = False

    # Quality
    hallucination_detected: bool = False
    shame_detected: bool = False

    # Error
    error: Optional[str] = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


# Tool to intent mapping
TOOL_TO_INTENT = {
    "get_spent_budget": Intent.SPENT_BUDGET.value,
    "get_weekly_summary": Intent.WEEKLY_SUMMARY.value,
    "get_debt_balance": Intent.DEBT_BALANCE.value,
    "get_goal_progress": Intent.GOAL_PROGRESS.value,
    "get_payday": Intent.PAYDAY.value,
    "get_recent_transactions": Intent.FALLBACK.value,
    "get_budget_status": Intent.FALLBACK.value,
    "get_savings_goals": Intent.FALLBACK.value,
}

# Keywords for shame detection (anti-patterns from spec)
SHAME_KEYWORDS = [
    "ใช้เยอะเกินไป",
    "ควรลด",
    "ไม่ควรซื้อ",
    "ใช้จ่ายมากเกินไป",
    "ฟุ่มเฟือย",
    "สิ้นเปลือง",
]


def extract_intent_from_tools(tool_calls: list[str]) -> Optional[str]:
    """Extract predicted intent from tool calls."""
    for tool in tool_calls:
        if tool in TOOL_TO_INTENT:
            return TOOL_TO_INTENT[tool]
    return None


def detect_shame(response_text: str) -> bool:
    """Detect shame/judgment language in response."""
    if not response_text:
        return False
    text_lower = response_text.lower()
    return any(keyword in text_lower for keyword in SHAME_KEYWORDS)


def extract_numbers_from_response(response_text: str) -> bool:
    """Check if response contains numerical data."""
    if not response_text:
        return False
    # Simple check: contains Thai numbers or Arabic digits
    import re
    return bool(re.search(r"[\d,]+\s*(บาท|%|วัน|เดือน)", response_text))


def calculate_accuracy(results: list[EvalMetrics]) -> dict:
    """Calculate accuracy metrics by intent."""
    by_intent = {}
    for intent in Intent:
        intent_results = [r for r in results if r.intent == intent]
        if intent_results:
            correct = sum(1 for r in intent_results if r.intent_correct)
            by_intent[intent.value] = {
                "total": len(intent_results),
                "correct": correct,
                "accuracy": correct / len(intent_results) * 100,
            }

    total_correct = sum(1 for r in results if r.intent_correct)
    return {
        "overall_accuracy": total_correct / len(results) * 100 if results else 0,
        "by_intent": by_intent,
    }


def calculate_latency_stats(results: list[EvalMetrics]) -> dict:
    """Calculate latency statistics."""
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if not latencies:
        return {"avg": 0, "p50": 0, "p95": 0}

    sorted_latencies = sorted(latencies)
    return {
        "avg": sum(latencies) / len(latencies),
        "p50": sorted_latencies[len(sorted_latencies) // 2],
        "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
    }


def print_evaluation_report(results: list[EvalMetrics]):
    """Print evaluation summary report."""
    accuracy = calculate_accuracy(results)
    latency = calculate_latency_stats(results)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # Overall accuracy
    print(f"\nOverall Accuracy: {accuracy['overall_accuracy']:.1f}%")

    # By intent
    print("\nBy Intent:")
    for intent, stats in accuracy["by_intent"].items():
        status = "✅" if stats["accuracy"] == 100 else "⚠️"
        print(f"  {status} {intent}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")

    # Latency
    print(f"\nLatency:")
    print(f"  Average: {latency['avg']:.0f}ms")
    print(f"  P50: {latency['p50']:.0f}ms")
    print(f"  P95: {latency['p95']:.0f}ms")

    # Quality issues
    shame_count = sum(1 for r in results if r.shame_detected)
    hallucination_count = sum(1 for r in results if r.hallucination_detected)

    if shame_count > 0:
        print(f"\n⚠️ Shame detected: {shame_count} cases")
    if hallucination_count > 0:
        print(f"\n⚠️ Hallucination detected: {hallucination_count} cases")

    # Failed tests
    failed = [r for r in results if not r.intent_correct]
    if failed:
        print(f"\nFailed Tests ({len(failed)}):")
        for r in failed:
            print(f"  - {r.test_id}: expected {r.intent}, got {r.predicted_intent}")

    print("\n" + "=" * 60)
