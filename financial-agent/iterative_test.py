#!/usr/bin/env python3
"""Iterative improvement loop for financial agent.

Steps:
    1. Ask question
    2. Verify against expected values
    3. Improve (edit agent/prompt/tools)
    4. Back to step 1
"""
import argparse
import asyncio
import json
import sys
import time
import httpx

STUDIO_URL = "http://127.0.0.1:2024"
GRAPH_ID = "financial_agent"

# Expected values for verification
EXPECTED = {
    "ฉันมีกระเป๋าอะไรบ้าง": {
        "ครอบครัว": {"value": 36970.23, "unit": "THB"},
        "TrueMonney": {"value": 13870.69, "unit": "THB"},
        "Pad shop": {"value": -216.50, "unit": "USD"},
    }
}


async def ask(question: str, user_id: str, thread_id: str = "iter-1") -> dict:
    """Ask question via LangGraph Studio."""
    payload = {
        "assistant_id": GRAPH_ID,
        "graph_id": GRAPH_ID,
        "input": {
            "user_id": user_id,
            "messages": [{"role": "user", "content": question}],
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


def verify(question: str, result: dict) -> tuple[bool, list[str]]:
    """Verify result against expected values. Returns (pass, issues)."""
    expected = EXPECTED.get(question, {})
    if not expected:
        return False, [f"No expected values defined for: {question}"]

    issues = []
    breakdown = result.get("current_result", {}).get("breakdown", [])

    # Build dict of results
    actual = {}
    for item in breakdown:
        label = item.get("label", "")
        try:
            value = float(item.get("value", 0))
        except (ValueError, TypeError):
            value = None
        unit = item.get("unit", "")
        actual[label] = {"value": value, "unit": unit}

    # Compare
    for label, exp in expected.items():
        act = actual.get(label, {})
        act_value = act.get("value")
        act_unit = act.get("unit", "")

        if act_value is None:
            issues.append(f"❌ {label}: MISSING (expected {exp['value']} {exp['unit']})")
            continue

        if act_unit != exp["unit"]:
            issues.append(f"❌ {label}: wrong unit {act_unit} (expected {exp['unit']})")
            continue

        diff = abs(act_value - exp["value"])
        tolerance = abs(exp["value"]) * 0.01  # 1% tolerance

        if diff <= tolerance:
            issues.append(f"✅ {label}: {act_value} {act_unit} (expected {exp['value']})")
        else:
            issues.append(f"❌ {label}: {act_value} {act_unit} (expected {exp['value']}, diff {diff:.2f})")

    passed = all("✅" in i for i in issues)
    return passed, issues


def improve(question: str) -> str:
    """Prompt user to improve and return description of changes."""
    print("\n" + "="*60)
    print("🔧 IMPROVE STEP")
    print("="*60)
    print(f"Question: {question}")
    print("\nOptions:")
    print("  1. Edit system_prompt.j2")
    print("  2. Edit tools code")
    print("  3. Edit prompt in system_prompt.j2")
    print("  4. Skip improvement (exit)")
    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        return "system_prompt.j2"
    elif choice == "2":
        return "tools code"
    elif choice == "3":
        return "prompt"
    else:
        return "exit"


async def iterative_loop(question: str, user_id: str, max_iterations: int = 5):
    """Run the iterative improvement loop."""
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}")

        # Step 1: Ask
        print(f"\n📤 Asking: {question}")
        result = await ask(question, user_id)

        if not result:
            print("❌ No result received")
            continue

        current = result.get("current_result", {})
        print(f"\n📊 Result:")
        print(json.dumps(current, indent=2, ensure_ascii=False))

        # Step 2: Verify
        print(f"\n🔍 Verifying:")
        passed, issues = verify(question, result)
        for issue in issues:
            print(f"  {issue}")

        if passed:
            print(f"\n🎉 ALL CHECKS PASSED!")
            return True

        # Step 3: Improve
        print(f"\n📝 Issues found: {len([i for i in issues if '❌' in i])}")
        improvement = improve(question)

        if improvement == "exit":
            print("Exiting loop.")
            return False

        print(f"\n⏳ After making changes, the graph will auto-reload.")
        print("Press Enter to continue with next iteration...")
        input()

    print(f"\n⚠️ Max iterations ({max_iterations}) reached.")
    return False


async def main():
    parser = argparse.ArgumentParser(description="Iterative improvement loop")
    parser.add_argument("question", help="Question to test")
    parser.add_argument("--user-id", required=True, help="User ID (UUID)")
    parser.add_argument("--max-iter", type=int, default=5, help="Max iterations")
    args = parser.parse_args()

    await iterative_loop(args.question, args.user_id, args.max_iter)


if __name__ == "__main__":
    asyncio.run(main())
