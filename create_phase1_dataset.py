"""Create Phase 1 dataset in LangSmith.

This script creates the "ai_friend_phase1" dataset covering all 5 intents
with their edge cases from docs/ai/ai_friend_phase1_spec.md.

Usage:
    python create_phase1_dataset.py
    python create_phase1_dataset.py --dry-run
    python create_phase1_dataset.py --delete  # Delete existing dataset first
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from langsmith import Client


# =============================================================================
# Dataset: Phase 1 Test Cases
# =============================================================================

PHASE1_EXAMPLES = [
    # -------------------------------------------------------------------------
    # SPENT_BUDGET — "เดือนนี้ฉันโอเคไหม?"
    # -------------------------------------------------------------------------
    {
        "id": "SB-001",
        "intent": "SPENT_BUDGET",
        "question": "เดือนนี้ใช้ไปเท่าไร",
        "edge_case": "NORMAL",
        "description": "Current month spending - happy path",
    },
    {
        "id": "SB-002",
        "intent": "SPENT_BUDGET",
        "question": "งบเหลือเท่าไร",
        "edge_case": "NORMAL",
        "description": "Budget remaining - happy path",
    },
    {
        "id": "SB-003",
        "intent": "SPENT_BUDGET",
        "question": "เดือนที่แล้วใช้ไปเท่าไร",
        "edge_case": "NORMAL",
        "description": "Last month spending",
    },
    {
        "id": "SB-004",
        "intent": "SPENT_BUDGET",
        "question": "สถานะเดือนนี้เป็นไง",
        "edge_case": "NORMAL",
        "description": "Check monthly budget status",
    },
    {
        "id": "SB-005",
        "intent": "SPENT_BUDGET",
        "question": "ดู budget หน่อย",
        "edge_case": "NORMAL",
        "description": "View budget",
    },
    {
        "id": "SB-006",
        "intent": "SPENT_BUDGET",
        "question": "งบหมดยัง",
        "edge_case": "NORMAL",
        "description": "Is budget exhausted",
    },
    {
        "id": "SB-007",
        "intent": "SPENT_BUDGET",
        "question": "ยังไม่มีรายจ่ายเดือนนี้เลยนะ",
        "edge_case": "NO_DATA",
        "description": "No transactions this month",
    },
    {
        "id": "SB-008",
        "intent": "SPENT_BUDGET",
        "question": "ยังไม่ได้ตั้งงบเดือนนี้นะ",
        "edge_case": "NO_BUDGET",
        "description": "No budget set",
    },
    {
        "id": "SB-009",
        "intent": "SPENT_BUDGET",
        "question": "เดือนนี้ใช้ไป 52000 จาก 50000 เกินไป 2000",
        "edge_case": "OVER_BUDGET",
        "description": "Over budget",
    },
    {
        "id": "SB-010",
        "intent": "SPENT_BUDGET",
        "question": "สถานะเดือนมิถุนายน 2027",
        "edge_case": "FUTURE_MONTH",
        "description": "Future month query",
    },

    # -------------------------------------------------------------------------
    # WEEKLY_SUMMARY — "สัปดาห์นี้ใช้ไปเท่าไร?"
    # -------------------------------------------------------------------------
    {
        "id": "WS-001",
        "intent": "WEEKLY_SUMMARY",
        "question": "สัปดาห์นี้ใช้ไปเท่าไร",
        "edge_case": "NORMAL",
        "description": "Current week spending - happy path",
    },
    {
        "id": "WS-002",
        "intent": "WEEKLY_SUMMARY",
        "question": "อาทิตย์นี้เป็นไง",
        "edge_case": "NORMAL",
        "description": "This week summary",
    },
    {
        "id": "WS-003",
        "intent": "WEEKLY_SUMMARY",
        "question": "สัปดาห์ที่แล้วเท่าไร",
        "edge_case": "NORMAL",
        "description": "Last week spending",
    },
    {
        "id": "WS-004",
        "intent": "WEEKLY_SUMMARY",
        "question": "สัปดาห์ก่อนเป็นยังไง",
        "edge_case": "NORMAL",
        "description": "Previous week summary",
    },
    {
        "id": "WS-005",
        "intent": "WEEKLY_SUMMARY",
        "question": "เปรียบเทียบสัปดาห์นี้กับอาทิตย์ที่แล้ว",
        "edge_case": "NORMAL",
        "description": "Week over week comparison",
    },
    {
        "id": "WS-006",
        "intent": "WEEKLY_SUMMARY",
        "question": "รายสัปดาห์ย้อนหลัง 2 อาทิตย์",
        "edge_case": "NORMAL",
        "description": "2 weeks ago summary",
    },
    {
        "id": "WS-007",
        "intent": "WEEKLY_SUMMARY",
        "question": "สัปดาห์นี้ยังไม่มีรายจ่ายเลยนะ",
        "edge_case": "NO_DATA_CURRENT",
        "description": "No transactions this week",
    },
    {
        "id": "WS-008",
        "intent": "WEEKLY_SUMMARY",
        "question": "ยังไม่มีข้อมูลสัปดาห์เลยนะ",
        "edge_case": "NO_DATA_BOTH",
        "description": "No transactions either week",
    },
    {
        "id": "WS-009",
        "intent": "WEEKLY_SUMMARY",
        "question": "สัปดาห์นี้ใช้ไป 12500 เยอะกว่าอาทิตย์ที่แล้ว 3500",
        "edge_case": "WEEK_OVER_WEEK_UP_50",
        "description": "Week-over-week up >50%",
    },

    # -------------------------------------------------------------------------
    # DEBT_BALANCE — "หนี้ฉันยังเหลือเท่าไร?"
    # -------------------------------------------------------------------------
    {
        "id": "DB-001",
        "intent": "DEBT_BALANCE",
        "question": "หนี้เหลือเท่าไร",
        "edge_case": "NORMAL",
        "description": "Total debt remaining - happy path",
    },
    {
        "id": "DB-002",
        "intent": "DEBT_BALANCE",
        "question": "ค่างวดเดือนนี้เท่าไร",
        "edge_case": "NORMAL",
        "description": "This month's debt payment",
    },
    {
        "id": "DB-003",
        "intent": "DEBT_BALANCE",
        "question": "ผ่อนรถเหลือเท่าไร",
        "edge_case": "NORMAL",
        "description": "Car loan remaining",
    },
    {
        "id": "DB-004",
        "intent": "DEBT_BALANCE",
        "question": "ดูสถานะหนี้หน่อย",
        "edge_case": "NORMAL",
        "description": "View debt status",
    },
    {
        "id": "DB-005",
        "intent": "DEBT_BALANCE",
        "question": "ยังต้องจ่ายอีกกี่เดือน",
        "edge_case": "NORMAL",
        "description": "How many months remaining",
    },
    {
        "id": "DB-006",
        "intent": "DEBT_BALANCE",
        "question": "outstanding บัตรเครดิต",
        "edge_case": "NORMAL",
        "description": "Credit card outstanding",
    },
    {
        "id": "DB-007",
        "intent": "DEBT_BALANCE",
        "question": "ไม่มีหนี้ active เลยนะ",
        "edge_case": "NO_DEBT",
        "description": "No active debts",
    },
    {
        "id": "DB-008",
        "intent": "DEBT_BALANCE",
        "question": "บัตรเครดิตใช้ไป 65000 จาก 100000 limit",
        "edge_case": "CC_ONLY",
        "description": "Credit card only, no obligations",
    },
    {
        "id": "DB-009",
        "intent": "DEBT_BALANCE",
        "question": "หนี้ก้อนนี้ปิดแล้ว ดีมาก",
        "edge_case": "DEBT_PAID_OFF",
        "description": "Debt fully paid",
    },
    {
        "id": "DB-010",
        "intent": "DEBT_BALANCE",
        "question": "มีงวดที่ค้างอยู่ 5000 บาทนะ",
        "edge_case": "PAST_DUE",
        "description": "Past due amount",
    },

    # -------------------------------------------------------------------------
    # GOAL_PROGRESS — "goal ฉันเป็นไงบ้าง?"
    # -------------------------------------------------------------------------
    {
        "id": "GP-001",
        "intent": "GOAL_PROGRESS",
        "question": "goal ผมเป็นไงบ้าง",
        "edge_case": "NORMAL",
        "description": "All goals progress - happy path",
    },
    {
        "id": "GP-002",
        "intent": "GOAL_PROGRESS",
        "question": "ออมไปเท่าไรแล้ว",
        "edge_case": "NORMAL",
        "description": "How much saved",
    },
    {
        "id": "GP-003",
        "intent": "GOAL_PROGRESS",
        "question": "ถึงเป้าไหมแล้ว",
        "edge_case": "NORMAL",
        "description": "Goal achieved",
    },
    {
        "id": "GP-004",
        "intent": "GOAL_PROGRESS",
        "question": "เงินก้อนสะสมเท่าไร",
        "edge_case": "NORMAL",
        "description": "Total saved amount",
    },
    {
        "id": "GP-005",
        "intent": "GOAL_PROGRESS",
        "question": "เหลืออีกเท่าไรถึงเป้า",
        "edge_case": "NORMAL",
        "description": "Amount remaining to goal",
    },
    {
        "id": "GP-006",
        "intent": "GOAL_PROGRESS",
        "question": "ดู goal หน่อย",
        "edge_case": "NORMAL",
        "description": "View goals",
    },
    {
        "id": "GP-007",
        "intent": "GOAL_PROGRESS",
        "question": "ยังไม่มี goal สักอันเลยนะ",
        "edge_case": "NO_GOALS",
        "description": "No goals",
    },
    {
        "id": "GP-008",
        "intent": "GOAL_PROGRESS",
        "question": "goal เงินดาวน์รถถึงเป้าแล้ว ดีมากเลย",
        "edge_case": "GOAL_ACHIEVED",
        "description": "Goal achieved",
    },
    {
        "id": "GP-009",
        "intent": "GOAL_PROGRESS",
        "question": "goal นี้อยู่ที่ 68% แล้วแต่ยังไม่ได้ตั้ง target date",
        "edge_case": "NO_TARGET_DATE",
        "description": "Goal without target date",
    },
    {
        "id": "GP-010",
        "intent": "GOAL_PROGRESS",
        "question": "goal เลยกำหนดไปแล้ว 12 วันนะ",
        "edge_case": "GOAL_OVERDUE",
        "description": "Goal overdue",
    },

    # -------------------------------------------------------------------------
    # PAYDAY — "เงินเดือนวันที่เท่าไร?"
    # -------------------------------------------------------------------------
    {
        "id": "PD-001",
        "intent": "PAYDAY",
        "question": "เงินเดือนวันที่เท่าไร",
        "edge_case": "NORMAL",
        "description": "Payday date - happy path",
    },
    {
        "id": "PD-002",
        "intent": "PAYDAY",
        "question": "ถึงวันจ่ายอีกกี่วัน",
        "edge_case": "NORMAL",
        "description": "Days until payday",
    },
    {
        "id": "PD-003",
        "intent": "PAYDAY",
        "question": "วันเงินเดือน",
        "edge_case": "NORMAL",
        "description": "Payday",
    },
    {
        "id": "PD-004",
        "intent": "PAYDAY",
        "question": "payday วันไหน",
        "edge_case": "NORMAL",
        "description": "When is payday",
    },
    {
        "id": "PD-005",
        "intent": "PAYDAY",
        "question": "เงินเดือนจะเข้าเมื่อไร",
        "edge_case": "NORMAL",
        "description": "When will salary arrive",
    },
    {
        "id": "PD-006",
        "intent": "PAYDAY",
        "question": "ยังไม่เห็น pattern เงินเดือนเลยนะ",
        "edge_case": "NO_INCOME_DATA",
        "description": "No income data",
    },
    {
        "id": "PD-007",
        "intent": "PAYDAY",
        "question": "เงินเดือนเข้าพรุ่งนี้แล้ว",
        "edge_case": "PAYDAY_TOMORROW",
        "description": "Payday tomorrow",
    },
    {
        "id": "PD-008",
        "intent": "PAYDAY",
        "question": "เงินเดือนเข้าวันนี้เลย",
        "edge_case": "PAYDAY_TODAY",
        "description": "Payday today",
    },
    {
        "id": "PD-009",
        "intent": "PAYDAY",
        "question": "มีรายได้หลายทาง เงินเดือนเข้าวันที่ 5 และ extra income วันที่ 20",
        "edge_case": "MULTIPLE_INCOME",
        "description": "Multiple income sources",
    },

    # -------------------------------------------------------------------------
    # FALLBACK — graceful deflection
    # -------------------------------------------------------------------------
    {
        "id": "FB-001",
        "intent": "FALLBACK",
        "question": "ทำอะไรได้บ้าง",
        "edge_case": "GENERAL",
        "description": "What can you do",
    },
    {
        "id": "FB-002",
        "intent": "FALLBACK",
        "question": "ช่วยแนะนำอะไรหน่อย",
        "edge_case": "GENERAL",
        "description": "Give recommendations",
    },
    {
        "id": "FB-003",
        "intent": "FALLBACK",
        "question": "ช่วยหาร้านอาหารอร่อยๆ หน่อย",
        "edge_case": "OUT_OF_SCOPE",
        "description": "Out of scope request",
    },
    {
        "id": "FB-004",
        "intent": "FALLBACK",
        "question": "สวัสดี วันนี้อากาศดีจัง",
        "edge_case": "OFF_TOPIC",
        "description": "Off topic",
    },
]


# =============================================================================
# LangSmith Operations
# =============================================================================

DATASET_NAME = "ai_friend_phase1"
DATASET_DESCRIPTION = "AI Friend Phase 1 — 5 intents with edge cases (2026-04-20)"


def create_or_get_dataset(client: Client) -> dict:
    """Create dataset or get existing one."""
    # Try to find existing
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
        print(f"Found existing dataset: {dataset.name} (id={dataset.id})")
        return dataset

    # Create new
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"Created new dataset: {dataset.name} (id={dataset.id})")
    return dataset


def delete_dataset(client: Client, dataset_name: str = DATASET_NAME):
    """Delete dataset by name."""
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        dataset = existing[0]
        client.delete_dataset(dataset_id=dataset.id)
        print(f"Deleted dataset: {dataset_name}")
    else:
        print(f"Dataset not found: {dataset_name}")


def add_examples(client: Client, dataset_id: str, examples: list[dict]):
    """Add examples to dataset."""
    # Format for LangSmith: list of dicts with keys matching input schema
    formatted = [
        {
            "question": ex["question"],
            "intent": ex["intent"],
            "edge_case": ex["edge_case"],
            "description": ex["description"],
        }
        for ex in examples
    ]

    client.create_examples(dataset_id=dataset_id, examples=formatted)
    print(f"Added {len(formatted)} examples to dataset")


def main():
    parser = argparse.ArgumentParser(description="Create Phase 1 dataset in LangSmith")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created")
    parser.add_argument("--delete", action="store_true", help="Delete existing dataset first")
    args = parser.parse_args()

    client = Client()

    if args.delete:
        delete_dataset(client)
        print()
        if args.dry_run:
            print("--dry-run: skipping re-creation")
            return

    if args.dry_run:
        print(f"Would create dataset: {DATASET_NAME}")
        print(f"Description: {DATASET_DESCRIPTION}")
        print(f"Total examples: {len(PHASE1_EXAMPLES)}")
        print()
        print("Examples by intent:")
        intents = {}
        for ex in PHASE1_EXAMPLES:
            intents.setdefault(ex["intent"], []).append(ex["id"])
        for intent, ids in intents.items():
            print(f"  {intent}: {len(ids)} examples")
        return

    dataset = create_or_get_dataset(client)
    add_examples(client, dataset.id, PHASE1_EXAMPLES)

    print()
    print(f"Dataset URL: https://smith.langchain.com/datasets/{dataset.id}")


if __name__ == "__main__":
    main()
