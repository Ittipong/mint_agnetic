"""Create/update LangSmith dataset from phase1_dataset.csv.

Run once before evaluation:
    cd agentic
    python -m tests.eva.dataset

    # Reset and re-upload
    python -m tests.eva.dataset --reset
"""

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langsmith import Client  # noqa: E402

DATASET_NAME = "ai_friend_phase1_eval"
CSV_PATH = Path(__file__).parent.parent / "evaluation" / "datasets" / "phase1_dataset.csv"


def create_or_update_dataset(reset: bool = False) -> None:
    client = Client()

    rows = _load_csv(CSV_PATH)

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
        if reset:
            for ex in client.list_examples(dataset_id=dataset.id):
                client.delete_example(ex.id)
            print(f"Reset {DATASET_NAME}: deleted existing examples")
        else:
            count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
            if count > 0:
                print(f"Dataset '{DATASET_NAME}' already has {count} examples. Use --reset to re-upload.")
                return
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "AI Friend Phase 1 — 52 test cases covering "
                "SPENT_BUDGET, WEEKLY_SUMMARY, DEBT_BALANCE, GOAL_PROGRESS, PAYDAY, FALLBACK"
            ),
        )
        print(f"Created dataset: {DATASET_NAME}")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {
                "messages": [{"role": "user", "content": row["question"]}],
                "user_id": row["user_id"],
            }
            for row in rows
        ],
        outputs=[
            {
                "intent": row["intent"],
                "edge_case": row["edge_case"],
                "description": row["description"],
            }
            for row in rows
        ],
    )
    print(f"Uploaded {len(rows)} examples to '{DATASET_NAME}'")


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete existing examples before uploading")
    args = parser.parse_args()
    create_or_update_dataset(reset=args.reset)
