"""Sync CSV datasets to/from LangSmith.

This module handles:
1. CSV → LangSmith dataset (upload)
2. LangSmith dataset → CSV (download)

Usage:
    # Upload CSV to LangSmith
    python -m tests.evaluation.langsmith_sync --upload

    # Download from LangSmith to CSV
    python -m tests.evaluation.langsmith_sync --download

    # Dry run
    python -m tests.evaluation.langsmith_sync --upload --dry-run
"""

import argparse
import csv
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from langsmith import Client


# Configuration
DATASET_NAME = "ai_friend_phase1"
CSV_PATH = Path(__file__).parent / "datasets" / "phase1_dataset.csv"


def load_csv_examples(csv_path: Path) -> list[dict]:
    """Load examples from CSV file."""
    examples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append({
                "question": row["question"],
                "intent": row["intent"],
                "edge_case": row["edge_case"],
                "description": row["description"],
                "user_id": row.get("user_id", ""),
            })
    return examples


def get_or_create_dataset(client: Client, name: str, description: str) -> dict:
    """Get existing dataset or create new one."""
    existing = list(client.list_datasets(dataset_name=name))
    if existing:
        dataset = existing[0]
        print(f"Found existing dataset: {name} (id={dataset.id})")
        return dataset

    dataset = client.create_dataset(
        dataset_name=name,
        description=description,
    )
    print(f"Created new dataset: {name} (id={dataset.id})")
    return dataset


def upload_to_langsmith(csv_path: Path, dataset_name: str, dry_run: bool = False, reset: bool = False):
    """Upload CSV examples to LangSmith dataset."""
    client = Client()
    examples = load_csv_examples(csv_path)

    print(f"\n{'='*60}")
    print(f"Uploading {len(examples)} examples to '{dataset_name}'")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Would create/update dataset with:")
        for ex in examples[:5]:
            print(f"  - {ex['intent']}: {ex['question'][:40]}...")
        if len(examples) > 5:
            print(f"  ... and {len(examples) - 5} more")
        return

    # Get or create dataset
    dataset = get_or_create_dataset(
        client,
        dataset_name,
        description="AI Friend Phase 1 — 5 intents with edge cases"
    )

    if reset:
        existing = list(client.list_examples(dataset_id=dataset.id))
        for ex in existing:
            client.delete_example(ex.id)
        print(f"🗑️  Deleted {len(existing)} existing examples")

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {
                "user_id": ex["user_id"],
                "messages": [{"role": "user", "content": ex["question"]}],
            }
            for ex in examples
        ],
        outputs=[
            {
                "intent": ex["intent"],
                "edge_case": ex["edge_case"],
                "description": ex["description"],
            }
            for ex in examples
        ],
    )
    print(f"✅ Successfully uploaded {len(examples)} examples")


def download_from_langsmith(csv_path: Path, dataset_name: str, dry_run: bool = False):
    """Download examples from LangSmith dataset to CSV."""
    client = Client()

    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if not datasets:
        print(f"Dataset '{dataset_name}' not found")
        return

    dataset = datasets[0]
    print(f"\nDataset: {dataset.name}")
    print(f"ID: {dataset.id}")

    examples = list(client.list_examples(dataset_id=dataset.id))
    print(f"\nFound {len(examples)} examples")

    if dry_run:
        print("[DRY RUN] Would write to CSV:")
        for ex in examples[:5]:
            print(f"  - {ex.inputs}")
        return

    if examples:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "intent", "edge_case", "description", "user_id"])
            writer.writeheader()
            for ex in examples:
                inputs = ex.inputs or {}
                outputs = ex.outputs or {}
                messages = inputs.get("messages", [])
                question = messages[0].get("content", "") if messages else ""
                writer.writerow({
                    "question": question,
                    "intent": outputs.get("intent", ""),
                    "edge_case": outputs.get("edge_case", ""),
                    "description": outputs.get("description", ""),
                    "user_id": inputs.get("user_id", ""),
                })
        print(f"✅ Written {len(examples)} examples to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Sync CSV ↔ LangSmith")
    parser.add_argument("--upload", action="store_true", help="Upload CSV to LangSmith")
    parser.add_argument("--download", action="store_true", help="Download from LangSmith to CSV")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--reset", action="store_true", help="Delete existing examples before upload")
    parser.add_argument("--dataset", default=DATASET_NAME, help=f"Dataset name (default: {DATASET_NAME})")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help=f"CSV path (default: {CSV_PATH})")

    args = parser.parse_args()

    if args.upload:
        upload_to_langsmith(args.csv, args.dataset, args.dry_run, args.reset)
    elif args.download:
        download_from_langsmith(args.csv, args.dataset, args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
