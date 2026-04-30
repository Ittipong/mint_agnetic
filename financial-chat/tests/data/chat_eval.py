"""Chat evaluation runner for LangSmith.

Combines dataset upload and evaluation into a single command.

Usage:
    python -m tests.data.chat_eval --file evaluation/dataset_poc.jsonl
    python -m tests.data.chat_eval --file evaluation/dataset_poc.jsonl --dataset-name "my-dataset"
    python -m tests.data.chat_eval --file evaluation/dataset_poc.jsonl --all-models
    python -m tests.data.chat_eval --file evaluation/dataset_poc.jsonl --langsmith
    python -m tests.data.chat_eval --file evaluation/dataset_poc.jsonl --export results.json

Make targets:
    make eval FILE=dataset_poc.jsonl
    make eval FILE=dataset_poc.jsonl ALL_MODELS=1
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from dotenv import dotenv_values

# Load environment
_env_test = dotenv_values(_project_root / ".env.test")
for key, value in _env_test.items():
    if value and key not in os.environ:
        os.environ[key] = value


def _upload_dataset(jsonl_path: Path, dataset_name: str) -> tuple[str, int]:
    """Upload JSONL to LangSmith. Returns (dataset_id, example_count).

    If dataset with same name exists, reuse it.
    """
    from langsmith import Client
    from evaluation.dataset import upload_jsonl

    # Check if dataset already exists
    client = Client()
    existing = list(client.list_datasets(dataset_name=dataset_name, limit=1))

    if existing:
        dataset_id = existing[0].id
        examples = list(client.list_examples(dataset_id=dataset_id, limit=1000))
        example_count = len(examples)
        print(f"[1/2] Using existing dataset '{dataset_name}' (ID: {dataset_id}, {example_count} examples)")
        return dataset_id, example_count

    print(f"[1/2] Uploading {jsonl_path} as '{dataset_name}'...")
    result = upload_jsonl(jsonl_path, dataset_name)

    if not result["success"]:
        raise RuntimeError(f"Upload failed: {result['errors']}")

    print(f"       Dataset ID: {result['dataset_id']}")
    print(f"       Examples: {result['example_count']}")
    return result["dataset_id"], result["example_count"]


def _run_eval(
    dataset_id: str,
    dataset_name: str,
    model: str | None,
    all_models: bool,
    group_by_usecase: bool,
    baseline_model: str | None,
    run_by: str,
    max_concurrency: int,
):
    """Run evaluation on uploaded dataset."""
    from evaluation.evaluate import run_evaluation, run_all_models

    print(f"[2/2] Running evaluation...")

    if all_models:
        results = asyncio.run(
            run_all_models(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                max_concurrency=max_concurrency,
                baseline_model=baseline_model,
                run_by=run_by,
            )
        )
    else:
        results = asyncio.run(
            run_evaluation(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                model=model,
                max_concurrency=max_concurrency,
                group_by_usecase=group_by_usecase,
                baseline_model=baseline_model,
                run_by=run_by,
            )
        )

    print("\nEvaluation complete!")
    print(f"  Dataset: {dataset_name}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Results: {results}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload + evaluate chat agent on LangSmith")
    parser.add_argument(
        "--file", "-f", type=str, required=True,
        help="Path to JSONL dataset file (relative to project root)"
    )
    parser.add_argument(
        "--dataset-name", "-n", type=str, default=None,
        help="Dataset name in LangSmith (default: derived from filename)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Model override (e.g. google/gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Run evaluation for all models in .env.test MODELS list"
    )
    parser.add_argument(
        "--group-by-usecase", action="store_true",
        help="Run separate experiment per usecase_id"
    )
    parser.add_argument(
        "--baseline-model", type=str, default=None,
        help="Baseline model for comparison"
    )
    parser.add_argument(
        "--run-by", type=str, default="manual",
        choices=["manual", "ci", "scheduled"],
        help="Who triggered this run"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4,
        help="Max concurrent evaluations"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to JSON file"
    )

    args = parser.parse_args()

    # Resolve file path
    jsonl_path = _project_root / args.file
    if not jsonl_path.exists():
        print(f"ERROR: File not found: {jsonl_path}")
        sys.exit(1)

    # Derive dataset name from filename if not provided
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = jsonl_path.stem  # filename without extension

    # Upload to LangSmith
    dataset_id, example_count = _upload_dataset(jsonl_path, dataset_name)

    # Run evaluation
    results = _run_eval(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        model=args.model,
        all_models=args.all_models,
        group_by_usecase=args.group_by_usecase,
        baseline_model=args.baseline_model,
        run_by=args.run_by,
        max_concurrency=args.max_concurrency,
    )

    # Export results if requested
    if args.export:
        export_path = _project_root / args.export
        with export_path.open("w") as f:
            json.dump({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "example_count": example_count,
                "results": str(results),
            }, f, indent=2, default=str)
        print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    main()
