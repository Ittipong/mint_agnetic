"""Dataset loader for LangSmith evaluation.

Supports loading from:
  - Local JSONL file (langsmith_dataset.jsonl or custom path)
  - LangSmith API (by dataset_id)

Usage:
    from evaluation.loader import load_dataset

    # From local file
    examples = load_dataset("evaluation/my_dataset.jsonl")

    # From LangSmith by dataset_id
    examples = load_dataset(dataset_id="abc-123")
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from langsmith import Client


# ── Config ────────────────────────────────────────────────────────────────────

def _get_client() -> Client:
    """Get LangSmith client from .env.test or .env."""
    base = Path(__file__).parent.parent
    api_key = None

    for env_file in [base / ".env.test", base / ".env"]:
        if env_file.exists():
            env = dotenv_values(env_file)
            api_key = env.get("LANGSMITH_API_KEY") or env.get("LANGCHAIN_API_KEY")
            if api_key:
                break

    if not api_key:
        api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")

    if not api_key:
        raise RuntimeError("Missing LANGSMITH_API_KEY — set in .env.test, .env, or environment")

    return Client(api_key=api_key)


# ── Local JSONL loader ────────────────────────────────────────────────────────

def load_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load examples from a local JSONL file.

    Each line must be a JSON object with 'inputs' (required) and 'outputs' (optional).
    LangSmith format: {"inputs": {...}, "outputs": {...}}
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    examples = []
    linenum = 0

    with path.open() as f:
        for linenum, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {linenum}: {e}")

            if not isinstance(row, dict):
                raise ValueError(f"Line {linenum}: expected object, got {type(row).__name__}")

            if "inputs" not in row:
                raise ValueError(f"Line {linenum}: missing 'inputs' field")

            example = {"inputs": row["inputs"]}
            if "outputs" in row:
                example["outputs"] = row["outputs"]

            examples.append(example)

    if not examples:
        raise ValueError(f"No examples found in {path}")

    return examples


# ── LangSmith SDK loader ──────────────────────────────────────────────────────

def load_from_langsmith(dataset_id: str) -> list[dict[str, Any]]:
    """Load examples from LangSmith by dataset_id."""
    client = _get_client()
    examples = list(client.list_examples(dataset_id=dataset_id, limit=1000))
    return [{"inputs": e.inputs, "outputs": e.outputs} for e in examples]


def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    """Get dataset metadata from LangSmith."""
    client = _get_client()
    dataset = client.read_dataset(dataset_id=dataset_id)
    return {"id": dataset.id, "name": dataset.name, "description": dataset.description}


# ── Unified loader ────────────────────────────────────────────────────────────

DatasetSource = Literal["jsonl", "langsmith"]


def load_dataset(
    source: str | Path | None = None,
    *,
    dataset_id: str | None = None,
    default_jsonl: str | Path = "evaluation/langsmith_dataset.jsonl",
) -> list[dict[str, Any]]:
    """Load evaluation dataset.

    Priority:
      1. If dataset_id is provided → load from LangSmith API
      2. If source is provided (str/Path) → load from local JSONL file
      3. Otherwise → load from default_jsonl path

    Args:
        source: Local file path (str or Path object)
        dataset_id: LangSmith dataset ID (takes priority)
        default_jsonl: Default JSONL path when no source specified

    Returns:
        List of examples, each with 'inputs' and optional 'outputs'
    """
    if dataset_id:
        return load_from_langsmith(dataset_id)

    if source is None:
        source = Path(__file__).parent / default_jsonl
    else:
        source = Path(source)

    return load_from_jsonl(source)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluation/loader.py <file.jsonl>")
        print("  python evaluation/loader.py --dataset-id <id>")
        sys.exit(1)

    if sys.argv[1] == "--dataset-id":
        if len(sys.argv) < 3:
            print("Usage: python evaluation/loader.py --dataset-id <dataset_id>")
            sys.exit(1)
        examples = load_from_langsmith(sys.argv[2])
    else:
        examples = load_from_jsonl(Path(sys.argv[1]))

    print(f"Loaded {len(examples)} examples")
    for i, ex in enumerate(examples, 1):
        msg = ex["inputs"].get("messages", [{}])[-1].get("content", "")
        print(f"  {i}. {msg[:60]}{'...' if len(msg) > 60 else ''}")
