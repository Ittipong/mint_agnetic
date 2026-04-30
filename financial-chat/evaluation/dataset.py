"""Dataset management for LangSmith evaluation.

Supports:
  - Validating JSONL files (LangSmith format)
  - Creating / updating datasets on LangSmith

Usage:
    # Validate a JSONL file
    python evaluation/dataset.py validate evaluation/my_dataset.jsonl

    # Upload to LangSmith (creates new dataset)
    python evaluation/dataset.py upload evaluation/my_dataset.jsonl --name "My Dataset"

    # Replace all examples in an existing dataset
    python evaluation/dataset.py update <dataset_id> evaluation/my_dataset.jsonl
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from langsmith import Client


# ── Schema ────────────────────────────────────────────────────────────────────

DATASET_INPUT_SCHEMA = {
    "type": "object",
    "required": ["user_id", "messages"],
    "properties": {
        "user_id": {"type": "string"},
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    },
}

DATASET_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reference": {"type": "object"},
    },
}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_row(row: dict, linenum: int) -> list[str]:
    """Validate a single JSONL row. Returns list of error messages."""
    errors = []

    if "inputs" not in row:
        errors.append(f"line {linenum}: missing 'inputs' field")
        return errors  # can't validate further

    inputs = row["inputs"]

    if not isinstance(inputs, dict):
        errors.append(f"line {linenum}: 'inputs' must be an object")
        return errors

    if "user_id" not in inputs:
        errors.append(f"line {linenum}: missing 'inputs.user_id'")
    elif not isinstance(inputs["user_id"], str) or not inputs["user_id"]:
        errors.append(f"line {linenum}: 'inputs.user_id' must be a non-empty string")

    if "messages" not in inputs:
        errors.append(f"line {linenum}: missing 'inputs.messages'")
    elif not isinstance(inputs["messages"], list):
        errors.append(f"line {linenum}: 'inputs.messages' must be an array")
    else:
        for i, msg in enumerate(inputs["messages"]):
            if not isinstance(msg, dict):
                errors.append(f"line {linenum}: messages[{i}] must be an object")
                continue
            if "role" not in msg:
                errors.append(f"line {linenum}: messages[{i}] missing 'role'")
            if "content" not in msg:
                errors.append(f"line {linenum}: messages[{i}] missing 'content'")
            elif not isinstance(msg["content"], str):
                errors.append(f"line {linenum}: messages[{i}] 'content' must be a string")

    if "outputs" in row:
        outputs = row["outputs"]
        if not isinstance(outputs, dict):
            errors.append(f"line {linenum}: 'outputs' must be an object")

    return errors


def validate_jsonl(path: Path) -> tuple[bool, list[str]]:
    """Validate a JSONL file. Returns (is_valid, errors)."""
    if not path.exists():
        return False, [f"File not found: {path}"]

    errors = []
    linenum = 0

    with path.open() as f:
        for linenum, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {linenum}: invalid JSON — {e}")
                continue

            if not isinstance(row, dict):
                errors.append(f"line {linenum}: root must be an object, got {type(row).__name__}")
                continue

            errors.extend(validate_row(row, linenum))

    return len(errors) == 0, errors


# ── LangSmith SDK ─────────────────────────────────────────────────────────────


def _get_client() -> Client:
    """Get LangSmith client from env vars or .env files."""
    from dotenv import dotenv_values

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
        raise RuntimeError("Missing LANGSMITH_API_KEY / LANGCHAIN_API_KEY — set in .env.test or .env")

    return Client(api_key=api_key)


def _parse_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Parse JSONL file into examples list."""
    examples = []
    with jsonl_path.open() as f:
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

            example = {"inputs": row.get("inputs", {})}
            if "outputs" in row:
                example["outputs"] = row["outputs"]

            examples.append(example)
    return examples


def create_dataset(name: str, description: str = "") -> str:
    """Create a new dataset. Returns dataset ID."""
    client = _get_client()
    dataset = client.create_dataset(dataset_name=name, description=description)
    return dataset.id


def upload_examples(dataset_id: str, jsonl_path: Path) -> int:
    """Upload examples from a JSONL file to an existing dataset. Returns count."""
    examples = _parse_jsonl(jsonl_path)
    if not examples:
        raise ValueError("No examples found in JSONL file")

    client = _get_client()
    client.create_examples(dataset_id=dataset_id, examples=examples)
    return len(examples)


def delete_all_examples(dataset_id: str) -> int:
    """Delete all examples from a dataset. Returns count of deleted examples."""
    client = _get_client()
    examples = list(client.list_examples(dataset_id=dataset_id, limit=1000))
    if not examples:
        return 0

    example_ids = [e.id for e in examples if e.id]
    if example_ids:
        client.delete_examples(example_ids=example_ids)
    return len(example_ids)


def replace_dataset(dataset_id: str, jsonl_path: Path) -> dict[str, Any]:
    """Replace all examples in an existing dataset with new ones from JSONL."""
    is_valid, errors = validate_jsonl(jsonl_path)
    if not is_valid:
        return {
            "success": False,
            "deleted_count": 0,
            "uploaded_count": 0,
            "errors": errors,
        }

    deleted = delete_all_examples(dataset_id)
    uploaded = upload_examples(dataset_id, jsonl_path)

    return {
        "success": True,
        "deleted_count": deleted,
        "uploaded_count": uploaded,
        "errors": [],
    }


def upload_jsonl(jsonl_path: Path, name: str, description: str = "") -> dict[str, Any]:
    """Upload a JSONL file as a new dataset to LangSmith."""
    is_valid, errors = validate_jsonl(jsonl_path)
    if not is_valid:
        return {
            "success": False,
            "dataset_id": None,
            "example_count": 0,
            "errors": errors,
        }

    dataset_id = create_dataset(name, description)
    count = upload_examples(dataset_id, jsonl_path)

    return {
        "success": True,
        "dataset_id": dataset_id,
        "example_count": count,
        "errors": [],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluation/dataset.py validate <file.jsonl>")
        print("  python evaluation/dataset.py upload <file.jsonl> --name 'Dataset Name' [--desc 'Description']")
        print("  python evaluation/dataset.py update <dataset_id> <file.jsonl>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "validate":
        if len(sys.argv) < 3:
            print("Usage: python evaluation/dataset.py validate <file.jsonl>")
            sys.exit(1)
        path = Path(sys.argv[2])
        is_valid, errors = validate_jsonl(path)
        if is_valid:
            print(f"OK: {path} is valid")
        else:
            print(f"FAILED: {path} has {len(errors)} error(s):")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)

    elif command == "upload":
        if len(sys.argv) < 3:
            print("Usage: python evaluation/dataset.py upload <file.jsonl> --name 'Dataset Name'")
            sys.exit(1)

        import argparse

        parser = argparse.ArgumentParser(description="Upload JSONL to LangSmith")
        parser.add_argument("jsonl_path", type=Path)
        parser.add_argument("--name", required=True, help="Dataset name in LangSmith")
        parser.add_argument("--desc", default="", help="Dataset description")
        args = parser.parse_args(sys.argv[2:])

        print(f"Uploading {args.jsonl_path} as '{args.name}'...")
        try:
            result = upload_jsonl(args.jsonl_path, args.name, args.desc)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        if result["success"]:
            print(f"Success! Dataset ID: {result['dataset_id']}")
            print(f"Examples uploaded: {result['example_count']}")
        else:
            print(f"Validation failed with {len(result['errors'])} error(s):")
            for e in result["errors"]:
                print(f"  - {e}")
            sys.exit(1)

    elif command == "update":
        if len(sys.argv) < 4:
            print("Usage: python evaluation/dataset.py update <dataset_id> <file.jsonl>")
            sys.exit(1)
        dataset_id = sys.argv[2]
        jsonl_path = Path(sys.argv[3])

        print(f"Updating dataset {dataset_id} with {jsonl_path}...")
        try:
            result = replace_dataset(dataset_id, jsonl_path)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        if result["success"]:
            print(f"Success! Deleted: {result['deleted_count']} examples")
            print(f"Uploaded: {result['uploaded_count']} examples")
        else:
            print(f"Validation failed with {len(result['errors'])} error(s):")
            for e in result["errors"]:
                print(f"  - {e}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
