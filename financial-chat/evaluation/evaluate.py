"""Evaluation runner for financial-chat agent.

Runs agent from local JSONL and uses LangSmith aevaluate for proper experiment tracking.

Usage:
    python evaluate.py --file dataset.jsonl [--model <model-name>]
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import dotenv_values

# Change to project root so .env is found and src module is importable
_project_root = Path(__file__).parent.parent
os.chdir(_project_root)
sys.path.insert(0, str(_project_root))

# Load .env.test and set environment variables
_env_test = dotenv_values(_project_root / ".env.test")
for key, value in _env_test.items():
    if value and key not in os.environ:
        os.environ[key] = value


# ── Config ────────────────────────────────────────────────────────────────────

def _get_client():
    """Get LangSmith client from .env.test or .env."""
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LANGSMITH_API_KEY")
    from langsmith import Client
    return Client(api_key=api_key)


# ── Load local JSONL ─────────────────────────────────────────────────────────

def load_jsonl(file_path: str) -> list[dict]:
    """Load dataset from local JSONL file."""
    if not Path(file_path).is_absolute():
        file_path = str(Path(__file__).parent / file_path)
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}")
    return examples


# ── Run evaluation with aevaluate ─────────────────────────────────────────────

async def run_evaluation(
    file: str,
    model: str | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Run evaluation on dataset from local JSONL file using aevaluate."""
    client = _get_client()
    eval_run_id = str(uuid.uuid4())[:8]
    model_name = model or os.environ.get("MODEL", "default")
    model_short = model_name.split("/")[-1]
    dataset_name_prefix = os.environ.get("EVAL_DATASET_NAME", Path(file).stem)
    exp_prefix = experiment_name or f"{dataset_name_prefix}/{model_short}"

    # Set model override
    if model:
        os.environ["MODEL"] = model
        os.environ["BASE_URL"] = "https://openrouter.ai/api/v1"
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src.graph") or mod_name.startswith("src.llm") or mod_name.startswith("src.config"):
                del sys.modules[mod_name]
    elif not os.environ.get("BASE_URL"):
        os.environ["BASE_URL"] = "https://openrouter.ai/api/v1"

    # Load local dataset
    raw_examples = load_jsonl(file)
    if not raw_examples:
        raise ValueError(f"No examples found in {file}")

    print(f"Running evaluation: {exp_prefix}")
    print(f"  Model: {model_name}")
    print(f"  Examples: {len(raw_examples)}")
    print()

    # Get or create dataset
    dataset_name = dataset_name_prefix
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name, limit=1))
        if datasets:
            dataset = datasets[0]
            dataset_id = dataset.id
            print(f"  Dataset: {dataset_name} (ID: {dataset_id})")
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description=f"Eval dataset: {dataset_name}",
            )
            dataset_id = dataset.id
            print(f"  Dataset created: {dataset_id}")
    except Exception as e:
        print(f"  Warning: Could not access dataset: {e}")
        dataset_id = None
        dataset_name = None

    # Build target for aevaluate
    from src.graph.agent_graph import graph

    async def target(inputs: dict) -> dict:
        usecase_id = inputs.get("usecase_id", "unknown")
        user_id = inputs.get("user_id", "unknown")
        example_id = inputs.get("example_id", f"local-{eval_run_id}-{user_id[:8]}")

        import time
        start_time = time.time()

        thread_id = f"eval-{eval_run_id}-{user_id[:8]}"
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "model": model_name,
                "model_short": model_short,
                "eval_run_id": eval_run_id,
                "usecase_id": usecase_id,
                "example_id": example_id,
                "user_id": user_id,
            },
            "tags": ["eval", f"usecase:{usecase_id}"],
        }

        result = await graph.ainvoke(inputs, config)
        end_time = time.time()

        last_msg = result["messages"][-1]
        latency_ms = (end_time - start_time) * 1000

        # Extract step count
        step_count = 0
        for msg in result.get("messages", []):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                marker_start = "__CODEACT_INFO_START__"
                marker_end = "__CODEACT_INFO_END__"
                if marker_start in msg.content:
                    try:
                        start_idx = msg.content.index(marker_start) + len(marker_start)
                        end_idx = msg.content.index(marker_end, start_idx)
                        info = json.loads(msg.content[start_idx:end_idx])
                        step_count = info.get("codeact_total_steps", 0)
                    except (json.JSONDecodeError, ValueError):
                        pass

        return {
            "response": last_msg.content if hasattr(last_msg, "content") else str(last_msg),
            "usecase_id": usecase_id,
            "example_id": example_id,
            "latency_ms": round(latency_ms, 2),
            "step_count": step_count,
        }

    # Create evaluator
    from langsmith.evaluation import EvaluationResult

    def simple_evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> EvaluationResult:
        """Simple evaluator that extracts metrics from agent output."""
        latency_ms = outputs.get("latency_ms", 0)
        step_count = outputs.get("step_count", 0)
        usecase_id = outputs.get("usecase_id", inputs.get("usecase_id", "unknown"))

        # Simple scoring based on latency and steps
        latency_target = 10000  # 10s
        latency_score = 1.0 if latency_ms <= latency_target else 0.5

        if step_count <= 2:
            step_score = 1.0
        elif step_count <= 4:
            step_score = 0.7
        else:
            step_score = 0.5

        # Response quality check
        response = outputs.get("response", "")
        if not response or response.strip() == "":
            quality_score = 0.0
        elif any(err in response.lower() for err in ["error", "exception", "ขออภัย"]):
            quality_score = 0.0
        else:
            quality_score = 1.0

        overall = (quality_score * 0.5 + latency_score * 0.25 + step_score * 0.25)

        return EvaluationResult(
            key=f"eval_{model_short}",
            score=overall,
            comment=f"quality={quality_score:.2f} latency={latency_score:.2f} steps={step_count}",
            metadata={
                # Dashboard columns
                "model": model_name,
                "usecase_id": usecase_id,
                "latency_ms": latency_ms,
                "step_count": step_count,
                "quality_score": quality_score,
                "latency_score": latency_score,
                "step_score": step_score,
                # For reference
                "response_preview": response[:200] if response else "",
            },
        )

    # Run aevaluate using dataset_id (not iterables of dicts)
    eval_metadata = {
        "eval_run_id": eval_run_id,
        "dataset_name": dataset_name,
        "model": model_name,
        "model_short": model_short,
        "usecase_count": len(raw_examples),
    }

    if dataset_id:
        # First add examples to dataset
        print("  Adding examples to dataset...")
        example_ids = []
        for ex in raw_examples:
            inputs = ex.get("inputs", ex)
            outputs = ex.get("outputs", {})
            try:
                example = client.create_example(
                    inputs=inputs,
                    outputs=outputs if outputs else None,
                    dataset_id=dataset_id,
                )
                example_ids.append(example.id)
                print(f"    Added example: {example.id}")
            except Exception as e:
                print(f"    Failed to add example: {e}")

        # Run aevaluate with dataset_id
        try:
            result = await client.aevaluate(
                target,
                data=dataset_id,  # Pass dataset ID, not raw examples
                evaluators=[simple_evaluator],
                experiment_prefix=exp_prefix,
                max_concurrency=1,
                description=f"model={model_name} | file={file}",
                metadata=eval_metadata,
            )
            print(f"  aevaluate completed: {result}")
        except Exception as e:
            print(f"  Warning: aevaluate failed: {e}")
            result = None
    else:
        print("  Skipping aevaluate - no dataset_id")
        result = None

    # Summary
    print()
    print("=" * 60)
    print(f"Evaluation complete!")
    print(f"  Experiment: {exp_prefix}")
    print(f"  Total examples: {len(raw_examples)}")

    if dataset_id:
        print(f"  Dataset ID: {dataset_id}")
        print(f"  View at: https://smith.langchain.com/o/165e46ef-6bc8-4faa-a2b5-71dd48f60320/datasets/{dataset_id}/compare")

    return {
        "experiment_name": exp_prefix,
        "eval_run_id": eval_run_id,
        "model": model_name,
        "dataset_id": dataset_id,
        "total_examples": len(raw_examples),
        "result": str(result) if result else None,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run financial-chat evaluation from local JSONL")
    parser.add_argument("--file", type=str, required=True, help="Path to local JSONL file")
    parser.add_argument("--model", type=str, default=None, help="Model override (e.g. google/gemini-2.5-flash-lite)")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom experiment name prefix")
    args = parser.parse_args()

    try:
        results = asyncio.run(run_evaluation(
            file=args.file,
            model=args.model,
            experiment_name=args.experiment_name,
        ))
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
