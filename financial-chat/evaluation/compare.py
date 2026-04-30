"""Compare model performance across datasets.

Uploads dataset, runs evaluation on all models in MODELS list,
generates comparison report and exports to CSV + JSON.

Usage:
    python evaluation/compare.py --file evaluation/dataset_poc.jsonl
    python evaluation/compare.py --file evaluation/dataset_poc.jsonl --dataset-name "my-dataset"
    python evaluation/compare.py --file evaluation/dataset_poc.jsonl --output-dir ./results
    python evaluation/compare.py --dataset-id <id> --dataset-name "existing-dataset"
"""

import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from dotenv import dotenv_values

# Load environment
_env_test = dotenv_values(_project_root / ".env.test")
for key, value in _env_test.items():
    if value and key not in os.environ:
        os.environ[key] = value


# ── Types ─────────────────────────────────────────────────────────────────────

ModelResult = dict[str, Any]  # {usecase_id: {"score": float, "reason": str, ...}}
ComparisonResult = dict[str, Any]  # Full comparison result


# ── Config ───────────────────────────────────────────────────────────────────

def _get_client():
    from langsmith import Client
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
        raise RuntimeError("Missing LANGSMITH_API_KEY — set in .env.test or .env")

    return Client(api_key=api_key)


def _get_models() -> list[str]:
    base = Path(__file__).parent.parent
    models_str = None

    for env_file in [base / ".env.test", base / ".env"]:
        if env_file.exists():
            env = dotenv_values(env_file)
            models_str = env.get("MODELS")
            if models_str:
                break

    if not models_str:
        models_str = os.environ.get("MODELS")

    if not models_str:
        return []

    try:
        models = json.loads(models_str)
        if isinstance(models, list):
            return [str(m) for m in models]
    except json.JSONDecodeError:
        pass

    if models_str:
        return [m.strip() for m in models_str.split(",") if m.strip()]

    return []


def _upload_dataset(jsonl_path: Path, dataset_name: str):
    from evaluation.dataset import upload_jsonl

    client = _get_client()
    existing = list(client.list_datasets(dataset_name=dataset_name, limit=1))

    if existing:
        ds = existing[0]
        examples = list(client.list_examples(dataset_id=ds.id, limit=1000))
        print(f"Using existing dataset '{dataset_name}' (ID: {ds.id}, {len(examples)} examples)")
        return ds.id, len(examples)

    print(f"Uploading {jsonl_path} as '{dataset_name}'...")
    result = upload_jsonl(jsonl_path, dataset_name)

    if not result["success"]:
        raise RuntimeError(f"Upload failed: {result['errors']}")

    print(f"Dataset ID: {result['dataset_id']}, Examples: {result['example_count']}")
    return result["dataset_id"], result["example_count"]


# ── Run single model evaluation ───────────────────────────────────────────────

async def _run_model_eval(
    dataset_id: str,
    dataset_name: str,
    model: str,
    eval_run_id: str,
    max_concurrency: int = 4,
) -> dict[str, ModelResult]:
    """Run evaluation for one model and collect per-usecase results."""
    from evaluation.evaluate import _build_target, _make_evaluator

    # Set model env
    os.environ["MODEL"] = model
    os.environ["BASE_URL"] = "https://openrouter.ai/api/v1"

    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src.graph") or mod_name.startswith("src.llm") or mod_name.startswith("src.config"):
            del sys.modules[mod_name]

    client = _get_client()
    model_short = model.split("/")[-1]

    # Load examples
    examples = list(client.list_examples(dataset_id=dataset_id, limit=1000))

    # Group by usecase
    from collections import defaultdict
    usecase_groups: dict[str, list] = defaultdict(list)
    for ex in examples:
        usecase_id = ex.inputs.get("usecase_id", "unknown") if ex.inputs else "unknown"
        usecase_groups[usecase_id].append(ex)

    results: dict[str, ModelResult] = {}

    for usecase_id, usecase_examples in usecase_groups.items():
        experiment_name = f"{dataset_name}/{usecase_id}/{model_short}"

        target = _build_target(model=model, dataset_name=dataset_name, eval_run_id=eval_run_id)

        eval_metadata = {
            "eval_run_id": eval_run_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "model": model,
            "model_short": model_short,
            "run_by": "compare",
            "run_timestamp": datetime.utcnow().isoformat() + "Z",
        }

        try:
            result = await client.aevaluate(
                target,
                data=iter(usecase_examples),
                evaluators=[_make_evaluator(model)],
                experiment_prefix=experiment_name,
                max_concurrency=max_concurrency,
                description=f"compare | usecase={usecase_id} | model={model}",
                metadata=eval_metadata,
            )

            # Extract metrics from runs in LangSmith after evaluation completes
            # With multi-metric evaluator, feedback_stats has keys: correctness, latency, code_execution, step_efficiency
            client = _get_client()
            example_ids = [e.id for e in usecase_examples]

            # Wait a moment for results to be stored
            import time
            time.sleep(2)

            runs = list(client.list_runs(reference_example=example_ids, limit=50))

            # Filter agent runs that match this model AND eval_run_id
            agent_runs = []
            for r in runs:
                if r.name == f"agent_{model_short}" and r.outputs and r.outputs.get("response"):
                    # Check metadata for eval_run_id match
                    run_meta = r.metadata or {}
                    if run_meta.get("eval_run_id") == eval_run_id:
                        agent_runs.append(r)

            if not agent_runs:
                print(f"    Warning: No agent runs found for eval_run_id={eval_run_id}")

            # Collect full trace data using client.get_run()
            raw_traces = []
            for run in agent_runs:
                try:
                    # Get full trace with all details
                    full_run = client.read_run(run.id)

                    # Extract inputs, outputs, metadata, and langsmith info
                    # Handle different attribute names
                    latency_ms = None
                    if hasattr(full_run, 'latency_s') and full_run.latency_s:
                        latency_ms = full_run.latency_s * 1000
                    elif hasattr(full_run, 'latency_ms') and full_run.latency_ms:
                        latency_ms = full_run.latency_ms

                    trace_data = {
                        "id": str(full_run.id),
                        "name": full_run.name,
                        "inputs": full_run.inputs,
                        "outputs": full_run.outputs,
                        "metadata": full_run.metadata,
                        "error": getattr(full_run, 'error', None),
                        "start_time": str(full_run.start_time) if getattr(full_run, 'start_time', None) else None,
                        "end_time": str(full_run.end_time) if getattr(full_run, 'end_time', None) else None,
                        "latency_ms": latency_ms,
                    }

                    # Get extra info from outputs
                    if full_run.outputs:
                        trace_data["response"] = full_run.outputs.get("response", "")
                        trace_data["step_count"] = full_run.outputs.get("step_count", 0)
                        trace_data["code_success"] = full_run.outputs.get("code_success", True)
                        trace_data["code_errors"] = full_run.outputs.get("code_errors", [])
                        trace_data["codeact_codes"] = full_run.outputs.get("codeact_codes", [])
                        trace_data["codeact_steps"] = full_run.outputs.get("codeact_steps", [])
                        trace_data["tools_used"] = full_run.outputs.get("tools_used", [])

                    raw_traces.append(trace_data)
                except Exception as e:
                    print(f"    Warning: Could not fetch full trace: {e}")

            # Store raw data for this usecase
            if raw_traces:
                first = raw_traces[0]
                results[usecase_id] = {
                    "traces": raw_traces,
                    "experiment_name": experiment_name,
                    "model": model,
                    "usecase_id": usecase_id,
                }
                print(f"  ✓ {usecase_id}: latency={(first.get('latency_ms') or 0):.0f}ms, steps={first.get('step_count') or 0}, codes={len(first.get('codeact_codes') or [])}")

        except Exception as e:
            print(f"  ✗ {usecase_id}: ERROR - {e}")
            results[usecase_id] = {
                "error": str(e),
                "traces": [],
                "experiment_name": experiment_name,
                "model": model,
                "usecase_id": usecase_id,
            }

    return results


# ── Compare all models ─────────────────────────────────────────────────────────

async def run_comparison(
    dataset_id: str,
    dataset_name: str,
    output_dir: Path | None = None,
    max_concurrency: int = 4,
) -> ComparisonResult:
    """Run comparison across all models."""
    import uuid

    models = _get_models()
    if not models:
        raise ValueError("No MODELS found in .env.test")

    eval_run_id = str(uuid.uuid4())[:8]
    print(f"\nComparing {len(models)} models on dataset '{dataset_name}'")
    print(f"Eval run ID: {eval_run_id}")
    print(f"Models: {models}\n")

    all_results: dict[str, dict[str, ModelResult]] = {}  # model -> {usecase_id -> result}
    usecases: set[str] = set()

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        results = await _run_model_eval(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            model=model,
            eval_run_id=eval_run_id,
            max_concurrency=max_concurrency,
        )

        all_results[model] = results
        usecases.update(results.keys())

    # Build comparison
    comparison = _build_comparison(all_results, list(usecases), models)

    # Export
    if output_dir:
        _export_comparison(comparison, output_dir)

    return comparison


def _build_comparison(
    all_results: dict[str, dict[str, ModelResult]],
    usecases: list[str],
    models: list[str],
) -> ComparisonResult:
    """Build structured comparison result with full traces."""
    # Raw data organized by model and usecase - keep full traces
    details_matrix: dict[str, dict[str, Any]] = {}

    for model, results in all_results.items():
        details_matrix[model] = {}
        for usecase_id, result in results.items():
            details_matrix[model][usecase_id] = result

    return {
        "dataset_name": all_results[models[0]][usecases[0]]["experiment_name"].split("/")[0] if usecases and models else "",
        "usecases": usecases,
        "models": models,
        "details_matrix": details_matrix,
    }


def _export_comparison(comparison: ComparisonResult, output_dir: Path):
    """Export comparison to CSV and JSON (raw data only)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # JSON export - full data including responses and code
    json_path = output_dir / f"comparison_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nExported JSON: {json_path}")

    # CSV export - raw metrics only
    csv_path = output_dir / f"comparison_{timestamp}.csv"
    usecases = comparison["usecases"]
    models = comparison["models"]
    details_matrix = comparison.get("details_matrix", {})

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["Model", "Usecase", "Latency (ms)", "Steps", "Code Success", "Code Errors", "Response Preview"])

        # Rows
        for model in models:
            for usecase in usecases:
                detail = details_matrix.get(model, {}).get(usecase, {})
                traces = detail.get("traces", [])
                if traces:
                    first_trace = traces[0]
                    lat = first_trace.get("latency_ms") or 0
                    steps = first_trace.get("step_count") or 0
                    code_success = first_trace.get("code_success") if first_trace.get("code_success") is not None else True
                    code_errors = first_trace.get("code_errors") or []
                    response_preview = (first_trace.get("response") or "")[:100]
                else:
                    lat = 0
                    steps = 0
                    code_success = None
                    code_errors = []
                    response_preview = ""

                writer.writerow([
                    model,
                    usecase,
                    f"{lat:.0f}",
                    steps,
                    code_success,
                    len(code_errors),
                    response_preview,
                ])

    print(f"Exported CSV: {csv_path}")

    # Print raw summary
    _print_summary(comparison)


def _print_summary(comparison: ComparisonResult):
    """Print comparison summary table to console (full traces)."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON - FULL TRACES")
    print(f"{'='*80}")
    print(f"Dataset: {comparison['dataset_name']}")
    print(f"Usecases: {len(comparison['usecases'])}")
    print(f"Models: {len(comparison['models'])}")
    print()

    details_matrix = comparison.get("details_matrix", {})

    # Summary table
    header = f"{'Model':<40}" + "".join(f"{u:<18}" for u in comparison["usecases"]) + f"{'Latency':>10} {'Steps':>6}"
    print(header)
    print("-" * len(header))

    for model in comparison["models"]:
        lats = []
        steps_list = []
        for u in comparison["usecases"]:
            detail = details_matrix.get(model, {}).get(u, {})
            traces = detail.get("traces", [])
            if traces:
                lats.append(traces[0].get("latency_ms") or 0)
                steps_list.append(traces[0].get("step_count") or 0)
            else:
                lats.append(0)
                steps_list.append(0)
        avg_lat = sum(lats) / len(lats) if lats else 0

        row = (f"{model:<40}" +
               "".join(f"{l:>17.0f} " for l in lats) +
               f"{avg_lat:>10.0f} " +
               f"{steps_list[0]:>6} " if steps_list else "      ")
        print(row)

    print()
    print(f"{'='*80}")
    print("RESPONSE COMPARISON (per model, per usecase)")
    print(f"{'='*80}")
    for model in comparison["models"]:
        print(f"\n{model}:")
        for usecase in comparison["usecases"]:
            detail = details_matrix.get(model, {}).get(usecase, {})
            traces = detail.get("traces", [])
            if traces:
                first_trace = traces[0]
                resp = first_trace.get("response") or ""
                latency = first_trace.get("latency_ms") or 0
                steps = first_trace.get("step_count") or 0
                code_success = first_trace.get("code_success") if first_trace.get("code_success") is not None else True
                code_errors = first_trace.get("code_errors") or []
                codes = first_trace.get("codeact_codes") or []

                print(f"  [{usecase}] latency={latency:.0f}ms, steps={steps}, code_success={code_success}, errors={len(code_errors)}")
                if codes:
                    print(f"    Code generated: {codes[0][:150]}...")
                preview = (resp or "(no response)")[:200].replace("\n", " ")
                print(f"    Response: {preview}...")
        print()
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare model performance on LangSmith evaluation")
    parser.add_argument("--file", "-f", type=str, help="Path to JSONL dataset file")
    parser.add_argument("--fetch-traces", type=str, metavar="FILE", help="Fetch traces for existing comparison JSON file")
    parser.add_argument("--dataset-id", type=str, help="Existing dataset ID")
    parser.add_argument("--dataset-name", "-n", type=str, help="Dataset name (derived from file if not set)")
    parser.add_argument("--output-dir", "-o", type=str, default="./eval_results", help="Output directory for exports")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Max concurrent evaluations")

    args = parser.parse_args()

    # Handle --fetch-traces mode
    if args.fetch_traces:
        fetch_traces_for_comparison(args.fetch_traces)
        return

    if not args.file and not args.dataset_id:
        parser.error("Must provide either --file or --dataset-id")

    output_dir = Path(args.output_dir)

    # Upload or use existing dataset
    if args.file:
        jsonl_path = _project_root / args.file
        if not jsonl_path.exists():
            print(f"ERROR: File not found: {jsonl_path}")
            sys.exit(1)

        dataset_name = args.dataset_name or jsonl_path.stem
        dataset_id, _ = _upload_dataset(jsonl_path, dataset_name)
    else:
        dataset_id = args.dataset_id
        dataset_name = args.dataset_name or "unknown"

    # Run comparison
    print("\n" + "="*60)
    print("STARTING MODEL COMPARISON")
    print("="*60)

    try:
        comparison = asyncio.run(
            run_comparison(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                output_dir=output_dir,
                max_concurrency=args.max_concurrency,
            )
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


def _fetch_with_retry(client, run_id: str, max_retries: int = 3, base_delay: float = 1.0):
    """Fetch run with retry logic for rate limiting."""
    import time
    import urllib.error

    for attempt in range(max_retries):
        try:
            return client.read_run(run_id)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Rate limited, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    return None


def fetch_traces_for_comparison(comparison_path: str | Path):
    """Fetch traces from LangSmith and update existing comparison file.

    Usage:
        python evaluation/compare.py --fetch-traces eval_results/comparison_20260428_054554.json
    """
    client = _get_client()
    comparison_path = Path(comparison_path)

    with comparison_path.open() as f:
        comparison = json.load(f)

    dataset_name = comparison.get("dataset_name", "dataset_poc")
    models = comparison.get("models", [])
    usecases = comparison.get("usecases", [])
    details_matrix = comparison.get("details_matrix", {})

    for model in models:
        model_short = model.split("/")[-1]
        for usecase in usecases:
            search_prefix = f"{dataset_name}/{usecase}/{model_short}"
            print(f"\nSearching projects: {search_prefix}-*")

            # Find matching projects
            matching_projects = []
            for proj in client.list_projects(name_contains=search_prefix, limit=10):
                matching_projects.append(proj)

            if not matching_projects:
                print(f"  ✗ No projects found matching: {search_prefix}")
                continue

            # Use the most recent project (last in list, sorted by name which includes timestamp)
            project = matching_projects[-1]
            experiment_name = project.name
            print(f"  Using project: {experiment_name}")

            try:
                # List runs for this experiment
                runs = list(client.list_runs(
                    project_name=experiment_name,
                    limit=50,
                ))

                traces = []
                for i, run in enumerate(runs):
                    # Get full trace details with retry
                    try:
                        full_run = _fetch_with_retry(client, run.id)

                        if full_run is None:
                            continue

                        # Extract latency
                        latency_ms = None
                        if hasattr(full_run, 'latency_s') and full_run.latency_s:
                            latency_ms = full_run.latency_s * 1000
                        elif hasattr(full_run, 'latency_ms') and full_run.latency_ms:
                            latency_ms = full_run.latency_ms

                        # Safely extract inputs
                        inputs = {}
                        if full_run.inputs:
                            if isinstance(full_run.inputs, dict):
                                inputs = full_run.inputs
                            else:
                                inputs = {"raw": str(full_run.inputs)}

                        # Safely extract outputs
                        outputs = {}
                        if full_run.outputs:
                            if isinstance(full_run.outputs, dict):
                                outputs = full_run.outputs
                            else:
                                outputs = {"raw": str(full_run.outputs)}

                        trace_data = {
                            "inputs": {
                                "inputs": inputs.get("inputs", {}) if isinstance(inputs.get("inputs"), dict) else inputs,
                                "usecase_id": inputs.get("usecase_id", usecase) if inputs else usecase,
                                "user_id": inputs.get("user_id", "") if inputs else "",
                                "example_id": inputs.get("example_id", "") if inputs else "",
                            },
                            "outputs": {
                                "response": outputs.get("response", "") if outputs else "",
                                "code_success": outputs.get("code_success", False) if outputs else False,
                                "code_errors": outputs.get("code_errors", []) if outputs else [],
                                "codeact_codes": outputs.get("codeact_codes", []) if outputs else [],
                                "codeact_steps": outputs.get("codeact_steps", []) if outputs else [],
                                "step_count": outputs.get("step_count", 0) if outputs else 0,
                                "tools_used": outputs.get("tools_used", []) if outputs else [],
                                "latency_ms": latency_ms,
                                "example_id": inputs.get("example_id", "") if inputs else "",
                                "usecase_id": usecase,
                            },
                            "metadata": {
                                "LANGSMITH_PROJECT": os.environ.get("LANGSMITH_PROJECT", "mint-agentic"),
                                "LANGSMITH_TRACING": "true",
                                "dataset_name": dataset_name,
                                "usecase_id": usecase,
                                "model": model,
                                "model_short": model_short,
                                "run_timestamp": datetime.utcnow().isoformat() + "Z",
                            },
                            "langsmith": {
                                "organization": {"name": "Personal"},
                                "workspace": {"name": "Workspace 1"},
                                "tracing_project": {"name": experiment_name},
                            },
                        }
                        traces.append(trace_data)

                        # Progress indicator every 5 traces
                        if (i + 1) % 5 == 0:
                            print(f"  Fetched {i + 1}/{len(runs)} traces...")

                    except Exception as e:
                        print(f"  Warning: Could not fetch trace {run.id}: {e}")
                        continue

                # Update details_matrix
                if model in details_matrix and usecase in details_matrix[model]:
                    details_matrix[model][usecase]["traces"] = traces
                    details_matrix[model][usecase]["experiment_name"] = experiment_name
                    if traces:
                        first = traces[0]
                        print(f"  ✓ Got {len(traces)} traces, latency={first.get('outputs', {}).get('latency_ms', 0):.0f}ms")
                    else:
                        print(f"  ✗ No traces found")
                else:
                    print(f"  Warning: Model/usecase not found in comparison")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                if model in details_matrix and usecase in details_matrix[model]:
                    details_matrix[model][usecase]["error"] = str(e)

    # Write updated comparison
    try:
        with comparison_path.open("w") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"JSON dump error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nUpdated: {comparison_path}")
    try:
        _print_summary(comparison)
    except Exception as e:
        print(f"Print summary error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
