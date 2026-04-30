#!/usr/bin/env python3
"""Fetch traces from LangSmith and update comparison file."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from langsmith import Client


def load_env_test():
    """Load .env.test into environment."""
    env_path = Path(__file__).parent / ".env.test"
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value


def fetch_traces_for_project(client: Client, project_name: str, dataset_name: str, usecase_id: str, limit: int = 50):
    """Fetch traces from a LangSmith project."""
    try:
        runs = client.list_runs(
            project_name=project_name,
            limit=limit,
            filter=f'and(eq("usecase_id", "{usecase_id}"))',
        )
        traces = []
        for run in runs:
            trace = {
                "inputs": run.inputs,
                "outputs": run.outputs,
                "metadata": {
                    "LANGSMITH_PROJECT": os.environ.get("LANGSMITH_PROJECT", "mint-agentic"),
                    "LANGSMITH_TRACING": "true",
                    "dataset_name": dataset_name,
                    "usecase_id": usecase_id,
                    "model": run.metadata.get("model", "unknown") if run.metadata else "unknown",
                },
            }
            traces.append(trace)
        return traces
    except Exception as e:
        print(f"Error fetching traces for {project_name}: {e}")
        return []


def update_comparison_file(comparison_path: str):
    """Update the comparison file with traces from LangSmith."""
    load_env_test()

    client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))

    with open(comparison_path) as f:
        data = json.load(f)

    dataset_name = data.get("dataset_name", "dataset_poc")
    usecases = data.get("usecases", [])
    models = data.get("models", [])

    # Fetch traces for each model and usecase
    for model in models:
        for usecase in usecases:
            project_name = f"{dataset_name}/{usecase}/{model}"

            print(f"Fetching traces for: {project_name}")

            traces = fetch_traces_for_project(
                client, project_name, dataset_name, usecase
            )

            # Update details_matrix
            if model in data["details_matrix"] and usecase in data["details_matrix"][model]:
                data["details_matrix"][model][usecase]["traces"] = traces
                data["details_matrix"][model][usecase]["experiment_name"] = project_name

    # Write updated file
    with open(comparison_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated: {comparison_path}")


if __name__ == "__main__":
    # Default path if not provided
    default_path = Path(__file__).parent / "eval_results" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    if len(sys.argv) > 1:
        comparison_path = sys.argv[1]
    else:
        # Find the latest comparison file
        eval_results = Path(__file__).parent / "eval_results"
        files = sorted(eval_results.glob("comparison_*.json"))
        if files:
            comparison_path = str(files[-1])
        else:
            print("No comparison file found")
            sys.exit(1)

    print(f"Processing: {comparison_path}")
    update_comparison_file(comparison_path)
