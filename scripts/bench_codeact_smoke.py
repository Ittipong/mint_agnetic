"""1-model × 1-case smoke test to validate the bench runner before
spending tokens on the full 3×5 grid.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from bench_codeact_accuracy import (  # noqa: E402
    CASES, REACT_MODEL, _swap_llms, run_one,
)


async def main() -> None:
    from src.graph.agent_graph import build_graph  # noqa: E402
    graph = build_graph()

    model = "openai/gpt-5-nano"
    case = CASES[0]  # easy case
    print(f"SMOKE: model={model} case#{case['id']} ({case['label']})", flush=True)
    _swap_llms(model)
    res = await run_one(graph, model, case)
    print(f"\nelapsed={res['elapsed_ms']}ms err={res['error']}")
    print(f"final_reply ({len(res['final_reply'])} chars):\n{res['final_reply']!r}\n")
    print(f"tool_message ({len(res['tool_message'])} chars):\n{res['tool_message']!r}\n")


if __name__ == "__main__":
    asyncio.run(main())
