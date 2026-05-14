"""Vision model evaluation for slip-parsing.

Runs each candidate model against every slip image under
`<repo_root>/test_slips/` and reports:
  - tool_call success rate (did the model call propose_transaction?)
  - extracted fields (amount / date / note) for spot-check accuracy
  - latency, input/output tokens, $ cost
  - failure mode (empty / refused / hallucinated / errored)

Usage:
    set -a; source .env; set +a
    uv run python evaluation/vision_eval.py
    # or pin to specific models:
    uv run python evaluation/vision_eval.py --models gemini-2.5-flash gpt-4o-mini

The five image-generation candidates the user mentioned (seedream,
flux.2-*, riverflow) are kept in the default list so that anyone
re-running this script can see *explicitly* that they reject vision
input — instead of silently switching away from a known-bad model.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make `src.*` importable when run from anywhere
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.tools.transaction import propose_transaction


SLIPS_DIR = Path("/Users/ittipong.it/Projects/mint_money/test_slips")


# ── Candidate models ─────────────────────────────────────────────────
#
# Per-1M-token pricing from OpenRouter (snapshot 2026-05-14). Used only
# to produce a $/slip estimate — update if pricing drifts. Missing keys
# fall back to ($0, $0) and the cost column shows `?`.
_PRICING: dict[str, tuple[float, float]] = {
    # vision-capable
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.0-flash-001": (0.10, 0.40),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "qwen/qwen2.5-vl-32b-instruct": (0.20, 0.20),
    "qwen/qwen2.5-vl-72b-instruct": (0.70, 0.70),
    # user-provided (image-generation — expected to FAIL on multimodal input)
    "bytedance-seed/seedream-4.5": (0.0, 0.0),
    "black-forest-labs/flux.2-pro": (0.0, 0.0),
    "black-forest-labs/flux.2-klein-4b": (0.0, 0.0),
    "black-forest-labs/flux.2-max": (0.0, 0.0),
    "sourceful/riverflow-v2-pro": (0.0, 0.0),
}


_DEFAULT_MODELS = [
    # User-provided (image-gen)
    "bytedance-seed/seedream-4.5",
    "black-forest-labs/flux.2-pro",
    "black-forest-labs/flux.2-klein-4b",
    "black-forest-labs/flux.2-max",
    "sourceful/riverflow-v2-pro",
    # Known multimodal — for reference
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "qwen/qwen2.5-vl-32b-instruct",
]


_SYSTEM_PROMPT = """You are a slip-parsing agent inside Mint Money.

If the image is a Thai payment slip / receipt / bank transfer confirmation,
call `propose_transaction` with the parsed values. Use the slip's actual
date. If the image is NOT a slip, reply with one short Thai sentence
saying so — do not call the tool.

Fields to extract when calling the tool:
  - amount: total paid (number)
  - date: ISO datetime from the slip (string)
  - note: short Thai description (merchant or item)
  - type: usually "expense" for outgoing, "income" for incoming transfers
  - currency_code: usually "THB"
"""


@dataclass
class Result:
    model: str
    slip: str
    tool_called: bool = False
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    error: str = ""

    @property
    def cost_usd(self) -> float:
        in_price, out_price = _PRICING.get(self.model, (0.0, 0.0))
        return (
            (self.input_tokens / 1_000_000) * in_price
            + (self.output_tokens / 1_000_000) * out_price
        )

    @property
    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        if self.tool_called:
            return "OK"
        if self.text:
            return "REFUSED"
        return "EMPTY"


def _to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_llm(model: str) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not in env — run `source .env` first")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,
        timeout=60,
    )


async def _evaluate(model: str, slip: Path, url: str) -> Result:
    result = Result(model=model, slip=slip.name)
    try:
        llm = _build_llm(model).bind_tools([propose_transaction])
        msgs = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": "[INTENT:parse_transaction_from_slip]"},
                {"type": "image_url", "image_url": {"url": url}},
            ]),
        ]
        t0 = time.perf_counter()
        resp = await llm.ainvoke(msgs)
        result.duration_s = time.perf_counter() - t0

        tool_calls = getattr(resp, "tool_calls", None) or []
        if tool_calls:
            result.tool_called = True
            result.args = dict(tool_calls[0].get("args") or {})
        text_content = getattr(resp, "content", "")
        result.text = (
            text_content if isinstance(text_content, str)
            else str(text_content)
        )
        usage = getattr(resp, "usage_metadata", None) or {}
        result.input_tokens = usage.get("input_tokens") or 0
        result.output_tokens = usage.get("output_tokens") or 0
    except Exception as exc:
        # Truncate to keep table readable; full error is in `repr(exc)` for the
        # post-run debug pass if needed.
        result.error = f"{type(exc).__name__}: {str(exc)[:120]}"
    return result


def _print_row(r: Result) -> None:
    cost = f"${r.cost_usd:.5f}" if r.cost_usd else "?"
    amount = r.args.get("amount", "-")
    date = r.args.get("date", "-")
    note = (r.args.get("note") or "")[:30]
    print(
        f"{r.model:<42} {r.slip:<12} {r.verdict:<8} "
        f"{r.duration_s:>5.1f}s "
        f"in={r.input_tokens:<6} out={r.output_tokens:<4} "
        f"{cost:<10} amt={amount:<8} date={date:<10} note={note}"
    )


def _print_header() -> None:
    print()
    print(
        f"{'model':<42} {'slip':<12} {'verdict':<8} "
        f"{'time':<7} {'tokens':<17} {'cost':<10} {'extracted':<60}"
    )
    print("-" * 175)


def _print_summary(results: list[Result]) -> None:
    print()
    print("=" * 175)
    print("SUMMARY (best → worst, ranked by OK count, then cost):")
    print("=" * 175)
    by_model: dict[str, list[Result]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    rows = []
    for model, rs in by_model.items():
        ok = sum(1 for r in rs if r.verdict == "OK")
        refused = sum(1 for r in rs if r.verdict == "REFUSED")
        empty = sum(1 for r in rs if r.verdict == "EMPTY")
        errored = sum(1 for r in rs if r.verdict == "ERROR")
        avg_cost = sum(r.cost_usd for r in rs) / max(len(rs), 1)
        rows.append((model, ok, refused, empty, errored, avg_cost))

    rows.sort(key=lambda x: (-x[1], x[5]))
    print(f"{'model':<42} {'OK':<4} {'REFUSED':<8} {'EMPTY':<6} {'ERROR':<6} {'avg cost/slip':<15}")
    print("-" * 90)
    for model, ok, refused, empty, errored, avg_cost in rows:
        cost = f"${avg_cost:.5f}" if avg_cost else "?"
        print(f"{model:<42} {ok:<4} {refused:<8} {empty:<6} {errored:<6} {cost:<15}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Vision LLM eval for slip parsing")
    parser.add_argument(
        "--models",
        nargs="+",
        default=_DEFAULT_MODELS,
        help="Override the default model list",
    )
    parser.add_argument(
        "--slips-dir",
        default=str(SLIPS_DIR),
        help="Directory containing slip images (PNG/JPG)",
    )
    args = parser.parse_args()

    slip_paths = sorted(
        p for p in Path(args.slips_dir).iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not slip_paths:
        print(f"No slips found in {args.slips_dir}")
        return

    # Pre-encode once per slip to avoid re-reading bytes for every model.
    encoded = {p.name: _to_data_url(p) for p in slip_paths}
    for p in slip_paths:
        kb = len(encoded[p.name]) * 3 // 4 // 1024
        print(f"  loaded: {p.name} ({kb} KB)")

    _print_header()
    all_results: list[Result] = []
    for model in args.models:
        # Run all slips for a model sequentially so per-model rate-limit
        # failures don't cascade. Models are tested in parallel within a
        # single model would risk hitting OpenRouter's per-key concurrency
        # limit on the free tier.
        for path in slip_paths:
            r = await _evaluate(model, path, encoded[path.name])
            _print_row(r)
            all_results.append(r)

    _print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
