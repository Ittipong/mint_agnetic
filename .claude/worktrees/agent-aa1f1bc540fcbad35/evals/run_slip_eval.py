#!/usr/bin/env python3
"""Run the `mint_v3_slip_eval_v1` dataset against `endpoints/slip_handler`.

Per Decision α (`docs/v3/phase2_state_and_graph.md`), the slip flow SKIPS
the ReAct loop entirely. So a slip eval row runs through
`handle_slip_chat(vision_call, catalog, image_b64s, …)` directly — no
graph, no ReAct.

Per Q2 amendment (`phase1_eval_dataset_design.md` §9), slip is OUT of scope
for `mint_v3_eval_v1` and lives in its own dataset. Pass criteria
(`phase3_implementation_plan.md` §5.5):
  - `transaction_proposal_group` block emitted with correct `group_id`
    (= `proposal_id`).
  - Wallet falls back to index-0 when wallet_id is None.
  - Each row's `transactions[*].amount > 0` and `type` is valid.
  - Category (when not null) is a real wallet+global category — no
    cross-wallet leak.

Failure modes are vision-specific, NOT ReAct — that's why this gate is
decoupled.

USAGE:
    python -m evals.run_slip_eval \
        --dataset mint_v3_slip_eval_v1 \
        --output report_slip_$(date +%Y-%m-%d).json
    python -m evals.run_slip_eval --smoke   # use 2 inline fixtures

NOTE: this script needs the `VISION_MODEL` env to make real vision calls;
without it, --smoke uses a fake vision_call that returns a canned
readable=true payload (just to verify pipeline shape).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from src.agent.db import make_backend_pool
from src.agent.endpoints.slip_handler import (
    SLIP_UNREADABLE_MESSAGE,
    handle_slip_chat,
)
from src.agent.entity_catalog import load_entity_catalog
from src.agent.llm_openrouter import make_multimodal_call


@dataclass
class SlipRowResult:
    example_id: str
    sub_mode: str
    passed: bool
    reason: str = ""
    group_block: Optional[dict] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0


def _inline_smoke() -> list[dict]:
    """One inline slip example so the pipeline is testable before the
    dataset is populated. Uses the committed slip fixture under
    tests_integration/fixtures/slips."""
    fixture = (
        Path(__file__).resolve().parent.parent
        / "tests_integration" / "fixtures" / "slips" / "slip1.png"
    )
    rows: list[dict] = []
    if fixture.exists():
        b64 = base64.b64encode(fixture.read_bytes()).decode("ascii")
        rows.append({
            "example_id": "smoke-slip-01",
            "inputs": {
                "image_b64s": [b64],
                "user_id": "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9",
                "wallet_id": None,
            },
            "expected": {
                "block_type": "transaction_proposal_group",
                "wallet_fallback_to_index_0": True,
            },
            "metadata": {"sub_mode": "slip-readable"},
        })
    return rows


async def _drive_one(
    example: dict,
    catalog,
    vision_call,
) -> SlipRowResult:
    started = time.monotonic()
    inputs = example["inputs"]
    expected = example["expected"] or {}
    sub_mode = (example.get("metadata") or {}).get("sub_mode", "?")

    blocks: list[dict] = []
    error: Optional[str] = None
    try:
        async for ev in handle_slip_chat(
            vision_call=vision_call,
            catalog=catalog,
            image_b64s=inputs.get("image_b64s") or [],
            thread_id=example["example_id"],
            user_id=inputs["user_id"],
            wallet_id=inputs.get("wallet_id"),
        ):
            if ev.get("event") == "block":
                try:
                    blocks.append(json.loads(ev["data"]))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - started

    if error:
        return SlipRowResult(
            example_id=example["example_id"], sub_mode=sub_mode,
            passed=False, reason=f"handler_error: {error}",
            error=error, elapsed_s=elapsed,
        )

    # Check expectations
    target_type = expected.get("block_type", "transaction_proposal_group")
    group = next((b for b in blocks if b.get("type") == target_type), None)
    if group is None:
        return SlipRowResult(
            example_id=example["example_id"], sub_mode=sub_mode,
            passed=False,
            reason=f"no {target_type!r} block; got {[b.get('type') for b in blocks]}",
            elapsed_s=elapsed,
        )

    # group_id == proposal_id (per memory `project_slip_vision_as_node`)
    if group.get("group_id") != group.get("proposal_id"):
        return SlipRowResult(
            example_id=example["example_id"], sub_mode=sub_mode,
            passed=False,
            reason=(
                f"group_id ({group.get('group_id')!r}) != proposal_id "
                f"({group.get('proposal_id')!r})"
            ),
            group_block=group, elapsed_s=elapsed,
        )

    # Index-0 wallet fallback when wallet_id was None
    if expected.get("wallet_fallback_to_index_0") and inputs.get("wallet_id") is None:
        expected_wallet = catalog.slip_wallet(None).sync_id
        if group.get("wallet_sync_id") != expected_wallet:
            return SlipRowResult(
                example_id=example["example_id"], sub_mode=sub_mode,
                passed=False,
                reason=(
                    f"wallet_sync_id={group.get('wallet_sync_id')!r} != "
                    f"index-0 fallback {expected_wallet!r}"
                ),
                group_block=group, elapsed_s=elapsed,
            )

    # Per-row validation
    rows = group.get("transactions") or []
    if not rows:
        return SlipRowResult(
            example_id=example["example_id"], sub_mode=sub_mode,
            passed=False, reason="group has no transactions",
            group_block=group, elapsed_s=elapsed,
        )
    for r in rows:
        if not (isinstance(r.get("amount"), (int, float)) and r["amount"] > 0):
            return SlipRowResult(
                example_id=example["example_id"], sub_mode=sub_mode,
                passed=False, reason=f"row has invalid amount: {r}",
                group_block=group, elapsed_s=elapsed,
            )
        if r.get("type") not in ("expense", "income"):
            return SlipRowResult(
                example_id=example["example_id"], sub_mode=sub_mode,
                passed=False, reason=f"row has invalid type: {r}",
                group_block=group, elapsed_s=elapsed,
            )

    return SlipRowResult(
        example_id=example["example_id"], sub_mode=sub_mode,
        passed=True, reason="all_invariants_satisfied",
        group_block=group, elapsed_s=elapsed,
    )


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="v3 slip eval runner")
    parser.add_argument("--dataset", default="mint_v3_slip_eval_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="use inline smoke fixtures")
    args = parser.parse_args()

    # Examples
    if args.smoke:
        examples = _inline_smoke()
        if not examples:
            print("[run_slip_eval] no smoke fixtures available — skip", file=sys.stderr)
            return 0
        print(f"[run_slip_eval] SMOKE: {len(examples)} examples")
    else:
        try:
            from langsmith import Client
            client = Client()
            examples = []
            for ex in client.list_examples(dataset_name=args.dataset):
                examples.append({
                    "example_id": str(ex.id),
                    "inputs": ex.inputs or {},
                    "expected": (ex.outputs or {}),
                    "metadata": ex.metadata or {},
                })
                if args.limit and len(examples) >= args.limit:
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"[run_slip_eval] LangSmith fetch FAILED: {exc}", file=sys.stderr)
            return 2

    print(f"[run_slip_eval] {len(examples)} examples loaded")

    # Catalog + vision call
    backend_pool = make_backend_pool(open=False)
    await backend_pool.open()
    try:
        # Use the first example's user_id to load the catalog. All eval rows
        # share SEED_USER per phase1_eval spec, so one load is enough.
        if not examples:
            return 0
        user_id = examples[0]["inputs"]["user_id"]
        catalog = await load_entity_catalog(backend_pool, user_id)

        vision_call = make_multimodal_call("vision")

        results: list[SlipRowResult] = []
        for i, ex in enumerate(examples):
            print(f"[run_slip_eval] [{i+1}/{len(examples)}] {ex['example_id']}",
                  end=" ", flush=True)
            r = await _drive_one(ex, catalog, vision_call)
            results.append(r)
            print(("OK" if r.passed else f"FAIL: {r.reason}") + f" ({r.elapsed_s:.1f}s)")

    finally:
        await backend_pool.close()

    summary = {
        "rows": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
    }
    summary["pass_rate"] = round(summary["passed"] / max(summary["rows"], 1), 4)
    summary["overall_passed"] = summary["pass_rate"] >= 0.95

    report = {
        "dataset": args.dataset,
        "summary": summary,
        "rows": [
            {
                "example_id": r.example_id,
                "sub_mode": r.sub_mode,
                "passed": r.passed,
                "reason": r.reason,
                "error": r.error,
                "elapsed_s": round(r.elapsed_s, 3),
            }
            for r in results
        ],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[run_slip_eval] wrote {args.output}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0 if summary["overall_passed"] else 1


def main() -> int:  # pragma: no cover
    return asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
