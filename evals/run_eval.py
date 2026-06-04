#!/usr/bin/env python3
"""Run the `mint_v3_eval_v1` dataset against the v3 ReAct + CodeAct graph.

This is the **eval gate** the v3 release MUST clear. Per
`docs/v3/phase3_implementation_plan.md` §5.3 + `phase1_eval_dataset_design.md` §5,
the gate scores 3 INDEPENDENT properties per row:

  - Numerical (≥ 99% per mode): every number in the answer text has a
    match (1% tolerance) in the row's `ExpectedNumber.sql` ground truth.
  - Block-shape (≥ 95% per mode): every block in `ExpectedBlock.blocks`
    appears in the SSE stream in order, with matching `match_fields`.
  - LLM-judge (≥ 90% per mode): the row's `text_rubric` graded by
    claude-4.7-sonnet (per Q3) returns `passed=true`.

Per-mode AND-gate semantics: dataset passes only if ALL modes pass ALL 3
gates. A mode at 89% LLM-judge fails the whole release even if numerical
and block-shape are at 100%.

Pipeline:

    1. CLI parse  (--dataset / --split / --limit / --judge-model / --output)
    2. Fetch examples from LangSmith via the langsmith SDK.
    3. seed_db ensures deterministic test data is in place (idempotent).
    4. For each example:
        a. build_graph() — fresh graph, in-memory checkpointer per row so
           threads are isolated.
        b. invoke with example.input — capture final state + the answer
           text reassembled from streamed messages.
        c. score 3 gates against example.expected.
    5. aggregate per-mode pass rates → exit 0 if ALL gates pass, else 1.

USAGE:
    python -m evals.run_eval \
        --dataset mint_v3_eval_v1 \
        --split regression \
        --judge-model anthropic/claude-sonnet-4 \
        --output report_regression_$(date +%Y-%m-%d).json

PHASE STATUS:
    The dataset `mint_v3_eval_v1` is Phase 1 deliverable — Phase 1 must
    populate it via `build_dataset.py` before this script can do useful
    work. Until then, `--smoke` uses 5 inline example fixtures from
    `phase1_examples_seed.md` so the eval pipeline itself is testable
    independent of LangSmith dataset availability.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env BEFORE any agent module reads OS env (DATABASE_URL, model env, etc).
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import build_graph
from src.agent.utils.store_factory import build_store
from src.agent.validators.numerical import (
    extract_money_numbers,
    validate_numerical_response,
)
from src.agent.db import ProposalRepo, make_backend_pool
from evals.judge import DEFAULT_JUDGE_MODEL, judge_row


# ─────────────────────────────────────────────────────────────────────────────
# Mode + gate thresholds (per phase1_eval_dataset_design.md §5)
# ─────────────────────────────────────────────────────────────────────────────

NUMERICAL_THRESHOLD = 0.99
BLOCK_THRESHOLD = 0.95
JUDGE_THRESHOLD = 0.90

MODES = {"ADD", "ADVISOR", "ANALYST", "CHITCHAT", "EMOTIONAL"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-row result + report aggregation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GateResult:
    passed: bool
    reason: str = ""


@dataclass
class RowResult:
    example_id: str
    mode: str
    sub_mode: str
    numerical: GateResult
    block_shape: GateResult
    llm_judge: GateResult
    answer_text: str = ""
    emitted_block_types: list[str] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_s: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Gate scorers
# ─────────────────────────────────────────────────────────────────────────────


def score_numerical(
    answer_text: str,
    expected_numbers: list[dict],
    backend_conn,
) -> GateResult:
    """Compare answer numbers to expected SQL truth.

    `expected_numbers` carries `{label, sql, sql_params, rel_tol,
    must_appear_in}` (per phase1_eval_dataset_design schema). For each
    expected number, we run the SQL against BACKEND_DATABASE_URL and check
    the answer contains a number within rel_tol of the SQL result.

    Empty expected_numbers → trivially passes (the phase1 spec calls this
    out for CHITCHAT/EMOTIONAL rows that don't carry numerical assertions).
    """
    if not expected_numbers:
        return GateResult(passed=True, reason="no_numbers_expected")

    answer_nums = extract_money_numbers(answer_text)
    if not answer_nums:
        return GateResult(passed=False, reason="no_numbers_in_answer")

    for spec in expected_numbers:
        label = spec.get("label", "?")
        sql = spec.get("sql")
        params = spec.get("sql_params", {})
        rel_tol = Decimal(str(spec.get("rel_tol", 0.01)))
        if not sql:
            return GateResult(passed=False, reason=f"{label}: missing sql")

        with backend_conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        if row is None or row[0] is None:
            return GateResult(passed=False, reason=f"{label}: empty sql truth")
        truth = Decimal(str(row[0]))

        matched = False
        for n in answer_nums:
            if truth == 0:
                matched = (n == 0)
            else:
                matched = abs(n - truth) / abs(truth) <= rel_tol
            if matched:
                break
        if not matched:
            return GateResult(
                passed=False,
                reason=f"{label}: truth={truth} not within {rel_tol*100}% of {answer_nums}",
            )
    return GateResult(passed=True, reason="all_numbers_matched")


_UUID_PLACEHOLDER = "<any-uuid>"


def _value_matches(actual: Any, expected: Any) -> bool:
    """Deep-equal with `<any-uuid>` placeholder support."""
    if isinstance(expected, str) and expected == _UUID_PLACEHOLDER:
        return isinstance(actual, str) and len(actual) >= 8
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(_value_matches(actual.get(k), v) for k, v in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _value_matches(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def _get_nested(d: dict, path: str) -> Any:
    """Fetch a dotted path from a nested dict ('transaction.amount')."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def score_block_shape(
    actual_blocks: list[dict],
    expected_blocks: list[dict],
) -> GateResult:
    """Check expected blocks appear in actual stream IN ORDER.

    `expected_blocks` is a list of `{block: {...}, match_fields: [...]}`
    where `match_fields` selects which keys to deep-equal. Default
    `match_fields` = every key in `expected_block.block`.

    Tolerant on EXTRA blocks: a stream that emits more blocks than expected
    passes as long as the expected ones appear in order. Strict on
    ORDER: out-of-order = fail (e.g. `transaction_proposal` MUST appear
    after `discard_proposal` in the atomic-discard sub-mode).
    """
    if not expected_blocks:
        return GateResult(passed=True, reason="no_blocks_expected")

    cursor = 0
    for i, exp in enumerate(expected_blocks):
        exp_block = exp.get("block") or {}
        fields = exp.get("match_fields") or list(exp_block.keys())
        # Walk actual_blocks from `cursor` looking for the next match.
        found_at = -1
        for j in range(cursor, len(actual_blocks)):
            actual = actual_blocks[j]
            if all(
                _value_matches(_get_nested(actual, f), _get_nested(exp_block, f))
                for f in fields
            ):
                found_at = j
                break
        if found_at < 0:
            return GateResult(
                passed=False,
                reason=(
                    f"expected block #{i} (type={exp_block.get('type')!r}) not "
                    f"found at-or-after position {cursor}; got types="
                    f"{[b.get('type') for b in actual_blocks]}"
                ),
            )
        cursor = found_at + 1
    return GateResult(passed=True, reason="all_blocks_matched_in_order")


async def score_llm_judge(
    user_question: str,
    answer_text: str,
    rubric: str,
    judge_model: Optional[str],
) -> GateResult:
    """Grade via the locked judge model (claude-4.7-sonnet per Q3)."""
    if not rubric:
        return GateResult(passed=True, reason="no_rubric")
    try:
        v = await judge_row(user_question, answer_text, rubric, model=judge_model)
        return GateResult(passed=bool(v["passed"]), reason=v.get("reason", ""))
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            passed=False,
            reason=f"judge_error: {type(exc).__name__}: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph driver — run ONE example through a fresh in-memory graph
# ─────────────────────────────────────────────────────────────────────────────


async def run_one_row(
    example: dict,
    graph,
    judge_model: Optional[str],
    backend_conn,
) -> RowResult:
    """Drive the graph with `example.inputs`, score gates, return RowResult."""
    inputs = example["inputs"]
    expected = example["outputs"] or {}
    metadata = example.get("metadata") or {}
    mode = metadata.get("mode", "?")
    sub_mode = metadata.get("sub_mode", "?")

    started = time.monotonic()
    error: Optional[str] = None
    answer_text = ""
    emitted_block_types: list[str] = []
    final_state: dict[str, Any] = {}

    try:
        # Build initial state. thread_prefix (multi-turn) is REPLAYED
        # before the test turn by sending them in `messages`. mobile
        # sends one turn at a time but for eval purposes we accept the
        # full prefix as the seed message history.
        prefix_msgs = []
        for role, content in (inputs.get("thread_prefix") or []):
            if role == "user":
                prefix_msgs.append(HumanMessage(content))
            elif role == "assistant":
                prefix_msgs.append(AIMessage(content))
        init = {
            "user_id": inputs["user_id"],
            "thread_id": example["example_id"],
            "messages": prefix_msgs + [HumanMessage(inputs["input_text"])],
            "default_wallet_sync_id": inputs.get("wallet_id") or "",
        }
        config = {
            "configurable": {"thread_id": example["example_id"]},
            "recursion_limit": 25,
        }
        final_state = await graph.ainvoke(init, config=config)

        # Final answer = the LAST AIMessage's content
        ai_msgs = [m for m in final_state.get("messages") or [] if isinstance(m, AIMessage)]
        if ai_msgs:
            content = ai_msgs[-1].content
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            answer_text = str(content or "")

        # Blocks for shape gate
        actual_blocks = list(final_state.get("emitted_blocks_this_turn") or [])
        emitted_block_types = [b.get("type") for b in actual_blocks if isinstance(b, dict)]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        actual_blocks = []

    elapsed = time.monotonic() - started

    if error:
        return RowResult(
            example_id=example["example_id"], mode=mode, sub_mode=sub_mode,
            numerical=GateResult(passed=False, reason=f"graph_error: {error}"),
            block_shape=GateResult(passed=False, reason=f"graph_error: {error}"),
            llm_judge=GateResult(passed=False, reason=f"graph_error: {error}"),
            answer_text="", emitted_block_types=[],
            error=error, elapsed_s=elapsed,
        )

    # Score 3 gates
    numerical = score_numerical(
        answer_text, expected.get("numbers") or [], backend_conn,
    )
    block_shape = score_block_shape(
        actual_blocks, expected.get("blocks") or [],
    )
    llm_judge = await score_llm_judge(
        inputs["input_text"], answer_text,
        expected.get("text_rubric") or "", judge_model,
    )

    return RowResult(
        example_id=example["example_id"],
        mode=mode, sub_mode=sub_mode,
        numerical=numerical, block_shape=block_shape, llm_judge=llm_judge,
        answer_text=answer_text, emitted_block_types=emitted_block_types,
        elapsed_s=elapsed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader — LangSmith OR inline smoke set
# ─────────────────────────────────────────────────────────────────────────────


def _smoke_examples() -> list[dict]:
    """5 inline examples from phase1_examples_seed.md — enough to verify the
    pipeline (graph build + scoring + judge call) before the full dataset is
    populated. Each row uses the same TypedDict schema as LangSmith examples."""
    return [
        {
            "example_id": "smoke-add-01",
            "inputs": {
                "input_text": "เพิ่ม 250 กาแฟ",
                "thread_prefix": [],
                "user_id": "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9",
                "wallet_id": None,
            },
            "outputs": {
                "blocks": [{
                    "block": {"type": "transaction_proposal"},
                    "match_fields": ["type"],
                }],
                "numbers": [],
                "text_rubric": (
                    "PASS: reply acknowledges saving 250 baht for coffee, in Thai, "
                    "mentions both the amount AND กาแฟ. FAIL: contradictory amount; English."
                ),
                "tool_calls": [],
            },
            "metadata": {"mode": "ADD", "sub_mode": "ADD-text-simple"},
        },
        {
            "example_id": "smoke-chitchat-01",
            "inputs": {
                "input_text": "สวัสดี",
                "thread_prefix": [],
                "user_id": "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9",
                "wallet_id": None,
            },
            "outputs": {
                "blocks": [],
                "numbers": [],
                "text_rubric": "PASS: friendly Thai greeting. FAIL: cold or non-Thai.",
                "tool_calls": [],
            },
            "metadata": {"mode": "CHITCHAT", "sub_mode": "small-talk"},
        },
        {
            "example_id": "smoke-emotional-01",
            "inputs": {
                "input_text": "เครียดเรื่องเงินมาก",
                "thread_prefix": [],
                "user_id": "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9",
                "wallet_id": None,
            },
            "outputs": {
                "blocks": [],
                "numbers": [],
                "text_rubric": (
                    "PASS: empathic Thai reply, no fabricated numbers, brief "
                    "actionable suggestion. FAIL: leads with numbers; cold tone."
                ),
                "tool_calls": [],
            },
            "metadata": {"mode": "EMOTIONAL", "sub_mode": "pure-emotional"},
        },
    ]


def _load_from_langsmith(
    dataset_name: str,
    split: Optional[str],
    limit: Optional[int],
) -> list[dict]:
    """Fetch dataset examples via langsmith SDK. Returns a list shaped like
    the smoke examples (inputs / outputs / metadata)."""
    from langsmith import Client

    client = Client()
    examples = []
    # `as_of` / split semantics: spec uses `metadata.split` tag for filtering.
    # If split is set, we filter by metadata key. Otherwise pull all.
    iter_kwargs: dict = {"dataset_name": dataset_name}
    if split:
        iter_kwargs["metadata"] = {"split": split}
    for ex in client.list_examples(**iter_kwargs):
        row = {
            "example_id": str(ex.id),
            "inputs": ex.inputs or {},
            "outputs": ex.outputs or {},
            "metadata": ex.metadata or {},
        }
        examples.append(row)
        if limit is not None and len(examples) >= limit:
            break
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Report aggregation + exit code
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_report(rows: list[RowResult]) -> dict:
    """Per-mode pass rates + per-gate AND-gate verdict."""
    by_mode: dict[str, dict] = {}
    for r in rows:
        m = by_mode.setdefault(r.mode, {"rows": 0, "num_pass": 0, "block_pass": 0, "judge_pass": 0})
        m["rows"] += 1
        m["num_pass"] += int(r.numerical.passed)
        m["block_pass"] += int(r.block_shape.passed)
        m["judge_pass"] += int(r.llm_judge.passed)

    summary: dict = {"per_mode": {}, "overall_passed": True}
    for mode, m in by_mode.items():
        rows_n = m["rows"]
        num_rate = (m["num_pass"] / rows_n) if rows_n else 0
        block_rate = (m["block_pass"] / rows_n) if rows_n else 0
        judge_rate = (m["judge_pass"] / rows_n) if rows_n else 0
        mode_passed = (
            num_rate >= NUMERICAL_THRESHOLD
            and block_rate >= BLOCK_THRESHOLD
            and judge_rate >= JUDGE_THRESHOLD
        )
        summary["per_mode"][mode] = {
            "rows": rows_n,
            "numerical_rate": round(num_rate, 4),
            "block_rate": round(block_rate, 4),
            "judge_rate": round(judge_rate, 4),
            "passed": mode_passed,
        }
        if not mode_passed:
            summary["overall_passed"] = False
    return summary


def _row_to_dict(r: RowResult) -> dict:
    return {
        "example_id": r.example_id,
        "mode": r.mode, "sub_mode": r.sub_mode,
        "numerical": {"passed": r.numerical.passed, "reason": r.numerical.reason},
        "block_shape": {"passed": r.block_shape.passed, "reason": r.block_shape.reason},
        "llm_judge": {"passed": r.llm_judge.passed, "reason": r.llm_judge.reason},
        "answer_text": r.answer_text[:500],
        "emitted_block_types": r.emitted_block_types,
        "error": r.error,
        "elapsed_s": round(r.elapsed_s, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────


_interrupted = False


def _install_sigint_handler() -> None:
    """Translate Ctrl-C into a clean shutdown — finish the current row, then
    write the partial report. Avoids losing 20 min of eval work on stray ^C."""
    def handler(signum, frame):  # noqa: ARG001
        global _interrupted
        _interrupted = True
        print("\n[run_eval] SIGINT received — finishing current row then exiting.")
    signal.signal(signal.SIGINT, handler)


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="v3 eval gate runner")
    parser.add_argument("--dataset", default="mint_v3_eval_v1")
    parser.add_argument("--split", default=None,
                        help="optional metadata.split filter (e.g. regression / dev)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap N examples (smoke / debugging)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--output", default=None,
                        help="path to write report JSON; stdout if omitted")
    parser.add_argument("--smoke", action="store_true",
                        help="use inline smoke examples (no LangSmith fetch)")
    args = parser.parse_args()

    _install_sigint_handler()

    # 1. Fetch examples
    if args.smoke:
        examples = _smoke_examples()
        print(f"[run_eval] SMOKE: {len(examples)} inline examples")
    else:
        print(f"[run_eval] fetching dataset={args.dataset} split={args.split}")
        try:
            examples = _load_from_langsmith(args.dataset, args.split, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[run_eval] LangSmith fetch FAILED: {exc}", file=sys.stderr)
            return 2
        print(f"[run_eval] fetched {len(examples)} examples")

    if not examples:
        print("[run_eval] no examples to run — exiting with success (vacuously)")
        return 0

    # 2. Build graph + open backend pool/conn
    print("[run_eval] building graph (in-memory checkpointer)")

    async def memsaver_factory():
        return MemorySaver()

    # Best-effort backend pool for `propose_transaction` audit table.
    # If BACKEND_DATABASE_URL is unset, the tool degrades gracefully — but
    # the SQL truth check needs it for numerical gate. We require a writable
    # connection (psycopg directly — NOT the MCP read-only).
    import psycopg
    backend_dsn = os.environ.get("BACKEND_DATABASE_URL")
    if not backend_dsn:
        print("[run_eval] WARN: BACKEND_DATABASE_URL unset — numerical gate "
              "will fail rows with `expected.numbers` (SQL truth needs DSN)",
              file=sys.stderr)
        backend_conn = None
    else:
        backend_conn = psycopg.connect(backend_dsn, autocommit=False)

    repo = None
    store = None
    try:
        if backend_dsn:
            # The backend pool is what propose_transaction's repo writes to.
            backend_pool = make_backend_pool(open=False)
            await backend_pool.open()
            repo = ProposalRepo(backend_pool)
            store = await build_store()
        graph = await build_graph(
            repo=repo, store=store,
            checkpointer_factory=memsaver_factory,
        )

        # 3. Run rows
        rows: list[RowResult] = []
        for i, ex in enumerate(examples):
            if _interrupted:
                print(f"[run_eval] interrupted at row {i}/{len(examples)}")
                break
            print(f"[run_eval] [{i+1}/{len(examples)}] {ex['example_id']} "
                  f"mode={ex.get('metadata', {}).get('mode')}",
                  end=" ", flush=True)
            row = await run_one_row(
                ex, graph, args.judge_model, backend_conn,
            )
            rows.append(row)
            tags = []
            for name, gate in (("num", row.numerical), ("blk", row.block_shape),
                               ("jdg", row.llm_judge)):
                tags.append(f"{name}={'OK' if gate.passed else 'X'}")
            print(" ".join(tags) + f" ({row.elapsed_s:.1f}s)")

    finally:
        if backend_conn is not None:
            backend_conn.close()

    # 4. Aggregate + write report
    summary = aggregate_report(rows)
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "judge_model": args.judge_model,
        "rows": [_row_to_dict(r) for r in rows],
        "summary": summary,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[run_eval] wrote {args.output}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 5. Exit code = AND-gate
    return 0 if summary["overall_passed"] else 1


def main() -> int:  # pragma: no cover
    return asyncio.run(_amain())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
