"""Numerical validator — post-model hook for the v3 ReAct loop.

NEW in Wave 1. Sourced from `docs/v3/phase2_validator.md` §1-§6.

Decision 6 (locked): pre-emit blocking validator + re-prompt once + emit
`event: error` on second failure.

Goal: 99% numerical accuracy SLO (decision 5). Every monetary number in
the final answer MUST be cross-checked against `state.tool_outputs_this_turn`
(run_python results + stdout).

Why this matters:
- Per memory `project_chat_money_column_float8`, `transactions.amount` is
  `double precision` in PostgreSQL. Latent drift.
- Per memory `project_codeact_dual_impl`, mismatched tool reference caused
  the 1,234.56 hallucination. Validator is the safety net — even if the
  prompt drifts, the validator catches numeric lies.

Q2 (locked Day 1): when retry fires, emit a `status_token`
"กำลังตรวจสอบตัวเลข..." via `get_stream_writer()` so the user sees the
slow retry is the system being careful, not just slow.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from src.agent.session_logger import slog


def _is_dev_env() -> bool:
    """True outside production — mirrors session_logger / sse_adapter gate.

    Used to decide whether the soft-warn footer also exposes the offending
    number (handy in dev, noise to a real user in prod).
    """
    return os.getenv("ENVIRONMENT", "development").lower() != "production"


# ── Soft-warn footer (appended to the answer on numerical mismatch) ──────────
#
# `_WARNING_MARKER` is the idempotency sentinel — its presence in the answer
# text means the footer was already appended this turn (don't stack it).
_WARNING_MARKER = "⚠️ หมายเหตุ: ตัวเลขบางส่วนอาจคลาดเคลื่อน"
_WARNING_FOOTER = (
    "\n\n---\n"
    f"{_WARNING_MARKER} ระบบยังตรวจสอบความถูกต้องไม่ผ่าน "
    "โปรดใช้วิจารณญาณก่อนตัดสินใจ"
)
# Dev-only addendum — surfaces the un-grounded value so a dev can fix the
# grounding. Gated behind `_is_dev_env()` so it never reaches a real user.
_DEV_WARNING_SUFFIX = (
    "\n[DEV] ungrounded={offending_number} | run_python={tool_numbers}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Number Extraction — Regex Patterns
# ─────────────────────────────────────────────────────────────────────────────


# Numeric magnitudes (Thai + English)
_TH_MAGNITUDES = {
    "ล้าน": Decimal("1000000"),
    "แสน": Decimal("100000"),
    "หมื่น": Decimal("10000"),
    "พัน": Decimal("1000"),
    "ร้อย": Decimal("100"),
}
_EN_MAGNITUDES = {
    "b": Decimal("1000000000"),
    "m": Decimal("1000000"),
    "k": Decimal("1000"),
}

# Match a numeric token: integer, decimal, or comma-grouped.
# Examples matched: "12500", "12,500", "12500.50", "1.5"
#
# IMPORTANT: `_NUM` itself contains ONE capturing group (the int-or-grouped
# alternation). When `_NUM` is embedded inside another pattern, that adds an
# extra group, shifting subsequent group indices. To make outer patterns
# easy to author, we wrap `_NUM` callers in a NON-CAPTURING `(?:...)` group
# so group 1 in callers is always the magnitude / currency marker.
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

# Pattern 1: "X พัน/หมื่น/แสน/ล้าน" (Thai magnitude phrases)
# Examples: "1.5 ล้าน", "3 หมื่น", "5พันบาท", "1.5 ล้านบาท"
# `(?:บาท)?` — entire word is optional. Earlier `บาท?` only made the trailing
# "ท" optional, so "1.5 ล้าน" (no บาท) silently missed.
_RE_TH_MAGNITUDE = re.compile(
    rf"({_NUM})\s*(ล้าน|แสน|หมื่น|พัน|ร้อย)(?:\s*บาท)?",
)

# Pattern 2: "Xk / Xm / Xb" (English magnitude shorthand)
# Examples: "32k", "1.5M", "2 b baht"
_RE_EN_MAGNITUDE = re.compile(
    rf"({_NUM})\s*([kmb])\b",
    flags=re.IGNORECASE,
)

# Pattern 3: "X บาท" (Thai currency, no magnitude)
# Examples: "12,500 บาท", "250 บาท"
_RE_TH_BAHT = re.compile(rf"({_NUM})\s*บาท")

# Pattern 4: "฿X" / "X THB" / "X B" (English currency markers)
_RE_EN_CURRENCY = re.compile(
    rf"(?:฿\s*({_NUM}))|(?:({_NUM})\s*(?:THB|baht|B)\b)",
    flags=re.IGNORECASE,
)

# Pattern 5: bare comma-grouped number — likely money in financial context.
# Only fires when 4+ digits (avoids matching "250" as money — too low-signal;
# but "12,500" is unambiguous).
_RE_BARE_GROUPED = re.compile(r"\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b")

# Permissive scanner for TOOL OUTPUTS only — those are structured dict / JSON
# strings (e.g. "{'amount': 250}") where the strict `extract_money_numbers`
# yields too few hits. Any bare integer or decimal that is NOT immediately
# preceded by an alphanumeric/underscore character (which would mean it is
# part of an identifier like 'tx1'). Used by `validate_numerical_response`
# for tool-output extraction — NEVER for answer text, to keep natural-language
# false-positives (e.g. "วันที่ 25") out of the validator's strict path.
_RE_TOOL_OUTPUT_NUMBER = re.compile(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tolerance Policy — 1% relative tolerance
# ─────────────────────────────────────────────────────────────────────────────


def _close_enough(a: Decimal, b: Decimal, rel_tol: Decimal = Decimal("0.01")) -> bool:
    """1% relative tolerance — covers rounding-for-display (12,500 vs 12,499.50)."""
    if a == b:
        return True
    larger = max(abs(a), abs(b))
    if larger == 0:
        return False
    return abs(a - b) / larger <= rel_tol


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extraction algorithm
# ─────────────────────────────────────────────────────────────────────────────


def _parse_decimal(token: str) -> Decimal:
    """Parse '12,500.50' or '12500' -> Decimal."""
    return Decimal(token.replace(",", ""))


def _overlaps(span1: tuple, span2: tuple) -> bool:
    """True if two regex spans share any character index."""
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])


def extract_money_numbers(text: str) -> list[Decimal]:
    """Extract every plausibly-monetary number from a Thai/English answer.

    Strict-mode extractor for ANSWER TEXT (natural language). Order of
    patterns by specificity, with overlap check to avoid double counting.
    Returns Decimal values in MAGNITUDE-EXPANDED form (e.g. "1.5 ล้าน" ->
    1500000).

    For TOOL OUTPUTS (structured dict / JSON-ish strings such as
    "{'amount': 250}") use the internal permissive scanner in
    `validate_numerical_response` instead — bare "250" must register as
    money there, but must NOT register as money in answer text (where it
    is too low-signal — "วันที่ 25", "10 ครั้ง", etc.).
    """
    if not text:
        return []
    nums: list[Decimal] = []
    consumed: list[tuple[int, int]] = []

    # 1. Thai magnitudes (must be first — strips "X ล้าน" before bare-numbers see "X")
    for m in _RE_TH_MAGNITUDE.finditer(text):
        base = _parse_decimal(m.group(1))
        magnitude = _TH_MAGNITUDES.get(m.group(2), Decimal(1))
        nums.append(base * magnitude)
        consumed.append(m.span())

    # 2. English magnitudes
    for m in _RE_EN_MAGNITUDE.finditer(text):
        if any(_overlaps(m.span(), c) for c in consumed):
            continue
        base = _parse_decimal(m.group(1))
        magnitude = _EN_MAGNITUDES.get(m.group(2).lower(), Decimal(1))
        nums.append(base * magnitude)
        consumed.append(m.span())

    # 3. Explicit Thai currency
    for m in _RE_TH_BAHT.finditer(text):
        if any(_overlaps(m.span(), c) for c in consumed):
            continue
        nums.append(_parse_decimal(m.group(1)))
        consumed.append(m.span())

    # 4. English currency
    for m in _RE_EN_CURRENCY.finditer(text):
        if any(_overlaps(m.span(), c) for c in consumed):
            continue
        token = m.group(1) or m.group(2)
        if token:
            nums.append(_parse_decimal(token))
            consumed.append(m.span())

    # 5. Bare comma-grouped numbers (only if 4+ digits — the regex enforces
    # at least one comma group, so minimum is 1,000)
    for m in _RE_BARE_GROUPED.finditer(text):
        if any(_overlaps(m.span(), c) for c in consumed):
            continue
        nums.append(_parse_decimal(m.group(1)))
        consumed.append(m.span())

    return nums


def _extract_numbers_from_tool_output(text: str) -> list[Decimal]:
    """Permissive number extractor for TOOL OUTPUTS only.

    Tool outputs are structured strings (Python dict repr, JSON, sandbox
    stdout) where the strict answer-text extractor misses bare values like
    `{'amount': 250}` or `{'amount_thb': 12499.50}`. This permissive scanner
    pulls every bare integer / decimal NOT immediately preceded by an
    alphanumeric/underscore character (which would mean it is part of an
    identifier such as `'tx1'`).

    Magnitude phrases (Thai/English) are still respected by running the
    strict extractor first — its hits are unioned with the permissive scan.
    This means a tool output like `"1500000"` and the strict result
    `[1500000]` from a magnitude phrase both end up in the validator's
    pool, giving the cross-check the best chance of finding a 1%-tolerance
    match.
    """
    if not text:
        return []
    # Start with the strict extractor — catches "1.5 ล้าน" / "32k" cases that
    # the bare-number scan would split into base+magnitude tokens.
    nums: list[Decimal] = list(extract_money_numbers(text))
    seen_spans: list[tuple[int, int]] = []
    for m in _RE_TOOL_OUTPUT_NUMBER.finditer(text):
        # Skip already-consumed spans implicitly: we re-extract independently,
        # then dedupe by value at the validator level. Duplicates here are
        # harmless because `any(_close_enough(...))` short-circuits in the
        # validator.
        nums.append(_parse_decimal(m.group(1)))
        seen_spans.append(m.span())
    return nums


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-check logic
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationResult:
    """Outcome of a single validation pass.

    `pass_` uses a trailing underscore to avoid shadowing the Python `pass`
    keyword. SSE adapter / post_model_hook reads `pass_` only.
    """
    pass_: bool
    offending_number: Optional[Decimal] = None
    tool_numbers: Optional[list[Decimal]] = None
    reason: str = ""


def validate_numerical_response(
    final_text: str,
    tool_outputs: list[str],
) -> ValidationResult:
    """Scan final_text for money numbers; ensure each has a 1%-tolerance match
    in the union of tool_outputs' numbers.

    Args:
      final_text: The LLM's final answer text (after streaming complete).
      tool_outputs: List of stringified run_python results + stdouts for this
                    turn. Order doesn't matter — we union all numbers.

    Returns:
      ValidationResult — pass_=True if all answer numbers match (or no
                         numbers were in the answer).
    """
    answer_nums = extract_money_numbers(final_text)
    if not answer_nums:
        return ValidationResult(pass_=True, reason="no_numbers_in_answer")

    # Tool outputs use the permissive scanner — they're structured-data
    # strings (Python dict repr, JSON, sandbox stdout) where bare numbers
    # like `{'amount': 250}` or `12499.50` carry the ground-truth amount.
    tool_nums: list[Decimal] = []
    for output in tool_outputs:
        tool_nums.extend(_extract_numbers_from_tool_output(output))

    if not tool_nums:
        # The LLM mentioned numbers but no tool produced any — instant fail.
        return ValidationResult(
            pass_=False,
            offending_number=answer_nums[0],
            tool_numbers=[],
            reason="no_tool_numbers_to_validate_against",
        )

    for n in answer_nums:
        if not any(_close_enough(n, t) for t in tool_nums):
            return ValidationResult(
                pass_=False,
                offending_number=n,
                tool_numbers=tool_nums,
                reason="number_not_in_tool_outputs",
            )

    return ValidationResult(pass_=True, reason="all_numbers_matched")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Re-prompt template (Thai + English)
# ─────────────────────────────────────────────────────────────────────────────


_RETRY_TEMPLATE = """\
[VALIDATOR_RETRY]
ตัวเลข {offending_number} ในคำตอบไม่ตรงกับ output ของ run_python ใน turn นี้.
ตัวเลขที่มีจาก run_python: {tool_numbers}

ตอบใหม่โดยใช้เฉพาะตัวเลขจาก run_python output ใน turn นี้เท่านั้น.
ถ้าจำเป็นต้องใช้ตัวเลขใหม่ → เรียก run_python อีกครั้งก่อนตอบ.
ห้ามเดาตัวเลข ห้ามใช้ตัวเลขจาก turn ก่อน.

System note (English, for clarity): the previous answer contained {offending_number}
which is not within 1% of any tool output in this turn. Re-emit the answer using
only values from the tool outputs listed above (or call run_python again to fetch
a missing value).
"""


# Status token shown to the user while validator retries — Q2 locked Day 1.
_RETRY_STATUS_TEXT = "กำลังตรวจสอบตัวเลข..."


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Tool-error finalize guard — structural R10 enforcement
# ─────────────────────────────────────────────────────────────────────────────
#
# v3's create_react_agent lets the LLM emit a final text answer even right
# after a tool returned an error — a weak model "gives up" on the first
# failure instead of retrying (the failing trace: compare_periods TypeError →
# canned apology). The v2 codeact subgraph prevented this STRUCTURALLY via a
# result-gated loop. We restore that guarantee here: when the model finalizes
# while THIS turn's most recent tool call is an unrecovered, RETRYABLE error,
# we force a loop-back instead of accepting the give-up. R10 in the prompt
# still tells the model HOW to retry; this guard makes sure it MUST.

# How many times the guard forces a retry before letting the Thai fallback
# (R10) stand. Mirrors the validator's single-retry budget but one higher —
# a wrong-signature call often needs read-error → fix → re-call.
_TOOL_ERROR_RETRY_MAX = 2

_TOOL_ERROR_RETRY_TEMPLATE = (
    "\n\n[TOOL_ERROR_RETRY {attempt}/{max}] The tool call above FAILED with "
    "the error shown. Do NOT apologise to the user and do NOT answer yet. "
    "Read the error, fix your code, and call the tool again now. If the error "
    "names a different helper to use instead (e.g. spending_trend), switch to "
    "it."
)

# Status token shown while the tool-error guard forces a retry — distinct from
# the numerical-validator status so logs/telemetry can tell them apart.
_TOOL_RETRY_STATUS_TEXT = "กำลังลองใหม่อีกครั้ง..."


def _parse_tool_payload(content: object) -> Optional[dict]:
    """Best-effort decode of a ToolMessage's content into a dict, else None."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _last_tool_error_this_turn(messages: list) -> Optional[tuple[str, ToolMessage]]:
    """Return (error_text, tool_message) when THIS turn's most recent tool
    result is an unrecovered, RETRYABLE error — else None.

    Turn-safe: walk backwards and stop at the current turn's HumanMessage so a
    stale error from a previous turn never triggers a retry. The first
    ToolMessage reached before that boundary is what the model saw right
    before finalizing — if the LAST tool succeeded (no `error`) the model has
    already recovered and we do nothing.

    Retryable ≠ terminal: `run_python` tags non-retryable terminal states
    (wallet_required, context_load_failed, invalid_state) with a `kind` field.
    Genuine sandbox code errors (TypeError / ValueError / SyntaxError) carry a
    bare `error` with NO `kind`. We only fire on the latter so the
    wallet-onboarding / clarification flows are left untouched.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return None
        if isinstance(m, ToolMessage):
            payload = _parse_tool_payload(m.content)
            if not isinstance(payload, dict):
                return None
            if payload.get("error") and not payload.get("kind"):
                return str(payload["error"]), m
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. In-graph hook — post_model_hook for create_react_agent
# ─────────────────────────────────────────────────────────────────────────────


def make_validator_post_model_hook() -> Callable:
    """Build the post-model hook that ReAct invokes after each LLM step.

    Returns a callable suitable for `create_react_agent`'s `post_model_hook`
    parameter. Behavior:
      - Inspect the latest AIMessage.
      - If it has tool_calls (intermediate) -> no-op (pass through).
      - If it's a FINAL answer (no tool_calls, has content) -> run validator.
        - Pass -> return state unchanged.
        - Fail + retries < 1 -> emit status_token (Q2), inject SystemMessage
                                 with retry template, increment
                                 `__validator_retries__`, return state
                                 (ReAct will re-invoke the model).
        - Fail + retries >= 1 -> set state["__validator_failed__"] = True;
                                  SSE adapter sees this and emits error.
    """
    def hook(state: dict) -> dict:
        messages = state.get("messages") or []
        if not messages:
            return {}
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return {}
        if getattr(last, "tool_calls", None):
            return {}    # intermediate — let ReAct keep going
        text = last.content or ""
        if isinstance(text, list):
            # multimodal content — flatten string parts
            text = " ".join(
                p.get("text", "") for p in text if isinstance(p, dict)
            )
        if not text or not text.strip():
            return {}

        # ── Tool-error finalize guard (structural R10) ──────────────────────
        # Runs BEFORE numerical validation: if the model is answering while
        # THIS turn's most recent tool call is an unrecovered retryable error,
        # force a loop-back rather than accept the give-up. Mirrors the
        # validator's mechanic below — drop the give-up AIMessage and rewrite
        # the failed ToolMessage so `messages[-1]` is a ToolMessage and the
        # ReAct router re-invokes the model.
        tool_error = _last_tool_error_this_turn(messages)
        if tool_error is not None:
            err_text, err_tool_msg = tool_error
            tool_retries = state.get("__tool_error_retries__", 0)
            if tool_retries < _TOOL_ERROR_RETRY_MAX:
                hint = _TOOL_ERROR_RETRY_TEMPLATE.format(
                    attempt=tool_retries + 1, max=_TOOL_ERROR_RETRY_MAX
                )
                updates: list = []
                hallu_id = getattr(last, "id", None)
                if hallu_id:
                    updates.append(RemoveMessage(id=hallu_id))
                # Rewrite the failed ToolMessage (same id → add_messages updates
                # in place) to carry the corrective hint. CRITICAL: embed the
                # hint INSIDE the JSON payload (a `retry_instruction` field) and
                # keep the original `error` key intact — appending the hint as
                # raw text would corrupt the JSON so `_last_tool_error_this_turn`
                # could no longer re-detect the error, and the guard would fire
                # at most ONCE per turn even if the model keeps giving up. By
                # preserving valid JSON, the guard re-fires (gated by the
                # counter) until the model actually retries or MAX is hit.
                payload = _parse_tool_payload(err_tool_msg.content) or {}
                payload["retry_instruction"] = hint
                updates.append(
                    ToolMessage(
                        content=json.dumps(payload, ensure_ascii=False, default=str),
                        tool_call_id=err_tool_msg.tool_call_id,
                        id=err_tool_msg.id,
                        name=getattr(err_tool_msg, "name", None),
                    )
                )
                # Tell the user the system is fixing itself, not stuck (Q2-style).
                try:
                    from langgraph.config import get_stream_writer
                    get_stream_writer()({"status": _TOOL_RETRY_STATUS_TEXT})
                except Exception:
                    pass
                return {
                    "messages": updates,
                    "__tool_error_retries__": tool_retries + 1,
                }
            # Retries exhausted — let the model's Thai R10 fallback stand and
            # fall through to numerical validation (the apology has no numbers,
            # so it passes cleanly).

        # Ground-truth pool for the answer's numbers:
        #   - run_python result + stdout (the canonical analyst source)
        #   - tool messages from THIS turn (e.g. propose_transaction returns
        #     JSON containing the amount the user typed — Q7 amendment lets
        #     user-typed numbers pass through `propose_transaction(amount=…)`)
        #   - the current turn's HumanMessage text (user-typed numbers like
        #     "เพิ่ม 250 กาแฟ" are ground truth)
        tool_outputs = [
            f"{item.get('result', '')} {item.get('stdout', '')}"
            for item in (state.get("tool_outputs_this_turn") or [])
        ]
        # Add every ToolMessage content from this turn — most contain the
        # tool's JSON payload which mirrors the user-typed values.
        for m in messages:
            if isinstance(m, ToolMessage):
                tool_outputs.append(str(m.content or ""))
        # Include the most-recent HumanMessage to satisfy Q7 pass-through
        # for user-typed amounts ("เพิ่ม 250 กาแฟ"). Multiple HumanMessages
        # would be unusual — the LAST one is the current turn's input.
        last_human_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_human_text = str(m.content or "")
                break
        if last_human_text:
            tool_outputs.append(last_human_text)

        result = validate_numerical_response(text, tool_outputs)
        if result.pass_:
            return {}  # no-op — avoid double-appending emitted_blocks_this_turn via append_reducer

        # ── Soft-warn instead of reject (NEVER drop, NEVER silent error) ────
        # Decision (Wave 7+): a numerical mismatch must NOT reject the answer or
        # surface a generic error. The user always gets the full streamed
        # answer; we APPEND a visible warning footer so they know some numbers
        # may be off, and slog the offending value so a dev can improve the
        # grounding later ("neet dev improve").
        #
        # Mechanics — why this lands on the wire without touching the SSE
        # adapter: the hook runs INSIDE the create_react_agent subgraph, BEFORE
        # the final AIMessage crosses the outer-graph boundary where
        # `stream_mode="messages"` turns it into `answer_token` (this is the
        # same reason the old RemoveMessage path could blank the answer). So we
        # REWRITE the same AIMessage in place (id preserved → add_messages
        # updates, not appends); the appended footer therefore reaches BOTH the
        # live answer_token stream AND the persisted history answer block.
        hallu_id = getattr(last, "id", None)

        # Idempotency: the hook can re-enter on a re-emitted final message.
        # Never stack the footer twice.
        if _WARNING_MARKER in text:
            return {}

        slog(
            "numerical_validator",
            f"soft-warn (no reject): offending={result.offending_number} "
            f"reason={result.reason} "
            f"tool_numbers={[str(t) for t in (result.tool_numbers or [])]}",
        )

        footer = _WARNING_FOOTER
        if _is_dev_env():
            footer += _DEV_WARNING_SUFFIX.format(
                offending_number=result.offending_number,
                tool_numbers=[str(t) for t in (result.tool_numbers or [])],
            )

        rewritten_answer = AIMessage(content=f"{text}{footer}", id=hallu_id)
        return {"messages": [rewritten_answer]}

    return hook
