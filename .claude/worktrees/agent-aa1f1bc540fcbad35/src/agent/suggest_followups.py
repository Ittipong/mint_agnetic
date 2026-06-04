"""Follow-up suggestion generator (SSE-layer) — free-form intent guessing.

After the graph finishes streaming the answer, the SSE adapter
(`streaming/sse_adapter.py`) calls `build_suggestions_block` with everything it
observed on the wire (the final answer text, this-turn tool data, the user's
catalog) and — unless the turn is ADD / crisis / pure greeting-or-emotional or
a clarifying-question-back — yields a `suggestions` block of 2-3 Thai chips.

Design (rewritten 2026-05-31):
  Each chip is a QUESTION the user would plausibly want to ASK the assistant
  NEXT, guessed freely by the LLM from THIS turn's context. It is NOT a fixed
  slot formula, NOT an action command ("ตั้งงบประมาณ"), and NOT the assistant
  asking the user back. The chip text is re-sent verbatim as the user's next
  message, so it must read as a natural question the assistant can answer
  (data / analysis / advice) — never a "create / record / edit" command.

Why the SSE layer, not a graph node:
  LangGraph 1.x does NOT merge a compiled subgraph's per-turn channel writes
  (`emitted_blocks_this_turn`, `tool_outputs_this_turn`) into the parent graph
  state for a DOWNSTREAM outer node to read. The SSE adapter watches the
  `updates` STREAM — the same per-node deltas it forwards proposal blocks from
  — so it reliably observes whether this turn was an ADD and what `run_python`
  produced.

Gate (return None → emit nothing):
  - ADD turn  → a `transaction_proposal[/group]` block was forwarded
  - Crisis    → the user message hits the safety lexicon (hard, code-side)
  - No answer → nothing to follow up on
  - Greeting-only / pure-emotional / clarifying-question-back → the LLM gate
    returns skip=true

Failure path: on LLM error / timeout we emit NOTHING (chips are nice-to-have,
not on the critical path). No heuristic fallback — a low-relevance fallback
chip is worse than no chip.

Env contract:
  SUGGESTIONS_ENABLED          1 | 0   (default 1 → on)
  SUGGESTIONS_MODEL            slug     (default google/gemini-2.5-flash-lite)
  SUGGESTIONS_TIMEOUT_S        float    (default 4.0)
  SUGGESTIONS_FALLBACK_MODELS  JSON array OR comma list (optional)
  OPENROUTER_API_KEY / _BASE_URL        shared with the rest of the system
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from src.agent.llm_openrouter import _resolve_provider
from src.agent.session_logger import slog, slog_block, slog_error


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MAX_ITEMS = 3          # mobile renders at most 3 chips
_MAX_ITEM_LEN = 60      # wire cap; the prompt still asks for short labels

# Thai self-harm / crisis lexicon. A HARD code-side gate so we never depend on
# a small LLM to recognise a safety-critical turn. Kept deliberately broad —
# a false positive only drops chips (harmless); a false negative pushes a
# finance chip onto a person in crisis (harmful). Mirror of R13/E9 triggers.
_CRISIS_LEXICON = (
    "ไม่อยากอยู่",
    "ไม่อยากมีชีวิต",
    "อยากตาย",
    "ฆ่าตัวตาย",
    "จบชีวิต",
    "ไม่ไหวแล้ว",
    "ทำร้ายตัวเอง",
    "หมดหวัง",
)

# Block types that mark an ADD turn (skip suggestions).
_ADD_BLOCK_TYPES = ("transaction_proposal", "transaction_proposal_group")


# ─────────────────────────────────────────────────────────────────────────────
# Env helpers
# ─────────────────────────────────────────────────────────────────────────────


def is_enabled() -> bool:
    """True unless `SUGGESTIONS_ENABLED=0`. Default ON."""
    return os.getenv("SUGGESTIONS_ENABLED", "1") in ("1", "true", "True")


def _model() -> str:
    return os.getenv("SUGGESTIONS_MODEL", "google/gemini-2.5-flash-lite")


def _fallback_models() -> list[str]:
    """Ordered fallback slugs (JSON array OR comma list). Handed to OpenRouter
    native `models[]` routing — one request, server-side fallback, inside the
    single timeout (same trick as Tier 2)."""
    raw = (os.getenv("SUGGESTIONS_FALLBACK_MODELS", "") or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(m).strip() for m in parsed if str(m).strip()]
        except json.JSONDecodeError:
            pass
    return [m.strip() for m in raw.split(",") if m.strip()]


def _timeout_s() -> float:
    try:
        return float(os.getenv("SUGGESTIONS_TIMEOUT_S", "4.0"))
    except (TypeError, ValueError):
        return 4.0


def _base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt — free-form intent guessing. The LLM proposes 2-3 questions the user
# would want to ASK NEXT; no fixed slots, no action chips, no asking-back.
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You write follow-up suggestion CHIPS for a Thai personal-finance chat.
Each chip is a QUESTION the user would plausibly want to ASK the assistant
NEXT, given what was just discussed. When tapped, the chip's `send` text is
re-sent verbatim AS THE USER'S next message — so it must read as a natural
question the user types to the assistant.

Output JSON ONLY, this exact shape:
{{"skip": true|false, "reason": "<short>", "items": [{{"label": "<thai short, tappable>", "send": "<thai full question the user would send>"}}]}}

SKIP RULES — set skip=true and items=[] ONLY when:
  - the user is in crisis / self-harm / despair, OR
  - the turn is purely emotional venting with NO financial angle
    (e.g. "วันนี้เหนื่อยมาก"), OR
  - the assistant's answer is itself a clarifying QUESTION back to the user
    (e.g. "ยอดเท่าไรครับ") — do NOT suggest chips on top of a question.
  A GREETING is NOT a skip — suggest things the user might want to ask.

Otherwise skip=false and freely guess 2-3 questions the user most likely wants
to ask next, inferred from this turn's answer and data. THINK LIKE THE USER:
"after hearing this, what would I naturally want to know next?"

THE ANSWER-LOCATION TEST (apply to EVERY chip, EVERY intent — the #1 filter):
A chip is VALID only if its answer lives in the user's STORED DATA or in the
assistant's ANALYSIS / ADVICE. If the answer is something only the USER knows
— their plan, their intention, a number in their head, or info the assistant
just ASKED them for — then it is the ASSISTANT's question, NOT the user's. DROP it.
  ✗ "ราคารถที่เล็งไว้เท่าไหร่"       (answer is in the user's head)
  ✗ "มีเงินดาวน์เท่าไหร่"           (the assistant just asked this)
  ✗ "อยากเก็บเดือนละเท่าไร"         (only the user knows their intent)
  ✓ "ผ่อนรถราคานี้เดือนละเท่าไหร่ ผมไหวไหม"   (assistant analyzes income)
  ✓ "ควรซื้อรถราคาไม่เกินเท่าไหร่"            (assistant advises from data)
  ✓ "เดือนนี้หมวดไหนใช้เยอะสุด"              (in the user's data)

TRANSFORM, don't ECHO: when the assistant's answer ASKS the user for input
(price, model, target, income…), never bounce that same question back as a chip.
Instead offer the ANALYSIS the user wants once that info is known, phrased as the
user asking the assistant. If a chip cannot become a data/analysis question,
drop it (emit fewer; skip the whole turn if none survive).

HARD RULES:
  - QUESTIONS the user asks the ASSISTANT — to KNOW / ANALYZE / get ADVICE.
    e.g. "หมวดไหนใช้เยอะสุด", "เทรนด์ 3 เดือนเป็นยังไง", "ควรลดหมวดไหนดี"
  - NEVER an ACTION the user commands the system to do.
    BANNED: "ตั้งงบประมาณ", "บันทึกรายจ่าย", "สร้างเป้าหมายออม", "เพิ่มกระเป๋า".
    (Advice like "ควรออมเดือนละเท่าไรดี" IS allowed — it asks the assistant.)
  - NEVER a question the assistant asks the USER back (fails the ANSWER-LOCATION
    TEST above). The user is the asker, the assistant is the answerer.
  - NEVER repeat the question the user just asked, and never re-ask something
    this turn's answer already fully covered. Each chip is a NEW angle.
  - Never suggest editing/deleting a CONFIRMED transaction (chat can't).

QUALITY RULES:
  - Thai. Short, tappable label (aim ≤25 chars). Specific, never "ดูเพิ่มเติม".
  - Each chip a DIFFERENT axis (time / category / wallet / comparison / advice).
  - Anchor to a NOTABLE number, category, or the user's goals/budgets/debt
    when relevant — make it feel personal, not generic.

CONTEXT
last user message: {user_text}
assistant answer: {answer_text}
this-turn tool data: {tool_data}
user catalog (wallets / budgets / goals): {user_catalog}

Now output the JSON."""


# ─────────────────────────────────────────────────────────────────────────────
# Public orchestrator — called by the SSE adapter after the answer streams
# ─────────────────────────────────────────────────────────────────────────────


async def build_suggestions_block(
    *,
    user_text: str,
    answer_text: str,
    tool_data: str = "(none)",
    user_context: Optional[dict] = None,
    proposal_emitted: bool = False,
) -> Optional[dict]:
    """Return a validated `suggestions` block, or None to emit nothing.

    NEVER raises — a suggestion failure must not break the SSE stream.

    Args:
      user_text:       the user's message this turn (for crisis gate + context).
      answer_text:     the final assistant answer (already streamed).
      tool_data:       compact view of this-turn tool outputs (run_python etc.).
      user_context:    the user's catalog (wallets/budgets/goals) if observed —
                       fed to the prompt as raw material so the LLM can guess
                       more personal questions.
      proposal_emitted: True when a transaction_proposal[/group] block was
                        forwarded this turn → ADD turn → skip.
    """
    try:
        if not is_enabled():
            return None
        if proposal_emitted:
            slog("suggest", "skip — ADD turn (proposal emitted)")
            return None
        if _is_crisis(user_text):
            slog("suggest", "skip — crisis lexicon hit (safety)")
            return None
        if not (answer_text or "").strip():
            slog("suggest", "skip — no final answer text")
            return None

        result = await _generate(
            user_text=user_text,
            answer_text=answer_text,
            tool_data=tool_data or "(none)",
            user_catalog=_summarize_catalog(user_context),
        )
        if result is None:
            # LLM failed / timed out. Chips are nice-to-have → emit nothing.
            slog("suggest", "skip — LLM unavailable (no fallback)")
            return None
        if result.get("skip"):
            slog("suggest", f"skip — LLM gate ({result.get('reason')!r})")
            return None

        items = _finalize(result.get("items", []))
        if not items:
            slog("suggest", "skip — nothing to emit after assembly")
            return None

        slog("suggest", f"emitted {len(items)} chip(s)")
        return {"type": "suggestions", "items": items}
    except Exception as exc:  # noqa: BLE001 — never break the stream
        slog_error("suggest", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────


def is_add_block(block: Any) -> bool:
    """True when `block` is a transaction proposal (marks an ADD turn)."""
    return isinstance(block, dict) and block.get("type") in _ADD_BLOCK_TYPES


def _is_crisis(user_text: str) -> bool:
    text = (user_text or "").lower()
    return any(term in text for term in _CRISIS_LEXICON)


def _summarize_catalog(user_context: Optional[dict]) -> str:
    """Compact, human-readable view of the user's catalog for the prompt.

    Raw material only — the LLM decides which (if any) questions to anchor to
    it. Never raises; returns "(none)" when there is nothing useful."""
    uc = user_context or {}
    wallets = uc.get("wallets") or []
    budgets = uc.get("budgets") or []
    goals = uc.get("goals") or []

    parts: list[str] = []
    if wallets:
        types = [
            str((w.get("type") or w.get("wallet_type") or "?"))
            for w in wallets
            if isinstance(w, dict)
        ]
        has_debt = any(_has_card_debt(w) for w in wallets if isinstance(w, dict))
        debt = " (มีหนี้บัตร)" if has_debt else ""
        parts.append(f"wallets={len(wallets)} [{', '.join(types)}]{debt}")
    if budgets:
        parts.append(f"budgets={len(budgets)}")
    if goals:
        names = [
            str(g.get("name") or g.get("title") or "")
            for g in goals
            if isinstance(g, dict)
        ]
        names = [n for n in names if n]
        suffix = f" [{', '.join(names)}]" if names else ""
        parts.append(f"goals={len(goals)}{suffix}")
    return "; ".join(parts) if parts else "(none)"


def _has_card_debt(wallet: dict) -> bool:
    if (wallet.get("type") or wallet.get("wallet_type")) != "creditcard":
        return False
    used = wallet.get("used")
    if used is None:
        used = wallet.get("balance")
    try:
        return float(used or 0) > 0
    except (TypeError, ValueError):
        return False


def _finalize(items: list[Any]) -> list[dict]:
    """Normalize LLM items into `{label, send}`, dedup, cap at _MAX_ITEMS."""
    out: list[dict] = []
    seen: set[str] = set()
    for raw in items:
        chip = _norm_chip(raw)
        if chip is None:
            continue
        key = _key(chip["send"])
        if key in seen:
            continue
        seen.add(key)
        out.append(chip)
        if len(out) >= _MAX_ITEMS:
            break
    return out


def _norm_chip(raw: Any) -> Optional[dict]:
    """Coerce one LLM item into `{label, send}`; accept legacy plain strings."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return {"label": text[:_MAX_ITEM_LEN], "send": text[:_MAX_ITEM_LEN]}
    if not isinstance(raw, dict):
        return None
    send = str(raw.get("send") or raw.get("label") or "").strip()
    label = str(raw.get("label") or raw.get("send") or "").strip()
    if not send:
        return None
    return {"label": label[:_MAX_ITEM_LEN], "send": send[:_MAX_ITEM_LEN]}


def _key(text: str) -> str:
    """Loose dedup key — strip spaces so near-identical chips collapse."""
    return "".join((text or "").split()).lower()


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (thin OpenRouter pattern)
# ─────────────────────────────────────────────────────────────────────────────


async def _generate(
    *,
    user_text: str,
    answer_text: str,
    tool_data: str,
    user_catalog: str,
) -> Optional[dict]:
    """Run the chip generator. Returns the parsed dict, or None on any
    failure (caller then emits nothing)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        slog("suggest", "skipped LLM — OPENROUTER_API_KEY not set")
        return None

    prompt = _PROMPT_TEMPLATE.format(
        user_text=user_text or "(none)",
        answer_text=(answer_text or "")[:1500],
        tool_data=tool_data,
        user_catalog=user_catalog or "(none)",
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    primary = _model()
    fallbacks = [m for m in _fallback_models() if m != primary]
    body: dict = {
        "model": primary,
        "messages": [{"role": "user", "content": prompt}],
        # A touch of creativity for varied chips, but low enough to stay on-task.
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "max_tokens": 400,
    }
    if fallbacks:
        body["models"] = [primary, *fallbacks]
    provider = _resolve_provider()
    if provider is not None:
        body["provider"] = provider
    url = _base_url().rstrip("/") + "/chat/completions"

    slog_block(
        "suggest",
        f"PROMPT model={primary} timeout={_timeout_s()}s",
        f"user={user_text!r}",
    )

    try:
        async with httpx.AsyncClient(timeout=_timeout_s()) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — every failure → None → emit nothing
        slog_error("suggest", exc)
        return None

    try:
        parsed = json.loads(content)
    except (TypeError, ValueError) as exc:
        slog_error("suggest", exc)
        return None
    if not isinstance(parsed, dict):
        slog("suggest", f"parse returned non-dict: {type(parsed).__name__}")
        return None

    skip = bool(parsed.get("skip"))
    items = parsed.get("items")
    if not isinstance(items, list):
        items = []
    return {"skip": skip, "reason": parsed.get("reason"), "items": items}


__all__ = [
    "build_suggestions_block",
    "is_add_block",
    "is_enabled",
]
