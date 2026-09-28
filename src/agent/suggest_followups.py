"""Follow-up suggestion generator (SSE-layer) — free-form intent guessing.

After the graph finishes streaming the answer, the SSE adapter
(`streaming/sse_adapter.py`) calls `build_suggestions_block` with everything it
observed on the wire (the final answer text, this-turn tool data, the user's
catalog) and — unless the turn is ADD / crisis / pure greeting-or-emotional —
yields a `suggestions` block of 2-3 Thai chips.

Design (two modes, the LLM picks per turn):
  MODE-NORMAL — the answer is a statement / analysis / advice / greeting. Each
    chip is a QUESTION the user would plausibly want to ASK the assistant NEXT,
    guessed freely from THIS turn's context. Not a fixed slot formula, not an
    action command ("ตั้งงบประมาณ"), and not the assistant asking the user back.
  MODE-QUICKREPLY — the answer ENDS WITH a clarifying question back to the user
    (e.g. "รายจ่ายคงที่เดือนละเท่าไร"). Instead of skipping, the chips become the
    short ANSWERS the user would give (a representative value, "ไม่มี/ไม่เกี่ยว",
    or "ช่วยประมาณให้") so a tap moves the conversation forward. Each chip's
    `send` carries the question's TOPIC (never a bare number) so the next turn
    is routed as a continuation, not mis-read as a new ADD transaction.
  In both modes the `send` text is re-sent verbatim as the user's next message.

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
  - Greeting-only / pure-emotional → the LLM gate returns skip=true
    (a clarifying-question-back answer is NOT skipped — it becomes
    MODE-QUICKREPLY chips above)

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

import ast
import asyncio
import json
import os
import re
from typing import Any, Optional

import httpx

from src.agent.llm_openrouter import _log_usage, _resolve_provider
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
When tapped, a chip's `send` text is re-sent verbatim AS THE USER'S next
message — so it must read as something the user would naturally type.

Output JSON ONLY, this exact shape:
{{"skip": true|false, "reason": "<short>", "items": [{{"kind": "decide"|"whatif"|"ahead"|"reply", "label": "<thai short, tappable>", "send": "<thai full message the user would send>"}}]}}

FIRST decide the MODE by looking at the assistant's answer:

MODE-QUICKREPLY — the answer ENDS WITH / CONTAINS a clarifying QUESTION back to
the user (e.g. "ยอดเท่าไรครับ", "รายจ่ายคงที่เดือนละเท่าไร", "อยากเก็บกี่บาท").
  The user now owes an ANSWER, not a new question. So make the chips the
  ANSWERS the user is most likely to give (kind="reply") — 2-3 of them:
    1. Hand it back to the assistant to work out FROM THE USER'S DATA
       (e.g. "ช่วยคำนวณจากข้อมูลของฉัน") — ALWAYS include this one.
    2. A "none / not applicable" reply (e.g. "ไม่มีรายจ่ายคงที่").
    3. A representative value ONLY when the answer truly lives in the user's
       head (a plan, a preference). If the asked figure can be derived from
       their stored data (fixed costs, income, average spend…), DO NOT offer a
       guessed number — a tapped guess becomes a "fact" the next answer builds
       a whole plan on.
  NEVER a "not now / skip / ยังไม่…ตอนนี้" chip — tapping it wastes a turn.
  Write 2 replies, then add 1 advisor QUESTION (kind="decide"|"whatif"|"ahead",
  see MODE-NORMAL) as a way out — otherwise reply chips trap the user in endless "ตึงไป / สูงกว่านี้"
  back-and-forth with the assistant.
  CRITICAL anti-misroute rule: the `label` is short, but the `send` MUST carry
  the TOPIC of the question so the next turn is read as a continuation, not as a
  new transaction to record. NEVER let `send` be a bare number.
    Q "รายจ่ายคงที่เดือนละเท่าไร"
      ✓ label="คำนวณจากข้อมูลของฉัน"  send="ช่วยคำนวณรายจ่ายคงที่จากข้อมูลของฉันให้หน่อย"
      ✓ label="ไม่มีรายจ่ายคงที่"  send="ไม่มีรายจ่ายคงที่ประจำ"
      ✗ label="ประมาณ 10,000"  (rent/bills are in the data — never guess them)
      ✗ label="ยังไม่เช็คตอนนี้"  (a "not now" chip dead-ends the chat)
      ✗ send="10000"   (bare number → mis-read as ADD a 10000 transaction)
  In MODE-QUICKREPLY the user's intended answer lives in THE USER'S HEAD ON
  PURPOSE — that is what a chip is FOR here. The ANSWER-LOCATION TEST below does
  NOT apply; do NOT drop these chips for "answer is in the user's head".

MODE-NORMAL — the answer is NOT a question back (a statement / analysis / advice
/ a greeting). Nimo is the user's money ADVISOR and a friend who never judges.
The app's screens already show every total, list, balance and due date — a
chip that only re-reads those is wasted. Write EXACTLY 3 ADVISOR chips, BEST
FIRST, one of each kind (every chip is shown — no spares, so each must pass
every rule below):
  • kind="decide" — so-what / should-I, about what this answer showed:
      "ช้อปเดือนนี้เกินไปไหม", "จ่ายบัตรเต็มเลยดีไหม", "ควรลดตรงไหนก่อน"
  • kind="whatif" — a what-if or a plan built on their numbers:
      "ถ้าลดช้อปครึ่งนึง ไปญี่ปุ่นเร็วขึ้นกี่เดือน", "เก็บเดือนละเท่าไหร่ถึงจะทัน"
  • kind="ahead" — looking forward, or a habit they would not spot alone:
      "เงินจะพอใช้ถึงสิ้นเดือนไหม", "ฉันชอบใช้เงินหนักวันไหน"
THE SCREEN TEST (drop the chip if it fails): "could the user get this by
opening a screen in the app?" → totals, item lists, balances, statement
amounts, due dates, "เทียบเดือนที่แล้ว" as a bare number are ALL screens.
  ✗ "ยอดรวมทุกบัญชี", "หมวดช้อปปิ้งมีรายการอะไรบ้าง", "เดือนที่แล้วใช้เท่าไหร่"
  ✓ "เดือนนี้ใช้เยอะกว่าปกติจนต้องห่วงไหม" (judgment, not a number)
Advisor questions run out fast if you recycle the same few ("ใช้เยอะไปไหม",
"เงินพอถึงสิ้นเดือนไหม"). When "already asked" covers this topic, move to a
NEW one the user has: a goal, a card, a budget, income, a habit.
The ANSWER-LOCATION TEST and TRANSFORM rule below apply to this mode.

SKIP RULES — set skip=true and items=[] ONLY when (both modes):
  - the user is in crisis / self-harm / despair, OR
  - the turn is purely emotional venting with NO financial angle
    (e.g. "วันนี้เหนื่อยมาก").
  A GREETING is NOT a skip. A clarifying question back is NOT a skip either —
  it is MODE-QUICKREPLY above.

THE ANSWER-LOCATION TEST (MODE-NORMAL ONLY — the #1 filter there):
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

TRANSFORM, don't ECHO (MODE-NORMAL only): when a MODE-NORMAL answer references
input the user might give (price, model, target, income…), never bounce that as
a chip QUESTION. Offer the ANALYSIS the user wants once that info is known,
phrased as the user asking the assistant. If a chip cannot become a
data/analysis question, drop it. (This does NOT apply to MODE-QUICKREPLY, where
chips ARE the user's answers to the assistant's question.)

HARD RULES (both modes):
  - NEVER an ACTION the user commands the system to do.
    BANNED: "ตั้งงบประมาณ", "บันทึกรายจ่าย", "สร้างเป้าหมายออม", "เพิ่มกระเป๋า".
    (Advice like "ควรออมเดือนละเท่าไรดี" IS allowed — it asks the assistant.)
  - Never suggest editing/deleting a CONFIRMED transaction (chat can't).
  - Stay on THE USER'S OWN MONEY. Never chips about external products,
    specific funds/stocks, other apps, or where to open an account — and no
    product-shopping questions either ("ลงทุนอะไรดี", "บัญชีดอกเบี้ยสูงมีแบบไหน",
    "ธนาคารไหนดี"). "ควรแบ่งไปลงทุนเท่าไหร่จากรายได้ของฉัน" is fine: it is
    about their numbers.
  - NEVER re-suggest anything in "already asked in this chat" below — not the
    same wording, and not the same question in other words (same metric +
    period + scope = same question; "วันสรุปยอด" = "วันตัดรอบบัญชี").
    Before writing a chip, check it against that list and against the
    figures this answer/tool data already shows; if it duplicates one, write
    a different angle instead.
  - Neutral wording. No worry/self-judgement framing ("ผมแย่ไหม",
    "จะเป็นอะไรไหม") — ask about the numbers, not about the user.

MODE-NORMAL HARD RULES:
  - QUESTIONS the user asks the ASSISTANT for JUDGMENT, a PLAN or a heads-up —
    never for a number the app already displays (THE SCREEN TEST).
  - NEVER a question the assistant asks the USER back (fails the ANSWER-LOCATION
    TEST above). The user is the asker, the assistant is the answerer.
  - NEVER repeat the question the user just asked, and never re-ask something
    this turn's answer already fully covered. Each chip is a NEW angle.

QUALITY RULES:
  - Thai, spoken and casual — how a user texts a friend who is good with
    money. Short (aim ≤25 chars). No office words: not "สถานะ…", "รายละเอียด
    …", "ของฉัน" at the end, "เป็นอย่างไรบ้าง". Prefer "ไหม / ดี / ยังไง".
      ✗ "สถานะเป้าหมายการเงินของฉัน"  ✓ "ไปญี่ปุ่นทันไหม"
  - Never judging: ask about the money, never scold the user.
      ✗ "ทำไมถึงใช้เยอะจัง"  ✓ "ช้อปเดือนนี้เกินไปไหม"
  - Anchor to a NOTABLE number, category, or the user's goals/budgets/debt
    when relevant — make it feel personal, not generic.

CONTEXT
already asked in this chat (oldest first): {asked_before}
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
    asked_before: Optional[list[str]] = None,
    speculative: Optional["asyncio.Task"] = None,
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
      asked_before:    the user's earlier questions in this thread — chips must
                       not repeat them (the generator used to see one turn only).
      speculative:     a chip call started while the answer was still being
                       written (`start_speculative`). Used unless the answer
                       ends by asking the user back, which needs reply chips.
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

        # Gates passed → the chip LLM below WILL run (a separate ~0–4s call).
        # Signal the client now so it can render a "fetching follow-ups"
        # placeholder during the silent gap before the `suggestions` block
        # lands. Best-effort: a missing writer (unit test / no active stream
        # context) must never break chip generation.
        _emit_pending()

        result = None
        if speculative is not None and not asks_back(answer_text):
            result = await speculative
            slog("suggest", f"speculative chips {'used' if result else 'failed → regenerate'}")
        if result is None:
            result = await _generate(
                user_text=user_text,
                answer_text=answer_text,
                tool_data=tool_data or "(none)",
                user_catalog=_summarize_catalog(user_context),
                asked_before=asked_before,
            )
        if result is None:
            # LLM failed / timed out. Chips are nice-to-have → emit nothing.
            slog("suggest", "skip — LLM unavailable (no fallback)")
            return None
        if result.get("skip"):
            slog("suggest", f"skip — LLM gate ({result.get('reason')!r})")
            return None

        # The question just asked is a repeat too (a chip once echoed it).
        items = _finalize(result.get("items", []),
                          asked_before=[*(asked_before or []), user_text])
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


# Markdown-fence + typographic-quote tolerance for the chip-generator reply.
# The fallback model (gemini-flash-lite, reached via OpenRouter `models[]`
# routing when the primary errs) frequently IGNORES `response_format` and
# returns the JSON wrapped in a ```json fence (`Expecting value … char 0`) or
# with Python-style single quotes / typographic “smart” quotes (`Expecting
# property name … char 1`). Both crashed the strict `json.loads` and dropped
# every chip for the turn. We recover all three shapes here.
_FENCE_LANG_RE = re.compile(r"^```[a-zA-Z0-9]*\n?")
_FENCE_END_RE = re.compile(r"\n?```$")
_SMART_QUOTES = {
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',  # “ ” „ ‟
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",  # ‘ ’ ‚ ‛
}


def _loads_lenient(content: str) -> Any:
    """Parse an LLM JSON reply, tolerating markdown fences, typographic quotes
    and Python-dict single quotes. Raises ValueError/SyntaxError if no usable
    object can be recovered (caller then emits no chips)."""
    s = (content or "").strip()
    # Strip a leading ```json / ``` fence (and any trailing one).
    if s.startswith("```"):
        s = _FENCE_END_RE.sub("", _FENCE_LANG_RE.sub("", s))
    # Isolate the outermost object — drops any prose the model wrapped around it.
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError("no JSON object in LLM reply")
    s = s[i : j + 1].translate(_SMART_QUOTES)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Last resort — single-quoted (Python-dict) JSON. Map JSON keywords to
    # Python literals, then `ast.literal_eval` (never executes code → safe on
    # untrusted text). Only reached after strict JSON already failed.
    py = re.sub(r"\btrue\b", "True", s)
    py = re.sub(r"\bfalse\b", "False", py)
    py = re.sub(r"\bnull\b", "None", py)
    return ast.literal_eval(py)


# Earlier questions sent to the chip prompt. Six covers a full chip chain.
_ASKED_BEFORE_MAX = 6


def asked_before(messages: list) -> list[str]:
    """The user's earlier questions in this thread, oldest first (excluding the
    current one). Internal `[INTENT:…]` markers are not questions."""
    from langchain_core.messages import HumanMessage

    texts = [
        m.content for m in messages
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
        and m.content.strip() and not m.content.startswith("[INTENT:")
    ]
    return texts[:-1][-_ASKED_BEFORE_MAX:]


def tool_data_of(state: dict) -> str:
    """This turn's tool outputs, flattened for the chip prompt."""
    return "\n".join(
        f"[{o.get('tool', '?')}] {o.get('stdout') or o.get('result') or ''}"
        for o in (state.get("tool_outputs_this_turn") or [])
        if isinstance(o, dict) and (o.get("stdout") or o.get("result"))
    ) or "(none)"


# The answer still being written when the chips are started speculatively.
_SPECULATIVE_ANSWER = (
    "(not written yet. It WILL present every row and total in the tool data "
    "below, so any chip whose answer is already in the tool data — the same "
    "list, the top item, a figure already shown — is a repeat: drop it. "
    "Use MODE-NORMAL.)"
)
_SPECULATIVE: dict[str, asyncio.Task] = {}

# The answer asks the user back when its last line is a question. Thai often
# drops the "?", so the common question words at the line end count too.
_ASKS_BACK_RE = re.compile(
    r"(\?|？|ไหม|มั้ย|เท่าไร|เท่าไหร่|อะไร|ยังไง|อย่างไร|หรือเปล่า|ไหน|กี่\S*)"
    r"\s*(ครับ|คะ|ค่ะ|คับ|นะครับ)?\s*[?？]?\s*\**\s*$"
)


def asks_back(answer_text: str) -> bool:
    """True when the answer ends by asking the user something (reply chips)."""
    lines = [ln.strip() for ln in (answer_text or "").splitlines() if ln.strip()]
    return bool(lines) and bool(_ASKS_BACK_RE.search(lines[-1]))


def start_speculative(
    key: str,
    *,
    user_text: str,
    tool_data: str,
    user_context: Optional[dict],
    asked_before: Optional[list[str]],
) -> None:
    """Start the chip call NOW, while the agent writes its answer.

    Why: chips waited for the finished answer and landed ~1.8s after it
    (docs/qa_chip_chain_2026-09-27.md). The answer mostly restates the tool
    data, so the tool data is enough to guess the next questions. Called when
    an agent step starts after tool results; a newer call for the same key
    cancels the older one (the agent went for another tool round)."""
    if not is_enabled() or not key:
        return
    cancel_speculative(key)
    _SPECULATIVE[key] = asyncio.create_task(_generate(
        user_text=user_text,
        answer_text=_SPECULATIVE_ANSWER,
        tool_data=tool_data or "(none)",
        user_catalog=_summarize_catalog(user_context),
        asked_before=asked_before,
    ))


def take_speculative(key: str) -> Optional[asyncio.Task]:
    """Hand over (and forget) the speculative chip call for this turn."""
    return _SPECULATIVE.pop(key, None)


def cancel_speculative(key: str) -> None:
    task = _SPECULATIVE.pop(key, None)
    if task is not None and not task.done():
        task.cancel()


def _emit_pending() -> None:
    """Push a `suggestions_pending` custom event onto the active graph stream.

    Fired once per turn, right before the chip LLM call, on turns that WILL
    produce chips (all gates already passed). The SSE adapter forwards it as
    an `event: suggestions_pending` so the client can show a placeholder in
    the gap. Swallows every failure — outside a node (unit tests / no live
    stream) `get_stream_writer` has no writer, and a hint must never break
    generation.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"suggestions_pending": True})
    except Exception:  # noqa: BLE001 — pending hint is strictly best-effort
        pass


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


# Chips that command something only the app can do. The prompt bans them, yet
# live runs still produced "เพิ่มบัตรเครดิต" / "ตั้งเป้าหมายออมเงิน" — tapping
# one only earns a "do it in the app" redirect, a dead end.
_APP_ONLY_CHIP = re.compile(
    r"^\s*(ช่วย)?\s*(เพิ่ม|สร้าง|ตั้ง|ลบ|แก้ไข|แก้|ย้าย|โอน|export|ส่งออก)\s*"
    r"(กระเป๋า|บัญชี|บัตร|เป้า|งบ|รายการประจำ|หมวด|แท็ก|รายการ|ข้อมูล|แจ้งเตือน|เตือน)"
    # Reminders: the chat cannot schedule one ("เตือนเมื่อถึงวันสรุปยอด").
    r"|^\s*(ช่วย)?\s*(ตั้ง)?\s*(แจ้ง)?เตือน"
)


# A chip that only re-reads what an app screen shows (owner, 2026-09-28: Nimo
# is an advisor, and 84% of chips were such lookups). A chip carrying an
# advisor word (should / enough / what-if / plan / in time) is kept even if it
# also names a total: "ถ้าลดช้อปครึ่งนึง เก็บได้เพิ่มเท่าไหร่" is a plan.
_LOOKUP_CHIP = re.compile(
    r"^(ช่วย|ขอ)?\s*(ดู|แสดง|สรุป|ยอด|รวม|รายการ|รายละเอียด|สถานะ|เทียบ|เงินเข้า|รายรับ|รายได้)"
    r"|อะไร(ไป)?บ้าง\s*\??$"
    r"|(ใช้|จ่าย|เข้า|เหลือ)(ไป|จ่าย|มา)?\s*(รวม)?\s*เท่า(ไหร่|ไร)(แล้ว)?\s*\??$"
    r"|เป็น(อย่างไร|ยังไง)(บ้าง)?\s*\??$"
    # Rankings the breakdown screens already sort: "หมวดไหนใช้เยอะสุด".
    r"|^(เดือนนี้|เดือนที่แล้ว)?\s*(หมวด|บัตร|บัญชี|กระเป๋า)(หมู่)?ไหน.*(เยอะ|สูง|มาก|บ่อย)(ที่)?สุด"
)
_ADVISOR_WORD = re.compile(
    r"ควร|พอ|ไหว|ถ้า|แผน|ทัน|ห่วง|เกินไป|ดีไหม|ดีมั้ย|ปกติ|ทำไม|เก็บเงิน|ออม|ลดได้|ลดตรง"
)


def is_lookup_chip(label: str) -> bool:
    """True for a chip whose answer is just a number/list an app screen shows."""
    text = (label or "").strip()
    return bool(_LOOKUP_CHIP.search(text)) and not _ADVISOR_WORD.search(text)


# "Not now" replies: tapping one only earns "ok, ask me anytime" — a dead turn.
_NOT_NOW_CHIP = re.compile(r"^\s*(ยังไม่|ไม่ต้อง|ไว้ก่อน|ไว้ทีหลัง|ไม่เป็นไร)")

# Product shopping — Nimo advises on the user's OWN money, never picks funds,
# stocks, coins, banks or apps. The prompt bans these, yet "แนะนำกองทุนรวม" /
# "แนะนำหุ้น" / "ควรลงทุนอะไรดี" still reached the wire (chip chains 09-28/29).
# "ควรแบ่งไปลงทุนเท่าไหร่" (how much of their income) stays allowed, and so does
# the provident fund (a payroll deduction, not a product pick).
_PRODUCT_CHIP = re.compile(
    r"กองทุน(?!สำรองเลี้ยงชีพ)|หุ้น|คริปโต|บิทคอยน์|crypto|bitcoin"
    r"|ลงทุน(อะไร|แบบไหน|ที่ไหน|ตัวไหน)|ธนาคารไหน|บัญชี(ดอกเบี้ยสูง|ไหนดี)|แอป(ไหน|อื่น)",
    re.IGNORECASE,
)


def _is_app_only(chip: dict) -> bool:
    return bool(_APP_ONLY_CHIP.match(chip["label"]) or _APP_ONLY_CHIP.match(chip["send"]))


def _finalize(items: list[Any], *, asked_before: Optional[list[str]] = None) -> list[dict]:
    """Normalize LLM items into `{label, send}`; drop app-only action chips,
    "not now" chips, lookups of what the app screens already show (reply chips
    excepted), and repeats of earlier questions; dedup; cap at _MAX_ITEMS."""
    kept: list[dict] = []
    seen: set[str] = {_key(q) for q in (asked_before or []) if q}
    for raw in items:
        chip = _norm_chip(raw)
        if chip is None:
            continue
        if _is_app_only(chip):
            slog("suggest", f"dropped app-only chip {chip['label']!r}")
            continue
        if _NOT_NOW_CHIP.match(chip["label"]):
            slog("suggest", f"dropped not-now chip {chip['label']!r}")
            continue
        if _PRODUCT_CHIP.search(chip["label"]) or _PRODUCT_CHIP.search(chip["send"]):
            slog("suggest", f"dropped product chip {chip['label']!r}")
            continue
        key = _key(chip["send"])
        if key in seen:
            slog("suggest", f"dropped repeat chip {chip['label']!r}")
            continue
        kind = str(raw.get("kind") or "") if isinstance(raw, dict) else ""
        if isinstance(raw, dict) and str(raw.get("repeats") or "").strip():
            slog("suggest", f"dropped self-flagged repeat {chip['label']!r} ~ {raw['repeats']!r}")
            continue
        if kind != "reply" and is_lookup_chip(chip["label"]):
            slog("suggest", f"dropped lookup chip {chip['label']!r}")
            continue
        seen.add(key)
        kept.append(chip)
    return kept[:_MAX_ITEMS]


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
    asked_before: Optional[list[str]] = None,
) -> Optional[dict]:
    """Run the chip generator. Returns the parsed dict, or None on any
    failure (caller then emits nothing)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        slog("suggest", "skipped LLM — OPENROUTER_API_KEY not set")
        return None

    prompt = _PROMPT_TEMPLATE.format(
        asked_before=json.dumps(asked_before or [], ensure_ascii=False),
        user_text=user_text or "(none)",
        answer_text=(answer_text or "")[:1500],
        tool_data=tool_data,
        user_catalog=user_catalog or "(none)",
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = _chips_messages(prompt)
    primary = _model()
    fallbacks = [m for m in _fallback_models() if m != primary]
    url = _base_url().rstrip("/") + "/chat/completions"
    slog_block(
        "suggest",
        f"PROMPT model={primary} timeout={_timeout_s()}s",
        f"user={user_text!r}",
    )

    # OpenRouter's `models[]` only falls back on a request ERROR. A 200 with
    # empty/garbled content (e.g. a reasoning model burning max_tokens on
    # thinking) sails through, so retry once on the next model client-side —
    # but only while the original time budget lasts: chips must not hold `done`.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _timeout_s()
    chain = [primary, *fallbacks]
    for i, model in enumerate(chain[:2]):
        remaining = deadline - loop.time()
        if i > 0:
            if remaining < _MIN_RETRY_S:
                slog("suggest", f"no time left to retry on {model} ({remaining:.1f}s)")
                return None
            slog("suggest", f"retry on fallback {model}")
        parsed = await _request_chips(url, headers, messages, model, chain[i + 1:], remaining)
        if parsed is not None:
            break
    else:
        return None

    skip = bool(parsed.get("skip"))
    items = parsed.get("items")
    if not isinstance(items, list):
        items = []
    return {"skip": skip, "reason": parsed.get("reason"), "items": items}


_MIN_RETRY_S = 1.5  # below this a retry can't finish inside the chip budget


_CONTEXT_MARKER = "\nCONTEXT\n"


def _chips_messages(prompt: str) -> list[dict]:
    """Split the rendered prompt into a cacheable system prefix (the fixed
    instructions, ~2.3K tokens) and a per-turn user message (CONTEXT block).

    Gemini on OpenRouter only caches content marked with `cache_control` —
    before this split every chip call paid full price for the same prefix
    (cached=0 on all calls, 2026-09-28)."""
    head, sep, tail = prompt.partition(_CONTEXT_MARKER)
    if not sep:
        return [{"role": "user", "content": prompt}]
    return [
        {"role": "system", "content": [
            {"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "user", "content": sep.lstrip("\n") + tail},
    ]


def _chips_body(messages: list[dict], model: str, fallbacks: list[str]) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        # A touch of creativity for varied chips, but low enough to stay on-task.
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        # 3 chips, no spares: output is ~70% of a chip call's cost (09-29).
        "max_tokens": 400,
    }
    if fallbacks:
        body["models"] = [model, *fallbacks]
    provider = _resolve_provider()
    if provider is not None:
        body["provider"] = provider
    # Reasoning models (qwen3.7-flash) think by default and can spend the whole
    # max_tokens on thinking → empty content → no chips. Chips need no thinking.
    if (os.getenv("SUGGESTIONS_REASONING") or "").strip().lower() == "off":
        body["reasoning"] = {"enabled": False}
    return body


async def _request_chips(
    url: str, headers: dict, messages: list[dict], model: str,
    fallbacks: list[str], timeout_s: float,
) -> Optional[dict]:
    """One chip request. Returns the parsed dict, or None when the call fails
    or the reply is empty / not a JSON object."""
    body = _chips_body(messages, model, fallbacks)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — every failure → None → emit nothing
        slog_error("suggest", exc)
        return None
    # Per-call cost so the bill can be split by role.
    _log_usage("suggest", data.get("model") or model, data.get("usage"))
    try:
        parsed = _loads_lenient(content)
    except (TypeError, ValueError, SyntaxError) as exc:
        # Log the head of the raw reply so a NEW malformation shape is
        # diagnosable instead of being a silent "no chips this turn".
        slog_error("suggest", exc)
        slog("suggest", f"unparseable content head={ (content or '')[:160]!r }")
        return None
    if not isinstance(parsed, dict):
        slog("suggest", f"parse returned non-dict: {type(parsed).__name__}")
        return None
    return parsed


__all__ = [
    "is_lookup_chip",
    "asked_before",
    "asks_back",
    "tool_data_of",
    "build_suggestions_block",
    "cancel_speculative",
    "start_speculative",
    "take_speculative",
    "is_add_block",
    "is_enabled",
]
