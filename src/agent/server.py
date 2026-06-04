"""FastAPI server for mint_agentic_v3 — wires the pure-ReAct graph to the
mobile contract.

Wave 6 (final integration). Spec sources:
  - `docs/v3/phase3_implementation_plan.md` §2 Wave 6 (this file's exit gate)
  - `docs/v3/phase3_decisions.md` Q1 (Postgres schema v3) + Q4 (no sync graph)
  - `docs/mint_agentic_v3_pure_react_spec.html` §3 (mobile contract — 7 surfaces)
  - Source TRANSFORM: `mint_agentic_v2/src/agent/server.py` (endpoints frozen,
    internals re-pointed at the v3 graph + streaming adapter + endpoint
    handlers).

Mobile contract surface (FROZEN — 0 endpoint or payload changes):
  POST  /chat/stream          — SSE text + image branches
  POST  /chat/voice           — SSE multipart audio
  POST  /transactions/confirm — REST proposal status flip
  POST  /transactions/cancel  — REST proposal status flip
  GET   /threads              — list user threads
  POST  /threads              — create thread
  PATCH /threads/{id}         — rename
  DELETE /threads/{id}        — delete
  GET   /threads/{id}/messages — rich history with blocks (memory:
                                  project_chat_history_block_replay)
  GET   /chat/{thread_id}     — legacy flat-text history alias
  GET   /healthz              — liveness probe

SSE event sequence (Pattern D, identical to v2 wire — memory
`project_sse_ngrok_framing`):
  1. status_token  — Thai status template, streamed word-by-word
  2. answer_token  — token stream of the final answer (raw text, coalesce-
                     immune over ngrok / Cloudflare)
  3. block         — one JSON block per emitted response_block; the `answer`
                     block is appended at end-of-stream by sse_adapter so
                     history replay can render it the same way as proposals
  4. done          — final marker with thread_id
  5. error         — on failure only (no traceback to client)

Cutover (memory `feedback_test_via_cloudflared`):
  CHAT_AGENT_VERSION=v3 → this server hosts the chat contract on :2026,
                          exposed at `https://chat.minttechdev.uk` via
                          Cloudflare tunnel.
  CHAT_AGENT_VERSION=v2 (or unset) → this server REFUSES to start so an
                                     unconfigured rollout can't accidentally
                                     hit v3. Operator runs v2 process instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from dotenv import load_dotenv

# Load .env BEFORE any module reads os.getenv (DATABASE_URL, model names,
# CHAT_AGENT_VERSION). Mirrors v2's pattern — required for uvicorn, pytest,
# and direct `python -m agent.server` entry paths.
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agent.db import (
    PENDING_PROPOSALS_DDL,
    ProposalRepo,
    make_backend_pool,
    make_pool,
)
from src.agent.endpoints.slip_handler import handle_slip_chat
from src.agent.endpoints.voice_handler import handle_voice_chat
from src.agent.entity_catalog import load_entity_catalog, set_catalog_pool
from src.agent.user_preferences import set_preferences_pool
from src.agent.graph import build_graph
from src.agent.llm_openrouter import make_llm_call, make_multimodal_call
from src.agent.messages_repo import MESSAGES_DDL, MessagesRepo
from src.agent.session_logger import (
    _logging_enabled,
    open_session_logger,
    session_log_dir,
    set_session_logger,
    slog,
    slog_block,
    slog_error,
)
from src.agent.preamble import astream_with_preamble
from src.agent.streaming.sse_adapter import stream_chat
from src.agent.threads_repo import THREADS_DDL, ThreadsRepo
from src.agent.utils.version_router import REFUSE_REASON, is_v3_active


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Max length for a server-derived (non-client) thread title — keeps the
# history list tidy; longer first messages are truncated with an ellipsis.
_TITLE_MAX_LEN = 40

# Status template streamed by the text branch before the answer token stream.
# v3 doesn't classify intents up-front (ReAct does it implicitly), so we
# always show the neutral "thinking..." status. The mobile UI animates it
# until the first answer_token arrives.
_STATUS_THINKING = "กำลังคิด..."

# Sentinel value the mobile client may send when the user hasn't picked a
# specific wallet (forwarded to propose_transaction's index-0 cascade).
_EMPTY_WALLET = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    """ISO 8601 UTC timestamp — used for confirm/cancel response payloads."""
    return datetime.now(timezone.utc).isoformat()


async def _stream_status_tokens(text: str) -> AsyncIterator[dict]:
    """Yield one `status_token` event per space-separated word.

    Mirrors v2 + Wave 5 voice/slip handlers so the mobile cubit's status
    animation behaves identically regardless of which endpoint produced it.
    """
    for word in text.split():
        yield {"event": "status_token", "data": word}
        await asyncio.sleep(0)  # cooperative yield — keeps SSE flush snappy


def _derive_thread_title(
    body: "ChatStreamRequest",
    *,
    slip_rows: Optional[list[dict]] = None,
    is_slip: bool = False,
) -> Optional[str]:
    """First-turn display title for a thread (v2 parity).

    Client-supplied title wins; otherwise name the thread from the turn's
    content so a brand-new thread is identifiable in history instead of the
    bare "แชตใหม่" default.

    - slip → first parsed line's note (the merchant); fallback to category;
      multi-line slips append "+N". No-readable returns None.
    - text → first user message, truncated to `_TITLE_MAX_LEN`.

    Returns None when there's nothing worth naming — the upsert COALESCEs on
    conflict so the existing "แชตใหม่" stays.
    """
    client = (getattr(body, "title", None) or "").strip()
    if client:
        return client
    if is_slip:
        if not slip_rows:
            return None
        first = slip_rows[0]
        label = (first.get("note") or "").strip()
        if not label or label == "สลิป":
            label = (first.get("category") or "").strip()
        label = label or "สลิป"
        extra = len(slip_rows) - 1
        return f"{label} +{extra}" if extra > 0 else label
    text = (body.message or "").strip()
    if not text:
        return None
    return text if len(text) <= _TITLE_MAX_LEN else text[:_TITLE_MAX_LEN].rstrip() + "…"


# ---------------------------------------------------------------------------
# Lifespan: version gate, open pools, run DDL, build graph
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan — boot all the singletons the request handlers reuse.

    Boot order (each step depends on the previous):
      1. Version gate — refuse if CHAT_AGENT_VERSION!=v3.
      2. Session log dir.
      3. Open both Postgres pools (checkpoint + backend).
      4. Run idempotent DDL for repos.
      5. Build the checkpointer + long-term-memory store.
      6. Build the v3 graph (ReAct + CodeAct) with the repo injected.
      7. Stash everything on `app.state` for handler access.

    The `agent_graph` slot uses an unusual name (not `graph`) to make
    accidental shadowing of LangGraph types impossible in handler code.
    """
    # 1. Version gate. v3 cannot host the v2 graph; refuse loudly.
    if not is_v3_active():
        raise RuntimeError(REFUSE_REASON)

    # 1b. LLM env contract — fail loud BEFORE opening DB pools / building
    # the graph so a missing/deprecated env surfaces as the first error in
    # the boot log (operators don't have to scroll past unrelated DDL).
    # Covers: deprecated vars (TRANSACTION_LLM_MODEL, INTENT_CLASSIFIER_MODEL,
    # UNDERSTAND_MODEL, TYPOHOON_API_KEY), missing primary models, and
    # fallback chains with fewer than 2 models per role.
    from src.agent.llm_env_validator import validate_llm_env
    validate_llm_env()

    # 2. Session log dir (no-op in production).
    if _logging_enabled():
        try:
            os.makedirs(session_log_dir(), exist_ok=True)
        except Exception:  # noqa: BLE001 — best-effort, debug only
            pass

    # 3. Open the agent's own metadata pool (DATABASE_URL; Q1 carries
    # `?options=-c%20search_path%3Dv3`) and the backend data pool
    # (BACKEND_DATABASE_URL or DATABASE_URL fallback for single-DB envs;
    # memory `reference_agentic_v2_two_databases`).
    pool = make_pool()
    await pool.open()
    backend_pool = make_backend_pool()
    await backend_pool.open()
    # Wire the backend pool into the entity_catalog module so the
    # get_user_context tool can call load_catalog_for_user() without
    # threading a pool through every tool signature.
    set_catalog_pool(backend_pool)
    # Same backend pool feeds the durable user-preferences read/write path
    # (pre_turn loads [about_user]; the set_user_preference tool upserts).
    set_preferences_pool(backend_pool)

    # 4. Idempotent DDL — agent owns its own metadata tables.
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(PENDING_PROPOSALS_DDL)
            await cur.execute(THREADS_DDL)
            await cur.execute(MESSAGES_DDL)

    # 5a. Checkpointer (Postgres AsyncPostgresSaver). v3 sits on the same
    # pool — the search_path in the DSN keeps its tables under the `v3`
    # schema (Q1) so they never collide with v2's checkpoint tables.
    saver_cm = None
    saver = None
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        dsn = os.getenv("DATABASE_URL")
        if dsn:
            saver_cm = AsyncPostgresSaver.from_conn_string(dsn)
            saver = await saver_cm.__aenter__()
            try:
                await saver.setup()
            except Exception:  # noqa: BLE001 — idempotent DDL
                pass
    except Exception as e:  # noqa: BLE001 — degrade gracefully
        slog("server", f"checkpointer disabled: {e!r}")
        saver = None

    # 5c. Repos (audit + transcript).
    repo = ProposalRepo(pool)
    threads = ThreadsRepo(pool)
    messages = MessagesRepo(pool)

    # 6. Build the v3 ReAct graph. `repo` is captured by the pre-turn hook
    # so propose_transaction reaches it via state["__repo__"] every turn.
    # No `store=` — the LangGraph BaseStore was retired with memory_*
    # (its only consumers). Durable user context now lives in user_preferences.
    agent_graph = await build_graph(
        repo=repo,
        checkpointer=saver,
    )

    # 7. Stash on app.state — handlers reach everything via `req.app.state`.
    app.state.pool = pool
    app.state.backend_pool = backend_pool
    app.state.repo = repo
    app.state.threads = threads
    app.state.messages = messages
    app.state.saver = saver
    app.state.agent_graph = agent_graph

    # Vision + STT callables — lazily constructed once so each request reuses
    # the same httpx connection pool. Tests override these via
    # `app.state.vision_call` / `app.state.stt_call` before issuing the
    # request, so the production wiring stays out of the test path.
    try:
        app.state.vision_call = make_multimodal_call("vision")
    except Exception as e:  # noqa: BLE001 — missing key in test env
        slog("server", f"vision_call unavailable at boot: {e!r}")
        app.state.vision_call = None
    try:
        app.state.stt_call = make_multimodal_call("stt")
    except Exception as e:  # noqa: BLE001 — missing key in test env
        slog("server", f"stt_call unavailable at boot: {e!r}")
        app.state.stt_call = None

    try:
        yield
    finally:
        # Tear down in reverse order. Pool close is best-effort — a failure
        # to release a pool mustn't mask a shutdown error from elsewhere.
        try:
            await backend_pool.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await pool.close()
        except Exception:  # noqa: BLE001
            pass
        if saver_cm is not None:
            try:
                await saver_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass


app = FastAPI(title="mint_agentic_v3", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Static web tool (vanilla HTML/JS) — optional, mounted only if shipped.
# ---------------------------------------------------------------------------
_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:  # noqa: D401 — static index passthrough
        return FileResponse(str(_WEB_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Models — Pydantic request bodies (mobile-frozen field set)
# ---------------------------------------------------------------------------


class ChatStreamRequest(BaseModel):
    """Body for `POST /chat/stream`.

    The field set mirrors v2 exactly (memory `project_chat_v2_contract_migration`).
    Optional fields the mobile client sometimes sends but the server doesn't
    require are accepted with sensible defaults so the request never fails
    a 422.
    """

    thread_id: str
    user_id: str
    message: str
    wallet_id: Optional[str] = None
    title: Optional[str] = None
    wallet_ids: Optional[list[str]] = None
    image_b64s: Optional[list[str]] = None
    default_currency_code: Optional[str] = None
    default_currency_symbol: Optional[str] = None


class ConfirmRequest(BaseModel):
    """Body for `POST /transactions/{confirm,cancel}`.

    `thread_id` locates the proposal in the checkpointer state (Add-only
    state machine, memory `project_chat_edit_repropose`); `user_id` is
    included for audit / owner validation.
    """

    proposal_id: str
    user_id: str
    thread_id: Optional[str] = None


class CreateThreadRequest(BaseModel):
    """Body for `POST /threads` (v1 contract)."""

    user_id: str
    title: Optional[str] = None


class RenameThreadRequest(BaseModel):
    """Body for `PATCH /threads/{thread_id}` (v1 contract)."""

    title: str


# ---------------------------------------------------------------------------
# Text branch — drive the v3 graph through sse_adapter.stream_chat
# ---------------------------------------------------------------------------


async def _text_graph_stream(
    app: FastAPI,
    body: ChatStreamRequest,
    *,
    user_kind: str = "text",
) -> AsyncIterator[dict]:
    """One text turn → SSE event dicts.

    Mirrors v2's `_text_graph_stream` but routes through Wave 5's
    `stream_chat` adapter (which owns the astream stream-mode wiring + block
    emission) instead of replicating the astream_events filter inline.

    Persists the transcript via the messages repo so reload replay sees the
    same blocks (memory `project_chat_history_block_replay`). Write failures
    are swallowed — the user already received every event live.

    `user_kind` is the discriminator persisted on the user message so history
    replay can tag its origin (voice vs text vs slip). Mobile renders a mic /
    photo bubble based on it.

    Intent routing lives INSIDE the graph: the stateful `classify_intent` node
    (src/agent/nodes/classify_intent.py, CLASSIFY_ROUTER_ENABLED, default OFF)
    can shortcut an unambiguous complete ADD to the deterministic `direct_propose`
    node; everything else flows through the full ReAct graph. No pre-graph
    classification happens here.
    """
    g = app.state.agent_graph

    # Tip-of-turn status — v3 doesn't pre-classify intent (ReAct decides via
    # tool calls), so we always start with the neutral "thinking" template.
    async for chunk in _stream_status_tokens(_STATUS_THINKING):
        yield chunk

    # The v3 graph reads from state["messages"] for ReAct's HumanMessage
    # input. Per Wave 4 graph design, the pre_turn_hook clears scratch
    # channels; we only need to seed the user message + wallet hint.
    init_state: dict = {
        "user_id": body.user_id,
        "thread_id": body.thread_id,
        # ReAct expects messages as the input channel. Append-only via
        # `add_messages` reducer — passing the latest user turn as a list
        # makes the reducer append (not replace) onto any prior state.
        "messages": [{"role": "user", "content": body.message}],
        # Client-chosen wallet → propose_transaction cascade step-2 (read as
        # state["wallet_id"], ahead of the catalog default). Empty string
        # honors the index-0 fallback (memory `project_wallet_index0_ordering`).
        "wallet_id": body.wallet_id or _EMPTY_WALLET,
    }
    config: dict[str, Any] = {
        "configurable": {"thread_id": body.thread_id},
        "recursion_limit": 25,
    }

    # Streaming preamble — "thinking out loud" warm-up that now races the
    # ReAct loop CONCURRENTLY (astream_with_preamble) instead of blocking ahead
    # of it: a 1-sentence Thai acknowledgment from a fast cheap model, emitted
    # as ONE ephemeral `narration_token` IF it wins the race against the first
    # answer token; dropped (and its task cancelled) the moment the real answer
    # starts or the stream ends — so a slow/dead preamble can never delay the
    # turn (previously an 8s ReadTimeout stalled ReAct). Never numbers/answers,
    # so it can't contradict the real answer; routed on the narration stream
    # (DYNAMIC LLM text), distinct from STATIC tool/progress `status_token`s.
    #
    # Capture blocks + answer for transcript persistence. The adapter yields
    # validated dicts; we re-parse the data field (cheap) instead of weaving
    # a side-channel through stream_chat. The preamble is NOT captured here — it
    # streams as an ephemeral `narration_token`, so it must not persist into the
    # assistant message (history would otherwise replay it in front of the
    # answer, the very leak we're fixing).
    turn_blocks: list[dict[str, Any]] = []
    answer_chars: list[str] = []
    error_for_log: Optional[BaseException] = None
    slog("server.response", "stream_chat ENTER")
    try:
        async for ev in astream_with_preamble(
            body.message,
            stream_chat(g, init_state, config, thread_id=body.thread_id),
            event_type="narration_token",
        ):
            etype = ev.get("event")
            if etype == "block":
                try:
                    blk = json.loads(ev["data"])
                except Exception:  # noqa: BLE001 — already validated, defensive
                    blk = None
                if isinstance(blk, dict):
                    turn_blocks.append(blk)
                    # Per-block log — captures wallet_required / proposal /
                    # answer blocks the moment they reach the wire.
                    slog_block(
                        "server.response",
                        f"BLOCK type={blk.get('type')}",
                        json.dumps(blk, ensure_ascii=False, indent=2),
                    )
            elif etype == "answer_token":
                tok = ev.get("data", "")
                answer_chars.append(tok)
            yield ev

        # Empty-LLM fallback: when the upstream model returns an error
        # (e.g. OpenRouter / Gemini `finish_reason="error"` — content="" with
        # no tool_calls) the stream ends with zero answer tokens and zero
        # blocks. Without this guard the mobile chat shows a blank assistant
        # bubble. Emit a Thai apology + matching answer block so the user
        # always sees SOMETHING and knows to retry. The preamble no longer
        # feeds `answer_chars` (it streams as an ephemeral status_token), so
        # any answer token at all means the main loop produced a real answer.
        main_answered = bool(answer_chars)
        if not main_answered and not turn_blocks:
            fallback = (
                "ขออภัยค่ะ ตอนนี้ระบบขัดข้องชั่วคราว ลองพิมพ์ใหม่อีกครั้งได้เลยนะคะ"
            )
            slog("server.response",
                 "LLM produced no answer + no blocks — emitting Thai fallback")
            yield {"event": "answer_token", "data": fallback}
            answer_chars.append(fallback)
            fb_block = {"type": "answer", "text": fallback}
            yield {"event": "block",
                   "data": json.dumps(fb_block, ensure_ascii=False)}
            turn_blocks.append(fb_block)

        # Thread metadata + transcript persistence. Title only persists on
        # first insert (COALESCE on conflict).
        try:
            await app.state.threads.upsert(
                thread_id=body.thread_id,
                user_id=body.user_id,
                title=_derive_thread_title(body),
            )
            user_content: dict[str, Any] = {"text": body.message}
            if user_kind != "text":
                user_content["kind"] = user_kind
            await app.state.messages.append(
                thread_id=body.thread_id, user_id=body.user_id,
                role="user", content=user_content,
            )
            await app.state.messages.append(
                thread_id=body.thread_id, user_id=body.user_id,
                role="assistant",
                content={
                    "blocks": turn_blocks,
                    "answer": ("".join(answer_chars)) or None,
                },
            )
        except Exception:  # noqa: BLE001 — debug log, never fail the turn
            slog("server", "transcript persistence failed (silently swallowed)")

    except Exception as e:  # noqa: BLE001 — forward all errors to client
        error_for_log = e
        tb = traceback.format_exc()
        slog_error("server", e, tb)
        yield {
            "event": "error",
            "data": json.dumps(
                {"code": type(e).__name__, "message": str(e)},
                ensure_ascii=False,
            ),
        }
    finally:
        # ALWAYS log final state — even on cancel/timeout/error — so the log
        # never has a silent gap between TURN header and footer.
        final_answer = "".join(answer_chars) or ""
        slog_block(
            "server.response",
            f"FINAL answer_chars={len(final_answer)} blocks={len(turn_blocks)}"
            + (f" error={type(error_for_log).__name__}" if error_for_log else ""),
            "ANSWER:\n" + (final_answer or "(empty)") + "\n\nBLOCKS:\n" + (
                json.dumps(turn_blocks, ensure_ascii=False, indent=2)
                if turn_blocks else "(none)"
            ),
        )


# ---------------------------------------------------------------------------
# Slip branch — short-circuit to slip_handler (skip ReAct per decision α)
# ---------------------------------------------------------------------------


async def _slip_stream(app: FastAPI, body: ChatStreamRequest) -> AsyncIterator[dict]:
    """One slip turn: vision -> transaction_proposal_group block.

    Per Wave 5 design (decision α), the slip flow SKIPS the ReAct loop
    entirely — one vision call + one group block. Persists the group
    proposal to the checkpointer state via `as_node="finalize"` so the
    confirm/cancel endpoint can resolve it later (memory:
    `project_slip_vision_as_node`).
    """
    if not body.image_b64s:
        # Defensive — caller should have routed to text branch.
        return

    # Load catalog scoped to the user. The slip handler picks the wallet +
    # categories from this.
    catalog = await load_entity_catalog(app.state.backend_pool, body.user_id)

    # Vision call — production wires it; tests override via app.state.
    vision_call = app.state.vision_call
    if vision_call is None:
        # Defer construction to handle test envs that boot without an API key.
        vision_call = make_multimodal_call("vision")

    # Capture blocks for persistence + confirm-state seeding.
    turn_blocks: list[dict[str, Any]] = []
    group_block: Optional[dict[str, Any]] = None
    async for ev in handle_slip_chat(
        vision_call=vision_call,
        catalog=catalog,
        image_b64s=body.image_b64s,
        thread_id=body.thread_id,
        user_id=body.user_id,
        wallet_id=body.wallet_id,
        default_currency_code=body.default_currency_code or "THB",
        default_currency_symbol=body.default_currency_symbol or "฿",
    ):
        if ev.get("event") == "block":
            try:
                blk = json.loads(ev["data"])
            except Exception:  # noqa: BLE001
                blk = None
            if isinstance(blk, dict):
                turn_blocks.append(blk)
                if blk.get("type") == "transaction_proposal_group":
                    group_block = blk
        yield ev

    # Persist the group proposal — dual write (audit table + checkpoint state)
    # so /transactions/confirm can resolve it. Skipped on unreadable / no-wallet
    # turns where no group block was emitted.
    if group_block is not None:
        try:
            await _persist_slip_proposal(app, body, group_block)
        except Exception as e:  # noqa: BLE001 — debug log, never fail turn
            slog_error("server", e)

    # Best-effort transcript persistence (mirrors v2 _persist_slip_turn).
    try:
        slip_rows = (group_block or {}).get("transactions") if group_block else None
        await app.state.threads.upsert(
            thread_id=body.thread_id, user_id=body.user_id,
            title=_derive_thread_title(body, slip_rows=slip_rows, is_slip=True),
        )
        await app.state.messages.append(
            thread_id=body.thread_id, user_id=body.user_id,
            role="user",
            content={
                "text": body.message,
                "kind": "slip",
                "image_count": len(body.image_b64s or []),
            },
        )
        await app.state.messages.append(
            thread_id=body.thread_id, user_id=body.user_id,
            role="assistant", content={"blocks": turn_blocks, "answer": None},
        )
    except Exception:  # noqa: BLE001 — transcript persistence is best-effort
        slog("server", "slip transcript persistence failed (swallowed)")


async def _persist_slip_proposal(
    app: FastAPI, body: ChatStreamRequest, group_block: dict
) -> None:
    """Dual-persist a slip group proposal (audit + checkpoint state).

    The slip flow short-circuits BEFORE the graph runs, so we have to seed
    state ourselves. Per memory `project_slip_vision_as_node`,
    `aupdate_state` MUST pass `as_node="finalize"` — otherwise LangGraph
    can't infer the owning node on a fresh thread and raises
    InvalidUpdateError.

    Append semantics: `AgentState.proposals` has no custom reducer, so the
    default reducer REPLACES the value on `aupdate_state`. To APPEND (so
    prior proposals on this thread survive) we read the current list,
    append our entry, then write the whole list back.
    """
    proposal_id = group_block.get("proposal_id") or group_block.get("group_id")
    if not proposal_id:
        return
    payload = {
        "group_id": group_block.get("group_id"),
        "total": group_block.get("total"),
        "wallet_sync_id": group_block.get("wallet_sync_id"),
        "currency_code": group_block.get("currency_code"),
        "user_id": body.user_id,
        "transactions": group_block.get("transactions") or [],
    }

    # 1. Audit table — covers legacy confirm/cancel paths.
    try:
        await app.state.repo.insert_pending_proposal(
            proposal_id=proposal_id,
            user_id=body.user_id,
            kind="ADD_TRANSACTION_GROUP",
            payload=payload,
        )
    except Exception as e:  # noqa: BLE001 — log + continue, state seed still works
        slog_error("server", e)

    # 2. Checkpointer state seed — APPEND, never clobber.
    entry = {
        "proposal_id": proposal_id,
        "intent_type": "ADD_TRANSACTION_GROUP",
        "type": "ADD_TRANSACTION_GROUP",
        "payload": payload,
        "status": "pending",
        "created_at": _iso_now(),
        "confirmed_at": None,
        "cancelled_at": None,
    }
    g = app.state.agent_graph
    cfg = {"configurable": {"thread_id": body.thread_id}}
    snap = await g.aget_state(cfg)
    existing = list(((snap.values or {}) if snap else {}).get("proposals") or [])
    existing.append(entry)
    # Attribute the out-of-graph write to `post_turn` — the v3 terminal node
    # (edge → END). v3 has NO "finalize" node (that was v2); using it raises
    # InvalidUpdateError → 500. `post_turn` schedules no pending tasks, so the
    # state reads as a cleanly completed turn.
    await g.aupdate_state(cfg, {"proposals": existing}, as_node="post_turn")


# ---------------------------------------------------------------------------
# /chat/stream — top-level entry that picks text vs slip branch
# ---------------------------------------------------------------------------


async def _chat_stream_generator(
    app: FastAPI, body: ChatStreamRequest
) -> AsyncIterator[dict]:
    """Pick the slip branch (when image_b64s present) or the text branch.

    Wraps both with the per-session debug logger so an investigator gets one
    file per thread_id across multiple turns.
    """
    # Per-session debug logger: 1 thread_id = 1 file (append across turns).
    session_logger = open_session_logger(body.thread_id, body.user_id)
    set_session_logger(session_logger)
    session_logger.turn_header(
        body.user_id, body.message, body.wallet_id,
        extra={"thread": body.thread_id},
    )
    _turn_start = time.monotonic()
    try:
        if body.image_b64s:
            slog("server", "slip branch — image attached")
            async for chunk in _slip_stream(app, body):
                yield chunk
        else:
            async for chunk in _text_graph_stream(app, body):
                yield chunk
    except Exception as e:  # noqa: BLE001 — forward all errors to client
        tb = traceback.format_exc()
        slog_error("server", e, tb)
        yield {
            "event": "error",
            "data": json.dumps(
                {"code": type(e).__name__, "message": str(e)},
                ensure_ascii=False,
            ),
        }
    finally:
        session_logger.turn_footer(time.monotonic() - _turn_start)


@app.post("/chat/stream")
async def chat_stream(req: Request, body: ChatStreamRequest):
    """SSE chat endpoint — text or slip branch picked by `image_b64s` presence."""
    return EventSourceResponse(_chat_stream_generator(req.app, body))


# ---------------------------------------------------------------------------
# /chat/voice — multipart audio
# ---------------------------------------------------------------------------


async def _voice_stream_generator(
    app: FastAPI, body: ChatStreamRequest, audio_bytes: bytes
) -> AsyncIterator[dict]:
    """One voice turn — Wave 5 voice_handler owns the STT + graph orchestration.

    The session logger header records `[voice]` as the placeholder text (the
    real transcript isn't known until STT completes — voice_handler logs it
    after STT inside the handler).
    """
    session_logger = open_session_logger(body.thread_id, body.user_id)
    set_session_logger(session_logger)
    session_logger.turn_header(
        body.user_id, "[voice]", body.wallet_id,
        extra={"thread": body.thread_id},
    )
    _turn_start = time.monotonic()

    stt_call = app.state.stt_call
    if stt_call is None:
        stt_call = make_multimodal_call("stt")

    # Transcript + voice blocks captured for transcript persistence.
    turn_blocks: list[dict[str, Any]] = []
    answer_chars: list[str] = []
    transcript_text: Optional[str] = None

    try:
        async for ev in handle_voice_chat(
            graph=app.state.agent_graph,
            stt_call=stt_call,
            audio_bytes=audio_bytes,
            thread_id=body.thread_id,
            user_id=body.user_id,
            wallet_id=body.wallet_id,
        ):
            etype = ev.get("event")
            if etype == "block":
                try:
                    blk = json.loads(ev["data"])
                except Exception:  # noqa: BLE001
                    blk = None
                if isinstance(blk, dict):
                    turn_blocks.append(blk)
                    if blk.get("type") == "transcript":
                        transcript_text = blk.get("text") or transcript_text
            elif etype == "answer_token":
                answer_chars.append(ev.get("data", ""))
            yield ev

        # Persist transcript so reload shows the voice turn. Tag the user
        # message with `kind=voice` and the heard text so the mobile UI
        # renders the mic bubble.
        try:
            user_text = transcript_text or body.message or ""
            await app.state.threads.upsert(
                thread_id=body.thread_id, user_id=body.user_id,
                title=None,  # voice carries no client title; let upsert COALESCE
            )
            await app.state.messages.append(
                thread_id=body.thread_id, user_id=body.user_id,
                role="user",
                content={"text": user_text, "kind": "voice"},
            )
            await app.state.messages.append(
                thread_id=body.thread_id, user_id=body.user_id,
                role="assistant",
                content={"blocks": turn_blocks,
                         "answer": "".join(answer_chars) or None},
            )
        except Exception:  # noqa: BLE001
            slog("server", "voice transcript persistence failed (swallowed)")

    except Exception as e:  # noqa: BLE001 — forward all errors to client
        tb = traceback.format_exc()
        slog_error("server", e, tb)
        yield {
            "event": "error",
            "data": json.dumps(
                {"code": type(e).__name__, "message": str(e)},
                ensure_ascii=False,
            ),
        }
    finally:
        session_logger.turn_footer(time.monotonic() - _turn_start)


@app.post("/chat/voice")
async def chat_voice(
    req: Request,
    user_id: str = Form(...),
    thread_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """Voice-to-proposal — multipart `audio` (m4a) + form fields.

    Voice carries no wallet selection, so propose_transaction's cascade
    falls back to index-0 (memory `project_wallet_index0_ordering`).
    """
    audio_bytes = await audio.read()
    body = ChatStreamRequest(
        thread_id=thread_id,
        user_id=user_id,
        message="",  # filled with the transcript after STT
        wallet_id=None,
    )
    return EventSourceResponse(
        _voice_stream_generator(req.app, body, audio_bytes)
    )


# ---------------------------------------------------------------------------
# Threads CRUD — v1 contract (thread_id / updated_at / etc.)
# ---------------------------------------------------------------------------


@app.get("/threads")
async def list_threads(req: Request, user_id: str, limit: int = 50):
    """List a user's threads sorted by recency.

    `next_cursor` is reserved for future pagination (always null for now —
    matches v2 contract).
    """
    rows = await req.app.state.threads.list_for_user(user_id, limit)
    return {"threads": rows, "next_cursor": None}


@app.post("/threads", status_code=201)
async def create_thread(req: Request, body: CreateThreadRequest):
    """Allocate a fresh thread_id + empty metadata row."""
    return await req.app.state.threads.create(
        user_id=body.user_id, title=body.title,
    )


@app.patch("/threads/{thread_id}")
async def rename_thread(req: Request, thread_id: str, body: RenameThreadRequest):
    """Update the display title."""
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "title cannot be empty")
    row = await req.app.state.threads.rename(thread_id, title)
    if not row:
        raise HTTPException(404, "thread not found")
    return row


@app.get("/threads/{thread_id}")
async def get_thread(req: Request, thread_id: str):
    """One thread's metadata (v1 ThreadOut)."""
    row = await req.app.state.threads.get(thread_id)
    if not row:
        raise HTTPException(404, "thread not found")
    return row


@app.delete("/threads/{thread_id}")
async def delete_thread(req: Request, thread_id: str):
    """Idempotent delete — `found=False` on a second call."""
    saver = req.app.state.saver
    if saver is not None:
        try:
            await saver.adelete_thread(thread_id)  # type: ignore[attr-defined]
        except Exception:
            # API exists only on newer langgraph-checkpoint-postgres versions.
            pass
    await req.app.state.messages.delete_for_thread(thread_id)
    found = await req.app.state.threads.delete(thread_id)
    return {"status": "deleted", "thread_id": thread_id, "found": found}


# ---------------------------------------------------------------------------
# History endpoints
#   /threads/{id}/messages — rich blocks (memory project_chat_history_block_replay)
#   /chat/{thread_id}      — legacy flat text (mobile fallback)
# ---------------------------------------------------------------------------


@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(req: Request, thread_id: str):
    """Full transcript, oldest -> newest, with backend-injected proposal status.

    Memory `project_chat_history_block_replay`: reopening a thread renders
    blocks (proposal_proposal_group, suggestions, ...) by reusing this
    response. Proposal blocks are annotated with their CURRENT status
    (pending / confirmed / cancelled / discarded) read from the live
    checkpointer state so a finalized card doesn't render as stale-pending.
    """
    messages = await req.app.state.messages.list_for_thread(thread_id)

    # Build proposal_id -> status from graph state (best-effort).
    status_by_id: dict[str, str] = {}
    try:
        snap = await req.app.state.agent_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        for p in ((snap.values or {}) if snap else {}).get("proposals") or []:
            pid = p.get("proposal_id")
            if pid:
                status_by_id[pid] = p.get("status", "pending")
    except Exception:  # noqa: BLE001 — best-effort, history still renders
        pass

    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or {}
        blocks = content.get("blocks") or []
        kept: list = []
        for blk in blocks:
            pid = blk.get("proposal_id")
            status = status_by_id.get(pid) if pid else None
            # Drop superseded cards (a merge replaced them) + transient
            # hide_block markers — mirrors v2 filter logic. Discarded
            # proposals stay (mobile renders the compact "ไม่ได้บันทึก" card).
            if status == "superseded" or blk.get("type") == "hide_block":
                continue
            if pid and status:
                blk["status"] = status
            kept.append(blk)
        content["blocks"] = kept

    # Lean-hybrid live status: annotate each saved proposal row with its
    # CURRENT transaction state (saved / edited / deleted / unknown) by
    # joining the live transactions table on sync_id. The snapshot body of
    # the block is left untouched — mobile renders the historical card and
    # uses `tx_state` only to badge it (so edit/delete done from the
    # transaction list shows up on reopen without a server-side rewrite).
    await _annotate_tx_state(
        messages, getattr(req.app.state, "backend_pool", None)
    )

    return {"messages": messages}


# Columns the edit-comparison + delete-detection read. `is_deleted` is the
# tombstone flag (soft delete); a purged tombstone simply yields no row.
_TX_STATE_SQL = (
    "SELECT sync_id, is_deleted, amount, category_sync_id, wallet_sync_id, note "
    "FROM transactions WHERE sync_id = ANY(%s)"
)


def _proposal_row_txns(blocks: Any) -> list[dict]:
    """Every per-row transaction dict across a turn's proposal blocks.

    A single `transaction_proposal` carries one row under `transaction`; a
    `transaction_proposal_group` (slip) carries N under `transactions`.
    Returned dicts are the live objects inside `blocks` so callers can
    annotate them in place.
    """
    out: list[dict] = []
    for blk in blocks or []:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype in ("transaction_proposal", "transaction_verify_proposal"):
            txn = blk.get("transaction")
            if isinstance(txn, dict):
                out.append(txn)
        elif btype == "transaction_proposal_group":
            for txn in blk.get("transactions") or []:
                if isinstance(txn, dict):
                    out.append(txn)
    return out


def _compute_tx_state(txn: dict, row: tuple | None) -> str:
    """Resolve one proposal row's live state against its DB record.

    `row` = (sync_id, is_deleted, amount, category_sync_id, wallet_sync_id,
    note) or None when no transaction carries that sync_id — a pre-stable-id
    card (saved before sync_id reuse) or a purged tombstone — which maps to
    `unknown` so mobile keeps the snapshot and shows NO badge (never a false
    "deleted").
    """
    if row is None:
        return "unknown"
    _sid, is_deleted, db_amount, db_cat, db_wallet, db_note = row
    if is_deleted:
        return "deleted"
    # Compare only user-editable fields. Amount is float8 → round to 2dp so
    # display-equal values don't read as edited. Date is intentionally
    # skipped (tz/precision noise would yield false "edited").
    try:
        snap_amount = round(float(txn.get("amount") or 0), 2)
    except (TypeError, ValueError):
        snap_amount = None
    same = (
        snap_amount == round(float(db_amount or 0), 2)
        and (txn.get("category_sync_id") or None) == (db_cat or None)
        and (txn.get("wallet_sync_id") or None) == (db_wallet or None)
        and (txn.get("note") or "") == (db_note or "")
    )
    return "saved" if same else "edited"


async def _annotate_tx_state(messages: list, backend_pool: Any) -> None:
    """Inject `tx_state` into every proposal row by joining the live
    transactions table on `sync_id`.

    Best-effort: a missing pool or query failure leaves rows unannotated
    (mobile → `unknown` → snapshot), so history still renders. One batched
    query per thread — no N+1.
    """
    if backend_pool is None:
        return
    txns: list[dict] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        blocks = (m.get("content") or {}).get("blocks") or []
        txns.extend(_proposal_row_txns(blocks))
    sync_ids = [
        s for s in {t.get("sync_id") for t in txns} if isinstance(s, str) and s
    ]
    if not sync_ids:
        return
    try:
        async with backend_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_TX_STATE_SQL, (sync_ids,))
                rows = await cur.fetchall()
    except Exception:  # noqa: BLE001 — best-effort; leave unannotated
        return
    by_sync = {r[0]: r for r in rows}
    for txn in txns:
        txn["tx_state"] = _compute_tx_state(txn, by_sync.get(txn.get("sync_id")))


def _flatten_assistant_block_text(blocks: Any) -> str:
    """Human-readable text for the v1 flat-history alias.

    Reads `blocks[].text` so proposal/clarification turns whose answer is in
    a block instead of the answer field don't flatten to an empty bubble.
    `suggestions` blocks become chips, not bubble text.
    """
    parts: list[str] = []
    for blk in blocks or []:
        if not isinstance(blk, dict) or blk.get("type") == "suggestions":
            continue
        text = blk.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _suggestion_items_from_blocks(blocks: Any) -> list[str]:
    """Pull the items of the first `suggestions` block so reload re-surfaces
    the follow-up chips."""
    for blk in blocks or []:
        if isinstance(blk, dict) and blk.get("type") == "suggestions":
            return [
                s.strip()
                for s in (blk.get("items") or [])
                if isinstance(s, str) and s.strip()
            ]
    return []


@app.get("/chat/{thread_id}")
async def get_chat_history(req: Request, thread_id: str):
    """v1 flat-history alias — bubbles `{role, content, suggestions}`.

    Mobile fallback (newer builds prefer /threads/{id}/messages). Empty
    bubbles are skipped to mirror v1.
    """
    rows = await req.app.state.messages.list_for_thread(thread_id)
    out: list[dict[str, Any]] = []
    for m in rows:
        role = m.get("role") or "assistant"
        content_obj = m.get("content") or {}
        if role == "user":
            text = content_obj.get("text") or ""
            suggestions: list[str] = []
        else:
            blocks = content_obj.get("blocks") or []
            text = content_obj.get("answer") or _flatten_assistant_block_text(blocks)
            suggestions = _suggestion_items_from_blocks(blocks)
        if not text:
            continue
        out.append({"role": role, "content": text, "suggestions": suggestions})
    return {"thread_id": thread_id, "messages": out}


# ---------------------------------------------------------------------------
# /transactions/{confirm,cancel} — proposal status flip
# ---------------------------------------------------------------------------


async def _finalize_proposal_in_state(
    app: FastAPI,
    *,
    thread_id: str,
    proposal_id: str,
    user_id: str,
    new_status: str,
) -> dict:
    """Common confirm/cancel logic.

    Resolves the proposal inside the LangGraph checkpointer state, flips its
    status in-place, and (for confirms) writes `last_txn` so the next turn's
    resolver can see the freshest confirmed txn.

    `as_node="finalize"` is REQUIRED on `aupdate_state` per memory
    `project_slip_vision_as_node` — both in-graph proposals (which actually
    ran the graph to `finalize`) and slip/voice proposals (seeded
    out-of-graph at `finalize`) end at the same terminal node, so attributing
    the write there is correct + ambiguity-free.
    """
    g = app.state.agent_graph
    values: dict = {}
    proposals: list = []
    if thread_id:
        cfg = {"configurable": {"thread_id": thread_id}}
        snap = await g.aget_state(cfg)
        values = (snap.values or {}) if snap else {}
        proposals = list(values.get("proposals") or [])

    idx = next(
        (i for i, p in enumerate(proposals) if p.get("proposal_id") == proposal_id),
        -1,
    )
    if idx == -1:
        raise HTTPException(404, "proposal_not_found")

    entry = dict(proposals[idx])
    current = entry.get("status", "pending")
    if current != "pending":
        raise HTTPException(
            409,
            {"error": "already_finalized", "current_status": current},
        )

    ts = _iso_now()
    entry["status"] = new_status
    if new_status == "confirmed":
        entry["confirmed_at"] = ts
    else:
        entry["cancelled_at"] = ts
    proposals[idx] = entry

    update: dict = {"proposals": proposals}
    if new_status == "confirmed" and entry.get("intent_type") == "ADD_TRANSACTION":
        # Persist the just-confirmed txn into short-term memory so the next
        # turn's resolver can find it as `last_txn`.
        p = entry.get("payload") or {}
        update["last_txn"] = {
            "id": entry.get("proposal_id"),
            "sync_id": p.get("sync_id"),
            "type": p.get("type"),
            "amount": p.get("amount"),
            "category": p.get("category"),
            "category_sync_id": p.get("category_sync_id"),
            "wallet_sync_id": p.get("wallet_sync_id"),
            "pending": False,
        }
    elif new_status == "cancelled":
        # Clear last_txn if it points at the now-cancelled proposal — a
        # follow-up "แก้..." must not target a cancelled card.
        lt = values.get("last_txn") or {}
        if lt.get("id") == proposal_id:
            update["last_txn"] = None

    # Attribute to `post_turn` — v3's terminal node (→ END). v3 has NO
    # "finalize" node; using it raises InvalidUpdateError → 500, which left
    # confirm/cancel silently failing (proposal status stuck at "pending", so
    # history replayed every finalized card as a re-savable proposal).
    await g.aupdate_state(cfg, update, as_node="post_turn")

    return {
        "ok": True,
        "proposal_id": proposal_id,
        "status": new_status,
        ("confirmed_at" if new_status == "confirmed" else "cancelled_at"): ts,
    }


@app.post("/transactions/confirm")
async def confirm_transaction(req: Request, body: ConfirmRequest):
    """Mark a transaction proposal confirmed.

    Python does NOT write the real `transactions` row — mobile-side
    CreateTransactionUseCase already did that via Drift+sync using the
    same `sync_id` embedded in the proposal payload. This endpoint only
    updates conversation state.
    """
    if not body.thread_id:
        raise HTTPException(400, "thread_id is required for v3 confirm")
    result = await _finalize_proposal_in_state(
        req.app,
        thread_id=body.thread_id,
        proposal_id=body.proposal_id,
        user_id=body.user_id,
        new_status="confirmed",
    )
    slog("proposal", f"confirm ok ({body.proposal_id})")
    return JSONResponse(result)


@app.post("/transactions/cancel")
async def cancel_transaction(req: Request, body: ConfirmRequest):
    """Mark a transaction proposal cancelled. `last_txn` is cleared when it
    points at the cancelled proposal (so the next "แก้..." doesn't latch on
    to a cancelled card)."""
    if not body.thread_id:
        raise HTTPException(400, "thread_id is required for v3 cancel")
    result = await _finalize_proposal_in_state(
        req.app,
        thread_id=body.thread_id,
        proposal_id=body.proposal_id,
        user_id=body.user_id,
        new_status="cancelled",
    )
    slog("proposal", f"cancel ok ({body.proposal_id})")
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Health + diagnostics
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz():
    """Liveness probe.

    Cloudflare tunnel + uvicorn watcher both hit this — keep it cheap.
    Reports the active version so an operator can verify the cutover at a
    glance.
    """
    return {"ok": True, "version": "v3"}


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------


def run():  # pragma: no cover — convenience for local dev only
    import uvicorn
    uvicorn.run(
        "agent.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "2026")),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
