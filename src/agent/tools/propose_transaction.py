"""`propose_transaction` — atomic discard + joint wallet/category resolve.

NEW in Wave 3, refactored to JOINT 1-LLM-call resolution.

Implements the atomic guard from Decision X (memory
`project_proposal_discard_state_machine` / `project_chat_edit_repropose`):

  "Auto-discard ALL pending proposals at the start of every ADD-like
   intent, replacing the v1 block-when-pending. Closes the late-Confirm
   hole; discarded ≠ superseded on reload."

PLUS — Wave 5 joint resolution: replaces the legacy 2-call chain
(`resolve_wallet_choice_async` → `resolve_category_for_add_async`) with a
single structured-output call against a dedicated `propose` LLM role
(env `PROPOSE_MODEL`). Cuts ~half the tail latency on the hot ADD path
and gives the LLM joint context (wallet hint informs category, vice versa).

Flow when invoked:
  1. Status emit "กำลังบันทึก..."
  2. Canonicalize amount via Decimal (NO float arithmetic).
  3. If a pending_proposal exists → mark discarded, append `discard_proposal`
     block BEFORE anything else (atomic guard — never stack pending).
  4. Read/auto-load `user_context`. No wallets → onboarding shortcut.
  5. WALLET RESOLUTION CHAIN:
       a) `wallet_label` given → joint LLM (all wallets, nested cats)
       b) else `state["wallet_id"]` (client pick) valid → use it (single-wallet LLM)
       c) else `user_context["wallet_id"]` (catalog default) valid → use it
       d) else index-0: general → creditcard → first wallet
  6. CATEGORY CANDIDATE FILTER:
       - `type` ∈ {expense, income} → filter cats to that type
       - `type` == "auto" → all cats; LLM picks type via category choice
  7. FAST PATH skip LLM: wallet known (b/c/d) + no `category_label` + no
     `note` → 0 calls; category=Other-floor.
  8. JOINT LLM CALL (`_resolve_pair_async`) — structured output `_PairChoice`.
     On failure (timeout/transport/validation/out-of-set wallet) → return
     error `joint_resolve_failed` so ReAct can retry/inform the user.
  9. Validators (post-LLM):
       - chosen wallet must be in candidates (else error)
       - chosen category must belong to chosen wallet (else reject cat,
         use Other-floor — Q9a cross-wallet leak guard)
       - chosen category type must match `type` arg when given (else reject)
 10. Final `transaction.type`:
       - chosen cat → `cat.type` (DB is source of truth)
       - no chosen cat + `type` ∈ {expense,income} → that
       - no chosen cat + `type=="auto"` → "expense" (Q12a default)
 11. Persist to audit table; emit `transaction_proposal` block.

Error handling: ALL failure paths return a Command (never raise). ReAct
would dead-loop on an unhandled exception inside a tool.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langsmith import traceable as _ls_traceable
from pydantic import BaseModel, Field, ValidationError

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover — older LangGraph fallback
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.entity_catalog import CategoryEntry, EntityCatalog, WalletEntry
from src.agent.llm_openrouter import OpenRouterError, make_llm_call
from src.agent.session_logger import slog
from src.agent.tools.get_user_context import load_user_context_dict
from src.agent.tools.wallet_required_cta import build_wallet_required_block


_STATUS_WORD = "กำลังบันทึก..."

# Other-floor display label when the LLM resolver fails / refuses to pick a
# category. Memory `project_chat_null_category_other_floor`: a null category
# used to default to whatever the first income/expense category was, which
# produced wrong proposals (slip discount shown as "เงินเดือน"). We surface
# "อื่นๆ" as the display name and keep category_sync_id=None so mobile can
# edit before confirm.
_OTHER_CATEGORY_LABEL = "อื่นๆ"


# ── Joint-resolver schema (PROPOSE_MODEL structured output) ───────────────


class _PairChoice(BaseModel):
    """LLM's joint pick of wallet + category.

    `wallet_sync_id` MUST be one of the candidates. `category_sync_id` MUST
    belong to that wallet, or be null when no candidate fits — the user will
    edit on the proposal card. `confidence` is logged for observability only
    (Q13a — we trust the pick regardless of confidence).
    """

    wallet_sync_id: str
    category_sync_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


# ── Core result envelope (tool_call_id-free) ───────────────────────────────


@dataclass
class ProposeResult:
    """Outcome of `_propose_core` — pure state-shaped, NO ToolMessage.

    Exactly ONE of {error, onboarding, success} describes the result:

      - `error` set        → pre-mutation failure (invalid amount, catalog
                             load, resolve fail). `state_updates` is empty;
                             callers preserve prior state. `result_payload`
                             carries {error, kind} for the tool ToolMessage.
      - `onboarding` True  → user has no wallets; `state_updates` carries the
                             wallet_required block. `result_payload` carries
                             the onboarding note.
      - otherwise success  → `state_updates` carries the canonical writes
                             (pending_proposal, last_txn, proposals,
                             emitted_blocks_this_turn, [user_context]).
                             `summary` holds human-facing fields for the
                             deterministic direct_propose confirmation.

    `state_updates` is the SAME dict the `@tool` wrapper merges into its
    Command(update=) — the wrapper only adds the terminating `messages`
    ToolMessage. The `direct_propose` node merges it into its own return + adds
    a deterministic AIMessage. Keeping ToolMessage out of here is what lets
    BOTH callers reuse one body without faking a tool_call_id.
    """

    state_updates: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_kind: Optional[str] = None
    onboarding: bool = False

    @property
    def ok(self) -> bool:
        """True only on the success path (not error, not onboarding)."""
        return self.error is None and not self.onboarding


# ── Main tool ─────────────────────────────────────────────────────────────


@tool
async def propose_transaction(
    amount: float,
    type: Literal["expense", "income", "auto"] = "expense",
    category_label: Optional[str] = None,
    wallet_label: Optional[str] = None,
    note: Optional[str] = None,
    date_iso: Optional[str] = None,
    *,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Propose a new transaction. The user must confirm via the mobile card.

    Use when the user has expressed clear intent to record a transaction with
    BOTH an amount AND a category/description. If a proposal is already
    pending, this tool atomically discards the old one before creating the new
    one — DO NOT call any discard tool manually.

    Args:
      amount: Numeric amount. Must be > 0.
      type: "expense" | "income" | "auto". Default "expense". Pass "auto"
        when the user's intent is ambiguous between expense and income — the
        joint resolver will infer the type from the chosen category.
      category_label: User's free-text category ("กาแฟ", "ค่าน้ำ"). None → "อื่นๆ".
      wallet_label: User's free-text wallet ("เงินสด", "KBank"). None → state
        wallet_id / catalog default / index-0.
      note: Free-text description. Defaults to category_label.
      date_iso: ISO date string. None → today.

    Returns:
      Command(update={
        "messages": [ToolMessage(<result or error JSON>)],
        "pending_proposal": <new proposal dict | None on error>,
        "last_txn": <short-term-memory dict | unchanged on error>,
      })

      On success the ToolMessage content is JSON: {proposal_id,
      transaction_sync_id, discarded_proposal_id, note}.
      On error the ToolMessage content is JSON: {error, kind}. The
      `pending_proposal` / `last_txn` keys are omitted from update on
      pre-mutation errors so the prior state is preserved by LangGraph's
      reducers.

    Thin wrapper (Wave 8): all logic lives in `_propose_core` (tool_call_id-
    free) so the `classify_intent → direct_propose` shortcut can reuse the SAME
    body without faking an InjectedToolCallId. This wrapper only wraps the
    ProposeResult in a terminating ToolMessage + Command. Payload/behavior is
    byte-identical to the pre-refactor monolith — the atomic tests guard it.
    """
    result = await _propose_core(
        amount=amount,
        type=type,
        category_label=category_label,
        wallet_label=wallet_label,
        note=note,
        date_iso=date_iso,
        state=state,
    )
    update: dict[str, Any] = dict(result.state_updates)
    update["messages"] = [
        ToolMessage(
            content=json.dumps(result.result_payload, ensure_ascii=False),
            tool_call_id=tool_call_id,
        ),
    ]
    return Command(update=update)


async def _propose_core(
    *,
    amount: float,
    type: Literal["expense", "income", "auto"] = "expense",
    category_label: Optional[str] = None,
    wallet_label: Optional[str] = None,
    note: Optional[str] = None,
    date_iso: Optional[str] = None,
    state: dict,
) -> ProposeResult:
    """Resolve + persist a transaction proposal — ToolMessage-free core.

    Shared by the `propose_transaction` @tool wrapper (ReAct path) and the
    `direct_propose` graph node (classify-router shortcut). Returns a
    `ProposeResult` describing state writes + a human summary; NEVER raises
    and NEVER touches tool_call_id / ToolMessage. See ProposeResult docstring
    for the error / onboarding / success contract.

    All numbered steps mirror the module docstring's flow.
    """
    _emit_status(_STATUS_WORD)

    # ── 1. Amount canonicalization (Decimal — never float arithmetic) ──────
    try:
        canon_amount = Decimal(str(amount))
    except (InvalidOperation, ValueError, TypeError):
        return _error_result(
            error=f"cannot parse amount {amount!r}",
            kind="invalid_amount",
        )
    if canon_amount <= 0:
        return _error_result(error="amount must be > 0", kind="invalid_amount")

    # ── 2. Atomic discard guard (Decision X) ──────────────────────────────
    # Mark the current pending as discarded BEFORE building the new one, and
    # emit the discard_proposal block FIRST. This is what distinguishes v3
    # from v1's block-when-pending: a correction = new ADD, not an EDIT.
    discarded_id: Optional[str] = None
    pending = _get_pending_proposal(state)
    if pending is not None:
        discarded_id = pending.get("proposal_id")
        pending["status"] = "discarded"
        pending["discarded_at"] = _now_iso()
        # Clear the convenience pointer so a later read sees no pending.
        state["pending_proposal"] = None
        # NOTE: discard_proposal block is propagated via Command(update=) only
        # to avoid append_reducer double-counting.
        slog(
            "propose_transaction",
            f"atomic discard: marked {discarded_id} discarded before new propose",
        )

    # ── 3. Context: cache hit or auto-load + onboarding shortcut ──────────
    # propose_transaction owns its catalog dependency. Three cache shapes
    # we tolerate without re-querying:
    #   1. wallets=[] — onboarding case, the branch below routes to the
    #      wallet_required shortcut without ever scoping categories.
    #   2. cache carries `categories` or `categories_all` keys — tests +
    #      legacy code paths supply these directly.
    # Anything else (cache missing entirely, OR has wallets but no
    # category fields) → load fresh from DB so per-wallet category
    # scoping has the full `categories_all` view.
    user_context = state.get("user_context")
    ctx_auto_loaded = False
    catalog: Optional[EntityCatalog] = None

    cache_is_usable = (
        user_context is not None
        and (
            # onboarding cache — wallets=[] needs no categories anyway
            not user_context.get("wallets")
            # OR the cache was hand-populated with category fields
            or "categories" in user_context
            or "categories_all" in user_context
        )
    )

    if cache_is_usable:
        catalog = EntityCatalog.from_dict(user_context)
    else:
        user_id_for_load = state.get("user_id")
        if not user_id_for_load:
            return _error_result(
                error="state.user_id missing",
                kind="invalid_state",
            )
        user_context, catalog, load_error = await load_user_context_dict(
            user_id_for_load,
        )
        if catalog is None or user_context is None:
            return _error_result(
                error=f"failed to load catalog: {load_error}",
                kind="catalog_load_failed",
            )
        ctx_auto_loaded = True
        slog(
            "propose_transaction",
            f"auto-loaded catalog — wallets={len(catalog.wallets)} "
            f"categories_all={len(catalog.categories_all)}",
        )

    if not catalog.wallets:
        # Onboarding shortcut: emit the wallet_required block directly so
        # onboarding completes in this same LLM round.
        return _onboarding_result(
            user_context=user_context,
            ctx_auto_loaded=ctx_auto_loaded,
        )

    # ── 4. Wallet candidate selection — priority chain ────────────────────
    # Priority (stop at first match):
    #   1. wallet_label given → joint LLM picks from ALL wallets (label is
    #      a strong user signal that overrides the state's stickied wallet).
    #   2. state["wallet_id"] (client's dropdown pick) valid → use it, cat-only LLM.
    #   3. user_context["wallet_id"] (catalog default) valid → use it.
    #   4. Index-0 fallback: general → creditcard → first wallet
    #      (memory `project_wallet_index0_ordering`).
    type_filter: Optional[str] = type if type in ("expense", "income") else None
    candidate_wallets: list[WalletEntry] = []
    wallet_pick_path = "unknown"

    if wallet_label:
        candidate_wallets = list(catalog.wallets)
        wallet_pick_path = "joint_label"
    else:
        for sid_source, sid_label in (
            (state.get("wallet_id"), "client_pick"),       # mobile dropdown pick
            (user_context.get("wallet_id"), "default"),    # catalog default
        ):
            if not sid_source:
                continue
            w = catalog.wallet_by_sync_id(sid_source)
            if w is None:
                # Q7a — stale ID: silently fall through to the next source.
                slog(
                    "propose_transaction",
                    f"wallet pick: {sid_label}={sid_source!r} not in catalog, "
                    "falling through",
                )
                continue
            candidate_wallets = [w]
            wallet_pick_path = sid_label
            break

        if not candidate_wallets:
            fallback = _index_zero_wallet(catalog.wallets)
            if fallback is None:
                # Defensive — onboarding shortcut should have caught this above.
                return _error_result(
                    error="no wallets available",
                    kind="wallet_resolution_failed",
                )
            candidate_wallets = [fallback]
            wallet_pick_path = "index_0"

    # ── 5. Category candidate filter (type-scoped) ────────────────────────
    candidate_cats_by_wallet: dict[str, list[CategoryEntry]] = {}
    for w in candidate_wallets:
        cats = catalog.categories_for(w.sync_id)
        if type_filter:
            cats = [c for c in cats if c.type == type_filter]
        candidate_cats_by_wallet[w.sync_id] = cats

    # Q18b — single-wallet path with 0 cats: surface as a hard error so ReAct
    # can prompt the user to create a category. Joint path (multiple wallets)
    # tolerates per-wallet empties because the LLM can pick a different wallet.
    if len(candidate_wallets) == 1:
        only = candidate_wallets[0]
        if not candidate_cats_by_wallet[only.sync_id]:
            return _error_result(
                error=f"wallet {only.name!r} has no {type_filter or 'any'} categories",
                kind="no_categories_for_wallet",
            )

    # ── 6. Fast path: skip LLM entirely (Q4a) ─────────────────────────────
    # Wallet is fixed (single-wallet path) AND no category hint at all
    # → 0 LLM calls, category = Other-floor.
    hint_text = (category_label or note or "").strip()
    skip_llm = (
        not wallet_label
        and not hint_text
    )

    chosen_wallet: WalletEntry
    chosen_category: Optional[CategoryEntry] = None

    if skip_llm:
        chosen_wallet = candidate_wallets[0]
        slog(
            "propose_transaction",
            f"fast path (no LLM): wallet={chosen_wallet.name!r} via {wallet_pick_path}, "
            "category=Other-floor",
        )
    else:
        # ── 7. Joint LLM call ─────────────────────────────────────────────
        pair = await _resolve_pair_async(
            candidate_wallets=candidate_wallets,
            candidate_cats_by_wallet=candidate_cats_by_wallet,
            amount=canon_amount,
            type_hint=type_filter,
            category_hint=category_label,
            description=note,
            wallet_label=wallet_label,
            multi_wallet=len(candidate_wallets) > 1,
        )
        if pair is None:
            # LLM transport/validation/out-of-set failure → Q10b: hard error
            # so ReAct can retry or inform the user. We've already discarded
            # the prior pending (Q11a) — user sees the discard + this error.
            return _error_result(
                error="joint wallet/category resolve failed",
                kind="joint_resolve_failed",
            )

        # ── 8. Validators ────────────────────────────────────────────────
        # 8a. wallet must be in candidates (out-of-set wallet = LLM failure).
        chosen_wallet_opt = next(
            (w for w in candidate_wallets if w.sync_id == pair.wallet_sync_id),
            None,
        )
        if chosen_wallet_opt is None:
            slog(
                "propose_transaction",
                f"LLM picked wallet_sync_id={pair.wallet_sync_id!r} "
                f"not in candidates {[w.sync_id for w in candidate_wallets]}",
            )
            return _error_result(
                error="LLM picked unknown wallet",
                kind="joint_resolve_failed",
            )
        chosen_wallet = chosen_wallet_opt

        # 8b. category must belong to the chosen wallet (Q9a cross-wallet
        # leak guard). Out-of-set OR cross-wallet leak → reject category,
        # use Other-floor — the user can still edit on the proposal card.
        if pair.category_sync_id:
            cats_for_chosen = candidate_cats_by_wallet.get(chosen_wallet.sync_id, [])
            chosen_category = next(
                (c for c in cats_for_chosen if c.sync_id == pair.category_sync_id),
                None,
            )
            if chosen_category is None:
                slog(
                    "propose_transaction",
                    f"cross-wallet/out-of-set category "
                    f"{pair.category_sync_id!r} not in wallet "
                    f"{chosen_wallet.sync_id!r}; falling back to Other-floor",
                )

        # 8c. type-mismatch guard — when caller specified a type, reject any
        # category of the opposite type. (Cannot happen when type_filter is
        # set because we already filtered candidates, but defend in depth.)
        if (
            chosen_category is not None
            and type_filter is not None
            and chosen_category.type != type_filter
        ):
            slog(
                "propose_transaction",
                f"type-mismatch: cat.type={chosen_category.type} vs "
                f"requested={type_filter}; falling back to Other-floor",
            )
            chosen_category = None

        slog(
            "propose_transaction",
            f"joint pick: wallet={chosen_wallet.name!r} "
            f"category={(chosen_category.name if chosen_category else 'Other-floor')!r} "
            f"confidence={pair.confidence:.2f} path={wallet_pick_path}",
        )

    # ── 9. Resolve final transaction.type ──────────────────────────────────
    # Source of truth: chosen_category.type (DB), then type arg, then default.
    if chosen_category is not None:
        final_type: Literal["expense", "income"] = chosen_category.type  # type: ignore[assignment]
    elif type in ("expense", "income"):
        final_type = type
    else:
        # Q12a — no LLM cat pick + type="auto": default expense.
        final_type = "expense"

    if chosen_category is not None:
        category_sync_id = chosen_category.sync_id
        category_name = chosen_category.name
    else:
        category_sync_id = None
        category_name = _OTHER_CATEGORY_LABEL

    # ── 10. Build the proposal payload (mobile write contract) ────────────
    proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
    transaction_sync_id = str(uuid.uuid4())
    final_date = date_iso or date.today().isoformat()
    amount_wire = _jsonable_amount(canon_amount)
    description = note or category_label or category_name

    payload: dict[str, Any] = {
        # Canonical mobile write contract:
        "sync_id": transaction_sync_id,
        "type": final_type,
        "amount": amount_wire,
        "category_sync_id": category_sync_id,
        "wallet_sync_id": chosen_wallet.sync_id,
        "note": description,
        "date": final_date,
        "currency_code": user_context.get("default_currency_code") or "THB",
        # Legacy keys retained for in-flight mobile clients (v2 contract):
        "user_id": state.get("user_id"),
        "wallet_id": chosen_wallet.sync_id,
        "category": category_name,
        "description": description,
    }

    # ── 11. Persist to audit table ────────────────────────────────────────
    repo = state.get("__repo__")
    if repo is not None:
        try:
            await repo.insert_pending_proposal(
                proposal_id=proposal_id,
                user_id=state.get("user_id"),
                kind="ADD_TRANSACTION",
                payload=payload,
            )
        except Exception as exc:
            # Persist failure must NOT lose the proposal — log loudly so the
            # confirm endpoint sees a missing row (and surfaces an error to
            # the user) instead of silently swallowing the write.
            slog(
                "propose_transaction",
                f"insert_pending_proposal FAILED for {proposal_id}: "
                f"{exc.__class__.__name__}: {exc}",
            )

    # ── 12. Update state (proposals + pending_proposal + last_txn) ────────
    new_proposal: dict[str, Any] = {
        "proposal_id": proposal_id,
        "intent_type": "ADD_TRANSACTION",
        "type": "ADD_TRANSACTION",
        "payload": payload,
        "status": "pending",
        "created_at": _now_iso(),
        "confirmed_at": None,
        "cancelled_at": None,
        "discarded_at": None,
    }
    # In-place list mutation — survives the pydantic shallow-copy boundary
    # because `state["proposals"]` is a shared LIST reference. Tests inspect
    # base_state["proposals"][-1] to verify the new proposal landed.
    state.setdefault("proposals", []).append(new_proposal)
    state["pending_proposal"] = new_proposal  # defensive (canonical write is Command below)

    last_txn = {
        "id": proposal_id,
        "type": payload["type"],
        "amount": payload["amount"],
        "category": payload["category"],
        "pending": True,
    }
    state["last_txn"] = last_txn  # defensive

    # ── 13. Build new blocks (discard FIRST, then proposal) ───────────────
    new_blocks: list[dict[str, Any]] = []
    if discarded_id is not None:
        new_blocks.append({"type": "discard_proposal", "target": discarded_id})
    proposal_block = {
        "type": "transaction_proposal",
        "proposal_id": proposal_id,
        "transaction": payload,
        "low_confidence": False,  # Q13a — field carried unused (always false)
        "text": "ยืนยันบันทึกรายการนี้ไหมครับ",
    }
    new_blocks.append(proposal_block)

    slog(
        "propose_transaction",
        f"proposed {proposal_id} amount={canon_amount} type={final_type} "
        f"cat={category_name} wallet={chosen_wallet.sync_id} "
        f"discarded={discarded_id}",
    )

    # ── 14. Build ProposeResult so BOTH callers share the state writes ─────
    full_proposals_list = list(state.get("proposals") or [])
    if new_proposal not in full_proposals_list:
        full_proposals_list.append(new_proposal)

    result_payload = {
        "proposal_id": proposal_id,
        "transaction_sync_id": transaction_sync_id,
        "discarded_proposal_id": discarded_id,
        "note": f"proposed {canon_amount} {final_type} as {category_name}",
    }
    state_updates: dict[str, Any] = {
        "pending_proposal": new_proposal,
        "last_txn": last_txn,
        "emitted_blocks_this_turn": new_blocks,
        "proposals": full_proposals_list,
    }
    if ctx_auto_loaded:
        state_updates["user_context"] = user_context
    return ProposeResult(
        state_updates=state_updates,
        result_payload=result_payload,
        summary={
            "amount": amount_wire,
            "amount_decimal": canon_amount,
            "type": final_type,
            "category_name": category_name,
            "discarded_id": discarded_id,
        },
    )


# ── Helpers (private) ──────────────────────────────────────────────────────


def _error_result(*, error: str, kind: str) -> ProposeResult:
    """Build a ProposeResult for the early-error paths (ToolMessage-free).

    No state mutation — `state_updates` stays empty so prior state is
    preserved by LangGraph's reducers. The LLM (or the direct_propose node)
    reads `error` + `kind` from `result_payload`.
    """
    return ProposeResult(
        result_payload={"error": error, "kind": kind},
        error=error,
        error_kind=kind,
    )


def _onboarding_result(
    *,
    user_context: dict,
    ctx_auto_loaded: bool,
) -> ProposeResult:
    """Emit the wallet_required CTA block in-place when the user has no wallets.

    Lets a one-shot ADD turn complete onboarding without a second LLM round
    routed through `wallet_required_cta`. Block shape matches the tool so the
    mobile renderer treats both paths identically.
    """
    block = build_wallet_required_block()
    payload = {
        "cta_emitted": True,
        "kind": "no_wallet",
        "note": "onboarding CTA emitted in-place by propose_transaction",
    }
    state_updates: dict[str, Any] = {"emitted_blocks_this_turn": [block]}
    if ctx_auto_loaded:
        state_updates["user_context"] = user_context
    slog(
        "propose_transaction",
        "onboarding shortcut: emitted wallet_required block "
        f"(ctx_auto_loaded={ctx_auto_loaded})",
    )
    return ProposeResult(
        state_updates=state_updates,
        result_payload=payload,
        onboarding=True,
    )


def _emit_status(word: str) -> None:
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"status": word})
    except Exception:
        pass


def _get_pending_proposal(state: dict) -> Optional[dict]:
    """Find the single currently-pending proposal.

    Prefer the explicit `pending_proposal` pointer, but fall back to scanning
    `proposals` so a turn that lost the pointer (e.g. through a partial state
    update) still discards correctly. Returns the LATEST pending entry.
    """
    pointer = state.get("pending_proposal")
    if pointer and pointer.get("status") == "pending":
        return pointer
    for p in reversed(state.get("proposals") or []):
        if p.get("status") == "pending":
            return p
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable_amount(canon: Decimal):
    """Decimal → JSON-safe number (int when integral, else float)."""
    if canon == canon.to_integral_value():
        return int(canon)
    return float(canon)


def _index_zero_wallet(wallets: list[WalletEntry]) -> Optional[WalletEntry]:
    """Index-0 fallback wallet (memory `project_wallet_index0_ordering`).

    SQL already orders general → creditcard → goal, but we re-assert on
    `wallet_type` so a hand-built test catalog (or a future SQL change)
    doesn't accidentally pick a goal wallet.
    """
    for wallet_type in ("general", "creditcard"):
        for w in wallets:
            if w.wallet_type == wallet_type:
                return w
    return wallets[0] if wallets else None


# ── Joint resolver (single LLM call) ───────────────────────────────────────


_PAIR_RESPONSE_SCHEMA_HINT = (
    'Return ONLY a JSON object with these fields: '
    '{"wallet_sync_id": "<one of the wallet sync_ids>", '
    '"category_sync_id": "<one of THAT wallet\'s category sync_ids, or null>", '
    '"confidence": <number 0.0-1.0>, '
    '"reason": "<short reason>"}'
)


@_ls_traceable(
    name="llm.propose",
    run_type="llm",
    tags=["classify_router", "propose_pair"],
)
async def _resolve_pair_async(
    *,
    candidate_wallets: list[WalletEntry],
    candidate_cats_by_wallet: dict[str, list[CategoryEntry]],
    amount: Decimal,
    type_hint: Optional[str],
    category_hint: Optional[str],
    description: Optional[str],
    wallet_label: Optional[str],
    multi_wallet: bool,
) -> Optional[_PairChoice]:
    """ONE LLM call that picks wallet + category jointly.

    Returns None on transport/validation/empty-content failure — the caller
    maps that to a `joint_resolve_failed` error envelope (Q10b). NEVER
    raises; the outer tool returns Command, not exceptions.
    """
    prompt = _build_pair_prompt(
        candidate_wallets=candidate_wallets,
        candidate_cats_by_wallet=candidate_cats_by_wallet,
        amount=amount,
        type_hint=type_hint,
        category_hint=category_hint,
        description=description,
        wallet_label=wallet_label,
        multi_wallet=multi_wallet,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You match a financial transaction to an existing wallet and "
                "category. Return ONLY a JSON object matching the schema in "
                "the user message. No markdown fences. No prose."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        call = make_llm_call("propose")
    except OpenRouterError as exc:
        slog("propose_transaction", f"PROPOSE_MODEL not configured: {exc}")
        return None
    try:
        raw = await call(messages)
    except OpenRouterError as exc:
        slog("propose_transaction", f"PROPOSE LLM call failed: {exc}")
        return None
    try:
        data = _extract_json_object(raw)
        return _PairChoice.model_validate(data)
    except (ValueError, ValidationError) as exc:
        slog(
            "propose_transaction",
            f"PROPOSE response parse/validate FAILED: "
            f"{type(exc).__name__}: {exc} — raw={raw!r}",
        )
        return None


def _build_pair_prompt(
    *,
    candidate_wallets: list[WalletEntry],
    candidate_cats_by_wallet: dict[str, list[CategoryEntry]],
    amount: Decimal,
    type_hint: Optional[str],
    category_hint: Optional[str],
    description: Optional[str],
    wallet_label: Optional[str],
    multi_wallet: bool,
) -> str:
    """Render the joint-resolve user-message.

    Layout: hints first, then candidate tree (wallet → its cats), then the
    response schema reminder. Wallet shows its `type` only in multi-wallet
    mode; cat shows its `type` only when the candidate set spans both
    expense AND income (i.e. `type_hint` is None / "auto").
    """
    lines: list[str] = []
    lines.append("Task — Pick BOTH a wallet AND a category for this transaction:")
    lines.append("  Step 1. Decide which wallet the user means.")
    lines.append("  Step 2. Pick a category that BELONGS TO THAT WALLET.")
    lines.append("")
    lines.append("Hard rules:")
    lines.append(
        "  - category_sync_id MUST belong to the chosen wallet_sync_id "
        "(NEVER pick a category listed under a different wallet)."
    )
    lines.append(
        "  - If no category fits well, set category_sync_id to null "
        "(the user will edit on the proposal card)."
    )
    lines.append(
        "  - Consider Thai/English semantics, typos, parent-child hierarchy, "
        "and the keywords field."
    )
    lines.append("")

    lines.append("User context:")
    lines.append(f"  - amount: {amount}")
    if type_hint:
        lines.append(f"  - type: {type_hint}")
    if wallet_label:
        lines.append(f"  - wallet hint: {wallet_label!r}")
    if category_hint:
        lines.append(f"  - category hint: {category_hint!r}")
    if description:
        lines.append(f"  - description: {description!r}")
    lines.append("")

    lines.append("Candidates (wallet → categories):")
    show_cat_type = type_hint is None
    for w in candidate_wallets:
        wtype = f" | type={w.wallet_type}" if multi_wallet else ""
        wcat = (
            f" | wallet_category={w.wallet_category!r}"
            if multi_wallet and w.wallet_category else ""
        )
        lines.append(
            f"- wallet_sync_id={w.sync_id} | name={w.name!r}{wtype}{wcat}"
        )
        cats = candidate_cats_by_wallet.get(w.sync_id, [])
        if not cats:
            lines.append("  -- (no categories)")
            continue
        for c in cats:
            ctype = f" | type={c.type}" if show_cat_type else ""
            parent = f" | parent={c.parent_id!r}" if c.parent_id else ""
            kw = f" | keywords={c.keywords!r}" if c.keywords else ""
            lines.append(
                f"  -- cat_sync_id={c.sync_id} | name={c.name!r}{ctype}{parent}{kw}"
            )
    lines.append("")
    lines.append(_PAIR_RESPONSE_SCHEMA_HINT)
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply, tolerating fences."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError(f"LLM response has no JSON object: {text!r}")
    return json.loads(s[i : j + 1])


__all__ = ["propose_transaction", "_propose_core", "ProposeResult"]
