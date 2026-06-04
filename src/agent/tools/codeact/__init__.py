"""`run_python` — CodeAct tool exposed to the ReAct loop.

NEW in Wave 2 — replaces v2's `codeact_subgraph` (a full sub-graph) with a
single LangChain `@tool` that the create_react_agent loop invokes.

The tool wraps the v2 sandbox (`sandbox.py`) + DB-aware namespace
(`namespace.py`). Both stay 1:1 with v2 — only this wrapper is new.

CONTRACT (mirrors `docs/v3/phase2_tools_design.md` Tool 5):
- Emit status word "กำลังคำนวณ..." via stream_writer at entry.
- Load the EntityCatalog inline every turn via `load_catalog_for_user` —
  this tool owns its catalog dependency, never relies on the ReAct LLM to
  pre-fetch via `get_user_context`.
- Gate empty wallets: zero active wallets (catalog.wallets=[]) → emit the
  shared wallet_required block via `build_wallet_required_block()` and
  return without executing user code. Mobile renders the create-wallet UI.
- Execute on a worker thread so the sandbox stays sync while the main
  event loop runs the asyncpg queries via run_coroutine_threadsafe.
- Translate ClarificationNeeded → clarification block (no LLM call needed).
- Record the result + stdout in state.tool_outputs_this_turn so the
  numerical validator (Wave 1) can cross-check every money number in the
  final answer.

REGRESSION GUARD: the keys this tool exposes are produced by
`namespace.build_namespace()` — UT-NS01 + UT-P01 lock them to the system
prompt's `# CODEACT TOOLBOX` section. Drift between the two is what caused
the 1,234.56 hallucination in v1 (memory `project_codeact_dual_impl`).
"""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import date
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

try:
    # Available on LangGraph >= 0.2; the writer pipes per-tool custom events
    # into the SSE adapter's `status_token` channel.
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover — older LangGraph fallback
    get_stream_writer = None  # type: ignore[assignment]

from src.agent.entity_catalog import load_catalog_for_user
from src.agent.session_logger import slog
from src.agent.tools.wallet_required_cta import build_wallet_required_block
from .exceptions import ClarificationNeeded
from .namespace import build_namespace
from .sandbox import execute


_MAX_OUTPUT_CHARS = 10_000     # cap to prevent context blowup
_STATUS_WORD = "กำลังคำนวณ..."


@tool
async def run_python(
    code: str,
    *,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Execute Python in a sandbox with read-only Postgres access.

    Available helpers (use ONLY these — see system prompt CODEACT TOOLBOX):
      Query: sum_income, sum_expense, sum_by_category, sum_by_wallet,
             sum_by_tag, list_transactions, balance, budget_remaining,
             budget_transactions, budget_list, creditcard_list, goal_list,
             goal_progress, goal_transactions, count_transactions,
             wallet_list, category_list, tag_list, spending_trend,
             transaction_stats, top_transactions, currency_rate,
             active_period, compare_periods, spending_pace, anomaly
      Resolvers: resolve_wallet, resolve_category, resolve_tag,
                 resolve_budget, resolve_goal, parse_period
      Helpers: Decimal, date, timedelta, today(), clarify

    MUST assign `result = <value>` to return data.
    NO imports. NO file/network I/O. NO INSERT/UPDATE/DELETE.

    Args:
      code: Python source (multi-line OK).

    Returns:
      Command(update={
        "tool_outputs_this_turn": [<result+stdout entry>],
        "emitted_blocks_this_turn": [<clarification block>] (only on ClarificationNeeded),
        "messages": [ToolMessage(JSON {result, stdout, error})],
      })
    """
    _emit_status(_STATUS_WORD)

    user_id = state.get("user_id")
    if not user_id:
        return _tool_command(
            tool_call_id,
            payload={"error": "state.user_id missing", "kind": "invalid_state",
                     "result": None, "stdout": "", "error_detail": None},
        )

    # Acquire the catalog: prefer the cached wire dict (test / parallel race
    # case), otherwise load fresh with retry-once-on-empty (memory
    # `project_wallet_index0_ordering`).
    user_context = state.get("user_context")
    catalog = None
    last_error: Exception | None = None

    if user_context and user_context.get("wallets"):
        from src.agent.entity_catalog import EntityCatalog
        catalog = EntityCatalog.from_dict(user_context)
        slog("run_python", "using cached user_context (wallets present)")
    else:
        for attempt in (1, 2):
            try:
                catalog = await load_catalog_for_user(user_id)
            except Exception as exc:
                last_error = exc
                catalog = None
                slog(
                    "run_python",
                    f"catalog load attempt {attempt} raised: "
                    f"{type(exc).__name__}: {exc}",
                )
            if catalog is not None and getattr(catalog, "wallets", None):
                break
            if attempt == 1:
                slog(
                    "run_python",
                    "catalog empty/failed on first load — retrying once",
                )

        if catalog is None:
            return _tool_command(
                tool_call_id,
                payload={"error": f"failed to load user context: {last_error}",
                         "kind": "context_load_failed",
                         "result": None, "stdout": ""},
            )

        # Zero active wallets after retry → emit the shared wallet_required
        # block in-place. Mobile renders the create-wallet UI; sandbox code
        # never runs without a wallet to scope queries against.
        # (Cached path above is trusted — tests + parallel race set it up
        # with non-empty wallets; the empty-cache state never reaches here.)
        if not catalog.wallets:
            block = build_wallet_required_block()
            _append_block(state, block)
            slog(
                "run_python",
                "wallets=[] after retry — emitted wallet_required block",
            )
            return _tool_command(
                tool_call_id,
                payload={
                    "result": None,
                    "stdout": "",
                    "error": "user has no active wallets",
                    "kind": "wallet_required",
                    "cta_emitted": True,
                },
                extra_updates={"emitted_blocks_this_turn": [block]},
            )

    main_loop = asyncio.get_running_loop()
    namespace = build_namespace(
        user_id=user_id,
        catalog=catalog,
        today=date.today(),
        main_loop=main_loop,
    )

    # Execute on worker thread — the sandbox is sync; namespace helpers
    # bridge back to the main loop via run_coroutine_threadsafe.
    try:
        result, stdout, error = await asyncio.to_thread(
            execute, code, namespace,
        )
    except ClarificationNeeded as cn:
        # Sandbox helper signaled the user must answer. Translate to a
        # clarification block (no LLM call needed).
        clarification_block = {
            "type": "clarification",
            "text": cn.question,
            "options": [{"label": o, "send": o} for o in cn.options] or [],
        }
        _append_block(state, clarification_block)
        slog("run_python",
             f"clarification_needed question={cn.question!r} "
             f"options={cn.options!r}")
        return _tool_command(
            tool_call_id,
            payload={
                "clarification_emitted": True,
                "question": cn.question,
                "options": list(cn.options),
            },
            extra_updates={"emitted_blocks_this_turn": [clarification_block]},
        )

    # Cap stdout + result repr to prevent context bloat.
    stdout_capped = (stdout or "")[-_MAX_OUTPUT_CHARS:]
    result_repr = _safe_repr(result, _MAX_OUTPUT_CHARS)

    # Record output for the numerical validator (Wave 1).
    output_entry = {
        "tool": "run_python",
        "result": result_repr,
        "stdout": stdout_capped,
    }
    _append_tool_output(state, output_entry)

    slog("run_python",
         f"executed code_len={len(code)} result_len={len(result_repr)} "
         f"error={error!r}")

    if error:
        return _tool_command(
            tool_call_id,
            payload={"result": None, "stdout": stdout_capped, "error": error},
            extra_updates={"tool_outputs_this_turn": [output_entry]},
        )

    # Silent-drop guard. The sandbox returns `namespace["result"]` (seeded to
    # None in build_namespace). When the code ran cleanly but NEVER assigned
    # `result` AND printed nothing, the fetched data died inside a temp variable
    # and NOTHING reached the model. A bare {"result": None} here looks like a
    # genuine empty result — that is exactly what made the model fabricate
    # "ไม่มีข้อมูลเปรียบเทียบ" from data that DID exist (compare_periods landed in
    # `diff`, never surfaced — trace 857a5761). Convert the silent drop into an
    # actionable retry instruction so the model fixes its code instead of
    # hallucinating. We detect it from the AST (`result` never assigned), NOT
    # `result is None`, so an intentional `result = None` is left alone.
    if (
        result is None
        and not stdout_capped.strip()
        and not _code_assigns_result(code)
    ):
        hint = _result_assignment_hint(code)
        nudge = (
            "Your code ran without error but produced NO output: you never "
            "assigned `result` and printed nothing, so I received nothing to "
            "answer from. Re-run: assign the value you want to use to `result`"
            + (f" (e.g. `{hint}`)" if hint else "")
            + ", then call run_python again. Do NOT answer from memory — the "
            "data only reaches me through `result` or print()."
        )
        slog("run_python", f"silent_drop: result unset + empty stdout; hint={hint!r}")
        return _tool_command(
            tool_call_id,
            payload={"result": None, "stdout": "", "error": nudge},
            extra_updates={"tool_outputs_this_turn": [output_entry]},
        )

    return _tool_command(
        tool_call_id,
        payload={"result": result_repr, "stdout": stdout_capped, "error": None},
        extra_updates={"tool_outputs_this_turn": [output_entry]},
    )


# ── Helpers (private) ──────────────────────────────────────────────────────


def _code_assigns_result(code: str) -> bool:
    """True when `code` assigns the `result` output variable ANYWHERE.

    Walks the whole tree (not just top level) so a conditional assignment like
    `if rows: result = rows[0]` still counts, and covers tuple targets
    (`a, result = …`), annotated/aug assignments, and walrus. A parse failure
    returns False — the caller already handled real SyntaxErrors via the sandbox
    error path, so reaching here with unparseable code is not expected; treating
    it as "no result assignment" is the safe default (it only gates a nudge).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        for tgt in targets:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Name) and sub.id == "result":
                    return True
    return False


def _result_assignment_hint(code: str) -> str | None:
    """Best-effort suggestion for the silent-drop nudge.

    When the model forgot `result = …`, the value it wanted almost always sits
    in the LAST top-level variable it assigned (e.g. `diff = compare_periods(...)`
    → suggest `result = diff`). Parse the module body, take the last simple
    `name = …` assignment whose target isn't already `result`, and surface it.
    Returns None on any parse failure or when no usable name is found — the
    nudge then degrades to the generic instruction.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    last_name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id != "result":
                last_name = target.id
    return f"result = {last_name}" if last_name else None


def _tool_command(
    tool_call_id: str,
    *,
    payload: dict,
    extra_updates: dict | None = None,
) -> Command:
    """Build a Command that closes the tool call with `payload` as the
    ToolMessage content and merges any extra channel updates."""
    update: dict[str, Any] = {
        "messages": [
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                tool_call_id=tool_call_id,
            ),
        ],
    }
    if extra_updates:
        update.update(extra_updates)
    return Command(update=update)


def _emit_status(word: str) -> None:
    """Stream a status word to the SSE adapter (best-effort, never raises)."""
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
        # Status updates are UX-only — never let a writer hiccup fail the tool.
        pass


def _append_block(state: dict, block: dict) -> None:
    """No-op — kept for callers that still pass `state` for clarity.

    Was previously `state.setdefault(...).append(block)`. Removed because
    LangGraph's append_reducer ALSO applies the Command(update=) the caller
    returns; doing both wrote the same block twice to the channel and
    produced duplicate `block` events on the SSE wire (issue #SSE-DUP, same
    pattern as emit_suggestions / wallet_required_cta). The Command(update=)
    return is the canonical write.
    """
    return None


def _append_tool_output(state: dict, entry: dict) -> None:
    state.setdefault("tool_outputs_this_turn", []).append(entry)


def _safe_repr(value: Any, cap: int) -> str:
    """Repr that survives Decimal, dict, list. Caps at `cap` chars."""
    try:
        s = repr(value)
    except Exception as exc:
        s = f"<unreprable: {type(exc).__name__}>"
    if len(s) > cap:
        s = s[:cap] + f" ...[truncated {len(s) - cap} chars]"
    return s


__all__ = ["run_python"]
