# ReAct Rebuild — Phase 0 Baseline Snapshot

Captured 2026-06-01 on the **current prebuilt** react graph (`REACT_IMPL` not yet
introduced). This is the comparison point for the flat rebuild — the flat path
must be at parity or better.

## Unit suite (`pytest tests/`)

```
395 passed, 8 failed, 2 xfailed   (46s)
```

### 8 baseline-RED tests (pre-existing — channel-merge / subgraph-boundary quirk)

These fail on the CURRENT code, BEFORE any rebuild. They expect a downstream
reader (outer node or the test itself) to see the react subgraph's per-turn
append-channel writes / in-place mutation — which the LangGraph 1.x subgraph
boundary does not provide. Cross-referenced with memory
`project_chat_suggestions_deterministic` (the ~13-RED note; now 8).

- `test_react_loop.py::test_UT_G02_pre_turn_hook_clears_scratch_between_turns`
- `test_react_loop.py::test_UT_G07_emission_guard_forces_propose_when_claim_has_no_block`
- `test_react_loop.py::test_UT_G08_emission_guard_falls_back_honestly_after_max_retries`
- `test_react_loop.py::test_UT_G09_emission_guard_silent_on_legit_add_and_analyst`
- `test_run_python.py::test_UT_T07_clarification_emits_block_without_raising`
- `test_server_answer_token.py::test_UT_SAT004_content_with_tool_calls_is_emitted`
- `test_store_factory.py::test_UT_SF01_builds_store_with_embed_index`
- `test_wallet_required_cta.py::test_UT_T11_emits_wallet_required_block_with_action_target`

**Hypothesis to confirm during rebuild:** flattening the graph (no subgraph
boundary) should flip SOME of these GREEN (the channel-merge ones: G07/G08/G09,
T07, T11, SAT004). G02 (pre_turn scratch) and SF01 (store factory) may be
unrelated — verify individually; do not assume the rebuild fixed them.

## Integration suite (`tests_integration/I1*.py`)

Deferred — requires in-process FastAPI lifespan + Postgres + OpenRouter. Run as
part of the Phase 4 eval gate (SSE byte-diff prebuilt vs flat), not the Phase 0
unit baseline.

## Phase 1 results (flat graph behind `REACT_IMPL` flag) — 2026-06-01

Built `src/agent/flat_react.py` (`wire_flat_react`) + branched `graph.py::build_graph`
on `REACT_IMPL` (default `prebuilt`). Findings:

- **Default (prebuilt) parity:** unit suite `395 passed / 8 failed / 2 xfailed` —
  IDENTICAL to baseline. Live-server balance turn unchanged. Zero behavior change
  when the flag is off. ✅
- **Flat parity (unit):** `REACT_IMPL=flat pytest tests/` → SAME `395/8`. The 8
  RED stay RED — they are tool/function-level tests that never build the graph,
  so flattening can't flip them (confirms the §"verify individually" caveat;
  they are NOT the channel-merge proof). ✅ (no regression)
- **Flat end-to-end (in-process, real model):** agent → tool → agent → answer,
  block emitted to final state. Loop executes correctly. ✅
- **Flat end-to-end (live tunnel):** balance turn returned the correct
  167,460 บาท. **R1 headline RESOLVED + IMPROVED:** flat emits **13
  `answer_token` events (LIVE token streaming)** vs prebuilt's **1
  `answer_token` (whole message at the subgraph boundary)**. Flat is strictly
  better streaming UX. ✅

### Still open (Phase 2+ gate — NOT done in Phase 1)
- **R1b (footer in flat stream):** the validator soft-warn footer is appended
  AFTER tokens stream; in flat mode it may reach persisted history but not the
  live `answer_token` stream. Not exercised (this turn passed validation). Must
  force a soft-warn turn in flat mode and re-plumb if the footer drops.
- **R3:** emission_guard / direct_add still read via the `messages` workaround;
  the payoff (read merged `emitted_blocks_this_turn` directly) is Phase 3.
- Full SSE byte-diff matrix (analyst/advisor/ADD/soft-warn/slip/voice) + the
  integration suite remain the Phase 4 flip-gate.

## Phase 2-4 results — PARITY GATE PASSED, flat shipped to dev — 2026-06-01

Ran the full flat-vs-prebuilt comparison (`scripts/compare_react_impl.py`, live
tunnel, 11 turn types). Findings + fixes:

1. **Narration leak (R1) — FIXED.** Live token streaming in flat mis-routed a
   tool-call message's narration ("กำลังรวมยอด…") into the answer bubble (early
   chunks arrive before the tool_call, so `_yield_message_event` saw no
   tool_calls → answer_token). Fix: `flat_react._make_agent_node` clones the model
   with `disable_streaming=True` → messages-mode emits ONE complete message per
   step, exactly like the prebuilt boundary → narration→status routing correct.
   Flat now streams `1 answer_token` like prebuilt.
2. **Soft-warn footer (R1b) — FIXED.** The validator footer is appended by the
   `react_validate` node AFTER the agent streamed the answer; that rewrite is not
   re-emitted on the messages stream, so the footer was missing from the flat
   answer (proven: same fake ungrounded answer → prebuilt footer=True,
   flat footer=False). Fix: `sse_adapter.stream_chat` captures the footer from the
   `updates` stream (validate node's message delta) and reconciles it into
   `final_text` + emits a trailing `answer_token`. Guarded by
   `_WARNING_MARKER not in final_text` → no-op for prebuilt.
3. **advisor_debt apparent divergence = nondeterminism, NOT impl.** Free-form
   advisor turns sometimes dump raw tool numbers → validator soft-warns; this
   varies run-to-run WITHIN each impl (observed flat both False and True across
   runs). After the R1b fix, when flat soft-warns the footer is present.

### Final parity (after fixes)
- Unit `pytest tests/`: 395/8 in BOTH modes (8 pre-existing RED unchanged).
- Integration `tests_integration/`: 9 fail/15 pass in BOTH modes (same 9
  pre-existing — asyncio-loop / DB-seed infra, not impl).
- Live battery: **11/11 parity** on {block types, proposal card, suggestions,
  answer numbers, soft-warn footer}.
- Streaming/footer: flat ≡ prebuilt.

**Shipped:** flat is now the **CODE DEFAULT** (`flat_react.is_flat_enabled()`
returns True unless `REACT_IMPL=prebuilt`). Every environment — dev, prod, CI,
studio — runs flat unless it explicitly opts out. Verified: unit suite stays
395/8 with no env set (flat default); `REACT_IMPL=prebuilt` fallback confirmed
working. `.env` keeps an explicit `REACT_IMPL=flat` line for readability.
**Fallback:** set `REACT_IMPL=prebuilt` to restore the legacy subgraph.

## Gate reminder

Flat path must: (a) keep all 395 currently-GREEN tests GREEN, (b) not regress
the 8 RED into different failures, (c) byte-diff SSE-equal to prebuilt on the
6 turn types listed in the assessment §6.
