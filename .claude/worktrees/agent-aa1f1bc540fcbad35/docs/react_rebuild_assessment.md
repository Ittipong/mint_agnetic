# ReAct Rebuild Assessment — replace `create_react_agent` with a hand-built flat graph

**Status:** assessment only (no code). Decision pending.
**Date:** 2026-06-01
**Why we're considering it:** the LangGraph 1.x subgraph boundary doesn't merge
the react subgraph's per-turn append-channel writes (`tool_outputs_this_turn`,
`emitted_blocks_this_turn`) into the parent for a downstream outer node to read.
That forced two workarounds (suggestion chips live in the SSE adapter;
`emission_guard` reconstructs tool-call state from `messages`). Flattening the
react loop into outer-graph nodes removes the boundary → those channels merge
naturally → suggestions / emission_guard / future nodes read this-turn data
directly. This doc weighs whether that benefit justifies the rebuild.

---

## 1. What `create_react_agent` gives us today (inventory)

From `graph.py::build_graph`:

| Concern | Current source | Reusable without prebuilt? |
|---|---|---|
| Model call (bind_tools + invoke) | prebuilt agent node, model from `_make_model` (ChatOpenAI→OpenRouter, REACT_MODEL, temp 0, max_tokens 8192, `extra_body.models[]` fallback, `ReActSessionLogCallback`) | ✅ model factory is ours already; just call `model.bind_tools(ALL_TOOLS).ainvoke(...)` in a node |
| Prompt per step | `_make_prompt(state)` callable (system + `_window_messages`) | ✅ call directly in the agent node |
| Tool execution | prebuilt `ToolNode` (parallel calls, `Command(update=)` handling, ToolMessage build, error capture) | ✅ **`ToolNode` is a public class** — compose it standalone (big risk reducer) |
| Loop routing | prebuilt `should_continue` (tool_calls → tools, else → end) | ✅ trivial conditional edge on last AIMessage `.tool_calls` |
| Post-model hook | `make_validator_post_model_hook()` (numerical validator + tool-error retry guard) wired via `post_model_hook=` | ⚠️ re-wire manually after the agent node — **timing-sensitive (see §4)** |
| Recursion guard | prebuilt `remaining_steps` (`RemainingSteps` channel) | ⚠️ replace with `compile(recursion_limit=…)` or a manual counter |
| State schema | `AgentState` (`add_messages`, `append_reducer` channels) | ✅ unchanged |
| Streaming | `stream_mode=["messages","custom","updates"]` re-emits inner AIMessage tokens at the **subgraph boundary** | 🔴 **highest-risk parity item (see §4)** |

**Conclusion:** this is NOT "from absolute zero." Model + prompt + `ToolNode` +
state are reusable. The real work is the loop wiring, hook re-timing, recursion
guard, and — critically — **preserving streaming + validator-footer behavior.**

---

## 2. Target architecture (flat, single scope)

```
START → pre_turn → classify_intent → [direct_add | agent]
                                          agent ⇄ tools   (loop on tool_calls)
                                          agent → validate (post-model hook as a node)
                                          validate → emission_guard
emission_guard → [agent (retry) | suggestions]
suggestions → post_turn → END
```

All nodes share ONE channel scope → `tool_outputs_this_turn` /
`emitted_blocks_this_turn` written by `tools` are directly readable by
`suggestions`, `emission_guard`, `post_turn`. No `messages`-reconstruction hacks.

---

## 3. Node-by-node build list

1. **`agent`** — render `_make_prompt`, `model.bind_tools(ALL_TOOLS)`, `ainvoke`, append AIMessage. (Streams via callbacks — verify §4.)
2. **`tools`** — `ToolNode(ALL_TOOLS)` (reuse). Keeps `Command(update=)` → `emitted_blocks_this_turn`, `last_txn`.
3. **`validate`** — call the existing `validator_hook(state)`; merge its returned `messages` update (footer rewrite / retry SystemMessage / tool-error rewrite).
4. **routing** — `agent → tools` if last AIMessage has `tool_calls`, else `agent → validate`. After a validator retry SystemMessage, route back to `agent`.
5. **`emission_guard`** — keep, but now read `emitted_blocks_this_turn` DIRECTLY (drop the `messages`-scan workaround).
6. **`suggestions`** — NEW node (the actual goal): read merged this-turn channels, call `build_suggestions_block`, write the block to a channel the SSE adapter forwards.
7. **recursion** — `compile(recursion_limit=N)` + drop `remaining_steps` from `AgentState` (it's prebuilt-only).

---

## 4. Risks & regression surface (ranked)

🔴 **R1 — Streaming parity (the killer risk).** `numerical.py` documents that
the validator appends its warning footer by REWRITING the AIMessage **inside the
subgraph, before it crosses the boundary where `stream_mode="messages"` turns it
into `answer_token`.** This implies tokens are re-emitted AT THE BOUNDARY, not
purely live during `ainvoke`. A flat graph has NO boundary — so the
streaming-to-`answer_token` timing and the footer-rewrite-reaches-the-stream
guarantee BOTH change. Must prove: (a) live answer tokens still stream, (b) the
validator footer still reaches both the live stream and persisted history.
**This alone can sink the rebuild if it doesn't hold.**

🟠 **R2 — Validator hook mechanics.** The hook uses `RemoveMessage` + in-place
`AIMessage` rewrite (id-preserved) + `get_stream_writer()` status tokens, and
re-enters via the ReAct router on a `ToolMessage` tail. Re-creating the exact
re-entry + RemoveMessage semantics outside the prebuilt is fiddly.

🟠 **R3 — `direct_add` / classify-router edges.** Wave 8 nodes wire around the
prebuilt today (`direct_add → post_turn` skips emission_guard). Must re-map onto
the flat graph without re-introducing the mis-fire `GOTCHA 2` documents.

🟡 **R4 — Pre-existing RED tests.** Memory `project_chat_suggestions_deterministic`
notes ~13 tests are already RED due to the channel-merge quirk (react_loop /
fast_path / tier2 / wallet_required / store_factory). A rebuild that REMOVES the
quirk may flip them GREEN — good — but distinguishing "fixed by rebuild" from
"broke something new" needs a clean baseline first.

🟡 **R5 — `remaining_steps` removal.** Any test/code asserting on it breaks; the
recursion-limit semantics differ subtly from the prebuilt's step budget.

🟢 **R6 — Tool emission, prompt, model** — low risk (reused verbatim).

---

## 5. Migration plan (phased, behind a flag)

- **Phase 0** — baseline: pin the current full test + eval results (LangSmith
  `mint_v3_eval`), record which ~13 tests are RED and why.
- **Phase 1** — build the flat graph behind `REACT_IMPL=flat|prebuilt` (default
  `prebuilt`). Zero behavior change when off.
- **Phase 2** — wire `agent`/`tools`/`validate`/routing; prove **R1 + R2** on a
  scratch thread (stream a normal answer, an ADD, and a validator-soft-warn turn;
  diff the SSE event sequence byte-for-byte against prebuilt).
- **Phase 3** — re-map `direct_add` + `emission_guard` (R3); add the
  `suggestions` node (the goal) reading merged channels.
- **Phase 4** — eval gate (below). Flip default to `flat` only if green.
- **Phase 5** — delete prebuilt path + `messages`-reconstruction hacks +
  SSE-adapter suggestion call.

## 6. Eval gate (must pass before flipping default)

- 100% of the existing integration suite (`tests_integration/I1*.py`) at parity
  or better; the ~13 known-RED either GREEN or explained.
- SSE byte-diff: prebuilt vs flat produce the SAME event order/shape for: plain
  analyst, advisor, ADD (proposal card), validator-soft-warn, slip, voice.
- LangSmith eval set `mint_v3_eval` numerical-accuracy + intent-routing scores ≥
  current baseline.
- Latency: flat ≤ prebuilt + 5% (no extra LLM round-trips introduced).

---

## 7. Recommendation

**Don't rebuild solely to make `suggestions` a node** — Option A (suggestions
node reading from `messages`) achieves the Studio-visible-node goal at a fraction
of the risk, and **R1 (streaming parity) is a genuine threat to the verified
answer/validator pipeline.**

**Do rebuild only if** you want the broader payoff together: full loop control,
dropping the `langgraph.prebuilt` dependency, and clearing ALL channel-merge
hacks at once (emission_guard, suggestions, the pre-existing RED tests). If so,
run it as its own flagged project through the Phase 0–5 + eval gate above — never
as an inline refactor.

**Cheapest unblock for the immediate ask:** Option A now; revisit the full
rebuild later if loop-control needs accumulate.
