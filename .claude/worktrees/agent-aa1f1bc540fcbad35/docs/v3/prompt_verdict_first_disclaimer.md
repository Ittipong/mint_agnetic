# Verdict-first answers + tiered disclaimer (v3 prompt)

**Date:** 2026-05-30
**File touched:** `src/agent/prompts.py` (system prompt only — no code/tool changes)

## Why

Mint Money is meant to be the user's *financial friend they lean on* — the
place they go precisely because they want a clear "this is what I'd do",
not a neutral menu they have to sort out alone. The old prompt undercut that:

- **R8** forced every major-advice turn to close with the identical
  boilerplate `"นี่เป็นมุมมองทั่วไป สำหรับการตัดสินใจสำคัญควรปรึกษาผู้เชี่ยวชาญ"`.
- Four few-shot examples (HOME, DEBT-PLAN, ADVISOR-1, ADVISOR-COMPARISON)
  all ended on that same line → the model learned to hedge by reflex.
- The ROLE said "don't punt every question to an expert," but the mechanics
  (R8 + examples) pushed the opposite way. Net effect: answers read like a
  defensive call-center script.

## What changed

1. **ROLE** — added a third differentiator: *commit to a verdict*. The
   caveat comes AFTER the stance, never instead of it. We trust the user to
   make the final call; our job is a strong, honest opinion to react to.
2. **R8 → TIERED-DISCLAIMER** — verdict first; a short, *varied*, friendly
   ownership-nudge caveat ONLY for irreversible/regulated decisions
   (specific stock/fund picks, complex tax, insurance, large loans /
   refinance / big transfers) or when app data is too thin. Everyday
   questions (how much to save, is this spend normal, which debt first) get
   NO disclaimer. The old boilerplate is now an explicit ❌ anti-pattern.
3. **R13 → COMMIT-TO-A-VERDICT** (new) — any "ควร…มั้ย / ทำไงดี / เลือก
   อะไรดี" turn MUST name the pick (or yes/no/"ตึงไป") before/with the
   trade-offs. Listing options with no recommendation is forbidden. When
   data is incomplete: provisional verdict + the one fact that would flip it.
4. **E6 (regulatory)** — take a grounded stance first, then one specific
   caveat; no whole-question punt.
5. **RESPONSE STYLE** — for Advisor turns, the *Lead* line IS the verdict.
6. **Few-shot examples** — removed all 4 boilerplate disclaimers; reworked
   each Final text to lead with / end on a verdict or a varied ownership
   nudge. Only the EX-* prose (`Final text`) was edited — no `Plan`/Python
   bodies, so UT-P04 / UT-P05 lockstep guards are untouched.

## Tier rule of thumb

| Question type                                  | Disclaimer? |
|------------------------------------------------|-------------|
| Analyst / budget / "ใช้ไปเท่าไร" / which debt first | none        |
| Save Xk/mo, emergency fund, everyday planning  | none        |
| Stock/fund picks, complex tax, insurance       | short, specific, varied |
| Large loan / refinance / big transfer / buy home | short, "เทียบก่อนเซ็น" nudge |
| App data too thin to be confident              | name the missing fact |

## Safety / tests

- No test asserts the disclaimer string verbatim → wording is free to change.
- `tests/test_prompts.py` 8/8 pass (UT-P01..P05 incl. clarify_wallet guard).
- `tests/test_advice_playbook.py` 8/8 pass.
- Prompt still references R8 in `advice_playbook.py` tone notes — semantics
  preserved (R8 is now "tiered" rather than "always append"); the playbook
  guidance "append R8 disclaimer" still resolves to the new tiered behavior.
