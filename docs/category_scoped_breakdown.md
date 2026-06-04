# Category/Tag-Scoped Questions Always Itemise (2026-06-04)

## Why

"ฉันกินข้าวเดือนนี้เป็นกี่บาท" used to get a bare lump sum ("970 บาท, 2
รายการ"). When a user scopes a question to one category or tag they are
asking about *real items* — hiding the rows behind a total forces a
follow-up question and erodes trust in the number. The product decision:
a scoped total must always arrive with its underlying transactions.

Plain monthly totals and single-wallet balances keep their short answers —
itemising those would bury the one number the user wanted (EX-ANALYST-12).

## What changed

1. **Prompt** (`src/agent/prompts.py`)
   - BREAKDOWN RULE: "one category total" is no longer a single-value
     answer; only a one-wallet *balance* is. A new bullet requires fetching
     `list_transactions(...)` for the scope in the SAME `run_python` call —
     no extra LLM round-trip, just one more SQL query.
   - New few-shot `EX-ANALYST-14` teaches the full pattern:
     `resolve_category` → `sum_expense` + `list_transactions` → show ≤10
     rows, else the 10 largest plus "และอีก N รายการ รวม X บาท" with N/X
     computed in the sandbox (R3 grounding).

2. **Resolver bug this surfaced** (`tools/codeact/resolvers.py`)
   Categories are cloned per wallet; the catalog dedupes by (name, type)
   keeping the *first* copy of each name — which can come from different
   wallets. A surviving child ('ร้านอาหาร') can therefore point at a parent
   copy of 'อาหาร' that dedupe dropped. `_expand_subcategories` compared
   `parent_id` against the chosen copy's sync_id only, so those children
   vanished from the name filter and the sum silently dropped their
   transactions (970 vs the true 1,290). It now collects the sync_ids of
   ALL copies sharing the chosen name and expands by name across them.
   Leaf queries are still never widened. Pinned by UT-RS05/UT-RS05b.

   The bug predates this feature: any `resolve_category` caller hit it, but
   the old few-shots steered the LLM to substring-filter `sum_by_category`
   buckets instead, so the resolver path was nearly never exercised.

## Cost

`resolve_category` is one LLM rerank call (~0.5–1s, flash-lite) per
category-scoped question — the price of hierarchy-correct totals. The
itemised answer adds completion tokens but no extra agent round.
