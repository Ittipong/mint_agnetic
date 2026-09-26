# Chat QA loop — 2026-09-26

Why this note exists: a four-round test → fix → retest pass over the live chat
(`mint_agentic`, via `https://chat.minttechdev.uk`) on fresh, checkable data.
It records what broke, why, and how to re-run the loop.

## Fixture

`scripts/seed_chat_qa_6m.py --apply` wipes wallets, transactions, categories,
budgets, tags, recurring and obligation data for **all users**, then seeds
`ittipong.it@gmail.com` with 6 months (Apr–Sep 2026) of a Bangkok office
worker: salary 42,000 on the 25th, rent on the 1st, daily coffee/lunch/BTS,
subscriptions on a card, a Chiang Mai trip (Aug), a bonus (Jun), and a
spending spike (Sep, iPad). Categories and icons are parsed from the mobile
`category_seed_data.dart`, so they match what the app seeds.

The mobile app keeps its own Drift copy. After a wipe, reinstall the app or
clear its data, or it can push the old rows back.

## Bugs found and fixed

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | "3 เดือนล่าสุด" averages ~30% too high (income 68,167 vs 44,167) | `parse_period("N เดือน…")` started on the 1st of month −N, so it spanned N months plus the current partial month, and the LLM divided by N | Rolling window: exactly N months ending today (UT-NS07, UT-NS07d) |
| 2 | "ค่า subscription ต่อเดือน" → "no subscriptions" | `spend_for` compared category names case-sensitively; with that fixed, a NULL `SUM` crashed `Decimal("None")` and looped to the recursion limit | Case-insensitive match + NULL-safe sum (UT-NS-SF01); EX-ANALYST-17 |
| 3 | "กาแฟ 60 ข้าว 80 BTS 44" dropped items 2..n silently | The classifier fast path proposed the first item and skipped R7 | Classifier `multi` flag → multi-item goes to react (UT-CR14) |
| 4 | Any failed fast-path ADD → "ระบบขัดข้องชั่วคราว" | `direct_propose → post_turn` was a static edge, so its "fall back to react" route was ignored | Conditional edge (UT-CR15) |
| 5 | "จ่ายค่าเช่า 9500 …" failed | The joint resolver put the category id in `wallet_sync_id` on the single-wallet path | Single-wallet path ignores the LLM wallet and recovers the category (UT-T04b) |
| 6 | The next ADD after a confirm discarded the CONFIRMED card | Confirm finalized `proposals` but left the `pending_proposal` pointer pending | Confirm/cancel clear the pointer; `proposals` is the source of truth (UT-T04c) |
| 7 | "โอนเงินเข้า TrueMoney 500" recorded as a 500 expense | No rule for own-wallet transfers | `transfer` capability + E11 + classifier rule |
| 8 | "บัตร KTC ต้องจ่ายวันไหน" → "app has no due date" | `creditcard_list` never returned the billing or due day | Adds `billing_cycle_day`, `payment_due_day`, `last_statement_date`, `next_due_date` (UT-NS-CC01) |
| 9 | "สิ้นเดือนจะใช้เท่าไหร่" repeated the month total | `spending_pace` had no description or example | EX-ANALYST-18 (a projection that leaves one-off spikes out) |
| 10 | Wrote Python code on request | No off-topic rule | E10 |
| 11 | Advisor asked "มีเงินสำรองแล้วหรือยัง" although a goal holds 60,000 | R11(b) did not include goals | R11 now pulls `goal_progress()` and never asks for data the app already holds |
| 12 | "ยกเลิก" → claimed "ยกเลิกให้แล้ว", with no discard sent | Chat cannot discard (locked decision #11) | ADD step 4: tell the user to tap ยกเลิก on the card |
| 13 | "1,250.5 บาท" | `_fmt_amount` stripped trailing zeros | Always 2 satang digits (UT-CR16) |
| 14 | "หมวดไหนเยอะสุด" ranked leaf categories, so the ranking did not match the app report | `sum_by_category` grouped by leaf name only | `by_parent=True` folds sub-categories into the parent, the app's "แยกตามหมวด" mode (UT-NS-CAT01); overview examples use it |
| 15 | "ต้องจ่ายบัตรเท่าไหร่" gave the live `used` (24,943), which includes swipes billed next month | No statement figure existed | `creditcard_list` adds `statement_balance` and `amount_due` = closed statement − payments since (KTC: 24,111) (UT-NS-CC01) |
| 16 | Slip / receipt saved with TODAY's date instead of the printed one | The vision prompt never asked for a `date` | `date` added to the output schema; `_valid_date` fixes an unconverted BE year and rejects future dates (UT-SL08) |

Result: round 1 passed 33/40 checks → round 4 passed 46/46. Voice (3 clips,
Thai TTS) and slip (transfer slip + 7-Eleven receipt) pass end-to-end.

Unit suite: 476 passed, 0 failed (it started at 447 passed, 9 failed). The 9
old failures:
- 7 stale tests. They asserted v2 `as_node="finalize"` (v3's node is
  `post_turn`), an old wire event set without `narration_token` /
  `suggestions_pending`, in-place block mirroring (removed on purpose for
  #SSE-DUP), and a dead `mint_agentic_v3/` path.
- 2 environment leaks. `server.py`'s `load_dotenv()` switched the classify
  router ON for every test after it, and the joint resolver called a live LLM.
  `conftest.py` now pins the router OFF and stubs the resolver.

## Still open

- Nothing from this loop. The "งบ/หมวด" level choice follows the app report
  (parent for overviews, leaf for drill-down). Revisit it if the report
  changes.

## Re-run

```bash
cd mint_agentic/evals/chat_loop
python3 chat_qa.py scenarios_core.json      # → results/<id>.json (via the tunnel)
python3 check.py results                     # ground-truth facts vs answers
python3 summarize.py 'S1*'                   # read transcripts
```

A turn of `"__CONFIRM__"` calls `/transactions/confirm` on the last proposal,
as tapping the card would. The facts in `check.py` match the seeded fixture
dated 2026-09-26. Re-derive them if the seed is re-run on another day,
because the window is relative to today.
