# Chip-chain QA — tap the suggested question 6 times in a row (2026-09-27)

**Question:** are the follow-up suggestion chips actually useful, are they fast
enough, and are answers still correct after a user taps chips 6 times in a row?

**How:** `evals/chat_loop/chip_chain.py` runs over the Cloudflare tunnel at the API
level (the UI is too slow for this). There are 4 seed questions. Each chain is the
seed plus 6 chip taps in one thread, 28 turns in total. The chip index rotates on
each hop, so the user follows a different angle each time. Every number in every
answer was checked against the dev DB (user `ba91d8a5…`, data as of 2026-09-27).

```
.venv/bin/python evals/chat_loop/chip_chain.py --hops 6 --pick rotate
```

## Correctness — 27/28 turns fully correct, 1 wrong

About 60 figures were checked. They include monthly totals, the per-category
split, month-over-month changes, the named purchases (iPad Air 16,900 and Nike
3,290), card spend per card per month, the 3-month card split, wallet balances,
goals, card limits, statement days, and the advice arithmetic (50/30/20 and months
to goal). All of them match the DB exactly. The context held across all 6 taps.

**Wrong, and it happens every time (4 out of 4 repeats):** "บัตรเครดิตต้องจ่ายเท่าไหร่ เมื่อไหร่"
→ "…ต้องชำระ 24,111 … ยังมียอดใช้จ่ายรวมในรอบปัจจุบันอีก **24,943 บาท** ซึ่งจะไปเรียกเก็บในรอบถัดไป".
In fact, only 832 was charged after the Sep 20 statement (982 − 150 cashback).
24,943 is the *total* owed, and it already includes the 24,111. The answer makes
the debt look about twice its real size.
Root cause: `creditcard_list()` returns `used` (the total owed) and `amount_due`,
but it has no field for the charges made after the statement date. The LLM
therefore reads `used` as "this cycle's charges".

## Usefulness — good at drilling down, weak across a whole conversation

What works:
- None of the 28 turns ended without chips. Every one offered 2–3.
- Chips are questions the user's own data can answer. They also follow a natural
  drill-down. C1 went: total → vs last month → why → which category to cut →
  why shopping → clothes. That is the path a real user would take.

Problems (from reading every chip set):
1. **Chips repeat questions already answered.** The generator sees only the
   current turn, not the thread. Examples: C1 hop 6 offers "which category
   is highest", which hop 0 already answered. C2 hop 5 asks the same question
   as hop 3 and gets the same answer, so the tap was wasted. C3 hop 4 offers
   "statement day", which hop 3 had just answered (the same thing under another
   name).
2. **Quick-reply chips put made-up facts in the user's mouth.** C4 hop 0 asked
   for fixed costs and offered the chip "ประมาณ 10,000 บาท". The whole 50/30/20
   plan on the next turn is built on that invented number. The data already
   holds the answer: housing 9,912, bills 2,177 and subscriptions 603, about
   12,700 a month. A "not now" chip (hop 1 "ยังไม่เช็คตอนนี้") also wastes a
   turn.
3. **Chips drift off scope.** C4 hops 5–6 follow chips into generic investing
   ("SET50 vs S&P 500", "แนะนำแอปเปิดบัญชีกองทุน"). This has nothing to do with
   the user's data or with Nimo's core job.
4. **Chip wording sets off the empathy script.** On neutral chip questions (C2
   hop 6, C4 hop 3) the answer opens with "ไม่ใช่ความล้มเหลวส่วนตัว…". The
   cause is the agent prompt, but chips phrased as worries lead into it.

## Performance — chips cost about 1.8 s after every answer

| per turn (n=28) | median | p90 | max |
|---|---|---|---|
| first answer token | 5.3 s | 7.6 s | 9.5 s |
| chips arrive after the last answer token | 1.8 s | 2.1 s | 2.4 s |
| turn done | 7.7 s | 10.2 s | 11.6 s |

The chip call (gemini-2.5-flash, about 1.5–1.9 s) runs only **after** the answer
is complete, because the `gen_suggestions` node needs the answer text. It adds
about 23% (median) to every turn. The answer text is already on screen by then.
What waits is the chips and the `done` event.

---

## Fixes applied (same day) — result after 6 rounds of the chain

| | before | after (round 6) |
|---|---|---|
| chips arrive after the answer (median / p90) | 1.8 s / 2.1 s | **0.3 s** / 1.7 s |
| turn done (median) | 7.7 s | 6.4 s |
| wrong turns (of 28) | 1 | 3 — all the category bug below |

What changed, and why:
- **Card owed amounts.** `creditcard_list()` now returns `unbilled` (`used` −
  the statement still owed), and the prompt says `used` = amount_due +
  unbilled. The answer now reads "24,111 due 5 Oct + 832 new, 24,943 in total".
  UT-NS-CC02.
- **No repeated chips.** The generator now sees the thread's earlier
  questions. Each chip must say which earlier question it `repeats`, and
  code drops those along with word-for-word repeats. UT-SG20/26.
- **The chip mix (owner request).** 2 drill-down chips plus 1 sideways chip
  (what-if, habits, links to goals/cards), still answered from the user's own
  data and always placed last. In reply mode the chips are 2 replies plus 1
  sideways question as a way out. Guessed values are gone ("คำนวณจากข้อมูล
  ของฉัน" instead), and "not now", reminder and product-shopping chips are
  dropped. UT-SG21/26/27.
- **Speed.** The chip call starts when the agent begins its step after a tool
  round, using the tool data (`start_speculative` in flat_react). It is
  replaced by a normal call only when the answer ends by asking the user
  back. UT-SG24/25.
- **Found by the chain, fixed:**
  - Streaming regression: Gemini wrote an invented answer ("เงินเดือน 32,000")
    *before* its tool call, and the 160-char rule released it. Now the first
    step is never released early, narration shows its first line only, and
    leaked text is dropped from the saved answer. UT-S21/22.
  - `sum_by_wallet` SQL had `AS bucket AS bucket`, so every call failed with a
    syntax error. The model then summed a capped `list_transactions` (KTC Aug
    14,574 vs DB 15,524). Fixed and run against the dev DB. UT-NS-WAL01.
  - `list_transactions` capped at 50 rows now prints TRUNCATED to the LLM, and
    the prompt forbids totalling its rows (a "fixed costs" pass had missed the
    rent). UT-NS-LIST01.

## Category filter follows the tree — fixed (round 7: 28/28 correct)

"หมวดช้อปปิ้ง" used to give 25,345 / 24,214 / 16,900 and "nothing last month"
in one thread. SQL matched a flat list of names against `t.category_name`, and
the list depended on which path the LLM took.

`_category_filter` now follows the category tree. A name that is mostly
top-level in the user's catalog (`resolvers.is_root_dominant`, the same majority
test as `_expand_subcategories`) matches the app report's grouping: the
category's parent name, or its own name if it has no parent. Mostly-nested names
("เสื้อผ้า", "ซุปเปอร์มาร์เก็ต", "น้ำมัน") stay leaf matches on every copy.

Checked on the dev DB. Every total equals the report, and each list adds up to its total:

| | Sep | Aug |
|---|---|---|
| ช้อปปิ้ง | 25,345 (8 items) | 7,244 (8), was "none" |
| อาหาร | 2,165 | 1,690 |
| เดินทาง | 507 | 8,442 |
| ที่อยู่อาศัย | 9,912 | 9,789 |
| ซุปเปอร์มาร์เก็ต (leaf) | 4,024 | 4,178 |

UT-NS-CAT02. Also a chip equal to the question just asked is now dropped (UT-SG28).

Round 7: 28/28 turns correct, including the fixed-cost total (ที่อยู่อาศัย 29,613
+ ค่าบิล 6,273 + Subscription 1,809 over Jun 28–Sep 27). Chips arrive 0.3 s
(median) after the answer.

## Still open

Tone: the agent still sometimes opens with empathy ("น่ากังวลใจ") on neutral
questions. That comes from the agent prompt, not the chips.
