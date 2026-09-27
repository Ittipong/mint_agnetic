# Chat starter chips (`GET /chat/starters`)

## Why

The empty chat screen is the moment the user decides what to type. It showed
the same ~30 canned prompts to everyone: "เติมน้ำมัน 700" to people without a
car, "ใช้จ่ายเกินงบไหม" to people without a budget. A chip that matches the
user's own life turns a record into one tap, which is the product's core
promise (Nimo = the easiest way to track money).

There is no LLM. The screen must render on every chat open, instantly
(~0.3 s through the tunnel), and chips built from rules are always right.

## What the user sees

Two parts: the records card, then a long list of example questions.

| kind | when | example |
|---|---|---|
| `log` — a habit | recorded ≥3× in 90 days near this hour (±2 h); not yet today; one chip per category; at most 3 | 08:30 → "Starbucks 95", 12:15 → "ข้าวแกง 65", 18:40 → "BTS 44" |
| `log` — a monthly item | ≤1.5×/month (salary, rent, a subscription): only ±1 day around its usual day, and only while this month's is unrecorded | "ได้เงินเดือน 42,000" on the 25th, "Netflix 419" on the 10th |
| `example` | a user with no history (no records card for a user who has history but no habit this hour) | "กาแฟ 60", "ข้าวเที่ยง 80" |
| `ask` | always: the fixed list of 14 example questions | "หมวดไหนใช้เยอะที่สุดเดือนนี้", "บัตรเครดิตรอบนี้ต้องจ่ายเท่าไหร่" |

The amount on a `log` chip is the user's usual one (the most common, and the
latest on a tie).

The questions are deliberately generic (owner, 2026-09-27). Their job is to
show how much Nimo can answer, so there are many, they scroll, and they quote
none of the user's numbers. An earlier version built questions from the
user's state ("งบอาหารเกินไปเท่าไหร่" at 103%) and was dropped. A user with no
data sees "Nimo ทำอะไรได้บ้าง" first.

## Contract for clients

```
GET /chat/starters?user_id=<uuid>&limit=3     # limit = max record chips
→ {"items": [{"label", "send", "kind": "log"|"ask"|"example", "reason",
              "hint"?, "category_id"?, "category_name"?}]}
```

- `label` always equals `send`. A chip that shows one sentence and sends
  another puts words in the user's mouth.
- `hint` is the one display line that says why the chip is there
  on record chips ("ปกติจดเวลานี้", "ปกติจดวันที่ 25"), null on questions.
- Record chips (`log`, and `example` records) carry the category:
  `category_id` is the user's own category sync_id, used to draw its icon.
  For examples it is looked up by name and is null if the user renamed it.
- Show `label`. On tap of `log` / `ask`, send `send` as the user's message and pass
  `origin: "starter:<kind>"` in the /chat/stream body. It is logged in the
  session header, so the tap rate can be measured per signal.
- On tap of an `example`, put `send` in the input box instead of sending: an
  example amount is not the user's, and sending it creates a card they never
  meant to make.
- The endpoint never errors. If it fails it serves the examples, and the app
  keeps its own static list as a last resort.
- `reason` (for example `habit:10/12@8h`, `monthly:day25`, `example`) is
  for analytics, not display.

## Not yet

- Recurring rules (the `recurring_transactions` table is empty in dev). The
  monthly-item rule covers salary, rent and bills from history for now.
- Insight cards ("ทำไมเดือนนี้ใช้ช้อปปิ้งเยอะ") — a cross-service signal, left
  for later.
- New users get the generic examples. That experience moves into onboarding
  (see the chat backlog).
