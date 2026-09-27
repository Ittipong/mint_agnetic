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

| kind | when | example |
|---|---|---|
| `log` — a habit | recorded ≥3× in 90 days near this hour (±2 h); not yet today; one chip per category | 08:30 → "Starbucks 95", 12:15 → "ข้าวแกง 65", 18:40 → "BTS 44" |
| `log` — a monthly item | ≤1.5×/month (salary, rent, a subscription): only ±1 day around its usual day, and only while this month's is unrecorded | "ได้เงินเดือน 42,000" on the 25th, "Netflix 419" on the 10th |
| `ask` | a card due within 5 days with a balance; a budget at ≥80% | "บัตร KTC รอบนี้ต้องจ่ายเท่าไหร่", "งบอาหารเดือนนี้เกินไปเท่าไหร่" |
| fill | up to 3 items: questions about their own data if they have history, generic examples if not | "สัปดาห์นี้ใช้ไปเท่าไหร่" / "กาแฟ 60" |

The amount on a `log` chip is the user's usual one (the most common, and the
latest on a tie).

## Contract for clients

```
GET /chat/starters?user_id=<uuid>&limit=5
→ {"items": [{"label", "send", "kind": "log"|"ask"|"example", "reason",
              "hint"?, "category_id"?, "category_name"?}]}
```

- `label` always equals `send`. A chip that shows one sentence and sends
  another puts words in the user's mouth.
- `hint` is the one display line that says why the chip is there
  ("ปกติจดเวลานี้", "ใช้ไป 103% ของงบ", "ครบกำหนดใน 3 วัน"), or null.
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
- `reason` (for example `habit:10/12@8h`, `monthly:day25`, `budget:103%`) is
  for analytics, not display.

## Not yet

- Recurring rules (the `recurring_transactions` table is empty in dev). The
  monthly-item rule covers salary, rent and bills from history for now.
- Insight cards ("ทำไมเดือนนี้ใช้ช้อปปิ้งเยอะ") — a cross-service signal, left
  for later.
- New users get the generic examples. That experience moves into onboarding
  (see the chat backlog).
