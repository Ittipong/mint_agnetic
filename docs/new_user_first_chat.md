# New user — the first chat

## Why this exists

Onboarding leaves a new user with **one wallet and no transactions**. Their
first chat decides whether they come back, and the product promise (Nimo = the
easiest way to track money) has to show in it. In that chat they should:

1. record something in seconds,
2. get a number that is **theirs**,
3. see a reason to open the chat again tomorrow.

Before this change, a brand-new user was answered as if they had months of
history. They were told "เงินสำรองฉุกเฉิน 6 เดือน = 0 บาท", or "รายได้เฉลี่ย
11,667" after one salary divided by 3 months. Questions with no data hit dead
ends ("ยังไม่มีข้อมูล ลองเริ่มบันทึกดูนะครับ"). Some chips offered things the
chat cannot do ("เพิ่มบัตรเครดิต"). The `[INTENT:wallet_created]` marker, which
mobile sends at the very first moment, reached the LLM raw, because v3 had no
handler (the v2-era `onboarding.py` was dead code).

## What the agent now knows

Every turn, `user_stage.py` loads the user's stage from the DB, which is
deterministic (no LLM):

| stage | meaning |
|---|---|
| `new` | no confirmed transaction yet |
| `starting` | first transaction less than 30 days ago: too little history for averages or trends |
| `established` | nothing special |

For `new` and `starting`, one line goes at the end of the system prompt. The
cached prefix is unchanged. That line points at the **NEW USER PLAYBOOK** in
`prompts.py`.

## The experience

- **Wallet created** → a welcome with no LLM call. It covers the three ways to
  record (type, speak, slip) and asks one question: monthly income. The chips
  lead into the starter plan.
- **Starter plan (about a minute)** → income → fixed monthly costs in one
  message → `starter_plan()` works out what they can spend a day until payday,
  a monthly saving (20% or 10%, depending on how much the fixed costs leave)
  and a starter emergency target (3 months). Only then does it ask, PDPA-gated,
  whether it may remember the income.
- **No hollow numbers** → no monthly averages or trends while history is days
  long. The agent uses the numbers the user states.
- **No dead ends** → a question with no data yet gets one line plus one
  concrete next step.
- **First record** → "รายการแรกของคุณ 🎉".
- **Chips** → app-only actions (add a card, set a goal, delete a record, …) are
  filtered out deterministically, and for early users the chips steer to the
  starter plan instead of trends.

## Guards that are code, not prompt

Live rounds showed the LLM breaking some rules even when they were in the
prompt. Consent and chips are therefore enforced deterministically; the plan-input
rule is still a prompt rule and is watched by the eval:

- **Consent (PDPA):** `set_user_preference(memory_consent=true)` succeeds only
  if the user's latest message agrees. That means they asked to be remembered,
  or said yes right after the assistant asked. Live, the LLM had set consent on
  its own while the user was listing bills.
- **Plan inputs are not records** (classifier prompt rule, held 3/3 live runs):
  a list of costs that answers "ค่าใช้จ่ายคงที่ต่อเดือนมีอะไรบ้าง" is an answer,
  not a multi-item ADD. Before the rule, the user's bills became a card to
  record them as expenses.
- **App-only chips** are filtered after the chip LLM.

## How to check it

`evals/chat_loop/run_newuser.sh` resets the fixed test user
(`qa-newuser@chat.test`: one wallet, 5,000 THB, the app's system categories).
It runs `scenarios_newuser.json` with the scenarios that write data running
alone. `check_newuser.py` then enforces the rules above, including the
starter-plan maths: 25,000 income with 6,998 fixed costs → 1,000 a day until
the 1st, 5,000 saved a month, 60,000 emergency target.
