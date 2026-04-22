# AI Financial Friend Chat — Phase 1 Spec
## Chat-First Q&A (Months 0–3)

> **Status**: Draft v1.0 | **Owner**: Product | **Date**: 2026-04-20
> **Parent doc**: `docs/ai/ai_friend_chat_spec.md` (Tier 1 + Tier 2 capability)
> **Scope**: This spec covers only Phase 1 — Read-only Q&A. Quick Log (T3) deferred to Phase 2.

---

## 1. Product Vision (Phase 1 Scope)

Phase 1 คือ **foundation layer** ของ AI Friend Chat — ทำให้ผู้ใช้ถามคำถาม 5 อย่างเกี่ยวกับตัวเองแล้วได้คำตอบที่ถูกต้อง เป็นมิตร ไม่มั่ว ใน ≤5 วินาที

**Phase 1 ไม่ใช่**: conversational AI ทั่วไป, emotional support, หรือ financial advisor
**Phase 1 คือ**: ถาม-ตอบ Read-only บน data จริง ด้วย friend voice

---

## 2. Technical Architecture

### 2.1 Data Flow

```
User Input (Thai text)
        │
        ▼
┌─────────────────┐
│  Intent Router  │  ← Keyword + embedding match → classify to 5 intents
└────────┬────────┘
         │ intent + entities (wallet_id, date_range, category)
         ▼
┌─────────────────┐
│   DB Query      │  ← Raw numbers from PostgreSQL ONLY
│   Layer         │     LLM must NEVER compute a number
└────────┬────────┘
         │ { spent: 32400, budget: 50000, remaining: 17600, ... }
         ▼
┌─────────────────┐
│   LLM Writer    │  ← Friend-voice paraphrase (no math, no hallucination)
│   (Claude Sonnet)│    System prompt: anti-shame + friend tone
└────────┬────────┘
         │ "เดือนนี้ใช้ไป 32,400 จาก 50,000 — เหลือ 17,600 นะ"
         ▼
┌─────────────────┐
│  Response +     │
│  Follow-up      │
│  Chips          │
└─────────────────┘
```

### 2.2 Two-Layer Separation (Critical)

| Layer | Responsibility | Forbidden |
|-------|---------------|-----------|
| **DB Query Layer** | Raw numbers, sums, counts, date calculations | LLM involvement |
| **LLM Writer Layer** | Paraphrase, friend tone, insight framing | Math, counting, data retrieval |

**Why**: LLM คำนวณตัวเลขผิดได้ง่าย (hallucination on numbers) — แยกเป็น 2 layer แก้ได้ที่ต้นทาง

### 2.3 Intent Router

Input: raw Thai text
Output: `IntentEnum + Map<String, dynamic> entities`

5 intents:

| Intent | Trigger Keywords (TH) | Entities |
|--------|----------------------|----------|
| `SPENT_BUDGET` | ใช้ไป, งบ, เหลือ, โอเคไหม, สถานะเดือนนี้ | month (optional, default current) |
| `WEEKLY_SUMMARY` | สัปดาห์นี้, อาทิตย์นี้, สัปดาห์ที่แล้ว, รายสัปดาห์ | week_offset (0 = current, -1 = last) |
| `DEBT_BALANCE` | หนี้, ค้าง, ค่างวด, ผ่อน, ยังเหลือเท่าไร, outstanding | none (aggregates all) |
| `GOAL_PROGRESS` | goal, เป้า, ออม, ถึงไหม, progress, เงินก้อน | goal_id (optional, default first active) |
| `PAYDAY` | เงินเดือน, วันเงินเดือน, ถึงวันจ่าย, วันจ่าย, payday | none |

Fallback: if no intent matched → `FALLBACK` → friendly deflection

### 2.4 Response Contract

Every response must contain:

```json
{
  "intent": "SPENT_BUDGET",
  "numbers": { ... },        // all raw numbers (for future debugging/audit)
  "message": "string",        // LLM-generated friend-voice response
  "chips": ["chip1", ...],   // max 3 follow-up options
  "confidence": 0.95          // for analytics
}
```

---

## 3. Intent Specifications

---

### Intent 1: SPENT_BUDGET — "เดือนนี้ฉันโอเคไหม?"

**Description**: แสดง spending vs budget ของเดือนปัจจุบัน หรือเดือนที่ระบุ

**User Query Examples (Thai)**:
- "เดือนนี้ใช้ไปเท่าไร"
- "เหลืองบเท่าไร"
- "สถานะเดือนนี้เป็นไง"
- "เดือนที่แล้วใช้ไปเท่าไร" (past month)
- "ดู budget หน่อย"
- "งบหมดยัง"

**DB Tables/Fields Needed**:
- `transactions`: `amount`, `type`, `date`, `wallet_sync_id`, `status`
- `budgets`: `amount`, `period`, `goal`
- `budget_wallets`: `budget_sync_id`, `wallet_sync_id`
- `general_wallets` / `creditcard_wallets`: `sync_id`, `user_id`

**Query Logic** (pseudo-SQL):

```sql
-- Step 1: Get user's wallets
SELECT sync_id FROM general_wallets
WHERE user_id = $userId AND is_deleted = false
UNION
SELECT sync_id FROM creditcard_wallets
WHERE user_id = $userId AND is_deleted = false;

-- Step 2: Get confirmed expenses for month
SELECT COALESCE(SUM(amount), 0) AS total_spent
FROM transactions
WHERE created_by_user_id = $userId
  AND type = 'expense'
  AND status = 'confirmed'
  AND is_deleted = false
  AND date >= $monthStart AND date < $monthStart + INTERVAL '1 month';

-- Step 3: Get budget for this month
SELECT b.amount AS budget_amount, b.goal
FROM budgets b
JOIN budget_wallets bw ON b.sync_id = bw.budget_sync_id
WHERE bw.wallet_sync_id = ANY($walletSyncIds)
  AND b.period = 'monthly'
  AND b.is_deleted = false
LIMIT 1;
```

**Response format (LLM Writer input)**:
```
numbers: {
  spent: 32400.00,
  budget: 50000.00,
  remaining: 17600.00,
  dailyAllowance: 1466.67,   -- remaining / daysLeft
  daysLeft: 12,
  percentUsed: 64.8,
  currency: "THB"
}
```

**Response (เพื่อน tone)**:
> "เดือนนี้ใช้ไป 32,400 จาก 50,000 นะ — เหลืออีก 17,600 บาท ใช้ได้อีก 12 วัน เฉลี่ยวันละ 1,466 บาท งบยังโอเคอยู่เลย 🙂"

**Follow-up chips**:
- "ดูรายละเอียดรายจ่าย"
- "หมวดไหนใช้เยอะสุด"
- "ตั้งงบใหม่"

**Edge cases**:
| Case | DB returns | LLM response |
|------|-----------|--------------|
| No transactions this month | spent = 0 | "ยังไม่มีรายจ่ายเดือนนี้เลยนะ — เริ่มบันทึกกันเลย!" |
| No budget set | budget = null | "ยังไม่ได้ตั้งงบเดือนนี้นะ — อยากตั้งไหม?" |
| Over budget | remaining < 0 | "เดือนนี้ใช้ไป 52,000 จาก 50,000 — เกินไป 2,000 นิดหน่อย ไม่ต้องกังวลนะ สังเกตุว่าหมวดไหนใช้ไปเยอะกว่าปกติ?" |
| Future month query | 0 rows | "ยังไม่ถึงเดือนนั้นนะ — ปัจจุบันเราอยู่ที่ [current month]" |

---

### Intent 2: WEEKLY_SUMMARY — "สัปดาห์นี้ใช้ไปเท่าไร?"

**Description**: Weekly spend breakdown by category, with comparison to previous week

**User Query Examples (Thai)**:
- "สัปดาห์นี้ใช้ไปเท่าไร"
- "อาทิตย์นี้เป็นไง"
- "สัปดาห์ที่แล้วเท่าไร"
- "เปรียบเทียบสัปดาห์นี้กับอาทิตย์ที่แล้ว"
- "รายสัปดาห์ย้อนหลัง 2 อาทิตย์"

**DB Tables/Fields Needed**:
- `transactions`: `amount`, `type`, `date`, `category_sync_id`, `category_name`, `wallet_sync_id`
- `categories`: `sync_id`, `name`

**Query Logic** (pseudo-SQL):

```sql
-- Get week date range
-- week_offset: 0 = current week (Mon-Sun), -1 = last week
currentWeekStart = date_trunc('week', CURRENT_DATE) + ($weekOffset * INTERVAL '1 week');
currentWeekEnd = currentWeekStart + INTERVAL '6 days';

-- Step 1: Weekly total
SELECT COALESCE(SUM(t.amount), 0) AS total_spent
FROM transactions t
WHERE t.created_by_user_id = $userId
  AND t.type = 'expense'
  AND t.status = 'confirmed'
  AND t.is_deleted = false
  AND t.date >= $weekStart AND t.date <= $weekEnd;

-- Step 2: Breakdown by category (top 5)
SELECT
  COALESCE(t.category_name, c.name, 'อื่นๆ') AS category,
  COALESCE(SUM(t.amount), 0) AS amount,
  COUNT(*) AS transaction_count
FROM transactions t
LEFT JOIN categories c ON t.category_sync_id = c.sync_id
WHERE t.created_by_user_id = $userId
  AND t.type = 'expense'
  AND t.status = 'confirmed'
  AND t.is_deleted = false
  AND t.date >= $weekStart AND t.date <= $weekEnd
GROUP BY 1
ORDER BY 2 DESC
LIMIT 5;

-- Step 3: Compare to last week
SELECT COALESCE(SUM(t.amount), 0) AS last_week_spent
FROM transactions t
WHERE t.created_by_user_id = $userId
  AND t.type = 'expense'
  AND t.status = 'confirmed'
  AND t.is_deleted = false
  AND t.date >= $lastWeekStart AND t.date <= $lastWeekEnd;
```

**Response format (LLM Writer input)**:
```
numbers: {
  totalSpent: 8420.00,
  categoryBreakdown: [
    { category: "อาหาร", amount: 3200, count: 12 },
    { category: "เดินทาง", amount: 1800, count: 8 },
    { category: "Shopping", amount: 2100, count: 3 },
    { category: "อื่นๆ", amount: 1320, count: 7 }
  ],
  lastWeekSpent: 9100.00,
  weekOverWeekChange: -7.5,  -- percent, negative = spent less
  currency: "THB"
}
```

**Response (เพื่อน tone)**:
> "สัปดาห์นี้ใช้ไป 8,420 บาทนะ — น้อยกว่าอาทิตย์ที่แล้ว 680 บาท ดีขึ้นเลย 🙂 หมวดอาหารใช้ไปเยอะสุดที่ 3,200 บาท (12 ครั้ง) ตามมาด้วย Shopping อีก 2,100"

**Follow-up chips**:
- "ดูหมวดอาหารอย่างละเอียด"
- "ทำไม Shopping เยอะจัง"
- "สัปดาห์หน้าใช้ด้วยได้เท่าไร"

**Edge cases**:
| Case | DB returns | LLM response |
|------|-----------|--------------|
| No transactions this week | spent = 0, lastWeek > 0 | "สัปดาห์นี้ยังไม่มีรายจ่ายเลยนะ — สบายไปก่อน!" |
| No transactions either week | spent = 0, lastWeek = 0 | "ยังไม่มีข้อมูลสัปดาห์เลยนะ ลองบันทึกสัก 2-3 วันแล้วคุยกันใหม่นะ" |
| Week-over-week up >50% | change > 50 | "สัปดาห์นี้ใช้ไป 12,500 — เยอะกว่าอาทิตย์ที่แล้ว 3,500 นิดหน่อย มีอะไรพิเศษหรือเปล่า?" (non-judgmental) |

---

### Intent 3: DEBT_BALANCE — "หนี้ฉันยังเหลือเท่าไร?"

**Description**: Total outstanding debt across all active obligations, with next due date and monthly payment

**User Query Examples (Thai)**:
- "หนี้ยังเหลือเท่าไร"
- "ค่างวดเดือนนี้เท่าไร"
- "ผ่อนรถเหลือเท่าไร"
- "ดูสถานะหนี้หน่อย"
- "ยังต้องจ่ายอีกกี่เดือน"
- "outstanding บัตรเครดิต"

**DB Tables/Fields Needed**:
- `obligation_wallets`: `sync_id`, `name`, `status`, `cached_total_paid`, `monthly_payment`, `due_day`
- `obligation_plan_versions`: `obligation_id`, `outstanding_principal`, `effective_from`, `effective_to`, `due_day`, `monthly_payment`
- `obligation_transactions`: `obligation_id`, `event_type`, `amount`, `event_date`
- `creditcard_wallets`: `sync_id`, `name`, `cached_used_amount`, `credit_limit`, `payment_due_day`
- `general_wallets`: `user_id`

**Query Logic** (pseudo-SQL):

```sql
-- Step 1: Active obligations (loan/credit type)
SELECT
  ow.name,
  ow.sync_id,
  ow.status,
  ow.monthly_payment,
  ow.due_day,
  ow.cached_total_paid,
  -- Get current plan version for outstanding principal
  (
    SELECT opv.outstanding_principal
    FROM obligation_plan_versions opv
    WHERE opv.obligation_id = ow.id
      AND opv.effective_from <= CURRENT_DATE
      AND (opv.effective_to IS NULL OR opv.effective_to >= CURRENT_DATE)
    ORDER BY opv.version DESC
    LIMIT 1
  ) AS outstanding_principal,
  -- Get last transaction for last payment date
  (
    SELECT ot.event_date
    FROM obligation_transactions ot
    WHERE ot.obligation_id = ow.id
      AND ot.event_type = 'payment'
      AND ot.deleted_at IS NULL
    ORDER BY ot.event_date DESC
    LIMIT 1
  ) AS last_payment_date,
  -- Total paid
  (
    SELECT COALESCE(SUM(ot.amount), 0)
    FROM obligation_transactions ot
    WHERE ot.obligation_id = ow.id
      AND ot.event_type = 'payment'
      AND ot.deleted_at IS NULL
  ) AS total_paid
FROM obligation_wallets ow
WHERE ow.user_id = $userId
  AND ow.status = 'active'
  AND ow.deleted_at IS NULL
ORDER BY ow.due_day;

-- Step 2: Credit card outstanding
SELECT
  cc.name,
  cc.cached_used_amount,
  cc.credit_limit,
  cc.payment_due_day
FROM creditcard_wallets cc
WHERE cc.user_id = $userId
  AND cc.deleted_at IS NULL
  AND cc.cached_used_amount > 0;

-- Step 3: Aggregate totals
SELECT
  COUNT(*) AS total_debts,
  SUM(COALESCE(outstanding_principal, 0)) AS total_outstanding,
  SUM(COALESCE(monthly_payment, 0)) AS total_monthly_payment,
  MIN(due_day) AS next_due_day,
  -- Next due date = this month or next month
  CASE
    WHEN MIN(due_day) >= EXTRACT(DAY FROM CURRENT_DATE) THEN
      DATE_TRUNC('month', CURRENT_DATE) + (MIN(due_day) - 1) * INTERVAL '1 day'
    ELSE
      DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' + (MIN(due_day) - 1) * INTERVAL '1 day'
  END AS next_due_date
FROM (
  SELECT outstanding_principal, monthly_payment, due_day FROM active_obligations
  UNION ALL
  SELECT cached_used_amount AS outstanding_principal, 0 AS monthly_payment, payment_due_day AS due_day FROM creditcards
) combined;
```

**Response format (LLM Writer input)**:
```
numbers: {
  totalOutstanding: 485000.00,
  totalMonthlyPayment: 18500.00,
  nextDueDate: "2026-04-25",
  daysUntilDue: 5,
  debts: [
    {
      name: "ผ่อนรถยนต์",
      outstanding: 420000.00,
      monthlyPayment: 12500.00,
      dueDay: 25,
      lastPayment: "2026-03-25"
    },
    {
      name: "บัตรเครดิต KBank",
      outstanding: 65000.00,
      monthlyPayment: 0,
      dueDay: 15,
      creditLimit: 100000.00,
      utilizationPercent: 65
    }
  ],
  currency: "THB"
}
```

**Response (เพื่อน tone)**:
> "หนี้ทั้งหมดเหลือ 485,000 บาทนะ — ค่างวดรวมเดือนนี้ 18,500 บาท วันที่ 25 นี่ต้องจ่ายงวดแรกเลย ดีนะที่บัตรเครดิตยังใช้ไปแค่ 65% ของ limit — เหลือพื้นที่ไว้ใช้ฉุกเฉินได้"

**Follow-up chips**:
- "ดูรายละเอียดแต่ละหนี้"
- "ผ่อนเร็วขึ้นได้ไหม"
- "วางแผนปิดหนี้"

**Edge cases**:
| Case | DB returns | LLM response |
|------|-----------|--------------|
| No active debts | total = 0 | "ไม่มีหนี้ active เลยนะ — โห ปลอดหนี้แล้ว! 🎉" |
| No obligation data but has credit card | cc used > 0 | "บัตรเครดิตใช้ไป 65,000 จาก 100,000 limit นะ — วันที่ 15 ต้องจ่ายขั้นต่ำ" |
| Debt fully paid | outstanding = 0 | "หนี้ก้อนนี้ปิดแล้ว ดีมาก! 🎉 ตอนนี้เหลืออีก [N] ก้อน รวม [X] บาท" |
| Past due | next_due_date < today | "มีงวดที่ค้างอยู่ 5,000 บาทนะ — แนะนำจ่ายก่อนเพื่อไม่ให้ดอกเบี้ยบวก" (non-alarmist) |

---

### Intent 4: GOAL_PROGRESS — "goal ฉันเป็นไงบ้าง?"

**Description**: Progress toward a savings goal, including amount remaining and days remaining

**User Query Examples (Thai)**:
- "goal เป็นไงบ้าง"
- "ออมไปเท่าไรแล้ว"
- "ถึงเป้าไหมแล้ว"
- "เงินก้อนสะสมเท่าไร"
- "เหลืออีกเท่าไรถึงเป้า"
- "ดู goal หน่อย"

**DB Tables/Fields Needed**:
- `goal_wallets`: `sync_id`, `name`, `target_amount`, `target_date`, `cached_balance`, `start_date`, `achieved_at`, `is_closed`
- `transactions`: `amount`, `type`, `date` (to verify cached_balance)
- `general_wallets`: `user_id`

**Query Logic** (pseudo-SQL):

```sql
-- Step 1: Get all active goals (not closed)
SELECT
  gw.sync_id,
  gw.name,
  gw.target_amount,
  gw.target_date,
  gw.cached_balance,
  gw.start_date,
  gw.achieved_at,
  gw.is_closed,
  -- Calculate derived fields
  CASE
    WHEN gw.cached_balance >= gw.target_amount THEN gw.target_amount
    ELSE gw.cached_balance
  END AS current_amount,
  CASE
    WHEN gw.cached_balance >= gw.target_amount THEN 0
    ELSE gw.target_amount - gw.cached_balance
  END AS amount_remaining,
  -- Days remaining (if target_date set)
  CASE
    WHEN gw.target_date IS NULL THEN NULL
    ELSE GREATEST(0, gw.target_date - CURRENT_DATE)
  END AS days_remaining,
  -- Progress percentage (cap at 100)
  LEAST(100, ROUND(
    CASE
      WHEN gw.cached_balance >= gw.target_amount THEN 100
      ELSE (gw.cached_balance / NULLIF(gw.target_amount, 0)) * 100
    END, 1
  )) AS progress_percent,
  -- Monthly required to reach goal (if target_date set and not achieved)
  CASE
    WHEN gw.target_date IS NULL OR gw.cached_balance >= gw.target_amount THEN NULL
    ELSE ROUND(
      (gw.target_amount - gw.cached_balance) /
      NULLIF(EXTRACT(EPOCH FROM (gw.target_date - CURRENT_DATE)) / 86400.0 / 30.0, 0)
    , 0)
  END AS monthly_required
FROM goal_wallets gw
WHERE gw.user_id = $userId
  AND gw.is_deleted = false
  AND gw.is_closed = false
ORDER BY
  CASE WHEN gw.achieved_at IS NULL THEN 0 ELSE 1 END,  -- active first
  gw.target_date ASC NULLS LAST;

-- Step 2: If specific goal requested, get detail
SELECT
  gw.name,
  gw.target_amount,
  gw.cached_balance,
  gw.target_date,
  -- Recent deposits (last 3)
  (
    SELECT json_agg(json_build_object(
      'amount', t.amount,
      'date', DATE(t.date)
    ) ORDER BY t.date DESC)
    FROM transactions t
    WHERE t.wallet_sync_id = gw.sync_id
      AND t.type = 'goalDeposit'
      AND t.status = 'confirmed'
      AND t.is_deleted = false
    LIMIT 3
  ) AS recent_deposits
FROM goal_wallets gw
WHERE gw.sync_id = $goalId AND gw.user_id = $userId;
```

**Response format (LLM Writer input)**:
```
numbers: {
  goals: [
    {
      id: "uuid",
      name: "เงินดาวน์รถ",
      target: 200000.00,
      current: 145000.00,
      remaining: 55000.00,
      progressPercent: 72.5,
      targetDate: "2026-12-31",
      daysRemaining: 255,
      monthlyRequired: 6471,
      isAchieved: false
    },
    {
      id: "uuid",
      name: "Emergency Fund",
      target: 100000.00,
      current: 102000.00,
      remaining: 0,
      progressPercent: 100,
      targetDate: "2026-06-01",
      daysRemaining: 42,
      monthlyRequired: null,
      isAchieved: true
    }
  ],
  totalGoals: 2,
  achievedGoals: 1,
  totalSaved: 247000.00,
  currency: "THB"
}
```

**Response (เพื่อน tone)**:
> "goal มี 2 อันนะ — **เงินดาวน์รถ** อยู่ที่ 72.5% แล้ว (145,000/200,000) เหลืออีก 55,000 ถึงเป้า ถ้าจะทันสิ้นปีต้องออมเดือนละ 6,471 บาท — **Emergency Fund** ถึงเป้าแล้ว! ดีมาก 🎉"

**Follow-up chips**:
- "ฝากเพิ่ม goal รถ"
- "ขยับ target date"
- "ถอนออกได้ไหม"

**Edge cases**:
| Case | DB returns | LLM response |
|------|-----------|--------------|
| No goals | goals = [] | "ยังไม่มี goal สักอันเลยนะ — อยากตั้ง goal แรกไหม? เช่น ออมเงินดาวน์รถ หรือเงินฉุกเฉิน!" |
| Goal achieved | progressPercent >= 100 | "goal [name] ถึงเป้าแล้ว! 🎉 ดีมากเลย — อยากตั้ง goal ใหม่ไหม?" |
| No target date | target_date = null | "[goal name] อยู่ที่ [X]% แล้ว ([current]/[target]) แต่ยังไม่ได้ตั้ง target date — อยากตั้งไหม?" |
| Goal overdue | daysRemaining < 0 AND not achieved | "goal [name] เลยกำหนดไปแล้ว 12 วันนะ — อยากขยับ target date ไปไหม?" |

---

### Intent 5: PAYDAY — "เงินเดือนวันที่เท่าไร?"

**Description**: Identify typical payday from income transactions and calculate days until next payday

**User Query Examples (Thai)**:
- "เงินเดือนวันที่เท่าไร"
- "ถึงวันจ่ายอีกกี่วัน"
- "วันเงินเดือน"
- "payday วันไหน"
- "เงินเดือนจะเข้าเมื่อไร"

**DB Tables/Fields Needed**:
- `transactions`: `amount`, `type`, `date`, `note` (for income)
- `recurring_transactions`: `frequency`, `day_of_month`, `type`, `amount`, `start_date`, `next_occurrence`
- `user_profile`: `user_id` (for context)

**Query Logic** (pseudo-SQL):

```sql
-- Step 1: Detect payday from recurring income transactions
SELECT
  rt.day_of_month,
  rt.frequency,
  rt.next_occurrence,
  rt.amount
FROM recurring_transactions rt
WHERE rt.created_by_user_id = $userId
  AND rt.type = 'income'
  AND rt.status = 'active'
  AND rt.is_deleted = false
ORDER BY rt.day_of_month;

-- Step 2: Detect payday from historical income transactions
-- Find most common day-of-month for income transactions
SELECT
  EXTRACT(DAY FROM t.date)::INTEGER AS day_of_month,
  COUNT(*) AS frequency,
  SUM(t.amount) AS total_amount,
  AVG(t.amount) AS avg_amount
FROM transactions t
WHERE t.created_by_user_id = $userId
  AND t.type = 'income'
  AND t.status = 'confirmed'
  AND t.is_deleted = false
  AND t.date >= CURRENT_DATE - INTERVAL '6 months'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 3;

-- Step 3: Calculate next payday
-- If recurring found: use next_occurrence
-- If historical pattern found: use most frequent day this/next month
WITH recurring_payday AS (
  SELECT day_of_month,
         next_occurrence,
         amount
  FROM recurring_transactions
  WHERE user_id = $userId AND type = 'income' AND status = 'active'
  LIMIT 1
),
historical_payday AS (
  SELECT day_of_month, COUNT(*) as freq
  FROM transactions
  WHERE user_id = $userId AND type = 'income' AND status = 'confirmed'
    AND date >= CURRENT_DATE - INTERVAL '6 months'
  GROUP BY 1
  ORDER BY 2 DESC LIMIT 1
),
combined AS (
  SELECT day_of_month, 'recurring' AS source FROM recurring_payday
  UNION ALL
  SELECT day_of_month, 'historical' AS source FROM historical_payday
)
SELECT
  c.day_of_month,
  c.source,
  CASE
    WHEN c.day_of_month >= EXTRACT(DAY FROM CURRENT_DATE)::INTEGER
    THEN DATE_TRUNC('month', CURRENT_DATE) + (c.day_of_month - 1) * INTERVAL '1 day'
    ELSE DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' + (c.day_of_month - 1) * INTERVAL '1 day'
  END AS next_payday,
  GREATEST(0, (
    CASE
      WHEN c.day_of_month >= EXTRACT(DAY FROM CURRENT_DATE)::INTEGER
      THEN DATE_TRUNC('month', CURRENT_DATE) + (c.day_of_month - 1) * INTERVAL '1 day'
      ELSE DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month' + (c.day_of_month - 1) * INTERVAL '1 day'
    END - CURRENT_DATE
  ))::INTEGER AS days_until,
  COALESCE(r.amount, (SELECT AVG(amount) FROM transactions WHERE user_id = $userId AND type = 'income' AND status = 'confirmed' AND date >= CURRENT_DATE - INTERVAL '6 months')) AS last_income_amount
FROM combined c
LEFT JOIN recurring_payday r ON c.source = 'recurring'
ORDER BY c.source = 'recurring' DESC, c.day_of_month DESC
LIMIT 1;
```

**Response format (LLM Writer input)**:
```
numbers: {
  nextPayday: "2026-05-05",
  daysUntil: 15,
  typicalPayday: 5,
  typicalAmount: 45000.00,
  source: "historical",  -- or "recurring"
  currency: "THB"
}
```

**Response (เพื่อน tone)**:
> "เงินเดือนเข้าทุกวันที่ 5 นะ — อีก 15 วัน ปกติเข้าประมาณ 45,000 บาท ถ้าวางแผนดีๆ เงินที่เหลือหลังจ่ายหนี้จะประมาณ 26,500 บาท"

**Follow-up chips**:
- "วางแผนเงินเดือนให้หน่อย"
- "ตั้งแผนออมหลังเงินเข้า"
- "ดูรายได้เดือนที่แล้ว"

**Edge cases**:
| Case | DB returns | LLM response |
|------|-----------|--------------|
| No income data | payday = null | "ยังไม่เห็น pattern เงินเดือนเลยนะ — ลองบันทึกรายได้สัก 2-3 เดือนแล้วถามใหม่นะ" |
| payday is today/tomorrow | daysUntil <= 1 | "เงินเดือนเข้าพรุ่งนี้แล้ว! 🎉" หรือ "เงินเดือนเข้าวันนี้เลย!" |
| Multiple income sources | multiple day_of_month | "เห็นว่ามีรายได้หลายทางนะ — เงินเดือนเข้าวันที่ 5 และ extra income วันที่ 20" |
| Only historical (no recurring) | source = "historical" | "ส่วนใหญ่เงินเข้าวันที่ [X] นะ (จากข้อมูล 6 เดือนที่ผ่านมา)" — แนะนำตั้ง recurring เพื่อ track แม่นขึ้น |

---

## 4. System Prompt Guidelines

### 4.1 Core System Prompt Template

```
You are "Fin" — a friendly Thai financial companion in the Mint Money app.
You are NOT a bank employee, financial advisor, or accountant.
You are a supportive friend who happens to have access to the user's financial data.

CORE RULES:
1. NUMBERS COME FROM THE DATA. Never invent, round, or calculate a number.
   All numbers you display must come verbatim from the `numbers` object provided.
2. FRIEND TONE. Speak like a supportive Thai friend — warm, casual, never judgmental.
3. NO SHAME. Never say "spend too much", "over budget", "shouldn't have", "waste".
   Use observational language: "สังเกตว่า...", "ลองดู...", "อาจจะ..."
4. CASUAL THAI. Use "ผม", "นะ", "เลย", "จ๊ะ". Avoid formal "ท่าน", "ครับ/ค่ะ" (too formal).
5. SHORT RESPONSES. Keep initial responses under 3 sentences. Expand only if user asks.
6. CONFIDENCE. If data is insufficient, say "ยังไม่เห็นข้อมูลพอ" — do not guess.
7. NOT HUMAN. If asked, you are AI but "อยู่ข้างผู้ใช้เสมอ"

ANTI-SHAME PHRASES:
- BAD: "ใช้เยอะเกินไป", "ควรลดค่าใช้จ่าย", "ไม่ควรซื้อ"
- GOOD: "สังเกตว่าหมวดนี้เยอะกว่าปกติ", "ลองระวังหมวด...ไหม?", "ไม่เป็นไร ทุกอย่างมี ups and downs"

THAI FRIEND VOICE EXAMPLES:
- "เดือนนี้ใช้ไป 32,400 จาก 50,000 นะ — เหลือ 17,600 งบยังโอเคเลย"
- "goal ออมเงินฉุกเฉินถึง 68% แล้ว ดีขึ้นเรื่อยๆ เลย"
- "หนี้เหลืออีก 485,000 — ทุกงวดที่จ่ายคือก้าวที่ดีนะ"
- "ยังไม่มีข้อมูลพอเลยนะ ลองบันทึกสัก 2-3 สัปดาห์แล้วถามใหม่นะ"
```

### 4.2 Response Length Guidelines

| Context | Max lines | Example |
|---------|-----------|---------|
| Quick answer (single metric) | 1-2 lines | "เดือนนี้ใช้ไป 32,400 นะ — เหลืออีก 17,600" |
| Summary (multiple metrics) | 3-4 lines | "เดือนนี้รวม 32,400 บาท งบเหลือ 17,600 วันละ 1,466 นะ" |
| Goal progress (multiple goals) | 5-6 lines | Bullet-style + emoji |
| Debt detail (multiple debts) | 5-6 lines | Per-debt summary + total |

### 4.3 Chip Generation Guidelines

Every response must include 1-3 follow-up chips. Rules:
- Chips must be **actionable** (tap → do something), not rhetorical
- Chips must be **specific** (name the category/goal/debt), not generic
- Max 3 chips per response
- Chips must be in Thai

---

## 5. Privacy & Consent Model

### 5.1 Data Access Control

Phase 1 ใช้ **per-intent opt-in** — user ต้อง consent ให้ AI เห็น data ก่อนถาม

```
Onboarding Flow:
1. First time open Chat tab → Consent screen
   "Fin จะถามได้ก็ต่อเมื่อคุณอนุญาตให้ดูข้อมูลการเงินของคุณ
    ข้อมูลจะถูกส่งไปประมวลผลบน server ของเราเท่านั้น
    [อนุญาต] [ไม่ตอนนี้]"
2. If denied → Chat still opens but shows empty state with "อนุญาตให้ Fin เห็นข้อมูล" chip
3. Consent stored in user_profile.consent_ai_chat = true/false + timestamp
```

### 5.2 Data Minimization

| Intent | Data accessed | Not accessed |
|--------|--------------|-------------|
| SPENT_BUDGET | transaction amounts, budget amounts | transaction notes, merchant names |
| WEEKLY_SUMMARY | category names, amounts, counts | full transaction list, notes |
| DEBT_BALANCE | obligation names, outstanding, due dates | lender details, loan contract numbers |
| GOAL_PROGRESS | goal names, amounts, dates | goal notes |
| PAYDAY | income dates, amounts | employer names, income sources |

### 5.3 Data Retention

- Chat history: 30 days rolling, then auto-delete
- AI response logs: 90 days (for quality review)
- No training data: responses not used to train LLM

---

## 6. Onboarding Chips UX

### 6.1 Empty State (First Launch)

```
┌─────────────────────────────────┐
│                                 │
│   [Fin Avatar - friendly]        │
│                                 │
│   สวัสดีครับ! ผม Fin            │
│   เพื่อนการเงินของคุณ           │
│                                 │
│   ลองถามอะไรก็ได้ เช่น:        │
│                                 │
│   [เหลือเท่าไรเดือนนี้]         │
│   [สัปดาห์นี้ใช้ไปเท่าไร]       │
│   [goal ผมเป็นไงบ้าง]           │
│   [หนี้เหลือเท่าไร]             │
│   [เงินเดือนวันที่เท่าไร]        │
│                                 │
└─────────────────────────────────┘
```

### 6.2 Chips Tap → Send to Chat

Each onboarding chip sends a pre-filled message to chat, triggering the appropriate intent.
This seeds the first interaction for users who don't know what to ask.

### 6.3 Post-Consent State

If user has consented but chat is empty:
- Show suggested chips: [สถานะเดือนนี้] [ดู goal] [หนี้ค้าง]
- No personal data visible until user initiates

---

## 7. Success Metrics & Acceptance Criteria

### 7.1 Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| p95 latency | < 5s (end-to-end) | Server-side timing, P95 of daily pings |
| p50 latency | < 2s | Server-side timing |
| Intent accuracy | > 95% | Manual eval on 100 random queries/week |
| Hallucination rate | < 2% | Weekly spot-check: verify numbers match DB |
| Response completeness | > 98% | All 5 intents return data when data exists |
| Fallback rate | < 5% | % queries falling to FALLBACK intent |

### 7.2 User Engagement Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Chat WAU / Total WAU | > 20% | Feature flag cohort |
| Avg turns per session | > 2.5 | Per-session turn count |
| Follow-up chip tap rate | > 30% | Click tracking on chips |
| Repeat chat users 7d | > 50% | D7 chat user retention |
| WMCU (Weekly Meaningful Conversations) | > 1.0 | ≥2 turns + action/chip/thumbs-up |

### 7.3 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Thumbs-up rate | > 70% | Explicit feedback |
| Thumbs-down rate | < 10% | Explicit feedback |
| "Feel worse after chat" | < 1% | Post-chat 1-question survey |
| Shame/judgment incidents | 0 | Weekly red-team review |

### 7.4 Phase 1 Acceptance Criteria (Go/No-Go)

Before advancing to Phase 2, all must pass for 4 consecutive weeks:

1. **Accuracy gate**: Hallucination rate < 2% AND Intent accuracy > 95%
2. **Latency gate**: p95 < 5s sustained
3. **Engagement gate**: Chat WAU / Total WAU > 15% AND thumbs-up > 65%
4. **Safety gate**: Zero shame/judgment incidents in weekly red-team review

---

## 8. Open Questions (Need Decision Before Build)

### OQ1: Holiday/Weekend Payday Handling

**Question**: If payday (e.g., day 5) falls on weekend/holiday, do we show day 5 or the actual deposit day?

**Options**:
- A: Show calendar day 5 (user knows it's "on or around")
- B: Show actual transaction date (more accurate but requires more complex query)
- C: Show both: "วันที่ 5 (เข้า ศุกร์ 5 พ.ค.)"

**Recommendation**: A for MVP (simplest), with a note when it falls on weekend.

### OQ2: Credit Card "Minimum Due" vs "Full Payment"

**Question**: For credit card debt intent, should "outstanding" = full statement balance or just minimum due?

**Options**:
- A: Full statement balance (more accurate for total debt view)
- B: Minimum due (less scary, more actionable)
- C: Show both: "ค้าง 65,000 จ่ายขั้นต่ำ 3,250"

**Recommendation**: A + C hybrid — show total outstanding prominently, mention minimum due as secondary.

### OQ3: Payday Source Priority

**Question**: If user has both recurring income AND historical income patterns, which takes priority?

**Options**:
- A: Recurring transactions always (user intentionally set it up)
- B: Historical pattern always (data-driven, more accurate for irregular income)
- C: Recurring if reliable (>3 months), else historical

**Recommendation**: C — show source in response so user can correct if wrong.

### OQ4: Goal Without Target Date

**Question**: For goals without a target date, how should the AI respond about "days remaining"?

**Options**:
- A: Omit days remaining entirely
- B: Show "no deadline set" chip → [ตั้ง target date]
- C: Calculate reasonable default (e.g., 12 months from now)

**Recommendation**: B — acknowledge goal exists but prompt to set deadline (increases engagement with goals feature).

### OQ5: Multi-Currency Users

**Question**: How to handle users with multiple currencies in wallets?

**Options**:
- A: Convert all to THB using latest exchange rate (show original + converted)
- B: Show per-currency breakdown with explicit currency labels
- C: Default to THB, show "includes X EUR / Y USD" as footnote

**Recommendation**: B for MVP (simplest, least surprising) — per-currency totals with explicit labels.

---

## 9. Technical Implementation Notes

### 9.1 Backend Endpoint

```
POST /api/v1/chat/ask
Headers: Authorization: Bearer <token>
Body: {
  "message": "เดือนนี้ใช้ไปเท่าไร",
  "sessionId": "uuid"  // for multi-turn context (future)
}

Response: {
  "intent": "SPENT_BUDGET",
  "numbers": { ... },
  "message": "เดือนนี้ใช้ไป 32,400...",
  "chips": ["ดูรายละเอียด", "หมวดไหนเยอะสุด", "ตั้งงบใหม่"],
  "confidence": 0.97
}
```

### 9.2 Rate Limiting

- 20 requests/minute/user (intent + fallback)
- Burst: max 5 requests in 10 seconds
- Over limit → 429 with friendly message: "ขอพักหายใจก่อนนะ รอสักครู่..."

### 9.3 Error Handling

| Error | HTTP Code | User Message |
|-------|-----------|--------------|
| DB timeout | 504 | "ข้อมูลมาช้านิดนึง ลองถามใหม่ได้ไหม?" |
| No data for intent | 200 (valid) | Friendly deflection per intent |
| Auth failure | 401 | "เซสชันหมดแล้ว ลองเปิด app ใหม่นะ" |
| Rate limited | 429 | "ขอพักหายใจก่อนนะ รอสักครู่..." |
| LLM timeout | 200 with fallback | "ขอเวลาคิดหน่อยนะ... ลองถามใหม่ได้ไหม?" |

### 9.4 Intent Classifier Implementation

For Phase 1 MVP, use keyword matching + embedding similarity (no fine-tuned model):

```
1. Keyword match: exact Thai keyword → high confidence (0.9+)
2. Embedding similarity: user query vs 5 intent examples → pick highest
3. Threshold: if best score < 0.6 → FALLBACK
4. Confidence = keyword_match ? 0.95 : embedding_score
```

Embed 20 example queries per intent (Thai, diverse phrasing).

---

## 10. Rollout Plan

### Week 1–2: Infrastructure
- [ ] Chat API endpoint with intent router
- [ ] DB query functions per intent
- [ ] LLM writer integration (Claude Sonnet)
- [ ] Consent flow UI

### Week 3–4: Core Chat UX
- [ ] Chat UI (message bubbles, loading states, chips)
- [ ] 5-intent full implementation
- [ ] Error handling + rate limiting
- [ ] Onboarding chips

### Week 5–6: Polish
- [ ] Friend voice tuning (prompt iteration)
- [ ] Red-team test (100 adversarial queries)
- [ ] Latency optimization
- [ ] Fallback rate < 5%

### Week 7–8: QA & Soft Launch
- [ ] Internal beta (team + alpha users)
- [ ] Hallucination audit
- [ ] Performance baseline established
- [ ] Go/No-go decision

---

*Spec status: Draft — pending review from CEO + CTO on open questions OQ1-OQ5 before dev begins*
