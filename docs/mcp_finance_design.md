# MCP Finance Query Server — Design Document

## Overview

**Purpose:** AI chat ที่ตอบคำถามเรื่องการเงินของ user โดย map คำถาม → Intent → SQL Template → Execute

**Architecture:** User → Intent Classifier → Template Resolver → Param Extractor → SQL Builder → PostgreSQL

**Security Model:**
- `userId` มาจาก auth context เท่านั้น (ไม่ใช่ AI ส่ง)
- ทุก query filter ด้วย `WHERE created_by_user_id = :userId` หรือ `WHERE user_id = :userId`
- อนุญาตเฉพาะ `SELECT` — ไม่มี INSERT/UPDATE/DELETE/DROP

---

## Architecture

### High-Level Flow
```
┌─────────────────────────────────────────────────────────────┐
│                         AI Chat                              │
│  "เดือนนี้ใช้ไปเท่าไหร่", "ค่ากินเท่าไหร่", "งบเหลือเท่าไหร่" │
└────────────────────────────┬────────────────────────────────┘
                             │ question
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  MCP Tool: query_finance                     │
│            (LangGraph Subgraph — 2 LLM calls)                │
└────────────────────────────┬────────────────────────────────┘
                             │ SQL result
                             ▼
                      AI Chat → Natural Language Response
```

### LangGraph Subgraph Detail

```
question + auth_context
        │
        ▼
┌─────────────────┐
│   llm_router    │ ◄── LLM #1: classify + select_template + extract_params
│  (confidence)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │ < 0.7   │ >= 0.7
    ▼         ▼
┌──────────────┐   ┌─────────────────┐
│ ask_clarify  │   │   llm_builder  │ ◄── LLM #2: validate + build_sql
└──────┬───────┘   └────────┬────────┘
       │ (END)               │
       │              ┌──────┴──────┐
       │              │ errors?     │ no
       │              ▼            ▼
       │     ┌──────────────┐   ┌──────────────┐
       │     │format_errors │   │   executor   │ ◄── deterministic DB call
       │     └──────┬───────┘   └──────┬───────┘
       │            │                  │
       │            │            ┌────┴────┐
       │            │            │ error?  │ no
       │            │            ▼        ▼
       │            │   ┌──────────────┐ ┌──────────────┐
       │            │   │format_result│ │ format_error │
       │            │   └──────┬──────┘ └──────────────┘
       │            │          │
       └────────────►└──────────┴──────► AI Chat Response
```

---

## Intent List (25 Intents)

### Group: Spending (4)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `spending_total_period` | "เดือนนี้ใช้ไปเท่าไหร่", "สัปดาห์นี้ใช้ไปเท่าไหร่" | "How much did I spend this month?" | `spending_by_period` |
| `spending_by_category` | "ค่ากินเท่าไหร่", "ค่ารถเท่าไหร่", "แต่ละหมวดเท่าไหร่" | "How much on food?" | `category_breakdown` |
| `spending_top_categories` | "หมวดไหนใช้เยอะสุด", "ใช้เงินไปกับอะไรมากสุด" | "What did I spend the most on?" | `top_categories` |
| `spending_trend` | "แนวโน้มใช้จ่าย", "เทียบช่วงเวลา" | "Spending trend" | `spending_trend` |

### Group: Income (2)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `income_total_period` | "เดือนนี้ได้เงินเท่าไหร่", "รายได้เดือนนี้" | "Income this month?" | `income_by_period` |
| `income_vs_expense` | "รายรับรายจ่ายเดือนนี้", "เหลือเท่าไหร่" | "Income vs expenses" | `cashflow_summary` |

### Group: Goals (3)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `goal_progress` | "เป้านี้ถึงไหมแล้ว", "เก็บได้กี่เปอร์เซ็นต์" | "How close to my goal?" | `goal_progress` |
| `goal_on_track` | "ถึงเป้าทันไหม", "ต้องเติมเดือนละเท่าไหร่" | "Am I on track?" | `goal_pacing` |
| `goals_summary` | "เป้าทั้งหมด", "สถานะเป้าออม" | "All my goals" | `all_goals` |

### Group: Budgets (3)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `budget_status` | "งบเหลือเท่าไหร่", "ใช้ไปกี่บาทจากงบ" | "Budget remaining?" | `budget_status` |
| `budget_over` | "เกินงบหรือยัง", "งบบานเกินไหม" | "Over budget?" | `budget_overcheck` |
| `budgets_all` | "งบทั้งหมด", "ดูงบประมาณ" | "All my budgets" | `all_budgets` |

### Group: Recurring (3)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `recurring_upcoming` | "ค่าที่กำลังจะโดนหัก", "รายการที่ต้องจ่ายเร็วๆนี้" | "Upcoming charges?" | `recurring_soon` |
| `recurring_monthly_list` | "ค่าที่ต้องจ่ายทุกเดือน", "รายการประจำเดือน" | "Monthly recurring payments" | `recurring_monthly` |
| `recurring_total_monthly` | "ค่าประจำเดือนรวมเท่าไหร่", "ต้องจ่ายเดือนละเท่าไหร่" | "Total monthly recurring" | `recurring_monthly_total` |

### Group: Wallets (2)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `wallet_balances` | "ยอดเงินกระเป๋า", "กระเป๋าไหนเหลือเยอะสุด" | "Wallet balances" | `wallet_balances` |
| `total_networth` | "ยอดรวมทั้งหมด", "มีเงินทั้งหมดเท่าไหร่" | "Total balance" | `networth_total` |

### Group: Comparison (2)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `compare_period` | "เทียบกับเดือนก่อน", "สัปดาห์นี้เทียบสัปดาห์ก่อน" | "vs last month" | `compare_periods` |
| `compare_year` | "ปีนี้เทียบปีที่แล้ว", "เทียบกับปีก่อน" | "vs last year" | `compare_yearly` |

### Group: Alerts (3)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `budget_warning` | "ใกล้เกินงบ", "งบจะหมดแล้ว" | "Budget warning" | `budget_warning` |
| `goal_reached` | "ถึงเป้าแล้ว", "เป้าสำเร็จ" | "Goal reached!" | `goal_reached` |
| `low_balance` | "กระเป๋าเงินใกล้หมด", "เงินเหลือน้อย" | "Low balance warning" | `low_balance_wallets` |

### Group: Tags (1)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `spending_by_tag` | "แท็กนี้เท่าไหร่", "ใช้เงินไปกับแท็ก..." | "Spending by tag" | `spending_by_tag` |

### Group: Time-Based (2)

| intent_id | Thai | English | Template |
|---|---|---|---|
| `spending_by_date_range` | "ระหว่างวันที่...ถึง...", "10-20 เมษายนใช้เท่าไหร่" | "Between dates" | `spending_date_range` |
| `spending_by_day_of_week` | "วันไหนใช้เยอะสุด", "วันศุกร์ใช้เท่าไหร่" | "Which day spend most?" | `spending_by_day` |

---

## SQL Templates

### 1. `spending_by_period`
```sql
SELECT
  SUM(amount) as total,
  COUNT(*) as transaction_count
FROM transactions
WHERE created_by_user_id = :userId
  AND type = 'expense'
  AND date >= :start_date
  AND date < :end_date
```

**Params:** `start_date`, `end_date`

---

### 2. `category_breakdown`
```sql
SELECT
  c.name as category_name,
  SUM(t.amount) as total,
  COUNT(*) as transaction_count
FROM transactions t
LEFT JOIN categories c ON t.category_sync_id = c.sync_id
WHERE t.created_by_user_id = :userId
  AND t.type = 'expense'
  AND (:category_name IS NULL OR c.name ILIKE '%' || :category_name || '%')
  AND t.date >= :start_date
  AND t.date < :end_date
GROUP BY c.name
ORDER BY total DESC
```

**Params:** `start_date`, `end_date`, `category_name` (optional)

---

### 3. `top_categories`
```sql
SELECT
  c.name as category_name,
  SUM(t.amount) as total,
  COUNT(*) as transaction_count,
  ROUND(SUM(t.amount) * 100.0 / NULLIF(:total_spending, 0), 1) as percent
FROM transactions t
LEFT JOIN categories c ON t.category_sync_id = c.sync_id
WHERE t.created_by_user_id = :userId
  AND t.type = 'expense'
  AND t.date >= :start_date
  AND t.date < :end_date
GROUP BY c.name
ORDER BY total DESC
LIMIT :limit
```

**Params:** `start_date`, `end_date`, `limit` (default 5)

---

### 4. `income_by_period`
```sql
SELECT
  SUM(amount) as total,
  COUNT(*) as transaction_count
FROM transactions
WHERE created_by_user_id = :userId
  AND type = 'income'
  AND date >= :start_date
  AND date < :end_date
```

**Params:** `start_date`, `end_date`

---

### 5. `cashflow_summary`
```sql
SELECT
  SUM(CASE WHEN type = 'income' THEN amount ELSE 0 END) as income_total,
  SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) as expense_total,
  SUM(CASE WHEN type = 'income' THEN amount ELSE -amount END) as net_balance
FROM transactions
WHERE created_by_user_id = :userId
  AND date >= :start_date
  AND date < :end_date
```

**Params:** `start_date`, `end_date`

---

### 6. `goal_progress`
```sql
SELECT
  id,
  name,
  cached_balance,
  target_amount,
  ROUND(cached_balance * 100.0 / NULLIF(target_amount, 0), 1) as percent_complete,
  (target_amount - cached_balance) as remaining,
  target_date,
  achieved_at
FROM goal_wallets
WHERE user_id = :userId
  AND is_deleted = false
  AND (:goal_name IS NULL OR name ILIKE '%' || :goal_name || '%')
ORDER BY target_date ASC
```

**Params:** `goal_name` (optional)

---

### 7. `goal_pacing`
```sql
SELECT
  name,
  cached_balance,
  target_amount,
  target_date,
  (target_amount - cached_balance) as remaining,
  CASE
    WHEN target_date IS NULL OR target_date <= CURRENT_DATE THEN NULL
    ELSE ROUND((target_amount - cached_balance) /
      NULLIF(EXTRACT(MONTH FROM AGE(target_date, CURRENT_DATE)), 0), 2)
  END as monthly_needed_to_reach
FROM goal_wallets
WHERE user_id = :userId
  AND is_deleted = false
  AND achieved_at IS NULL
```

**Params:** none

---

### 8. `budget_status`
```sql
SELECT
  id,
  name,
  amount,
  spent_amount,
  (amount - spent_amount) as remaining,
  ROUND(spent_amount * 100.0 / NULLIF(amount, 0), 1) as percent_used,
  start_date,
  end_date
FROM budgets
WHERE user_id = :userId
  AND is_deleted = false
  AND start_date <= CURRENT_DATE
  AND end_date >= CURRENT_DATE
```

**Params:** none

---

### 9. `budget_overcheck`
```sql
SELECT
  id,
  name,
  amount,
  spent_amount,
  (amount - spent_amount) as remaining,
  CASE WHEN spent_amount > amount THEN true ELSE false END as is_over,
  CASE WHEN (spent_amount * 100.0 / NULLIF(amount, 0)) >= :threshold THEN true ELSE false END as is_warning
FROM budgets
WHERE user_id = :userId
  AND is_deleted = false
  AND start_date <= CURRENT_DATE
  AND end_date >= CURRENT_DATE
```

**Params:** `threshold` (default 80)

---

### 10. `recurring_soon`
```sql
SELECT
  id,
  type,
  amount,
  frequency,
  next_occurrence,
  note
FROM recurring_transactions
WHERE created_by_user_id = :userId
  AND status = 'active'
  AND next_occurrence IS NOT NULL
  AND next_occurrence <= CURRENT_DATE + INTERVAL ':days_ahead days'
ORDER BY next_occurrence ASC
```

**Params:** `days_ahead` (default 7)

---

### 11. `wallet_balances`
```sql
SELECT
  id,
  name,
  cached_balance,
  currency,
  wallet_category
FROM general_wallets
WHERE user_id = :userId
  AND deleted_at IS NULL
ORDER BY cached_balance DESC
```

**Params:** none

---

### 12. `networth_total`
```sql
SELECT
  SUM(cached_balance) as total_balance,
  COUNT(*) as wallet_count
FROM general_wallets
WHERE user_id = :userId
  AND deleted_at IS NULL
```

**Params:** none

---

### 13. `compare_periods`
```sql
SELECT
  (SELECT SUM(amount) FROM transactions
   WHERE created_by_user_id = :userId
     AND type = :type
     AND date >= :current_start AND date < :current_end) as current_period,
  (SELECT SUM(amount) FROM transactions
   WHERE created_by_user_id = :userId
     AND type = :type
     AND date >= :previous_start AND date < :previous_end) as previous_period,
  (SELECT SUM(amount) FROM transactions
   WHERE created_by_user_id = :userId
     AND type = :type
     AND date >= :current_start AND date < :current_end) -
  (SELECT SUM(amount) FROM transactions
   WHERE created_by_user_id = :userId
     AND type = :type
     AND date >= :previous_start AND date < :previous_end) as difference
```

**Params:** `current_start`, `current_end`, `previous_start`, `previous_end`, `type`

---

### 14. `budget_warning`
```sql
SELECT
  id,
  name,
  amount,
  spent_amount,
  (amount - spent_amount) as remaining,
  ROUND(spent_amount * 100.0 / amount, 1) as percent_used
FROM budgets
WHERE user_id = :userId
  AND is_deleted = false
  AND (spent_amount * 100.0 / NULLIF(amount, 0)) >= :threshold
  AND start_date <= CURRENT_DATE
  AND end_date >= CURRENT_DATE
```

**Params:** `threshold` (default 80)

---

### 15. `goal_reached`
```sql
SELECT
  id,
  name,
  cached_balance,
  target_amount,
  achieved_at
FROM goal_wallets
WHERE user_id = :userId
  AND cached_balance >= target_amount
  AND is_deleted = false
```

**Params:** none

---

### 16. `low_balance_wallets`
```sql
SELECT
  id,
  name,
  cached_balance,
  currency
FROM general_wallets
WHERE user_id = :userId
  AND deleted_at IS NULL
  AND cached_balance < :threshold
ORDER BY cached_balance ASC
```

**Params:** `threshold` (default 1000)

---

## Data Types

### Date Range Types
```go
type Period string

const (
    PeriodDay   Period = "day"
    PeriodWeek  Period = "week"
    PeriodMonth Period = "month"
    PeriodYear  Period = "year"
)
```

### Transaction Types
```go
type TransactionType string

const (
    TypeIncome    TransactionType = "income"
    TypeExpense   TransactionType = "expense"
    TypeTransfer  TransactionType = "transfer"
)
```

---

## Project Structure

```
agentic/
├── mcp_finance/
│   ├── cmd/
│   │   └── main.go                 # Entry point, server bootstrap
│   ├── internal/
│   │   ├── auth/
│   │   │   └── context.go           # Extract userId from auth token
│   │   ├── config/
│   │   │   └── config.go            # DB config, server config
│   │   ├── domain/
│   │   │   ├── intent.go            # Intent definition struct
│   │   │   └── result.go            # Query result struct
│   │   ├── intents/
│   │   │   ├── registry.go          # Intent registry + pattern matcher
│   │   │   └── patterns.go          # TH/EN patterns for each intent
│   │   ├── extractor/
│   │   │   └── param_extractor.go   # Extract date/category/amount from text
│   │   ├── templates/
│   │   │   ├── registry.go          # Template registry
│   │   │   └── finance.go           # All SQL templates
│   │   ├── executor/
│   │   │   └── query_executor.go    # Execute SQL, handle errors
│   │   └── validator/
│   │       └── sql_validator.go     # Validate generated SQL
│   └── mcp/
│       └── handlers.go              # MCP protocol handlers
```

---

## MCP Tool Definition

```json
{
  "name": "mint_money_finance",
  "description": "Query Mint Money financial data — spending, wallets, goals, budgets",
  "tools": [
    {
      "name": "query_finance",
      "description": "ถามคำถามเรื่องการเงินของคุณ — เช่น รวมรายจ่ายเดือนนี้, งบเหลือเท่าไหร่, เป้าถึงไหมแล้ว",
      "inputSchema": {
        "type": "object",
        "properties": {
          "question": {
            "type": "string",
            "description": "คำถามภาษาไทยหรืออังกฤษ เช่น 'เดือนนี้ใช้ไปเท่าไหร่' หรือ 'How much did I spend this month?'"
          }
        },
        "required": ["question"]
      }
    }
  ]
}
```

---

## LangGraph Subgraph Architecture (Optimized 1-Tool)

### Overview

ใช้ LangGraph subgraph ภายใน MCP tool handler เพื่อ orchestration ของ 1 tool ที่รองรับทุก intent

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Tool: query_finance                       │
│                     (External Interface)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ question + auth_context
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LangGraph Subgraph                              │
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│   │  llm_router  │────►│ llm_builder  │────►│   executor   │    │
│   │  (LLM #1)   │     │  (LLM #2)    │     │ (deterministic)│   │
│   └──────────────┘     └──────────────┘     └──────────────┘    │
│        │                    │                    │              │
│        │              params + template    SQL result            │
│        │                    │                    │              │
│   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐        │
│   │ classify│          │ validate │          │ format   │        │
│   │+select  │          │ + build  │          │ response │        │
│   │+extract │          │    SQL   │          │          │        │
│   └─────────┘          └─────────┘          └──────────┘        │
│                                                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │ structured_result
                              ▼
                         AI Chat Response
```

### State Schema

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

class FinanceQueryState(TypedDict, total=False):
    # Input
    question: str                          # Raw user question
    user_id: str                           # From auth context (never from user)

    # LLM #1 Output
    intent_id: Optional[str]               # e.g. "spending_total_period"
    template_id: Optional[str]             # e.g. "spending_by_period"
    params: dict                           # Extracted parameters

    # LLM #2 Output
    sql: Optional[str]                     # Final validated SQL
    validation_errors: Optional[list[str]] # If params invalid

    # Executor Output
    raw_result: Optional[list[dict]]       # DB rows
    error: Optional[str]                   # Execution error

    # Metadata
    confidence: float                      # 0.0-1.0 for routing decisions
    needs_clarification: bool              # True if confidence < threshold
    clarification_question: Optional[str]  # What to ask user
```

### Node Definitions

#### 1. `llm_router` (LLM #1 — classify + select + extract)

**Responsibility:** Single LLM call เพื่อ:
- Classify intent (จาก 25 intents)
- Select SQL template
- Extract parameters (date range, category, etc.)

**Prompt Strategy:** Few-shot พร้อม examples ของทุก intent

```
Input:  question, user_id
Output: intent_id, template_id, params, confidence

Edge: confidence >= 0.7 → llm_builder
       confidence < 0.7 → ask_clarification
```

#### 2. `ask_clarification` (Deterministic)

**Responsibility:** สร้างคำถามชี้แจงเมื่อ confidence ต่ำ

```
Input:  question, intent_id, confidence
Output: clarification_question, needs_clarification=true

Edge: → END (return to user for clarification)
```

#### 3. `llm_builder` (LLM #2 — validate + build SQL)

**Responsibility:** Single LLM call เพื่อ:
- Validate extracted params against template schema
- Build final SQL with user_id injected

```
Input:  template_id, params, user_id
Output: sql, validation_errors

Edge: validation_errors == [] → executor
       validation_errors != [] → format_error_response
```

#### 4. `executor` (Deterministic)

**Responsibility:** Execute validated SQL against PostgreSQL

```
Input:  sql, user_id
Output: raw_result, error

Edge: error == None → format_response
       error != None → format_error_response
```

#### 5. `format_response` (Deterministic)

**Responsibility:** Format DB result เป็น natural language response

```
Input:  raw_result, intent_id, template_id
Output: final_response (str)
```

#### 6. `format_error_response` (Deterministic)

**Responsibility:** Format error เป็น user-friendly message

```
Input:  error OR validation_errors
Output: final_response (str)
```

### Edge Conditions

```python
def route_after_router(state: FinanceQueryState) -> str:
    """Route after llm_router based on confidence."""
    if state.get("needs_clarification"):
        return "ask_clarification"
    return "llm_builder"

def route_after_builder(state: FinanceQueryState) -> str:
    """Route after llm_builder based on validation."""
    if state.get("validation_errors"):
        return "format_error_response"
    return "executor"

def route_after_executor(state: FinanceQueryState) -> str:
    """Route after executor based on error status."""
    if state.get("error"):
        return "format_error_response"
    return "format_response"

# Graph construction
builder.add_edge(START, "llm_router")
builder.add_edge("llm_router", "llm_builder", cond=route_after_router)
builder.add_edge("llm_router", "ask_clarification", cond=route_after_router)
builder.add_edge("llm_builder", "executor", cond=route_after_builder)
builder.add_edge("llm_builder", "format_error_response", cond=route_after_builder)
builder.add_edge("executor", "format_response", cond=route_after_executor)
builder.add_edge("executor", "format_error_response", cond=route_after_executor)
builder.add_edge("format_response", END)
builder.add_edge("format_error_response", END)
builder.add_edge("ask_clarification", END)
```

### LLM Call Count Analysis

| Scenario | LLM #1 | LLM #2 | Total |
|----------|--------|--------|-------|
| High confidence (≥0.7) | 1 | 1 | **2** |
| Low confidence (<0.7) | 1 | 0 | **1** (+ clarify) |
| Validation error | 1 | 1 | **2** (+ retry) |

### Optimizations Applied

1. **Combined nodes**: classify + select + extract = 1 LLM call (router)
2. **Combined nodes**: validate + build SQL = 1 LLM call (builder)
3. **Conditional edges**: skip builder/executor if clarification needed
4. **Early exit**: validation errors return immediately without DB call
5. **Confidence threshold**: 0.7 cutoff ลด hallucination risk

### Comparison: Original 6-Node vs Optimized

| Aspect | Original (6 nodes) | Optimized (6 nodes) |
|--------|-------------------|---------------------|
| LLM calls | 3 | **2** |
| Nodes | intent → validate → template → param → builder → executor | router → builder → executor |
| Latency | ~1.5-2s | **~1s** |
| Complexity | High | Medium |

### Project Structure (Updated for LangGraph)

```
agentic/
├── mcp_finance/
│   ├── cmd/
│   │   └── main.go                     # Entry point, server bootstrap
│   ├── internal/
│   │   ├── auth/
│   │   │   └── context.go              # Extract userId from auth token
│   │   ├── config/
│   │   │   └── config.go              # DB config, server config
│   │   ├── domain/
│   │   │   ├── intent.go              # Intent definition struct
│   │   │   ├── result.go              # Query result struct
│   │   │   └── state.go               # FinanceQueryState schema
│   │   ├── graph/
│   │   │   ├── builder.go             # LangGraph StateGraph construction
│   │   │   ├── router_node.go         # LLM #1: classify + select + extract
│   │   │   ├── builder_node.go        # LLM #2: validate + build SQL
│   │   │   ├── executor_node.go       # Deterministic DB executor
│   │   │   └── formatter_node.go      # Response formatter
│   │   ├── intents/
│   │   │   ├── registry.go            # Intent registry + pattern matcher
│   │   │   └── patterns.go           # TH/EN patterns for each intent
│   │   ├── extractor/
│   │   │   └── param_extractor.go    # Extract date/category/amount from text
│   │   ├── templates/
│   │   │   ├── registry.go           # Template registry
│   │   │   └── finance.go           # All SQL templates
│   │   ├── executor/
│   │   │   └── query_executor.go    # Execute SQL, handle errors
│   │   └── validator/
│   │       └── sql_validator.go      # Validate generated SQL
│   └── mcp/
│       ├── handlers.go               # MCP protocol handlers
│       └── tool_handler.go           # Wraps LangGraph subgraph
```

### MCP Integration

```go
// tool_handler.go
type ToolHandler struct {
    graph *langgraph.Graph  // Compiled LangGraph subgraph
}

func (h *ToolHandler) HandleToolCall(ctx context.Context, req ToolRequest) (ToolResponse, error) {
    // 1. Extract userId from auth context (NOT from req)
    userId := auth.UserIDFromContext(ctx)

    // 2. Build initial state
    initialState := map[string]interface{}{
        "question": req.Question,
        "user_id":  userId,
    }

    // 3. Invoke LangGraph subgraph
    result, err := h.graph.Invoke(ctx, initialState)
    if err != nil {
        return ToolResponse{Error: err.Error()}, nil
    }

    // 4. Extract final response
    return ToolResponse{
        Result: result["final_response"].(string),
    }, nil
}
```

### Checkpointer (Optional)

สำหรับ session ที่ต้องการ resume หลัง clarify:

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/checkpoints"
)
graph = builder.compile(checkpointer=checkpointer)

# Resume after clarification:
# result = graph.invoke(Command(resume={"clarification_answer": "..."}), config)
```
```

---

## Security Rules

### 1. Auth Context
- `userId` มาจาก `Authorization: Bearer <token>` เท่านั้น
- ไม่รับ `userId` จาก user input

### 2. SQL Validation
```go
func ValidateSQL(sql string) error {
    // 1. Must be SELECT only
    // 2. No DDL: DROP, TRUNCATE, ALTER, CREATE
    // 3. Tables must be in whitelist
    // 4. No multiple statements (;)
    // 5. No subquery depth > 2
}
```

### 3. Table Whitelist
```go
var AllowedTables = []string{
    "transactions",
    "general_wallets",
    "goal_wallets",
    "categories",
    "budgets",
    "tags",
    "recurring_transactions",
}
```

### 4. Query Timeout
- Max execution: 5 seconds
- Max rows returned: 1000

---

## MVP Scope (Phase 1)

Implement 6 intents ก่อน:

1. `spending_total_period`
2. `spending_by_category`
3. `budget_status`
4. `goal_progress`
5. `recurring_upcoming`
6. `income_total_period`

Phase 2: เพิ่ม 8 intents ถัดไป
Phase 3: ครบทั้ง 25 intents

---

## Dependencies

- Go 1.24+
- `github.com/jmoiron/sqlx` — PostgreSQL
- `github.com/google/uuid` — UUID parsing
- Standard library for MCP protocol
