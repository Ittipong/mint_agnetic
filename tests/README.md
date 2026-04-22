# AI Friend Evaluation

## ภาพรวม

Testing framework สำหรับ AI Friend Phase 1 — ครอบคลุม 5 intents พร้อม edge cases

## โครงสร้าง

```
tests/
├── unit/                      # Unit tests (ไม่ต้องเชื่อมต่อ external)
│   ├── test_intent_router.py  # ทดสอบ intent classification
│   └── test_response_gen.py   # ทดสอบ response quality (shame detection)
├── integration/               # Integration tests (ต้องมี Studio รันอยู่)
│   └── test_agent_flow.py    # ทดสอบ full agent flow
└── evaluation/                # Evaluation & datasets
    ├── evaluators.py         # Metrics และ evaluation logic
    ├── run_eval.py          # CLI สำหรับ run evaluation
    ├── langsmith_sync.py    # Sync CSV ↔ LangSmith
    └── datasets/
        └── phase1_dataset.csv  # 52 test cases
```

## เริ่มต้นใช้งาน

### 1. เริ่ม LangGraph Studio

```bash
cd agentic
make studio
```

### 2. Run Evaluation

```bash
# Dry run — ดูว่าจะ run อะไรบ้าง
python -m tests.evaluation.run_eval --dry-run

# Run ทั้ง 52 test cases
python -m tests.evaluation.run_eval

# Run เฉพาะ intent
python -m tests.evaluation.run_eval --intent SPENT_BUDGET

# Run พร้อม log ไป LangSmith
python -m tests.evaluation.run_eval --langsmith

# Export ผลลัพธ์เป็น JSON
python -m tests.evaluation.run_eval --export results.json
```

### 3. Run Unit Tests

```bash
# Unit tests ทั้งหมด
python -m pytest tests/unit/ -v

# Specific test file
python -m pytest tests/unit/test_intent_router.py -v

# พร้อม coverage
python -m pytest tests/unit/ -v --cov
```

### 4. Run Integration Tests

```bash
# ต้องมี Studio รันอยู่ก่อน
python -m pytest tests/integration/ -v
```

## CSV Dataset Format

```csv
question,intent,edge_case,description
เดือนนี้ใช้ไปเท่าไร,SPENT_BUDGET,NORMAL,Current month spending
```

### Fields

| Field | Description |
|-------|-------------|
| `question` | คำถามภาษาไทย |
| `intent` | Expected intent (SPENT_BUDGET, WEEKLY_SUMMARY, DEBT_BALANCE, GOAL_PROGRESS, PAYDAY, FALLBACK) |
| `edge_case` | Edge case type (NORMAL, NO_DATA, NO_BUDGET, etc.) |
| `description` | คำอธิบาย |

## Intent Coverage

| Intent | จำนวน Test Cases | Edge Cases |
|--------|------------------|------------|
| SPENT_BUDGET | 10 | NORMAL, NO_DATA, NO_BUDGET, OVER_BUDGET, FUTURE_MONTH |
| WEEKLY_SUMMARY | 9 | NORMAL, NO_DATA_CURRENT, NO_DATA_BOTH, WEEK_OVER_WEEK_UP_50 |
| DEBT_BALANCE | 10 | NORMAL, NO_DEBT, CC_ONLY, DEBT_PAID_OFF, PAST_DUE |
| GOAL_PROGRESS | 10 | NORMAL, NO_GOALS, GOAL_ACHIEVED, NO_TARGET_DATE, GOAL_OVERDUE |
| PAYDAY | 9 | NORMAL, NO_INCOME_DATA, PAYDAY_TOMORROW, PAYDAY_TODAY, MULTIPLE_INCOME |
| FALLBACK | 4 | GENERAL, OUT_OF_SCOPE, OFF_TOPIC |
| **รวม** | **52** | |

## LangSmith Sync

### Upload CSV ไป LangSmith

```bash
python -m tests.evaluation.langsmith_sync --upload

# Dry run ก่อน
python -m tests.evaluation.langsmith_sync --upload --dry-run
```

### Download จาก LangSmith มาเป็น CSV

```bash
python -m tests.evaluation.langsmith_sync --download
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STUDIO_URL` | LangGraph Studio URL | http://localhost:8080 |
| `TEST_USER_ID` | User ID สำหรับ testing | ba91d8a5-... |
| `LANGSMITH_API_KEY` | LangSmith API key | (from .env) |

## Evaluation Metrics

### Intent Accuracy

- Pass: predicted intent ตรงกับ expected intent
- Fallback: ตอบ FALLBACK ถูกต้องสำหรับ out-of-scope questions

### Latency

| Metric | เป้าหมาย | Phase 1 |
|--------|----------|---------|
| p50 | < 2s | - |
| p95 | < 5s | - |

### Quality Checks

- **Shame detection**: ไม่มี judgmental language (เช่น "ใช้เยอะเกินไป")
- **Number presence**: Response มีตัวเลขเมื่อควรมี

## Makefile Commands

```bash
make studio        # เริ่ม LangGraph Studio
make studio-stop  # หยุด LangGraph Studio
make test-chat    # Test chat endpoint
make dev          # Run agent CLI
```
