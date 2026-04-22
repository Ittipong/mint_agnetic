# Mint Agentic AI

LangGraph ReAct Agent with template-based tools for Thai financial Q&A.

## Architecture

```
User Input (Thai)
    │
    ▼
ReAct Agent (Gemma via OpenRouter)
    │
    ├─── Tool: get_spent_budget(user_id, month_offset?)
    ├─── Tool: get_weekly_summary(user_id, week_offset?)
    ├─── Tool: get_debt_balance(user_id)
    ├─── Tool: get_goal_progress(user_id, goal_id?)
    ├─── Tool: get_payday(user_id)
    └─── Tool: get_recent_transactions, get_budget_status, get_savings_goals
              │
              ▼
         Raw JSON numbers
              │
              ▼
         LLM responds in friendly Thai voice
```

## Quick Start

```bash
# 1. Start LangGraph Studio
make studio

# 2. Test chat (separate terminal)
make test-chat USER_MSG="goal ผมเป็นไงบ้าง"

# 3. Run evaluation
python -m tests.data.chat_eval
```

## Project Structure

```
src/
├── graph/
│   ├── agent_graph.py    # ReAct agent graph
│   ├── nodes.py          # Node functions + tools
│   └── state.py          # AgentState schema
├── tools/
│   └── db_tools.py      # Template tools (5 intents) + ReAct tools
├── llm.py               # OpenRouter LLM setup
├── config.py            # Environment config
└── main.py              # CLI entry point

tests/
└── data/
    └── chat_eval.py      # Evaluation dataset + LangSmith integration
```

## 5 Supported Intents (Template Tools)

| Tool | Questions |
|------|-----------|
| `get_spent_budget` | "เดือนนี้ใช้ไปเท่าไร", "งบเหลือเท่าไร" |
| `get_weekly_summary` | "สัปดาห์นี้ใช้ไปเท่าไร" |
| `get_debt_balance` | "หนี้เหลือเท่าไร", "ค่างวดเท่าไร" |
| `get_goal_progress` | "goal ผมเป็นไงบ้าง", "ออมไปเท่าไรแล้ว" |
| `get_payday` | "เงินเดือนวันที่เท่าไร" |

## Database Setup

```bash
# Create the mint_agentic database on the shared postgres
psql postgresql://postgres:postgres@localhost:5432/mint_money_dev \
  -c "CREATE DATABASE mint_agentic;"

# Initialize the schema
psql postgresql://postgres:postgres@localhost:5432/mint_agentic \
  -f schema.sql
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `DATABASE_URL` | Agentic DB (mint_agentic) |
| `BACKEND_DATABASE_URL` | Backend DB (mint_money_dev) for queries |
| `MODEL` | Gemma for responses |

## Makefile Commands

```bash
make studio              # Start LangGraph Studio (port 8080)
make studio-stop        # Stop LangGraph server
make test-chat          # Test chat endpoint (need USER_MSG=)
make dev                # Run agent CLI
make dev-chat           # Run CLI in chat mode
make dev-react          # Run CLI in ReAct mode
make help               # Show all commands
```

## Evaluation

```bash
# Run all tests
python -m tests.data.chat_eval

# Run specific intent
python -m tests.data.chat_eval --intent GOAL_PROGRESS

# Export results
python -m tests.data.chat_eval --export results.json

# Log to LangSmith
python -m tests.data.chat_eval --langsmith
```

## Design Principles

### Hybrid: ReAct + Templates
- LLM decides **which tool** to call (accurate intent understanding)
- Tools use **fixed SQL templates** (no hallucination risk)
- LLM generates **friendly Thai response** from raw numbers

### Safety
- LLM **never generates SQL** — hallucination risk eliminated
- All numbers come **verbatim from database**
- Friend tone: supportive, never judgmental
