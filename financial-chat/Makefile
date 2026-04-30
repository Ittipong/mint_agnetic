# Mint Agentic — Makefile

PY := python
UV := uv
DB_URL ?= postgresql://mint:mint123@localhost:5433/mint_agentic
BACKEND_DB_URL ?= postgresql://postgres:postgres@localhost:5432/mint_money_dev
STUDIO_PORT ?= 2024

# === Development ===

dev: ## Run the agent CLI in development mode
	$(UV) sync
	$(UV) run $(PY) -m src.main

dev-chat: ## Run the agent in chat mode
	$(UV) run $(PY) -m src.main --mode chat

dev-react: ## Run the agent in ReAct mode
	$(UV) run $(PY) -m src.main --mode react

# === FastAPI Server ===

SERVER_HOST ?= 0.0.0.0
SERVER_PORT ?= 8000

server: ## Start FastAPI chat server (streaming SSE)
	$(UV) sync
	$(UV) run uvicorn src.server:app --host $(SERVER_HOST) --port $(SERVER_PORT) --reload

server-prod: ## Start FastAPI chat server (production, no reload)
	$(UV) run uvicorn src.server:app --host $(SERVER_HOST) --port $(SERVER_PORT) --workers 2

# Test streaming endpoint
test-stream: ## Test SSE stream (usage: USER_ID=<uuid> THREAD_ID=<uuid> MSG="question")
	@curl -sN -X POST http://localhost:$(SERVER_PORT)/chat/stream \
		-H "Content-Type: application/json" \
		-d '{"user_id":"$(USER_ID)","thread_id":"$(THREAD_ID)","message":"$(MSG)"}' \
		| while IFS= read -r line; do echo "$$line"; done

# === LangGraph Studio ===

studio: ## Start LangGraph Studio API server
	-lsof -ti :$(STUDIO_PORT) | xargs kill -9 2>/dev/null; sleep 1
	@echo "Starting LangGraph Studio on http://localhost:$(STUDIO_PORT)"
	@echo "Studio UI: https://smith.langchain.com/studio/?baseUrl=http://localhost:$(STUDIO_PORT)"
	$(UV) run langgraph dev \
		--config langgraph.json \
		--port $(STUDIO_PORT) \
		--no-browser

studio-stop: ## Stop LangGraph dev server
	-lsof -ti :$(STUDIO_PORT) | xargs kill -9 2>/dev/null || true
	-pkill -f "langgraph dev" 2>/dev/null || true

studio-restart: studio-stop studio ## Restart LangGraph Studio

# === Testing via API ===

TEST_USER_ID ?= ba91d8a5-46b2-46f7-aaf4-189a54e17fe9
STUDIO_URL ?= http://localhost:$(STUDIO_PORT)

test-chat: ## Test chat endpoint (usage: make test-chat USER_MSG="your question")
	@curl -s -X POST $(STUDIO_URL)/runs/wait \
		-H "Content-Type: application/json" \
		-d '{"assistant_id": "agent", "graph_name": "agent", "input": {"user_id": "$(TEST_USER_ID)", "messages": [{"role": "user", "content": "$(USER_MSG)"}]}}' \
		| python3 -c "import sys,json; d=json.load(sys.stdin); [print(m.get('content','')[:300]) for m in d.get('messages',[]) if m.get('type')=='ai' and not m.get('tool_calls')]"

# === Database ===

db-up: ## Start PostgreSQL via docker-compose
	docker compose up -d

db-down: ## Stop PostgreSQL container
	docker compose down

db-reset: db-down db-up ## Reset database (destroy + recreate + migrate)
	sleep 2
	$(MAKE) db-migrate

db-migrate: ## Apply schema.sql to database
	psql "$(DB_URL)" -f schema.sql

db-create: ## Create mint_agentic database on shared postgres
	psql postgresql://postgres:postgres@localhost:5432/mint_money_dev \
	  -c "CREATE DATABASE mint_agentic;"

db-backend: ## Show backend database URL
	@echo "$(BACKEND_DB_URL)"

# === Dependencies ===

install: ## Install dependencies via uv sync
	$(UV) sync

update: ## Upgrade all dependencies
	$(UV) sync --upgrade

# === Evaluation ===

eval: ## Run upload + eval (usage: make eval FILE=evaluation/dataset_poc.jsonl [ALL_MODELS=1] [BASELINE_MODEL=x] [RUN_BY=ci])
	$(UV) run python -m tests.data.chat_eval --file $(FILE) \
		$(if $(ALL_MODELS),--all-models,) \
		$(if $(BASELINE_MODEL),--baseline-model $(BASELINE_MODEL),) \
		$(if $(RUN_BY),--run-by $(RUN_BY),)

eval-all-models: eval ## Alias for ALL_MODELS=1
eval-intent: ## Run evaluation for specific intent (usage: make eval-intent INTENT=GOAL_PROGRESS)
	$(UV) run python -m tests.data.chat_eval --intent $(INTENT)

compare: ## Run model comparison (usage: make compare FILE=evaluation/dataset_poc.jsonl [OUTPUT_DIR=./results])
	$(UV) run python evaluation/compare.py --file $(FILE) --output-dir $(if $(OUTPUT_DIR),$(OUTPUT_DIR),./eval_results)

compare-dataset-id: ## Run comparison on existing dataset (usage: make compare-dataset-id DATASET_ID=xxx DATASET_NAME=xxx)
	$(UV) run python evaluation/compare.py --dataset-id $(DATASET_ID) --dataset-name $(DATASET_NAME) --output-dir $(if $(OUTPUT_DIR),$(OUTPUT_DIR),./eval_results)

eval-export: ## Run evaluation and export to JSON (usage: make eval FILE=dataset_poc.jsonl OUTPUT=results.json)
	$(UV) run python -m tests.data.chat_eval --file $(FILE) --export $(OUTPUT)

eval-langsmith: ## Run evaluation and log to LangSmith
	$(UV) run python -m tests.data.chat_eval --file $(FILE)

# === Testing ===

test: ## Run pytest
	$(UV) run pytest

lint: ## Run ruff linter
	$(UV) run ruff check src/

typecheck: ## Run mypy type checker
	$(UV) run mypy src/

# === Docker ===

docker-build: ## Build docker image
	docker compose build

# === Clean ===

clean: ## Remove venv and cache directories
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache src/__pycache__

# === Help ===

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
