# Financial Agent - Makefile
#
# SETUP
#   make install          Install package
#   make install-dev      Install package + dev dependencies
#
# RUN
#   make run                                        Run with default user & task
#   make run TASK="show my wallets"                 Custom task, default user
#   make run USER_ID="<uuid>" TASK="my expenses"    Custom user and task
#   make run-complex TASK="forecast next month"     Extended thinking mode
#
# DEV / DEBUG
#   make dev                                        Run with --dev flag (DEBUG logs + step JSONL to stderr)
#   make dev TASK="my expenses"                     Custom task in dev mode
#   make dev-log                                    Run dev mode, save step JSONL to /tmp/steps.jsonl
#   make dev-log TASK="my expenses"
#
# EXAMPLES
#   make run TASK="What is my current balance?"
#   make run TASK="Show my expenses this month"
#   make run TASK="List all my wallets"
#   make run TASK="How much did I spend on food last week?"
#   make run USER_ID="ba91d8a5-46b2-46f7-aaf4-189a54e17fe9" TASK="Show income vs expenses"
#
# TEST
#   make test             Run unit tests (no DB/API required)
#   make test-e2e         Run end-to-end tests (requires DB + API key)
#   make test-all         Run all tests
#   make bench            Run benchmark
#
# CODE QUALITY
#   make lint             Check code style
#   make format           Auto-format code
#   make clean            Remove cache and build artifacts

.PHONY: install install-dev test test-e2e test-all run run-complex dev dev-log bench lint format clean

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]" || uv pip install -e . && uv pip install pytest pytest-asyncio

test:
	pytest -m "not e2e"

test-e2e:
	pytest -m e2e

test-all:
	pytest

USER_ID ?= ba91d8a5-46b2-46f7-aaf4-189a54e17fe9
TASK ?= "What is my current balance?"
DEV_LOG ?= /tmp/steps.jsonl

run:
	uv run financial-agent --user-id $(USER_ID) $(TASK)

run-complex:
	uv run financial-agent --user-id $(USER_ID) --complex $(TASK)

dev:
	uv run financial-agent --user-id $(USER_ID) --dev $(TASK)

dev-log:
	uv run financial-agent --user-id $(USER_ID) --dev $(TASK) 2>$(DEV_LOG) \
		&& echo "Step trace saved to $(DEV_LOG)" \
		&& cat $(DEV_LOG) | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"

bench:
	python benchmark.py

lint:
	ruff check financial_agent tests || true
	ruff format --check financial_agent tests || true

format:
	ruff format financial_agent tests
	ruff check --fix financial_agent tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist .ruff_cache
