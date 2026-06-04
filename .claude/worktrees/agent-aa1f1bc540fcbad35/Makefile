PY := .venv/bin/python
PIP := .venv/bin/pip

PORT ?= 2026
STUDIO_PORT ?= 2024

.PHONY: install test dev dev-api dev-studio stop compile clean

install:
	$(PIP) install -r requirements.txt

compile:
	$(PY) -m py_compile src/agent/*.py src/agent/tools/*.py src/agent/tools/codeact/*.py src/agent/endpoints/*.py src/agent/streaming/*.py src/agent/validators/*.py src/agent/utils/*.py src/agent/llm/*.py

test:
	$(PY) -m pytest tests/ -v

## dev — run FastAPI (web chat + REST/SSE) and LangGraph Studio together.
## Web chat:  http://localhost:$(PORT)/
## API docs:  http://localhost:$(PORT)/docs
## Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$(STUDIO_PORT)
dev:
	@echo "[dev] starting FastAPI on :$(PORT) and LangGraph Studio on :$(STUDIO_PORT)"
	@echo "[dev]   web chat : http://localhost:$(PORT)/"
	@echo "[dev]   api docs : http://localhost:$(PORT)/docs"
	@echo "[dev]   studio   : https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:$(STUDIO_PORT)"
	@trap 'kill 0' INT TERM EXIT; \
	( $(PY) -m uvicorn agent.server:app --reload --port $(PORT) --app-dir src 2>&1 \
		| sed -u 's/^/[api]    /' ) & \
	( .venv/bin/langgraph dev --port $(STUDIO_PORT) --no-browser 2>&1 \
		| sed -u 's/^/[studio] /' ) & \
	wait

dev-api:
	$(PY) -m uvicorn agent.server:app --reload --port $(PORT) --app-dir src

dev-studio:
	.venv/bin/langgraph dev --port $(STUDIO_PORT) --no-browser

## stop — kill processes on both ports.
stop:
	@for port in $(PORT) $(STUDIO_PORT); do \
		pids=$$(lsof -ti tcp:$$port 2>/dev/null); \
		if [ -n "$$pids" ]; then \
			echo "Stopping port $$port (pid: $$pids)"; \
			kill $$pids 2>/dev/null; sleep 1; \
			still=$$(lsof -ti tcp:$$port 2>/dev/null); \
			if [ -n "$$still" ]; then kill -9 $$still 2>/dev/null; fi; \
		else \
			echo "Nothing listening on port $$port"; \
		fi; \
	done

clean:
	find src tests -name __pycache__ -type d -exec rm -rf {} +
