# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Makefile — uses .env.test
#
# Workflow:
#
#   # 1. Create JSONL file (each example must have usecase_id)
#   cat > my_dataset.jsonl << 'EOF'
#   {"inputs":{"user_id":"u1","usecase_id":"wallet_balance","messages":[{"role":"user","content":"ตอนนี้ฉันมีเงินในบัญชีไหนบ้าง แต่ละบัญชีเหลืออยู่เท่าไร"}]}}
#   {"inputs":{"user_id":"u1","usecase_id":"wallet_list","messages":[{"role":"user","content":"ข้อมูลของแต่ละ wallet"}]}}
#   EOF
#
#   # 2. Validate & Run evaluation
#   make validate FILE=my_dataset.jsonl
#   make evaluate FILE=my_dataset.jsonl [MODEL=google/gemini-2.5-flash-lite]
#
# Config (via .env.test):
#   EVAL_DATASET_NAME=compare-llm  # prefix for dataset names in LangSmith
#
# ══════════════════════════════════════════════════════════════════════════════

.PHONY: help validate evaluate fetch-traces

PYTHON = ../.venv/bin/python3
ENV_FILE = ../.env.test

help:
	@echo "Evaluation targets (uses .env.test):"
	@echo ""
	@echo "  make validate FILE=my_dataset.jsonl"
	@echo "      Validate a JSONL file (each example needs usecase_id)"
	@echo ""
	@echo "  make evaluate FILE=my_dataset.jsonl [MODEL=<model>]"
	@echo "      Run evaluation from local JSONL file"

# ── Validation ────────────────────────────────────────────────────────────────

validate:
	@if [ -z "$(FILE)" ]; then echo "Usage: make validate FILE=my_dataset.jsonl"; exit 1; fi
	. $(ENV_FILE) && $(PYTHON) dataset.py validate $(FILE)

# ── Evaluation ────────────────────────────────────────────────────────────────

evaluate:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make evaluate FILE=my_dataset.jsonl [MODEL=<model>]"; \
		exit 1; \
	fi
	. $(ENV_FILE) && $(PYTHON) evaluate.py --file $(FILE) \
		$(if $(MODEL),--model "$(MODEL)",)

# ── Fetch traces ────────────────────────────────────────────────────────────────

fetch-traces:
	@if [ -z "$(DATASET_NAME)" ] && [ -z "$(DATASET_ID)" ]; then \
		echo "Usage: make fetch-traces DATASET_NAME=compare-llm [OUTPUT_DIR=/tmp/traces]"; \
		exit 1; \
	fi
	. $(ENV_FILE) && $(PYTHON) fetch_traces.py \
		$(if $(DATASET_NAME),--dataset-name "$(DATASET_NAME)",) \
		$(if $(DATASET_ID),--dataset-id "$(DATASET_ID)",) \
		$(if $(OUTPUT_DIR),--output-dir "$(OUTPUT_DIR)",)
