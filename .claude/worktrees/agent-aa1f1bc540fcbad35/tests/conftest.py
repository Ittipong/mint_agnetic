"""Minimal pytest fixtures + path setup for v3 Wave 1.

Heavier fixtures (mock LLM, mock proposal repo, mock tools) will be added
in later waves as more components come online. Wave 1 only exercises
sandbox-local utilities (safe_json, exceptions, schemas, state, validator)
and a single env-var contract for the DB pool — no shared mocks needed yet.

The `pythonpath = ["src", "tests_integration"]` block in pyproject.toml
already makes `src.agent…` importable; this file only adds defensive
fallback for direct invocations.
"""

from __future__ import annotations

import os
import sys

# Make `src` importable when pytest is run from the v3 root directly
# (matches the v2 conftest pattern). Also add the project root so
# `from src.agent...` imports resolve — `pythonpath` in pyproject only
# adds `src/`, which gives `agent.X`; tests in this tree use the
# explicit `src.agent.X` style to match production source imports.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Disable streaming preamble in tests — production UX feature that hits
# OpenRouter live; unit tests assert exact answer tokens and shouldn't
# depend on a real LLM round-trip. Real preamble behavior is covered by
# the live curl smoke test, not unit tests.
os.environ.setdefault("PREAMBLE_ENABLED", "0")
