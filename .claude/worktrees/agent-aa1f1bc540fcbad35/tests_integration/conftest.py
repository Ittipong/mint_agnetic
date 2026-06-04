"""Path + env setup for v3 integration tests (I1001..I1007).

Mirrors `tests/conftest.py` so direct invocations from the v3 root resolve
`from src.agent...` imports. Each I-test then imports `agent.server` (which
calls `load_dotenv()` at module import) BEFORE the module-level `skipif`
check — order matters per memory `project_integration_test_pattern`.

We do NOT call `load_dotenv()` here because each test module owns its env
contract: integration env vars are gated by per-module `pytestmark` skips,
so accidentally pre-populating env from a stale `.env` would hide a
"missing env" skip.
"""

from __future__ import annotations

import os
import sys

# Make `src` and the v3 root importable when pytest is run from anywhere
# (CI cwd vs local cwd vs uvicorn cwd). Matches v2's tests_integration setup.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
