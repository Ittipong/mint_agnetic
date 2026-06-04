"""Version routing for the chat-agent cutover (v2 -> v3).

NEW in Wave 6. Spec: `docs/v3/phase3_implementation_plan.md` §2 (Wave 6) +
`docs/v3/phase3_decisions.md` (Q1, Q4).

PURPOSE
-------
The Mint Money mobile + Cloudflare-tunnel front-end speaks a fixed wire
contract (`/chat/stream`, `/transactions/confirm`, ...). Two implementations
of that contract exist:

  - **v2** (`mint_agentic_v2/`): legacy multi-node graph (Understand →
    Dispatcher → CodeAct / Add / Rule → Finalize).
  - **v3** (this directory): pure ReAct + CodeAct (single agent, one
    SystemMessage, tool-call loop).

The cutover happens at the **deployment level**, not inside one Python
process:

  - `CHAT_AGENT_VERSION=v3 uvicorn agent.server:app` → starts the v3 server
    (this codebase). The lifespan checks `is_v3_active()` and proceeds.
  - `CHAT_AGENT_VERSION=v2` (or unset) → start the v2 process instead
    (`mint_agentic_v2/src/agent/server.py`). The v3 process refuses to
    start with a clear error: v3 cannot impersonate v2 because v2 is a
    sibling directory, not an importable package from here.

This separation is deliberate:
  - v2 and v3 have independent venvs, model configs, prompts, and DB
    schemas (v2 = public; v3 = `?options=-c%20search_path%3Dv3` per Q1).
    Loading both into one process would risk env-var collisions and
    double-DDL on the same checkpoint tables.
  - Roll-forward / roll-back is a single env-var flip on the deployment
    target (`launchctl setenv CHAT_AGENT_VERSION v3` + restart) — no code
    change, no rebuild, no migration.

PUBLIC API
----------
  get_chat_version() -> Literal["v2", "v3"]
      Read `CHAT_AGENT_VERSION` env var. Default `"v2"` so an unset env in
      legacy deployments keeps routing to v2. Unknown values fall back to
      v2 with a logged warning (typo guard).

  is_v3_active() -> bool
      Convenience predicate (== "v3"). Used by server.py lifespan to
      decide whether to build the v3 graph or abort with a "wrong server"
      error.

  REFUSE_REASON
      One-line Thai/English error message the lifespan raises when v3 is
      asked to host the v2 contract. Centralized so the message stays
      consistent across all rejection paths (lifespan, healthz hint, ...).

NON-GOALS
---------
- This module does NOT import or instantiate either graph. It only
  reports which version SHOULD be active. The actual `build_graph()` call
  lives in `server.py`'s lifespan.
- This module does NOT manage env loading. `server.py` calls
  `load_dotenv()` BEFORE any module-level `os.getenv` (mirroring v2's
  pattern), so by the time this code runs the env is populated.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowed values for CHAT_AGENT_VERSION. Anything else falls back to the
# default and logs once for traceability.
_ALLOWED: tuple[str, ...] = ("v2", "v3")

# Default when env unset OR malformed. v2 stays the safe default during the
# rollout window — an unconfigured deployment must NOT silently flip to v3
# (the v3 server would then refuse to boot, which is the explicit guarantee).
_DEFAULT: Literal["v2", "v3"] = "v2"

# One-shot warning guard so a malformed env var doesn't spam logs every call.
_WARNED: dict[str, bool] = {"emitted": False}

# Reason returned when the v3 server is started against CHAT_AGENT_VERSION!=v3.
# Surfaced through the lifespan error AND any /healthz diagnostics so an
# operator immediately knows which knob to flip.
REFUSE_REASON: str = (
    "v3 chat-agent server refusing to start: CHAT_AGENT_VERSION!=v3. "
    "v3 is a parallel-directory implementation — it cannot host the v2 "
    "graph. To run v2, start the mint_agentic_v2 process instead. "
    "To run v3, set CHAT_AGENT_VERSION=v3 before launching uvicorn."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_chat_version() -> Literal["v2", "v3"]:
    """Return the active chat-agent version per `CHAT_AGENT_VERSION` env.

    Defaults to `"v2"` so an unset env keeps the legacy deployment path.
    Unknown values (typos like `v4`, `V3`, ` v3 `) fall back to the default
    + log a one-shot warning so the operator can fix the misconfiguration
    without log spam.

    The string `"v3"` (lower-case, no padding) is the ONLY value that
    activates this codebase — match the v2 deployment convention exactly.
    """
    raw = os.getenv("CHAT_AGENT_VERSION", "").strip()
    if raw in _ALLOWED:
        return raw  # type: ignore[return-value]
    if raw and not _WARNED["emitted"]:
        # Only warn once per process — repeated calls per request would spam.
        _WARNED["emitted"] = True
        _LOG.warning(
            "CHAT_AGENT_VERSION=%r is not one of %s; falling back to %r",
            raw, _ALLOWED, _DEFAULT,
        )
    return _DEFAULT


def is_v3_active() -> bool:
    """True iff `CHAT_AGENT_VERSION=v3`. Used by the v3 lifespan."""
    return get_chat_version() == "v3"


__all__ = [
    "get_chat_version",
    "is_v3_active",
    "REFUSE_REASON",
]
