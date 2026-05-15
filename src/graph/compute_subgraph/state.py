"""State for the analyze subgraph (Smart CodeAct).

Internal state is isolated from parent AgentState — only the bridge
(`act_node`) reads/writes the parent's `messages`.
"""

from __future__ import annotations

from typing import NotRequired
from typing_extensions import TypedDict


class ComputeSubState(TypedDict):
    # Input — set by act_node before invoking the subgraph
    task: str
    user_id: str
    today: str  # ISO date
    # Entity catalog passed in from the parent — every codeact iteration
    # reads it via the resolve_*() helpers, but no node mutates it.
    catalog: NotRequired[object]  # EntityCatalog (kept as `object` to avoid circular import)

    # CodeAct loop bookkeeping — populated by codeact_step_node
    codeact_history: NotRequired[list]   # list[{step, code, stdout, error, result}]
    codeact_done: NotRequired[bool]
    codeact_final: NotRequired[object]   # last non-null `result` from the loop

    # Clarification — raised inside the sandbox by `clarify(question, options)`
    needs_clarification: NotRequired[bool]
    clarification_question: NotRequired[str]
    clarification_options: NotRequired[list]

    # Output — set by respond_node and read by act_node
    answer: NotRequired[str]
