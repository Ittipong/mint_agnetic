"""Loop control for the CodeAct subgraph."""

from __future__ import annotations

from src.graph.compute_subgraph.state import ComputeSubState


def codeact_route(state: ComputeSubState) -> str:
    """Branch after each codeact step — done → respond, else → next step."""
    return "respond" if state.get("codeact_done") else "codeact_step"
