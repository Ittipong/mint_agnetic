"""Graph nodes that are NOT the prebuilt ReAct agent.

The classify-router shortcut. Holds the `classify_intent` node (in-graph
stateful intent classifier) and the `direct_propose` node (a ReAct-free ADD path
that reuses `propose_transaction._propose_core`).

The classifier is STATEFUL — it sees windowed conversation history, which is
what lets it tell "รถ 2 ล้าน" (a fresh ADD) apart from "รถ 2 ล้าน" answering a
prior advisor question (the E9 case). It is the single intent classifier in
v3; gate it via CLASSIFY_ROUTER_ENABLED (default OFF → every turn → react).
"""

from .classify_intent import (
    classify_intent_node,
    direct_propose_node,
    is_classify_router_enabled,
    route_after_classify,
)

__all__ = [
    "classify_intent_node",
    "direct_propose_node",
    "is_classify_router_enabled",
    "route_after_classify",
]
