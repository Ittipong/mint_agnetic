"""Templates-as-Tools CodeAct branch.

Used when the planner picks `metric=freeform_codeact` — the LLM composes
SQL templates programmatically rather than producing a single QuerySpec.
The sandbox only exposes pre-built templates and Decimal helpers, so the
LLM can never write raw SQL or invent numbers.
"""
