# Plan-and-Execute Pattern Analysis: Financial-Chat vs Reference Implementation

**Date:** 2026-04-30
**Status:** Draft for Review

---

## Executive Summary

The financial-chat workflow implements a **ReAct-based Plan-Execute pattern** with dedicated Planner (Gemini) and Executor (CodeAct). It works well for single-turn tool use but has architectural gaps compared to the reference notebook's **structured Plan-and-Execute** pattern.

**Key Finding:** Your workflow is **reactive** (adapts step-by-step via ReAct), while the reference is **proactive** (creates complete plan upfront). Both are valid — the question is which fits your use case better.

---

## Current Architecture Analysis

### Current Flow (ReAct)
```
START → reason → [route] → act (CodeAct) → observe → reason (loop)
                         ├→ tool (ToolNode)
                         └→ respond → END
```

### ReAct Loop: Reason → Act → Observe
- **Reason**: คิดว่าจะทำอะไรต่อ
- **Act**: รัน CodeAct หรือ tool
- **Observe**: เห็นผล แล้ววนกลับไป Reason

### Reference Flow
```
START → planner → executor → [More steps?] → YES: executor (loop)
                                        └→ NO: finalizer → END
```

---

## Comparison Matrix

| Aspect | Reference (Notebook) | Financial-Chat (ReAct) | Gap |
|--------|---------------------|------------------------|-----|
| **Planning** | Structured plan with `ExecutionPlan` schema (Pydantic) | Implicit via tool_calls in Reasoner | Medium |
| **Step tracking** | `current_step`, `step_results[]`, `plan` in state | `messages[]` only | High |
| **Plan visibility** | User can see full plan before execution | No visible plan until execution | High |
| **Multi-step planning** | Full plan upfront, then execute | One step at a time only | Medium |
| **Finalizer node** | Dedicated node combines all results | No dedicated finalizer | Medium |
| **Observe node** | Actually processes results | No-op (`return {}`) | High |
| **Step schema** | `Step{step_number, description, tool_needed}` | No explicit step schema | High |
| **Execution routing** | Static: executor loops until done | Dynamic: Reasoner decides each step | Low |

---

## Strengths of Current Implementation

1. **Clean separation** — Reasoner (Gemini) vs Actor (CodeAct) is well-defined
2. **CodeAct subgraph isolation** — Per-step tracing works correctly
3. **Good tool routing** — Distinguishes CodeAct, regular tools, direct response
4. **Emotional support integration** — Friend voice is well-documented in system prompt
5. **Error handling** — Graceful degradation when CodeAct fails

---

## Identified Gaps

### Gap 1: No Structured Execution Plan (HIGH Priority)

**Reference:**
```python
class Step(BaseModel):
    step_number: int
    description: str
    tool_needed: Optional[str]

class ExecutionPlan(BaseModel):
    task: str
    steps: List[Step]
```

**Current:** No equivalent. Reasoner produces `AIMessage` with tool_calls, but no structured `ExecutionPlan` object.

**Impact:**
- Users can't see what the agent plans to do before execution
- Hard to debug "why did it do this?" questions
- No way to validate plan structure before execution

**Recommendation:** Add `ExecutionPlan` schema. Reasoner outputs structured plan instead of (or alongside) tool_calls.

---

### Gap 2: State Lacks Step Tracking (HIGH Priority)

**Reference state:**
```python
class PlanExecuteState(TypedDict):
    task: str
    plan: Optional[ExecutionPlan]
    current_step: int
    step_results: List[str]
    final_answer: str
```

**Current state:**
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: NotRequired[str]
```

**Impact:**
- Can't easily answer "which step is it on?"
- No `step_results[]` to accumulate execution history
- Final answer is implicit in messages, not explicit

**Recommendation:** Extend `AgentState` with plan tracking fields.

---

### Gap 3: Observe Node is a No-Op (HIGH Priority)

**Current implementation (nodes.py):**
```python
async def observe_node(state: AgentState) -> dict:
    return {}  # No-op
```

**Reference intent:** Observe processes results, checks plan completion, prepares context for next step.

**Impact:**
- Loop exists but observe doesn't actually "observe"
- `reason_node` re-evaluates with new data but has no structured plan to check against

**Recommendation:** Implement actual observation logic — check plan completion, aggregate results, prepare summary for final response.

---

### Gap 4: No Dedicated Finalizer Node (MEDIUM Priority)

**Reference:** Finalizer combines all `step_results[]` into `final_answer`.

**Current:** Direct response goes to END, tool execution loops back to plan.

**Impact:**
- Multi-step tasks (e.g., "analyze Q1-Q4 and compare to industry") don't have a dedicated synthesis phase
- Response quality depends on ReAct's last turn rather than structured aggregation

**Recommendation:** Add `finalizer_node` that runs when plan is complete, explicitly synthesizing results.

---

### Gap 5: One Step at a Time Philosophy (MEDIUM Priority)

**Current (nodes.py:131):**
> "🎯 ONE STEP AT A TIME: Plan one tool call or one response. Don't plan everything at once."

**Reference:** Creates full plan upfront, then executes sequentially.

**Trade-off analysis:**

| Approach | Pros | Cons |
|----------|------|------|
| **Current (ReAct)** | Adaptive, handles surprises | Less predictable, harder to trace |
| **Reference (Plan-first)** | Predictable, visible plan, better debugging | Less flexible if task changes mid-way |

**For financial-chat, the ReAct approach may be intentional** — financial queries can be ambiguous, and one-step-at-a-time allows course correction.

**Recommendation:** Consider hybrid — create explicit plan for multi-step queries, use ReAct for exploratory/unclear queries.

---

## Recommendations (Priority Order)

### 1. Add `ExecutionPlan` Schema (High)

```python
class Step(BaseModel):
    step_number: int
    description: str
    tool_needed: Optional[str] = None

class ExecutionPlan(BaseModel):
    task: str
    steps: List[Step]
```

Add to `state.py`. Planner outputs this schema instead of just AIMessage.

### 2. Extend `AgentState` with Plan Tracking (High)

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: NotRequired[str]
    # New fields:
    plan: Optional[ExecutionPlan]
    current_step: int
    step_results: list[str]
    final_answer: str
```

### 3. Implement Real Observe Node (High)

```python
async def observe_node(state: AgentState) -> dict:
    """Check plan completion, aggregate results."""
    plan = state.get("plan")
    current_step = state.get("current_step", 0)

    if not plan or current_step >= len(plan.steps):
        return {"plan_complete": True}

    return {
        "plan_complete": False,
        "step_results": state.get("step_results", []) + [latest_result]
    }
```

### 4. Add Finalizer Node (Medium)

```python
async def finalizer_node(state: AgentState) -> dict:
    """Combine all step results into final answer."""
    # Synthesize from step_results and plan
    return {"final_answer": synthesized_response}
```

### 5. Conditional Multi-Step Planning (Low)

For complex queries (quarterly analysis, comparisons), use upfront planning. For simple queries, continue with one-step ReAct.

---

## Decision Points

1. **Should we make planning explicit?** If yes → implement ExecutionPlan schema and planner restructure.

2. **Should we keep ReAct flexibility?** The current one-step-at-a-time approach may be a deliberate choice for handling ambiguous financial queries. Confirm this is intentional.

3. **What triggers multi-step vs single-step?** Define rules for when to create full plan vs use ReAct.

4. **Finalizer scope:** Should it only run for multi-step plans, or also synthesize single-step results?

---

## Appendix: Key File References

| Component | Path |
|-----------|------|
| Main Graph | `mint_agnetic/financial-chat/src/graph/agent_graph.py` |
| State | `mint_agnetic/financial-chat/src/graph/state.py` |
| Nodes | `mint_agnetic/financial-chat/src/graph/nodes.py` |
| CodeAct Subgraph | `mint_agnetic/financial-chat/src/graph/codeact_subgraph.py` |
| Reference Notebook | Chapter 13: Plan-and-Execute Agent |

---

*Analysis based on code review conducted 2026-04-30*
