# Agentic Orchestration Workflow — Mint Money AI Financial Friend

เอกสารสรุปภาพรวม workflow ของ agentic orchestrator (LangGraph) ที่ใช้ขับเคลื่อนฟีเจอร์ AI Financial Friend ใน `mint_agnetic/financial-chat/`

> Source code อ้างอิงหลัก: `src/graph/agent_graph.py`, `src/graph/nodes.py`, `src/graph/slip_node.py`, `src/graph/compute_subgraph/graph.py`

---

## 1. ภาพรวมสถาปัตยกรรม

ระบบใช้ **LangGraph StateGraph** จัดการ flow แบบ stateful และมี 2 เลน (lanes) หลักที่แยกกันชัดเจน:

```
                      ┌──────────────────────────────────┐
                      │              START               │
                      └────────────────┬─────────────────┘
                                       │
                          _route_by_input()
                       ┌───────────────┴───────────────┐
                       │                               │
                  มีรูปแนบมา?                     ไม่มีรูป
                       │                               │
                       ▼                               ▼
              ┌─────────────────┐            ┌──────────────────┐
              │   SLIP LANE     │            │   ReAct LANE     │
              │ (Vision flow)   │            │ (Text reasoning) │
              └────────┬────────┘            └────────┬─────────┘
                       │                              │
                       ▼                              ▼
                     END                            END
```

- **Slip lane** = straight-line, ไม่ loop — ใช้ vision LLM ครั้งเดียว
- **ReAct lane** = loop จนกว่า LLM จะตอบโดยไม่เรียก tool

---

## 2. State Schema

`AgentState` ใน `src/graph/state.py` เก็บข้อมูลที่ทุก node อ่าน/เขียนผ่านได้:

| Field | Type | บทบาท |
|---|---|---|
| `messages` | `list[AnyMessage]` | ประวัติแชท (HumanMessage / AIMessage / ToolMessage) |
| `user_id` | `str` | ใช้ดึง entity catalog + scope ข้อมูล |
| `current_date` | `str` (ISO) | วันที่ปัจจุบัน — ส่งให้ system prompt และ CodeAct |
| `images` | `list[str]` | data URLs — มีค่า ⇒ route ไป slip lane |

State persistence ใช้ **AsyncPostgresSaver checkpointer** เพื่อเก็บประวัติแบบ thread-based

---

## 3. Workflow Step-by-Step

### Step 0 — Entry Routing (`_route_by_input`)

ตรวจสอบ `state["images"]`:
- ถ้ามี → ไป `slip` node
- ถ้าไม่มี → ไป `reason` node

---

### Slip Lane (Vision Flow)

#### Step S1 — `slip_node` (Vision LLM)
1. โหลด user's wallet + category catalog จาก database
2. สร้าง system prompt ฝัง slip context (wallets + categories พร้อม `sync_id`)
3. เรียก **vision LLM** ด้วย image data URL
4. LLM extract: amount, date, merchant, line items → match wallet + category → เรียก tool `propose_transaction`

#### Step S2 — `propose_validation_node` (Tool execution)
- Execute `propose_transaction` → validate schema → dispatch SSE custom event ไปที่ mobile client (เพื่อ render proposal card)

#### Step S3 — `slip_cleanup_node`
- ใช้ `RemoveMessage` ตัด slip turn ออกจาก checkpoint
- เพื่อไม่ให้ vision context กิน token ใน ReAct turn ถัดไป

จบที่ **END** — ไม่ loop กลับ

---

### ReAct Lane (Text Reasoning Loop)

แกนหลักของ workflow — **Reason → Act → Reason** จนกว่าจะตอบ

#### Step R1 — `reason_node`
1. **Compress history**: ตัด `AIMessage(tool_calls)` + `ToolMessage` ของ turn ก่อนๆ ทิ้ง เหลือแค่ HumanMessage + final AIMessage text
2. **Trim**: cap ที่ ~6,000 tokens (strategy=`last`) เพื่อ control cost
3. **Fetch entity catalog** ใหม่ทุก turn (wallets / categories / tags) — รองรับการ add/rename/delete ทันที
4. **Build system prompt** ฝัง:
   - Today's date
   - Slip intent rules
   - Time disambiguation rules (WINDOW vs SINGLE-POINT)
   - Tool routing hints (wallet_list, top_transactions, spending_trend, ...)
   - Entity catalog (verbatim names)
5. เรียก LLM `llm.bind_tools(ALL_TOOLS)` แล้วได้ AIMessage กลับ

#### Step R2 — `_should_route` (Conditional edge)
ดู `tool_calls` ใน AIMessage ล่าสุด:

| เงื่อนไข | ไปที่ |
|---|---|
| ไม่มี tool_calls (LLM ตอบเลย) | `respond` → END |
| มี `analyze_user_finances` | `act` (CodeAct subgraph) |
| มี tool อื่น (เช่น `get_financial_advice`) | `tool` (ToolNode มาตรฐาน) |

#### Step R3a — `act_node` → Compute Subgraph (CodeAct)

นี่คือ **subgraph** ที่ embed อยู่ใน parent graph (`src/graph/compute_subgraph/graph.py`)

```
   START → codeact_step ⟲ (loop จนกว่า codeact_done=True) → respond → END
```

- **`codeact_step_node`**: LLM เขียน Python code เรียก wrapper functions ใน sandboxed namespace (`balance()`, `sum_expense()`, `top_transactions()`, `spending_trend()`, `compare_periods()`, `note_query`, …) ที่แปลงเป็น SQL ต่อ PostgreSQL
- **`respond_node`**: สรุปผลเป็นภาษาธรรมชาติ ใส่ใน `answer`
- **Bridge back**: `act_node` ห่อ `answer` เป็น `ToolMessage` ส่งกลับ parent state (พร้อมแนบ `entity_catalog` ใน `additional_kwargs` เพื่อกัน LLM paraphrase ชื่อ)

#### Step R3b — `tool` (Regular ToolNode)
Execute tool ทั่วไป (เช่น `get_financial_advice`) → return ToolMessage

#### Step R4 — Loop กลับ `reason`
- จาก `act` หรือ `tool` ⇒ edge ตรงกลับ `reason`
- LLM อ่าน ToolMessage ที่เพิ่งเพิ่มเข้า state → ตัดสินใจรอบใหม่
- วน until LLM ตอบโดยไม่เรียก tool → `respond` → END

---

## 4. Tools Available

| Tool | ใช้ตอนไหน | Executor |
|---|---|---|
| `analyze_user_finances(task)` | คำถามที่ต้องดึงข้อมูลจริง (transactions, balance, budget, ฯลฯ) | CodeAct subgraph |
| `get_financial_advice` | คำแนะนำการเงินทั่วไป ไม่ต้องดึงข้อมูล | Regular ToolNode |
| `propose_transaction` | Slip lane เท่านั้น — สร้าง draft transaction | `propose_validation_node` |

> ⚠️ `propose_transaction` ตั้งใจ **ไม่** expose ให้ ReAct lane — ป้องกัน LLM hallucinate card จาก text-only context

---

## 5. Cross-cutting Concerns

### History Compression (กัน token explosion)
- เฉพาะ past turns: drop tool-call/tool-message pairs, เก็บแค่ user prompt + final AI reply
- Current turn: เก็บครบทั้ง chain (กัน tool_call_id orphan)
- Trim ที่ ~6K tokens ⇒ cost ต่อ turn ≈ $0.018 input

### Entity Name Lock
- ทุก node ที่เรียก LLM (`reason_node`, `slip_node`, CodeAct `respond_node`) ฉีด entity catalog เข้าไป
- บังคับให้ LLM ใช้ชื่อตัวอักษรตรง catalog (เช่น `TrueMonney` ห้ามเป็น `TrueMoney`)

### Streaming (SSE)
- FastAPI server (`src/server.py`) stream messages + custom events
- `propose_transaction` ใช้ `adispatch_custom_event` ส่ง proposal card payload แยกจาก text stream

### Debug Logging
- File-backed log: `logs/reason_debug_YYYY-MM-DD.log` + `logs/slip_debug_YYYY-MM-DD.log`
- ทุก node เขียน structured log พร้อม `thread_id` + `run_id` เพื่อ correlate

---

## 6. Lifecycle ของ 1 Turn (สรุป Sequence)

```
User message → FastAPI /chat (SSE)
    ↓
server.py ainvoke(graph, input, config={thread_id, user_id})
    ↓
checkpointer load state → enter graph at START
    ↓
_route_by_input → reason (text) หรือ slip (image)
    ↓
[ReAct loop หรือ slip linear]
    ↓
END → checkpointer save state
    ↓
SSE stream ปิด → mobile render response + suggestions chips
```

---

## 7. ทำไมออกแบบแบบนี้

| ปัญหา | วิธีแก้ใน workflow |
|---|---|
| LLM hallucinate ตัวเลข | บังคับเรียก tool ก่อนตอบเรื่องเงิน + CodeAct รัน SQL จริง |
| Token cost บานปลายเมื่อแชทยาว | History compression + trim 6K |
| LLM paraphrase ชื่อ wallet/tag | ฉีด entity catalog ทุก turn + ระบุ "verbatim" ใน prompt |
| Vision cost สูง ถ้าใช้ทุก turn | แยก slip lane ใช้ vision เฉพาะ turn ที่มีรูป |
| Slip context รบกวน chat ถัดไป | `slip_cleanup_node` ลบ slip turn ออกจาก checkpoint |
| LLM คำนวณเลขผิด | CodeAct ใช้ Python + SQL ทำเลข ไม่ปล่อยให้ LLM บวกเอง |
| Tool ถูกเรียกผิดประเภท | Routing hints ใน system prompt + `_should_route` แยก CodeAct vs ToolNode |

---

## 8. ไฟล์อ้างอิง

- `src/graph/agent_graph.py` — graph topology + routing
- `src/graph/state.py` — AgentState schema
- `src/graph/nodes.py` — `reason_node`, tool declarations, system prompt
- `src/graph/slip_node.py` — vision flow nodes
- `src/graph/compute_subgraph/graph.py` — CodeAct subgraph + `act_node` bridge
- `src/graph/compute_subgraph/codeact/` — sandboxed Python execution + DB wrappers
- `src/entity_catalog.py` — wallet/category/tag catalog fetcher
- `src/server.py` — FastAPI + SSE layer
