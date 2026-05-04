# Redesign Research — NL→Financial-Data Agents on GitHub

> งานวิจัยพื้นหลังสำหรับ redesign `mint_agnetic/financial-chat`
> เป้าหมาย: ตอบคำถามภาษาไทยเกี่ยวกับข้อมูลการเงิน เช่น "ดูรายได้ 5 เดือนของ wallet เกี่ยวกับสัตว์เลี้ยง"
> วันที่: 2026-05-03 · Stack ปลายทาง: Python + LangGraph + Postgres

---

## TL;DR

- **Schema linking ไม่ใช่ string match** — repos ระดับ SoTA (CHESS, Wren AI, Vanna) ทำ schema/value retrieval ด้วย hybrid: embedding (semantic) + BM25/LSH/trigram (lexical) แล้วค่อย rerank ผ่าน LLM. การจับ "wallet เกี่ยวกับสัตว์เลี้ยง" → `pat show` ต้องใช้ embedding บนทั้ง name + description ไม่ใช่ fuzzy string เพียงอย่างเดียว
- **Semantic Layer คือสิ่งที่ Wren AI ใช้ป้องกัน hallucination ของตัวเลข** — แทนที่จะให้ LLM เขียน SQL เอง ให้ encode business logic (วิธีคำนวณ income, balance, currency separation) ลงใน MDL/view ก่อน. LLM แค่เลือก metric + filter เท่านั้น เป็น pattern ที่เหมาะกับ financial domain ที่สุด
- **อย่าให้ LLM ทำเลข** — ทุก aggregation (`SUM`, `AVG`, currency conversion) ต้องเกิดใน Postgres ด้วย `numeric` (Decimal). LLM แค่ produce structured query plan แล้ว Python convert → SQL → execute
- **Plan→Resolve→Execute→Verify→Respond** เป็น pattern ที่ MAC-SQL, CHESS, LangChain SQL Agent ใช้ตรงกัน — แทน ReAct loop. เพิ่ม Verifier node (LLM-as-judge หรือ NL-back-translation) เพื่อจับ silent failures
- **Clarification ใช้ `interrupt()` ของ LangGraph** — เมื่อ entity resolver คืน confidence ต่ำ หรือมี match หลายตัวแบบใกล้เคียงกัน → pause graph, ให้ user เลือก, ต่อด้วย `Command(resume=...)`. AmbiSQL และ Odin ทำ pattern นี้พร้อมกับ taxonomy ของ ambiguity
- **Time parsing ใช้ library + LLM fallback** — `dateparser` รองรับไทยอยู่แล้ว ("1 เดือนตุลาคม 2005"). expressions ที่ซับซ้อน เช่น "ไตรมาส 2 ปีก่อน" ให้ LLM แปลงเป็น Pydantic struct (`{quarter:2, year_offset:-1}`) แล้วโค้ด Python คำนวณ `(start, end)` แบบ deterministic
- **Conversation memory ใช้ 2 ชั้น** — `checkpointer` (thread state, ระยะสั้น, มี history) + `BaseStore` พร้อม semantic search (ระยะยาว, scoped per user). Entity ที่ resolve ไว้ใน turn ก่อน ให้ใส่ใน state และ reuse ใน turn ถัดไปอัตโนมัติ
- **Three-tier confidence model** — แทนที่จะตอบทุกครั้ง: (1) confidence สูง → ตอบเลย, (2) ปานกลาง → ตอบพร้อมแสดงสมมติฐาน ("ใช้ wallet 'pat show' จาก 3 wallet ที่ใกล้เคียง"), (3) ต่ำ → interrupt ถาม

---

## Problem 1: Entity Resolution

ปัญหา: map "wallet เกี่ยวกับสัตว์เลี้ยง" → `wallet_id` แม้ชื่อจริงคือ `pat show` (typo ของ "pet shop")

### Pattern A — Vanna AI: RAG over schema metadata
- Repo: [vanna-ai/vanna](https://github.com/vanna-ai/vanna) (~23.4k stars)
- **Idea:** เก็บ DDL + documentation + (question, SQL) pairs ลง vector store. ตอนถาม → embed คำถาม → retrieve top-K ที่ใกล้เคียง → ใส่เป็น context ของ prompt
- **Why it works:** ถ้า train ด้วย question-SQL pair จริง (เช่น "wallet สัตว์เลี้ยง" → `WHERE wallet.id = '...'`) คำถามคล้าย ๆ จะ retrieve ได้ตรง. RAG มี grounding effect สูงกับการตั้งชื่อภายในบริษัท
- **Limitation:** ต้องมี training data เริ่มต้น. cold start แย่. Vanna ไม่ได้ออกแบบ specifically สำหรับ disambiguation — เมื่อ retrieval คืนหลาย wallet ที่เกี่ยวข้อง มันก็ส่งทั้งหมดให้ LLM แล้วหวังว่า LLM จะเลือกถูก

### Pattern B — CHESS: Information Retriever + LSH + semantic similarity
- Repo: [ShayanTalaei/CHESS](https://github.com/ShayanTalaei/CHESS) (269 stars) · [paper](https://arxiv.org/html/2405.16755v1)
- **Idea:** สอง-ชั้น value retrieval:
  1. **LSH index** บน cell values สำหรับ syntactic similarity (จับ typo "pat" ↔ "pet")
  2. **Embedding** บน column descriptions + table catalogs สำหรับ semantic similarity (จับ "สัตว์เลี้ยง" → wallet ที่ description มี "pet")
- **Why it works:** แยก concern ระหว่าง "พิมพ์ผิด" กับ "ความหมายต่าง" ชัดเจน. ใน Postgres เรา substitute LSH ด้วย `pg_trgm` ได้ฟรี
- **Limitation:** ต้อง pre-index ทุก wallet/category/tag ทุกครั้งที่ user เพิ่มข้อมูล. ใน MVP ใช้ on-the-fly retrieval ก็พอ

### Pattern C — Wren AI: Semantic Layer (MDL) + named entity catalog
- Repo: [Canner/WrenAI](https://github.com/Canner/WrenAI) (~15.1k stars) · [Wren Engine](https://github.com/Canner/wren-engine)
- **Idea:** กำหนด business semantic layer ก่อน — table aliases, column descriptions, calculated metrics, relationships. LLM ไม่ได้เห็น raw schema แต่เห็น "concept" ของ business
- **Why it works:** ใน Mint Money เราควบคุม schema เอง — สร้าง view เช่น `wallet_with_searchable_text` ที่รวม `name || ' ' || description || ' ' || tag_names` แล้ว index ด้วย trigram + embedding ได้
- **Limitation:** ต้อง maintain semantic layer ให้ sync กับ schema จริง

### Pattern D — RASL: vector index per semantic entity
- Paper: [RASL: Retrieval Augmented Schema Linking](https://assets.amazon.science/1b/95/8f62e89647348f4c4836f6c3040d/rasl-retrieval-augmented-schema-linking-for-massive-database-text-to-sql.pdf)
- **Idea:** decompose schema ออกเป็น semantic entity (table-level + column-level) แต่ละ entity มี name, alias, description, value-format. embed ทุก entity แล้ว query top-K ตอน inference
- **Why it works:** สเกลได้กับ schema ขนาดใหญ่. คำถามเดียวอาจ retrieve ทั้ง table "wallet" + column "name" + value "pat show" พร้อมกัน
- **Limitation:** complexity สูง ไม่จำเป็นสำหรับ schema เล็ก ๆ ของ Mint Money (มีไม่กี่ table)

### Pattern E — Postgres pg_trgm + LLM rerank
- Reference: [Medium: PostgreSQL Trigrams + LLM Agents](https://medium.com/@gnananveshm/how-i-used-postgresql-trigrams-llm-agents-to-query-structured-tables-intelligently-7b0b43f1010b)
- **Idea:** ใช้ `pg_trgm` เป็น first-pass filter (`SELECT … WHERE name % 'pat show' ORDER BY similarity DESC LIMIT 10`) แล้วให้ LLM rerank top-10 ด้วย context (description, recent usage)
- **Why it works:** trigram index = O(log n), LLM call แค่ครั้งเดียวบน ≤10 candidates. ไม่ต้องใช้ embedding infrastructure เลย ถ้าไม่อยากเพิ่ม dependency
- **Limitation:** ไม่จับ semantic distance ของ Thai (เช่น "สัตว์เลี้ยง" ↔ "pet"). ต้องเสริมด้วย bilingual embedding (ใช้ `BAAI/bge-m3` หรือ `intfloat/multilingual-e5`)

### สรุปสำหรับ Mint Money
ใช้ **hybrid 3-stage**: (1) `pg_trgm` candidate (top-20) → (2) embedding rerank ด้วย bilingual model (top-5) → (3) LLM ตัดสินใจขั้นสุดท้าย พร้อม confidence score. ถ้า top-1 < threshold → interrupt ถาม user.

---

## Problem 2: Time-Expression Parsing

ปัญหา: "5 เดือนที่แล้ว", "เดือนนี้", "ไตรมาส 2 ปีก่อน" → `(start_date, end_date)` แม่นยำ

### Pattern A — dateparser (deterministic library)
- Repo: [scrapinghub/dateparser](https://github.com/scrapinghub/dateparser) (~2.6k stars) · supports 200+ locales
- **Idea:** library deterministic, รองรับไทย out-of-the-box. `dateparser.parse('5 เดือนที่แล้ว', languages=['th'])` คืน `datetime`
- **Why it works:** zero LLM cost, 100% reproducible. มี `RELATIVE_BASE` setting ให้ inject "today" เพื่อทดสอบ
- **Limitation:** coverage ของ expression แบบ "ไตรมาส 2 ปีก่อน" ไม่ครอบคลุม. ภาษาผสม ("Q2 ปีก่อน") ก็ไม่ work

### Pattern B — Microsoft Recognizers-Text
- Repo: [microsoft/Recognizers-Text](https://github.com/microsoft/Recognizers-Text)
- **Idea:** parse datetime expressions เป็น TIMEX expression (ISO standard) — เช่น "next week" → `XXXX-WXX` แล้ว resolve ผ่าน timex helper
- **Why it works:** robust สำหรับ EN/ZH/ES. คืน range ที่แน่ชัด
- **Limitation:** **ไม่รองรับไทยเต็มรูปแบบ** (เป็น partial JA/KO/AR/SV เท่านั้น). ตัด option นี้ออกสำหรับ Mint Money

### Pattern C — LLM → Pydantic struct → Python computation
- Pattern ทั่วไปใน LangGraph SQL agent ([docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/sql-agent))
- **Idea:** LLM แปลง expression → structured object (เช่น `RelativePeriod(unit='month', offset=-5, length=1)`), แล้ว Python คำนวณ `(start, end)` ด้วย `dateutil.relativedelta` — *ห้าม* ให้ LLM คำนวณวันที่ตรง ๆ
- **Why it works:** LLM เก่งเข้าใจ intent, Python เก่งเลขกับ timezone. inject `today = getIt<AppClock>().today()` (Mint Money convention) เป็น context
- **Limitation:** LLM อาจ structure ผิด (เช่น "เดือนที่แล้ว" → offset=0 แทน -1). ต้องมี unit test crawl

### Pattern D — Hybrid: dateparser ก่อน, LLM fallback
- **Idea:** ลอง `dateparser.parse(text, languages=['th'])` ก่อน ถ้า return None หรือดูแปลก → ส่งให้ LLM extract เป็น Pydantic struct
- **Why it works:** ครอบคลุม 80% ของ expression ทั่วไปด้วย library ฟรี, อีก 20% ใช้ LLM. ลด token cost
- **Limitation:** ต้องมี logic ตัดสินใจว่า "ดูแปลก" คืออะไร (เช่น parsed date > vันนี้ ในบริบทย้อนหลัง)

### สรุปสำหรับ Mint Money
ใช้ **Pattern D**. function `parse_time_range(text, today) -> TimeRange | None` ที่ลำดับ: regex ตัวเลข+หน่วยภาษาไทย → `dateparser` → LLM (Pydantic). คืนเสมอเป็น `(start: date, end: date, granularity: enum)`.

---

## Problem 3: Numerical Accuracy

ปัญหา: ตัวเลขการเงินต้องถูก 100%. LLM ห้ามทำเลข. รองรับ Decimal, multi-currency

### Pattern A — Wren AI Semantic Layer (MDL): metric definitions
- Repo: [Canner/wren-engine](https://github.com/Canner/wren-engine)
- **Idea:** declare metric ใน MDL เช่น `total_income = SUM(transactions.amount) WHERE type = 'income'`. LLM แค่เลือก metric + filter ไม่ได้เขียน aggregation
- **Why it works:** ป้องกัน LLM เผลอใช้ `AVG` แทน `SUM`, ลืม filter เงินเข้า/ออก, หรือ hallucinate column. คำนวณจริงที่ Postgres ด้วย `numeric` precision
- **Limitation:** ต้อง pre-define metric ทุกตัวที่ business support. expression แบบ "เงินเข้า - เงินออก ของ wallet X" อาจไม่อยู่ใน catalog

### Pattern B — Tool-calling with structured query plan
- Pattern: LangGraph SQL agent + Pydantic ([Pydantic AI](https://ai.pydantic.dev/output/))
- **Idea:** LLM เรียก tool `query_transactions(QuerySpec)` ที่ `QuerySpec` เป็น Pydantic เช่น
  ```python
  class QuerySpec(BaseModel):
      metric: Literal["sum_income", "sum_expense", "balance", "count"]
      wallet_ids: list[UUID] | None = None
      category_ids: list[UUID] | None = None
      tag_ids: list[UUID] | None = None
      start_date: date
      end_date: date
      group_by: Literal["day", "week", "month", "year"] | None = None
      currency: Literal["THB", "USD"] = "THB"
  ```
  Python ใช้ spec นั้นสร้าง parameterized SQL ด้วย sqlalchemy/asyncpg ที่ใช้ `numeric`
- **Why it works:** LLM ไม่เห็น raw SQL, ไม่เผลอเขียน arithmetic. validation ด้วย Pydantic catch type mismatch ทันที. คุม injection ได้
- **Limitation:** flexibility ต่ำกว่า raw SQL — ถ้า user ถาม corner case ที่ไม่อยู่ใน schema ต้องเพิ่ม metric ใหม่

### Pattern C — Decimal everywhere
- Reference: [LLM Hallucinations in Finance](https://biztechmagazine.com/article/2025/08/llm-hallucinations-what-are-implications-financial-institutions)
- **Idea:** ใน Postgres เก็บเงินเป็น `numeric(20, 4)`. ใน Python รับด้วย `Decimal` (asyncpg + `init_connection` register codec). serialize → JSON ด้วย string. ใน prompt แสดงตัวเลขเป็น string ไม่ใช่ number (LLM token tokenizer ทำให้ float ผิดได้)
- **Why it works:** ไม่มี floating point error. ตรงตาม Mint Money convention (`Drift` + `Decimal` ใน mobile)
- **Limitation:** การ format display ต้อง round explicitly (`Decimal.quantize(Decimal('0.01'))`)

### Pattern D — Multi-currency separation
- **Idea:** ห้าม `SUM` ข้าม currency. SQL pattern: `SUM(amount) FILTER (WHERE currency = $1)` แยกเป็นแถวละ currency. final output คืน `dict[Currency, Decimal]`
- **Why it works:** ไม่มี implicit FX conversion → ไม่มี hallucinated rate
- **Limitation:** UI ต้องเตรียมรับ multi-row response

### สรุปสำหรับ Mint Money
**Pattern B + C + D รวมกัน**. ไม่มี code generation หรือ raw SQL จาก LLM. ทุก query ผ่าน whitelisted `QuerySpec` → SQL template ที่ test ครบ → Postgres compute ด้วย `numeric`.

---

## Problem 4: Verification / Self-Checking

ปัญหา: agent อาจเลือก wallet ผิด, date range ผิด แล้วตอบมั่ว — ต้องมี gate

### Pattern A — Query Checker Node (LangChain SQL agent)
- Reference: [LangChain SQL Agent docs](https://docs.langchain.com/oss/python/langgraph/sql-agent)
- **Idea:** dedicated node `check_query` ก่อน execute. LLM อ่าน SQL อีกครั้ง ตรวจ "NOT IN with NULL", "data type mismatch", "missing JOIN condition"
- **Why it works:** จับ syntax/semantic error ระดับ SQL ได้
- **Limitation:** ไม่จับ "เลือก wallet ผิดตัว" — เพราะ checker ไม่รู้ว่า user หมายถึง wallet ไหน

### Pattern B — NL Back-Translation (LLM-as-judge)
- Reference: [Arize: Text-to-SQL Evaluation](https://arize.com/blog/text-to-sql-evaluating-sql-generation-with-llm-as-a-judge/)
- **Idea:** LLM #2 อ่าน SQL → generate "คำถามภาษาธรรมชาติที่ SQL นี้น่าจะตอบ" → LLM #3 เปรียบเทียบกับคำถามเดิม. ถ้า semantic equivalent → pass
- **Why it works:** จับ "เลือก wallet ผิด" ได้ — เพราะ back-translated question จะเป็น "income of wallet 'salary'" แทนที่จะเป็น "wallet เกี่ยวกับสัตว์เลี้ยง"
- **Limitation:** เพิ่ม 2 LLM call ต่อ turn. ต้อง prompt judge ให้ binary (yes/no + reason) ตามที่ Patronus AI/Evidently AI แนะนำ

### Pattern C — CHESS Unit Tester
- Repo: [ShayanTalaei/CHESS](https://github.com/ShayanTalaei/CHESS)
- **Idea:** generate "natural language unit test" จากคำถาม (เช่น "result must include only transactions in 2025-01") แล้ว validate ผลลัพธ์ post-execution
- **Why it works:** จับ logical error เช่น "filter ผิดเดือน" หลัง run แล้ว
- **Limitation:** complex, ทำให้ debug ยาก

### Pattern D — Confidence-based gating
- Reference: [Amazon: Confidence Scoring for LLM-Generated SQL](https://assets.amazon.science/03/bb/db4fedf948cebfd88475be8bb191/11-confidence-scoring-for-llm.pdf)
- **Idea:** แต่ละ resolution step (entity, time, metric) แนบ confidence (0-1). aggregate confidence ของ pipeline. ถ้า < threshold → trigger verifier; ถ้า < lower threshold → interrupt user
- **Why it works:** ประหยัด token, verify เฉพาะ case ไม่ชัดเจน
- **Limitation:** confidence ของ LLM ไม่ calibrate ดี — ต้อง test และ tune threshold

### Pattern E — Result Sanity Checks (deterministic)
- **Idea:** post-execute checks เช่น
  - ถ้า user ถามรายได้ แต่ผลลัพธ์ทั้งหมดเป็น expense → fail
  - ถ้า date range คือเดือนนี้ แต่มี transaction outside → fail
  - ถ้า wallet specified แต่ผลลัพธ์มาจาก wallet อื่น → fail
- **Why it works:** ถูก 100% เพราะเป็น code, ไม่พึ่ง LLM
- **Limitation:** ต้องเขียน rule ทุกตัวเอง

### สรุปสำหรับ Mint Money
**3-layer verifier**: (1) Pydantic validation ของ `QuerySpec` (free), (2) deterministic sanity check บน result (Pattern E), (3) NL back-translation judge (Pattern B) เฉพาะตอน confidence < 0.8.

---

## Problem 5: Clarification / Human-in-the-Loop

ปัญหา: เมื่อ ambiguous → ถาม user แทนที่จะเดา

### Pattern A — LangGraph `interrupt()` + `Command(resume=...)`
- Reference: [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [Blog](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/)
- **Idea:** ใน node ที่เจอ ambiguity เรียก
  ```python
  from langgraph.types import interrupt, Command
  choice = interrupt({
      "type": "entity_disambiguation",
      "candidates": [{"id": w.id, "name": w.name, "score": w.score} for w in top3],
      "question": "หมายถึง wallet ไหน?"
  })
  return {"resolved_wallet_id": choice}
  ```
  Graph pause, save state ผ่าน checkpointer, รอ frontend ส่ง `Command(resume=wallet_id)` กลับ
- **Why it works:** native LangGraph primitive, ทำงานข้าม session ได้ (resume วันถัดไปได้)
- **Limitation:** parallel `interrupt()` มี [bug เรื่อง ID collision](https://github.com/langchain-ai/langgraph/issues/6626) — ต้อง interrupt ทีละครั้ง

### Pattern B — Odin / AmbiSQL: ambiguity taxonomy
- Papers: [Odin](https://arxiv.org/pdf/2505.19302), [AmbiSQL](https://arxiv.org/html/2508.15276)
- **Idea:** จำแนก ambiguity เป็น category: schema (column ไหน?), value (entity ตัวไหน?), aggregation (SUM vs AVG?), time (start/end ไม่ชัด?). แต่ละ category มี clarification prompt template เฉพาะ
- **Why it works:** UX ดีขึ้น — คำถาม clarify ของ entity ("เลือก wallet") ต่างจาก aggregation ("รวม หรือ เฉลี่ย")
- **Limitation:** ต้อง build taxonomy เอง

### Pattern C — Reactive HITL (only when needed)
- Reference: [Elastic: HITL with LangGraph](https://www.elastic.co/search-labs/blog/human-in-the-loop-hitllanggraph-elasticsearch)
- **Idea:** อย่า interrupt ทุก turn — interrupt ก็ต่อเมื่อ resolver flag `needs_clarification=True`. มี 3-tier confidence (high → ตอบเลย, mid → ตอบพร้อมระบุสมมติฐาน, low → ถาม)
- **Why it works:** ลด friction ของ user, ระบบยังตอบได้เร็วเมื่อข้อมูลชัด
- **Limitation:** ต้อง design "ตอบพร้อมสมมติฐาน" UX ให้ user แก้ได้ง่าย (เช่น chip "wallet: pat show ✕")

### Pattern D — Multiple candidates inline
- Pattern จาก Odin
- **Idea:** เมื่อ ambiguous, ส่ง **N candidate SQLs** + ผลลัพธ์ทั้งหมด ให้ user เลือก แทนที่จะถามแบบ open question
- **Why it works:** user เห็นข้อมูลจริงก่อนตัดสินใจ
- **Limitation:** ต้อง execute หลาย query, แพง

### สรุปสำหรับ Mint Money
**Pattern A + B + C**. ใช้ `interrupt()` พร้อม payload ตาม taxonomy (entity/time/metric/aggregation). 3-tier confidence: > 0.85 ตอบเลย, 0.6-0.85 ตอบพร้อม disclosure (เช่น "ใช้ wallet 'pat show' — ไม่ใช่? พิมพ์ 'เปลี่ยน wallet'"), < 0.6 interrupt.

---

## Problem 6: Conversation Memory & Entity Carry-Over

ปัญหา: "แล้วเดือนที่แล้วล่ะ" ต้อง reuse wallet จาก turn ก่อน

### Pattern A — Checkpointer (thread state)
- Reference: [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory) · [Persistence](https://www.baihezi.com/mirrors/langgraph/how-tos/persistence/index.html)
- **Idea:** `PostgresSaver` checkpoint เก็บ full state ของแต่ละ thread. resolved entities (wallet_ids, category_ids, last_time_range) อยู่ใน state field — ถูก persist อัตโนมัติ
- **Why it works:** native, no code. resume ข้าม session ได้
- **Limitation:** scope แค่ thread เดียว — ถ้า user เริ่มแชทใหม่ entity preferences หาย

### Pattern B — Long-term Store + Semantic Search
- Reference: [Semantic Search for LangGraph Memory](https://blog.langchain.com/semantic-search-for-langgraph-memory/)
- **Idea:** `BaseStore` (Postgres-backed) + embedding index. เก็บ "user prefers wallet 'pat show' for pet-related queries" เป็น memory item. ตอนเริ่ม turn → semantic search top-3 memory ใส่ใน context
- **Why it works:** ข้าม thread ได้, scope ที่ user_id
- **Limitation:** ต้องระวัง retrieve memory ที่ outdated (user เปลี่ยน wallet name แล้ว)

### Pattern C — Explicit Entity State (Pydantic)
- Pattern จาก [LangGraph Smart Agent (KBTG)](https://medium.com/kbtg-life/building-a-smarter-agent-with-langgraph-a-guide-to-short-term-memory-and-context-engineering-25b1f7ae155d)
- **Idea:** state schema มี field `current_context: ConversationContext` ที่เก็บ
  ```python
  class ConversationContext(BaseModel):
      last_wallets: list[UUID] = []
      last_categories: list[UUID] = []
      last_tags: list[UUID] = []
      last_time_range: TimeRange | None = None
      last_metric: str | None = None
  ```
  Resolver node อ่าน context ก่อนเริ่ม → ถ้า user ไม่ระบุ wallet ใหม่ → reuse `last_wallets`
- **Why it works:** explicit, debug ง่าย, ไม่ต้องพึ่ง embedding
- **Limitation:** ต้อง manual update ทุก node

### Pattern D — Coreference resolution via LLM
- **Idea:** node แรกรับ message + context summary → LLM rewrite "แล้วเดือนที่แล้วล่ะ" → "income ของ wallet pat show เดือนเมษายน 2026"
- **Why it works:** standard preprocessing pattern, ทำให้ pipeline ที่เหลือไม่ต้องรู้เรื่อง history
- **Limitation:** LLM อาจ rewrite ผิด — ต้อง verify ว่า rewrite ตรง intent

### สรุปสำหรับ Mint Money
**Pattern A + C + D**. checkpointer + explicit `ConversationContext` ใน state. Node แรก (`coreference_resolver`) ใช้ LLM rewrite query โดยอ้าง context. Pattern B (long-term store) ไว้สำหรับ phase 2 — ตอนนี้ thread-scoped พอ

---

## Notable Repos Referenced

| Repo | Stars | What we borrow |
|---|---|---|
| [vanna-ai/vanna](https://github.com/vanna-ai/vanna) | ~23.4k | RAG over (DDL + docs + SQL pairs); training pattern; user-aware SQL filtering |
| [Canner/WrenAI](https://github.com/Canner/WrenAI) | ~15.1k | Semantic Layer (MDL); intent classification node; SQL correction loop; pre-defined metrics |
| [eosphoros-ai/DB-GPT](https://github.com/eosphoros-ai/DB-GPT) | ~14k | AWEL workflow expression; multi-agent orchestration concepts |
| [eosphoros-ai/Awesome-Text2SQL](https://github.com/eosphoros-ai/Awesome-Text2SQL) | curation | survey of techniques (RESDSQL, PICARD, RAT-SQL, DAIL-SQL) |
| [wbbeyourself/MAC-SQL](https://github.com/wbbeyourself/MAC-SQL) | 336 | Selector → Decomposer → Refiner pipeline |
| [ShayanTalaei/CHESS](https://github.com/ShayanTalaei/CHESS) | 269 | LSH + embedding hybrid value retrieval; LLM-based unit tester |
| [defog-ai/sqlcoder](https://github.com/defog-ai/sqlcoder) | ~3.8k | metadata pruning; joinable column dictionary pattern |
| [defog-ai/introspect](https://github.com/defog-ai/introspect) | not verified | structured-data deep research pipeline |
| [microsoft/Recognizers-Text](https://github.com/microsoft/Recognizers-Text) | ~1.9k | TIMEX expression normalization (EN only — reference, not adoption) |
| [scrapinghub/dateparser](https://github.com/scrapinghub/dateparser) | ~2.6k | Thai relative date parsing — direct dependency |
| [PyThaiNLP/pythainlp](https://github.com/PyThaiNLP/pythainlp) | ~1k | Thai tokenization / `thai_strftime` formatting (display only) |
| [pgvector/pgvector](https://github.com/pgvector/pgvector) | ~17k | embedding storage + KNN in our existing Postgres |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | ~13k | StateGraph, `interrupt()`, `PostgresSaver`, `BaseStore` |
| [HKUSTDial/NL2SQL_Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook) | reference | survey of techniques + benchmarks |
| Odin paper / AmbiSQL paper | n/a | ambiguity taxonomy + clarification UX |
| [pydantic/pydantic-ai](https://ai.pydantic.dev/output/) | ~5k | structured output via tool-call binding |

---

## Synthesis: Recommended Design

### High-level architecture

```
                       ┌────────────────────────┐
   user message ─────▶ │ 1. Coreference Rewrite │  ◀── ConversationContext
                       │    (LLM, optional)     │
                       └───────────┬────────────┘
                                   │ rewritten_query
                                   ▼
                       ┌────────────────────────┐
                       │ 2. Intent Classifier   │   labels:
                       │    (LLM, structured)   │   {balance, sum, list,
                       └───────────┬────────────┘    trend, compare, …}
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ 3. Plan Builder        │   produces QueryPlan:
                       │    (LLM, Pydantic)     │   slots to fill
                       └───────────┬────────────┘
                                   │
              ┌────────────────────┴──────────────────┐
              ▼                                       ▼
   ┌──────────────────────┐           ┌──────────────────────────┐
   │ 4a. Entity Resolver  │           │ 4b. Time Resolver        │
   │  pg_trgm  ─►         │           │  regex → dateparser →    │
   │  embed rerank ─►     │           │  LLM(Pydantic)           │
   │  LLM choose          │           │  + AppClock context      │
   │  (confidence)        │           │  (confidence)            │
   └──────────┬───────────┘           └─────────────┬────────────┘
              │                                     │
              └─────────────────┬───────────────────┘
                                ▼
                  ┌─────────────────────────────┐
                  │ 5. Confidence Gate          │   if any < 0.6
                  │                             │ ──────────────▶ interrupt()
                  │                             │   user picks → resume
                  └─────────────┬───────────────┘
                                ▼
                  ┌─────────────────────────────┐
                  │ 6. SQL Builder (Python)     │   QuerySpec → SQL template
                  │    deterministic            │   asyncpg/sqlalchemy core
                  └─────────────┬───────────────┘   numeric / Decimal
                                ▼
                  ┌─────────────────────────────┐
                  │ 7. Executor (Postgres)      │   execute, return rows
                  └─────────────┬───────────────┘
                                ▼
                  ┌─────────────────────────────┐
                  │ 8. Verifier                 │   layer 1: Pydantic
                  │    deterministic checks +   │   layer 2: sanity rules
                  │    NL back-translate (opt)  │   layer 3: LLM judge if
                  └─────────────┬───────────────┘            confidence < 0.8
                                ▼
                  ┌─────────────────────────────┐
                  │ 9. Responder (LLM)          │   format Decimal with
                  │    Thai answer + numbers    │   thousand sep + currency
                  │    pre-formatted by code    │   *no LLM math*
                  └─────────────┬───────────────┘
                                │
                                ▼
                  update ConversationContext → checkpointer save
```

### Node-by-node responsibility

| Node | Type | Input | Output | LLM call? | Confidence |
|---|---|---|---|---|---|
| `coreference_rewrite` | LLM (small) | message + context | rewritten text | yes (cheap) | implicit |
| `intent_classify` | LLM + Pydantic | rewritten | `Intent` enum | yes | yes |
| `plan_build` | LLM + Pydantic | rewritten + intent | `QueryPlan` (slots) | yes | yes |
| `entity_resolve` | hybrid (SQL + embed + LLM) | mentioned terms | `[(id, score)]` per slot | yes (rerank only) | yes (per entity) |
| `time_resolve` | dateparser → LLM fallback | time phrase + today | `TimeRange` | sometimes | yes |
| `confidence_gate` | code | all confidences | continue / interrupt | no | aggregate |
| `interrupt_clarify` | LangGraph `interrupt()` | candidates + reason | user choice | no | reset to 1.0 |
| `sql_build` | code | `QuerySpec` | parameterized SQL | no | n/a |
| `execute` | asyncpg | SQL + params | rows (Decimal) | no | n/a |
| `verify` | code + LLM | spec + rows + question | pass/fail + reason | conditional | n/a |
| `respond` | LLM | rows + spec + intent | Thai natural-language | yes | n/a |
| `commit_context` | code | resolved entities | new state | no | n/a |

### State schema (Pydantic — actual implementation hint)

```python
from pydantic import BaseModel, Field
from datetime import date
from decimal import Decimal
from typing import Literal
from uuid import UUID

class TimeRange(BaseModel):
    start: date
    end: date
    granularity: Literal["day", "week", "month", "quarter", "year"]
    confidence: float

class ResolvedEntity(BaseModel):
    id: UUID
    display_name: str
    score: float                     # 0..1
    alternatives: list[dict] = []    # for disclosure UI

class QuerySpec(BaseModel):
    metric: Literal["sum_income", "sum_expense", "balance",
                    "count", "avg_per_day", "list"]
    wallets: list[ResolvedEntity] = []
    categories: list[ResolvedEntity] = []
    tags: list[ResolvedEntity] = []
    time_range: TimeRange
    group_by: Literal["day", "week", "month", "year", "wallet",
                      "category", "tag"] | None = None
    currency: Literal["THB", "USD", "ALL"] = "THB"

class ConversationContext(BaseModel):
    last_wallets: list[UUID] = []
    last_categories: list[UUID] = []
    last_tags: list[UUID] = []
    last_time_range: TimeRange | None = None
    last_metric: str | None = None
    last_currency: str | None = None

class AgentState(BaseModel):
    # input
    user_message: str
    user_id: UUID
    today: date                      # injected from AppClock-equivalent

    # progressive enrichment
    rewritten_query: str | None = None
    intent: str | None = None
    plan: dict | None = None
    spec: QuerySpec | None = None
    confidence: float = 1.0
    needs_clarification: bool = False
    clarification_payload: dict | None = None

    # results
    rows: list[dict] | None = None     # Decimal serialized as str
    verifier_verdict: dict | None = None
    answer: str | None = None

    # carry-over
    context: ConversationContext = Field(default_factory=ConversationContext)
```

### Where each "good idea" maps in

| Good idea | Source | Where it lives |
|---|---|---|
| RAG over schema metadata + question/SQL pairs | Vanna | offline indexing job; `entity_resolve` reads top-K |
| Semantic Layer / pre-defined metrics | Wren AI | `QuerySpec.metric` enum + SQL templates in Python |
| LSH + embedding hybrid value retrieval | CHESS | `entity_resolve` (pg_trgm = LSH substitute) + pgvector |
| Plan→Resolve→Execute→Verify pipeline | LangChain SQL agent + MAC-SQL | overall graph topology |
| Schema selector / pruning | MAC-SQL Selector | not needed (small schema); replaced by `intent_classify` |
| Refiner / self-correct | MAC-SQL Refiner | `verify` → if fail, regenerate with reason loop (max 2 iter) |
| LLM-based unit test / NL back-translation | CHESS UT + Arize blog | `verify` layer 3 (conditional on low confidence) |
| `interrupt()` HITL | LangGraph docs | `interrupt_clarify` node |
| Ambiguity taxonomy | Odin / AmbiSQL | clarification payload `type` field |
| dateparser for Thai | scrapinghub/dateparser | `time_resolve` first pass |
| Pydantic structured output | Pydantic AI | every LLM call uses `with_structured_output` |
| pg_trgm fast fuzzy | Postgres + Medium ref | candidate generation in `entity_resolve` |
| Postgres long-term Store | LangGraph BaseStore | phase 2 (skip in MVP) |
| Checkpointer threading | LangGraph PostgresSaver | thread persistence + resume |
| Decimal everywhere, never LLM math | finance domain | asyncpg codec + `respond` formats string before LLM sees it |

### Open questions / risks

1. **Embedding model choice** — `BAAI/bge-m3` (multilingual, 568M) vs `intfloat/multilingual-e5-large`. ต้อง benchmark กับ Thai+English wallet name + description. ถ้าโฮสต์เองไม่ไหว → ใช้ Cohere embed-multilingual-v3 หรือ OpenAI `text-embedding-3-small`
2. **Cold start** — Vanna pattern ต้อง training pairs. เริ่มต้นไม่มี → ต้องใช้ synthetic pairs จาก LLM แล้ว iterate ตามคำถามจริงของ user
3. **Verifier cost** — NL back-translation = 2 extra LLM call. trigger เฉพาะ confidence < 0.8 และ cache verdict per (spec hash) ลด traffic
4. **Multi-currency follow-up** — ถ้า turn ก่อนเป็น THB แล้วถาม "USD ล่ะ" — context carry-over ต้อง override currency ไม่ใช่ keep ค่าเดิม. logic ใน `coreference_rewrite` ต้อง explicit
5. **Confidence calibration** — LLM-reported confidence ไม่ reliable. ต้อง replace ด้วย heuristic (top-1 score - top-2 score) สำหรับ entity, regex-coverage สำหรับ time
6. **Interrupt UX in mobile** — Flutter client ต้องรองรับการ display candidate list และ resume thread ได้. ต้อง coordinate กับ mobile team ก่อน
7. **Schema drift** — ถ้า user เพิ่ม/ลบ/rename wallet, embedding index ต้อง update. ใช้ trigger-based ETL หรือ refresh on-demand
8. **Test corpus ภาษาไทย** — ต้องเขียน eval set ของคำถามจริง (~200 ตัวอย่าง) ครอบคลุม 6 ปัญหา ก่อนเริ่ม implement — เป็นการตั้ง regression baseline

### Phase plan (suggestion)

| Phase | Scope | Estimate |
|---|---|---|
| **0. Eval harness** | 200-question Thai test set + LangSmith dataset | 2 days |
| **1. Skeleton graph** | nodes 1-9 with rule-based stubs (no LLM, no fuzzy) | 3 days |
| **2. Time + Entity** | dateparser + pg_trgm + embedding rerank | 4 days |
| **3. Plan + SQL build** | Pydantic `QuerySpec` + SQL templates + Decimal | 4 days |
| **4. Verifier + interrupt** | sanity checks + clarification flow | 3 days |
| **5. Memory** | checkpointer + ConversationContext carry-over | 2 days |
| **6. Polish** | NL back-translation judge, response formatting, eval pass | 3 days |

รวม ~21 working days สำหรับ feature parity + เกินเดิมในเรื่อง verification และ clarification

---

## Sources

- [Vanna — vanna-ai/vanna](https://github.com/vanna-ai/vanna)
- [How Vanna works (Medium)](https://medium.com/vanna-ai/how-vanna-works-how-to-train-it-data-security-8d8f2008042)
- [WrenAI — Canner/WrenAI](https://github.com/Canner/WrenAI)
- [Wren Engine — Canner/wren-engine](https://github.com/Canner/wren-engine)
- [Why Semantic Layer is Essential — getwren.ai](https://www.getwren.ai/post/why-the-semantic-layer-is-essential-for-reliable-text-to-sql-and-how-wren-ai-brings-it-to-life)
- [DB-GPT — eosphoros-ai/DB-GPT](https://github.com/eosphoros-ai/DB-GPT)
- [Awesome-Text2SQL — eosphoros-ai/Awesome-Text2SQL](https://github.com/eosphoros-ai/Awesome-Text2SQL)
- [MAC-SQL — wbbeyourself/MAC-SQL](https://github.com/wbbeyourself/MAC-SQL)
- [CHESS — ShayanTalaei/CHESS](https://github.com/ShayanTalaei/CHESS) · [paper](https://arxiv.org/html/2405.16755v1)
- [SQLCoder — defog-ai/sqlcoder](https://github.com/defog-ai/sqlcoder)
- [Introspect — defog-ai/introspect](https://github.com/defog-ai/introspect)
- [LangGraph SQL Agent docs](https://docs.langchain.com/oss/python/langgraph/sql-agent)
- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory)
- [Semantic Search for LangGraph Memory](https://blog.langchain.com/semantic-search-for-langgraph-memory/)
- [Plan-and-Execute Agents — LangChain blog](https://blog.langchain.com/planning-agents/)
- [Making it easier to build HITL agents with interrupt](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/)
- [HITL with LangGraph & Elastic](https://www.elastic.co/search-labs/blog/human-in-the-loop-hitllanggraph-elasticsearch)
- [Microsoft Recognizers-Text](https://github.com/microsoft/Recognizers-Text)
- [scrapinghub/dateparser](https://github.com/scrapinghub/dateparser)
- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)
- [pgvector](https://github.com/pgvector/pgvector)
- [Postgres pg_trgm docs](https://www.postgresql.org/docs/current/pgtrgm.html)
- [Trigrams + LLM Agents (Medium)](https://medium.com/@gnananveshm/how-i-used-postgresql-trigrams-llm-agents-to-query-structured-tables-intelligently-7b0b43f1010b)
- [RASL (Amazon Science)](https://assets.amazon.science/1b/95/8f62e89647348f4c4836f6c3040d/rasl-retrieval-augmented-schema-linking-for-massive-database-text-to-sql.pdf)
- [Confidence Scoring for LLM-Generated SQL (Amazon)](https://assets.amazon.science/03/bb/db4fedf948cebfd88475be8bb191/11-confidence-scoring-for-llm.pdf)
- [Text-to-SQL: LLM-as-Judge (Arize)](https://arize.com/blog/text-to-sql-evaluating-sql-generation-with-llm-as-a-judge/)
- [Odin paper — schema ambiguity](https://arxiv.org/pdf/2505.19302)
- [AmbiSQL paper — interactive ambiguity](https://arxiv.org/html/2508.15276)
- [HKUSTDial/NL2SQL_Handbook](https://github.com/HKUSTDial/NL2SQL_Handbook)
- [Pydantic AI — Output](https://ai.pydantic.dev/output/)
