# AI Friend Chat — Product Spec

> **Status**: Draft v1.0 | **Owner**: Product | **Date**: 2026-04-20
> **North Star reference**: `docs/ai/old/vision/Mint Money วิสัยทัศน์ AI Financial Friend.md`

---

## 1. Positioning Statement

Chat ของ Mint Money **ไม่ใช่ general chatbot** และ**ไม่ใช่ financial advisor ทางการ** — มันคือ **เพื่อนการเงินคนหนึ่ง** ที่:

1. **รู้ข้อมูลของผู้ใช้** (กระเป๋า, หนี้, transaction, goal) — ตอบได้เฉพาะเจาะจง ไม่ใช่คำตอบกว้างๆ แบบ ChatGPT
2. **พูดไทยแบบเพื่อน** — อบอุ่น ไม่ตัดสิน ให้กำลังใจ
3. **ลดแรงเสียดทานการใช้แอป** — แทนที่จะกดเมนู 5 ชั้น ก็แค่พิมพ์ถาม

> เราชนะ ChatGPT ตรงที่ "มี data + พูดไทยแบบเพื่อน + actionable" และชนะ Money Lover ตรงที่ "คุยได้"

---

## 2. User Jobs-to-be-Done (7 อันดับแรก)

ผู้ใช้จะ "จ้าง" chat ไปทำอะไร (เรียงตามความถี่คาด)

| # | JTBD | Trigger | Pain point (จาก Vision) | Persona หลัก |
|---|---|---|---|---|
| J1 | **"ช่วยจดให้หน่อย"** | เหนื่อย/รีบ พิมพ์สั้นๆ | การบันทึกเป็นอุปสรรค (2-3 นาที → เลิก) | ทุกคน |
| J2 | **"เดือนนี้ฉันโอเคไหม?"** | สิ้นเดือน / ก่อนซื้อของ | ข้อมูลมี แต่ไม่เข้าใจภาพรวม | เอมม่า, อนุชา |
| J3 | **"ผ่อนหนี้เร็วขึ้นได้ยังไง?"** | หลังจ่ายค่างวด / เงินเดือนเข้า | กับดักหนี้ซ้อนหนี้ ไม่มีที่ปรึกษา | อนุชา |
| J4 | **"เดือนนี้เหลือกี่บาท ใช้ได้ถึงสิ้นเดือนไหม?"** | กลางเดือน ก่อนใช้เงิน | เงินชนเดือน ไม่มี buffer check | เอมม่า |
| J5 | **"ออม/ตั้ง goal ยังไงดี?"** | มีแรงบันดาลใจ เช่น เห็นเพื่อนออม | อยากออมแต่ไม่รู้เริ่มยังไง | เอมม่า |
| J6 | **"ทำไมเดือนนี้ใช้เยอะจัง?"** | ตกใจยอดบัตรเครดิต | ใช้ 50K ไม่บอกว่าทำไม | สมชาย, อนุชา |
| J7 | **"ฉันกังวลเรื่องเงิน..."** (ระบาย) | late night / หลังเห็นยอด | ไม่มีใครคุยด้วย ธนาคารน่ากลัว | ทุกคน |

**หมายเหตุ**: J1 (Quick Log) คาดจะ claim >50% ของ chat volume — จึงต้องเร็วและแม่นที่สุด

---

## 3. Conversation Patterns

### Pattern A — Quick Log (จ้าง AI จดให้)

- **Trigger**: ผู้ใช้พิมพ์ลักษณะ transaction ("กาแฟ 50", "ข้าวเที่ยง 80 BTS 34")
- **Flow**:
  1. User: *"กาแฟ 50 ข้าวเที่ยง 80 BTS 34"*
  2. AI: แสดง 3 รายการเป็น card พร้อมหมวด/กระเป๋าที่เดา → ปุ่ม **[ยืนยันทั้งหมด]** / [แก้ไข]
  3. หลังยืนยัน: *"จดให้แล้ว 3 รายการ รวม 164฿ เดือนนี้ใช้อาหารไป 4,200 แล้วนะ"*
- **Outcome**: เร็วกว่ากรอกเอง ≥5x (≤5 วินาที)
- **Metric**: confirm rate ≥85%, median latency <2s

### Pattern B — Ask About My Money (ถามสถานะตัวเอง)

- **Trigger**: *"เดือนนี้ฉันโอเคไหม?"*, *"เหลือเท่าไร?"*, *"ใช้ค่ากาแฟไปเท่าไร?"*
- **Flow**:
  1. AI ดึง data จริง (wallets, transactions, budget) → เรียบเรียงเป็นเรื่องเล่า
  2. *"เดือนนี้ใช้ไป 32,000 จาก 50,000 — เหลืออีก 18K ใช้ได้ถึงสิ้นเดือน 12 วันเฉลี่ย 1,500/วัน กำลังดี งบอาหารเหลือสบายๆ แต่หมวดช้อปปิ้งใกล้ชนเพดานแล้วนะ 🙂"*
  3. Quick follow-up chips: [ดูรายการล่าสุด] [ช้อปปิ้งใช้ไปเท่าไร?]
- **Outcome**: รู้สึก "เห็นภาพ" ไม่ต้องกดดู dashboard
- **Metric**: follow-up rate ≥30% (ถามต่อ = engaged), thumbs-up ≥70%

### Pattern C — Coaching / Advice (ขอคำแนะนำ)

- **Trigger**: *"อยากปลดหนี้เร็วขึ้น"*, *"ออมได้เดือนละกี่บาทดี?"*, *"ควรปิดบัตรไหนก่อน?"*
- **Flow**:
  1. AI วิเคราะห์ context (หนี้, cash flow, goal) → เสนอ **1 action ที่ทำได้ทันที**
  2. *"ดูจากเงินเดือนและหนี้ของคุณ ลองเพิ่มค่างวดรถเดือนละ 3,000 จะปลดหนี้เร็วขึ้น 7 เดือน ประหยัดดอก ~8,400฿ เริ่มเดือนหน้าดีไหม?"* → [ตั้ง reminder] [ดูแผนละเอียด]
- **Outcome**: ออกจาก chat พร้อม 1 action ที่ตั้งใจทำ
- **Metric**: action-accept rate ≥25% (ภายใน 7 วัน)
- **Anti-pattern**: ห้ามให้คำแนะนำทั่วไป ("ควรประหยัด" / "ควรออม") — ต้องระบุจำนวน + timing + เริ่มยังไง

### Pattern D — Emotional Support (ระบาย)

- **Trigger**: *"เครียดเรื่องเงิน"*, *"รู้สึกใช้เยอะไป"*, *"กลัวไม่พอจ่าย..."*
- **Flow**:
  1. AI **ไม่เริ่มด้วยตัวเลข** → เริ่มด้วย acknowledge *"เข้าใจเลย เรื่องเงินเป็นเรื่องกังวลที่คุยกับใครยาก..."*
  2. ถามกลับแบบเปิด *"อยากเล่าให้ฟังไหมว่ากังวลเรื่องอะไรที่สุดตอนนี้?"*
  3. เมื่อผู้ใช้เล่า → สะท้อนกลับ + เสนอ **action เล็กๆ ที่ทำได้ทันที** (ไม่ใช่แผนใหญ่)
  4. *"ลองทำแค่ step แรกดูก่อนนะ — เดี๋ยวเราเช็คดูด้วยกันอาทิตย์หน้า ok?"*
- **Outcome**: ผู้ใช้รู้สึก "ไม่ได้อยู่คนเดียว"
- **Metric**: session length ≥3 turns, sentiment improvement (self-report 1-5 pre/post)
- **Guardrail**: หลีกเลี่ยงคำว่า "เยอะ", "เกิน", "ผิดพลาด" — ใช้คำเช่น "ลองดู", "ค่อยๆ", "ทำได้"

### Pattern E — Onboarding / Exploration (เพิ่งเริ่มใช้)

- **Trigger**: 3 sessions แรก หรือ chat ว่างเปล่า
- **Flow**:
  1. AI ทักก่อนแบบเปิด *"สวัสดี! ผมคือเพื่อนการเงินของคุณ — ลองถามอะไรก็ได้ เช่น 'ฉันเหลือเท่าไร' หรือแค่พิมพ์สิ่งที่ซื้อวันนี้ให้ผมจดก็ได้"*
  2. แสดง suggested chips 3 อัน: [เหลือเท่าไรเดือนนี้] [บันทึกรายจ่าย] [ตั้ง goal แรก]
- **Outcome**: ผู้ใช้เข้าใจว่าทำอะไรได้บ้างภายใน 30 วินาที
- **Metric**: chip tap rate ≥60% ใน session แรก

---

## 4. Anti-patterns (ห้ามทำเด็ดขาด)

1. **ห้ามตัดสิน / shame** — ไม่พูด *"ใช้เยอะเกินไปนะ"*, *"ไม่ควร..."*, *"ควรจะ..."* ใช้ *"สังเกตว่า..."*, *"ลองดูไหม..."*
2. **ห้ามฮาร์ดเซลล์ Premium กลางบทสนทนา** — paywall ได้เฉพาะตอนจะใช้ feature ที่ถูก gate (เช่น scan ที่ 6 ของเดือน) และต้องเป็น soft mention ไม่ตัดบทสนทนา
3. **ห้ามคำแนะนำกว้างๆ** — *"ควรประหยัด"*, *"ควรออม"*, *"ตั้งงบดูสิ"* → ต้องเจาะจง: จำนวน + หมวด + timing + first step
4. **ห้ามใช้ tone แบบ bank manager** — คำอย่าง *"สถานะทางการเงินของท่าน"*, *"โปรดพิจารณา..."*, *"ขอแนะนำ..."* ไม่ใช้
5. **ห้ามหลอกว่าเป็นมนุษย์** — เมื่อผู้ใช้ถาม "เป็นคนไหม" → ยอมรับตรงๆ ว่าเป็น AI แต่เป็น AI ที่ "อยู่ข้างคุณ"
6. **ห้ามให้คำปรึกษาการลงทุน/กฎหมาย/ภาษีเชิงลึก** — redirect ไปผู้เชี่ยวชาญจริง (compliance guardrail)
7. **ห้ามตอบเมื่อไม่แน่ใจ data** — ถ้า context ไม่พอ ต้องบอก *"ผมยังไม่เห็นข้อมูลพอ ลองบันทึกอีกสัก 2 สัปดาห์แล้วเราคุยใหม่นะ"* ไม่มั่ว

---

## 5. Differentiation (ทำไมไม่เปิด ChatGPT แทน?)

| มิติ | ChatGPT/Gemini ทั่วไป | Cleo (UK) / Plum (EU) | **Mint Money Chat** |
|---|---|---|---|
| Context ของผู้ใช้ | ไม่รู้อะไรเลย | รู้จาก bank sync | **รู้จาก Drift DB + obligation + goal** |
| ภาษา | ไทยได้ แต่ไม่เนทีฟ | ไม่รองรับไทย | **ไทยแบบเพื่อน ตั้งแต่วันแรก** |
| Actionable | คำแนะนำทั่วไป | กด button ใน app Cleo | **สั่ง action ในแอปได้ทันที (จด, ตั้ง goal, reminder)** |
| Obligation depth | ไม่รู้ | ไม่มี | **รู้ทุกค่างวด + PlanVersion** |
| ราคา | ฟรี/$20 | $5.99/เดือน | **ฟรี + Premium 99฿** |

**Moat**: data + language + obligation domain — 3 ชั้นนี้ลอกยากและต้องใช้เวลา

---

## 6. Capability Tier Roadmap

> **หลักคิด**: จัด phase ตาม **ความซับซ้อนของ AI capability** ไม่ใช่ตาม timeline — เพราะ timeline เปลี่ยนตามทีม/ทุน/learning แต่ **ลำดับความซับซ้อนของ capability เปลี่ยนไม่ได้** (ของยากต้องสร้างบนของง่าย) ทีมจะผ่าน tier เร็วแค่ไหนขึ้นกับ gate ไม่ใช่วันปฏิทิน

### Tier overview (เรียงจากง่ายไปยาก)

| Tier | Name | Capability หลัก | Effort | Gate เข้า tier ถัดไป |
|---|---|---|---|---|
| **T1** | Read-only Q&A (Template) | ตอบสถานะจาก DB ด้วย template | S | confirm factual >95% |
| **T2** | Read-only Q&A (LLM-paraphrased) | LLM เรียบเรียงเป็นเพื่อน single-turn | M | thumbs-up ≥70%, hallucination <2% |
| **T3** | Text Quick Log (Chat Input → Transaction) | parse text → confirm card → mutate DB | M | confirm rate ≥85%, mis-categorize <15% |
| **T4** | Multi-turn Context + Action Execution | จำ context ใน session + สั่ง action อื่น (goal, reminder, budget) | L | multi-turn coherence ≥80%, action-accept ≥25% |
| **T5** | Multimodal Input (Photo/OCR → Chat) | ถ่ายใบเสร็จใน chat → parse + confirm | L | OCR accuracy ≥80% Thai receipt |
| **T6** | Voice + Proactive + Cross-session Memory | Thai STT, AI ทักก่อน, จำข้าม session | XL | nudge open rate ≥40%, voice WER <20% |
| **T7** | Coaching / Advice (Actionable) | Pattern C — คำแนะนำเจาะจงพร้อม action | L | action-accept 7-day ≥25%, hallucination <1% |
| **T-X** | **Anti-tier (Don't build yet)** | Emotional Support (Pattern D) | — | ดูเหตุผลด้านล่าง |

---

### T1 — Read-only Q&A (Template-based)

- **Capability**: ตอบคำถามสถานะตัวเองด้วย **structured template** ไม่มี LLM เรียบเรียง (เช่น *"เหลือ: 18,240฿ | ใช้ไป: 31,760฿ | งบเหลือ 12 วัน"*)
- **Complexity drivers**: ง่ายที่สุด — แค่ intent classify + DB query + string template. Deterministic, ไม่มี hallucination risk
- **Dependencies**: ไม่มี (entry tier)
- **Effort**: **S** (1 engineer × 2-3 สัปดาห์)
- **Key risks**: ตอบดูแข็งเหมือน bot / Thai users รู้สึก "ไม่ใช่เพื่อน" → **mitigation**: ยอมรับว่า tier นี้เป็น foundation ไม่ใช่ UX win — ให้ data layer + intent router ทำงานได้ก่อน
- **Exit criteria**: 
  - รองรับ 5 intent หลัก (balance, month spend, category spend, goal progress, obligation status)
  - Factual accuracy 100% (เทียบกับ DB direct query)
  - p95 latency <1s
- **User value**: ถามได้โดยไม่ต้องกด dashboard (ประหยัดเวลา แต่ยังไม่ "รู้สึก" เหมือนเพื่อน)

### T2 — Read-only Q&A (LLM-paraphrased, Single-turn)

- **Capability**: เหมือน T1 แต่ให้ LLM เรียบเรียงเป็น **friend voice** + เพิ่ม insight เล็กๆ (เช่น *"เหลือ 18K ใช้ได้อีก 12 วันสบายๆ นะ แต่หมวดช้อปใกล้ชนเพดานแล้ว 🙂"*) — single-turn เท่านั้น
- **Complexity drivers**: เริ่มมี LLM ใน loop → **hallucination risk**, prompt injection, cost control. Math ยังคง deterministic (inject ตัวเลขเข้า prompt, ห้าม LLM คำนวณ)
- **Dependencies**: T1 (reuse intent router + DB query)
- **Effort**: **M** (เพิ่ม prompt engineering + eval harness)
- **Key risks**: 
  - LLM มั่วตัวเลข → **mitigation**: separate "numbers layer" จาก "language layer" — ตัวเลขมาจาก DB, LLM แค่ห่อ
  - Tone ผิด (bank manager / shame) → **mitigation**: guardrail prompt + red-team 50 ตัวอย่าง/สัปดาห์
- **Exit criteria**:
  - Thumbs-up ≥70%
  - Hallucination rate <2% (weekly sample review)
  - Shame/judgment incidents = 0
- **User value**: รู้สึก "เห็นภาพ" + "เพื่อนพูด" ไม่ใช่ bot — นี่คือจุดที่ **moat language เริ่มทำงาน**

### T3 — Text Quick Log (Chat Input → Transaction Write)

- **Capability**: Pattern A — พิมพ์ *"กาแฟ 50 ข้าวเที่ยง 80"* → AI parse → confirm card → **write transaction จริง**
- **Complexity drivers**: **Jump ใหญ่** — เปลี่ยนจาก read-only → **mutation**. ต้อง:
  - NLU: entity extraction (amount, category, wallet) จากภาษาไทยไม่มี format
  - Confirm UX: ให้ user แก้ได้ก่อน commit
  - Undo/rollback mechanism
  - Learning loop: จำว่า user แก้หมวดอะไรบ่อย
- **Dependencies**: T2 (ต้องมี LLM layer ที่ stable ก่อน)
- **Effort**: **M** (แต่ surface area ใหญ่: parser + confirm UI + DB write + learning)
- **Key risks**:
  - Parse ผิดหมวด/จำนวน → user เสียข้อมูล → **mitigation**: confirm card บังคับก่อน commit, undo 30s
  - Ambiguous input (*"กาแฟกับข้าว 130"* — คือ 1 รายการหรือ 2?) → **mitigation**: LLM ถามกลับเมื่อ confidence <70%
- **Exit criteria**:
  - Confirm rate ≥85% (ไม่ต้องแก้)
  - Mis-categorize rate <15%
  - Share of daily transactions ที่มาจาก chat ≥20%
- **User value**: **นี่คือ killer feature ตาม J1** — บันทึก 3 รายการใน 5 วินาที

### T4 — Multi-turn Context + Action Execution (Non-transaction)

- **Capability**: จำ context ใน session (*"เมื่อกี้ฉันถามเรื่องกาแฟ"* → AI ยังจำ) + สั่ง action อื่น (*"ตั้ง goal ออม 5K"* → สร้าง goal จริง, *"เตือนฉันจ่ายบัตร"* → สร้าง reminder)
- **Complexity drivers**: 
  - **Conversation state management** (ต่างจาก T3 ที่ turn-by-turn)
  - **Tool-calling / function-calling** LLM pattern
  - **Permission model**: action ไหน confirm ได้ทันที, action ไหนต้อง explicit confirm (เช่น "ลบ goal" ต้อง confirm)
  - Idempotency: ถ้า user พูดซ้ำ ไม่สร้าง 2 อัน
- **Dependencies**: T3 (reuse mutation + confirm pattern)
- **Effort**: **L** (state management + tool-call orchestration + multiple action handlers)
- **Key risks**:
  - LLM สั่ง action ที่ user ไม่ได้ขอ (tool-call hallucination) → **mitigation**: whitelist action + confirm card สำหรับ mutation
  - State leak ข้าม session (ไม่ตั้งใจ) → **mitigation**: clear session on app restart, ยังไม่ cross-session ใน tier นี้
- **Exit criteria**:
  - Multi-turn coherence ≥80% (3-turn test set)
  - Action-accept rate ≥25% สำหรับ non-transaction actions
  - Zero unauthorized mutation incidents
- **User value**: chat กลายเป็น **universal command** — ไม่ต้องหาเมนู

### T5 — Multimodal Input: Photo/OCR

- **Capability**: ถ่ายใบเสร็จส่งใน chat → AI อ่าน → confirm card → write transaction (reuse T3 confirm flow)
- **Complexity drivers**:
  - **Thai OCR accuracy** (ใบเสร็จไทยมีหลาย format, ลายมือบางร้าน)
  - **Image pipeline**: upload, compress, store or transient?
  - **Cost**: OCR call per image > text token cost
  - **Privacy**: รูปใบเสร็จอาจมีข้อมูลอื่น (ชื่อ, ที่อยู่)
- **Dependencies**: T3 (reuse transaction write + confirm)
- **Effort**: **L** (OCR integration + image UX + cost gating)
- **Key risks**:
  - OCR accuracy ต่ำใน Thai receipts → **mitigation**: fallback ให้ user แก้ง่าย + learning loop
  - Cost blow-up → **mitigation**: free tier 5 scan/เดือน, Premium unlimited
- **Exit criteria**:
  - OCR field accuracy ≥80% (amount + merchant) on Thai receipt test set
  - Per-scan cost < ฿2 (to keep unit economics viable)
- **User value**: ไม่ต้องพิมพ์ — แก้ pain point "ขี้เกียจจด" ระดับสูงสุด

### T6 — Voice Input + Proactive Nudges + Cross-session Memory

- **Capability** (bundle 3 อันเพราะ complexity ใกล้กัน):
  - **Voice**: Thai STT → reuse T3 parser
  - **Proactive**: AI ทักก่อนตาม AI Insights trigger (ไม่รอ user เปิด chat)
  - **Cross-session memory**: จำ goal/preference ข้าม session (*"เดือนที่แล้วคุณบอกอยากออม 5K — เดือนนี้ทำได้ไหม?"*)
- **Complexity drivers**:
  - **Thai STT quality** (ภาษาพูด, dialect, background noise)
  - **Proactive = notification system** — timing, frequency cap, content quality
  - **Memory = privacy implication** — ต้องมี memory viewer + eraser + consent
- **Dependencies**: T4 (action execution for proactive), T3 (reuse parser)
- **Effort**: **XL** (3 big systems; อาจแตกย่อยเป็น T6a/T6b/T6c หลัง build T5)
- **Key risks**:
  - Notification fatigue → **mitigation**: max 2 proactive/week, user mute
  - Memory ผิด → **mitigation**: memory ที่ AI ใช้ต้องแสดงให้ user เห็นเป็น "fact card" แก้ได้
  - STT WER สูงในสภาพแวดล้อมจริง → **mitigation**: voice เป็น optional input ไม่ใช่ default
- **Exit criteria**:
  - Voice WER <20% on Thai daily speech
  - Proactive nudge open rate ≥40%, mute rate <15%
  - Memory correctness ≥90% (user spot-check)
- **User value**: จาก "เพื่อนที่ตอบเมื่อถาม" → **"เพื่อนที่จำและทักก่อน"**

### T7 — Coaching / Advice (Actionable, Deterministic Math)

- **Capability**: Pattern C — *"ปลดหนี้เร็วขึ้นได้ยังไง?"* → วิเคราะห์ cash flow + obligation → เสนอ **1 action ที่เจาะจง** (จำนวน + timing + first step) + สั่งตั้ง reminder/plan ได้
- **Complexity drivers**:
  - ต้องมี **financial computation layer** (debt avalanche/snowball simulator, savings projection) ที่ deterministic, ไม่ใช่ LLM คิด
  - Edge cases เยอะ (คนมีหนี้ 5 ก้อน, มี obligation + credit card + goal พร้อมกัน)
  - Compliance: อย่าข้ามไปคำแนะนำการลงทุน/ภาษี
- **Dependencies**: T4 (action execution), T6 (memory — coaching ต้องจำ plan ที่เคยเสนอ)
- **Effort**: **L** (financial sim engine + action plan generator + compliance guardrail)
- **Key risks**:
  - คำแนะนำผิด (math ผิด → ผู้ใช้เสียเงิน) → **mitigation**: sim engine มี unit test ครอบคลุม, LLM ห้ามแตะตัวเลข
  - คำแนะนำกว้างๆ ("ควรประหยัด") → **mitigation**: enforce schema "action ต้องมี {amount, category, timing, first step}"
- **Exit criteria**:
  - Action-accept rate 7-day ≥25%
  - User self-report "คำแนะนำใช้ได้จริง" ≥60%
  - Hallucination rate <1%
- **User value**: เปลี่ยนจาก "รู้สถานะ" → "รู้จะทำอะไรต่อ" — **นี่คือจุดที่ moat obligation+data ทำงานเต็มที่**

### T-X — Anti-tier: Emotional Support (Don't build yet)

- **Capability**: Pattern D — acknowledge ความรู้สึก, ระบาย, สะท้อนกลับ
- **ทำไม technical ง่าย**: ไม่มี math, ไม่มี mutation, แค่ prompt engineering + tone guardrail — looks like an easy T2-level feature
- **ทำไมห้ามสร้างก่อน**:
  1. **Product risk สูงสุดในทุก tier** — AI พูดผิดบริบทเรื่องอารมณ์ = ผู้ใช้รู้สึก "AI ไม่เข้าใจฉัน" → churn มากกว่า feature อื่นพัง
  2. **Feedback loop ช้า** — thumbs-up จับ "useful" ได้ แต่จับ "felt worse" ยาก (ต้อง survey + long-term retention)
  3. **Regulatory/ethical gray area** — ถ้าผู้ใช้พูดเรื่อง suicidal ideation ที่เกี่ยวกับหนี้ → AI ต้องมี protocol ไม่งั้น liability
  4. **Moat เท่ากับศูนย์** — ChatGPT ทำได้ดีพอ ในขณะที่ T3/T7 ลอกยาก (ต้องมี data + domain)
- **เงื่อนไขจะสร้าง**: หลัง T2-T7 เสถียร + มี qualitative research ≥20 user interview + มี safety protocol + มี human-in-loop reviewer
- **ในระหว่างนี้**: ถ้า user พิมพ์ข้อความ emotional (เช่น *"เครียด"*, *"กลัว"*) → AI แค่ **acknowledge สั้นๆ + redirect ไป actionable** (*"เข้าใจเลยครับ งั้นลองดูสถานการณ์ด้วยกันนะ — เดือนนี้เหลือ..."*) **ไม่เปิด emotional mode เต็มรูปแบบ**

---

### Mapping JTBD/Pattern ↔ Tier

| JTBD | Pattern | Tier ที่ทำได้ | หมายเหตุ |
|---|---|---|---|
| J1 "ช่วยจดให้" | A | **T3** (text), **T5** (photo), **T6** (voice) | สำคัญสุด — T3 ต้องเร็ว |
| J2 "เดือนนี้โอเคไหม" | B | **T1** (template) → **T2** (friend voice) | T2 คือจุดที่ "รู้สึก" |
| J3 "ผ่อนหนี้เร็วขึ้น" | C | **T7** | ต้อง sim engine |
| J4 "เหลือกี่บาท" | B | **T1/T2** | intent ง่าย, เลขชัด |
| J5 "ออมยังไง" | C + action | **T4** (ตั้ง goal) → **T7** (คำแนะนำจำนวน) | T4 ทำได้ partial, T7 ครบ |
| J6 "ทำไมใช้เยอะ" | B (insight) | **T2** + (optional proactive ใน **T6**) | T2 answer ได้ on-demand |
| J7 "กังวลเรื่องเงิน..." | D | **T-X** (don't build yet) | redirect เป็น actionable ใน T2+ |

**JTBD ที่ไม่มี tier รองรับ**: J7 — โดยตั้งใจ (ดู T-X rationale)

---

### Tier ที่แนะนำให้เริ่มก่อน: **T1 → T2 → T3** เป็น MVP bundle

**เหตุผล**:
1. **Riskiest assumption** = "Thai users จะยอมพิมพ์ chat แทนกด UI ไหม" — T3 (Quick Log) ตอบคำถามนี้ตรงที่สุด แต่ต้อง build บน T1+T2
2. **Moat เริ่มทำงานที่ T2** (language + data context) — ถ้า T2 ผ่าน thumbs-up ≥70% แปลว่า moat voice ใช้ได้จริง
3. **T3 เป็น volume driver** (J1 คาด >50% ของ chat volume) — ไม่มี T3 = chat จะ underused
4. **ไม่ควรเริ่มที่ T4/T7** — action execution ซับซ้อนและมี mutation risk สูงก่อนจะรู้ว่า user อยาก chat ไหม
5. **ไม่ควรเริ่มที่ T5 (OCR)** — มี OCR อยู่นอก chat แล้ว (AddTransactionsPage photo tab) — ทำ T5 ก่อน = reinvent

**MVP success gate** (ผ่านแล้วค่อยขยับ T4): 
- T3 confirm rate ≥85% × 4 สัปดาห์ติด
- WMCU ≥1.5
- D7 retention chat user > non-chat user ≥+5pp

---

### Tier Gate — เงื่อนไขผ่านจาก T_N → T_{N+1}

**Quantitative gate** (ทุก tier):
- Tier นั้นผ่าน exit criteria ต่อเนื่อง ≥4 สัปดาห์
- Hallucination rate <2% (strict <1% สำหรับ T7)
- p95 latency <5s
- No regression ใน tier ก่อนหน้า

**Qualitative gate** (ทุก tier):
- User interview ≥8 คน บอกว่า "ใช้ได้จริง" ไม่ใช่ "เจ๋งดี"
- Red team 50 ตัวอย่าง ไม่มี shame/judgment/hallucination incident
- Unit economics: cost per active chat user < ฿15/เดือน (Premium coverage)

**Kill criteria** (หยุด tier นั้น ไม่ไปต่อ):
- Thumbs-down rate >15%
- "Felt worse after chat" >1%
- Cost per user > ฿30/เดือน ไม่มี path ลง

---

## 7. Success Metrics

**North Star**: **Weekly Meaningful Conversations per User** (WMCU)
- นิยาม: chat session ที่มี ≥2 turn AND จบด้วย confirm action OR thumbs-up OR follow-up question
- Target Q1 หลัง launch: 1.5 WMCU/active chat user

**Leading indicators**:
- Chat adoption: % WAU ที่ใช้ chat ≥1 ครั้ง/สัปดาห์ (target ≥25%)
- Quick Log share of transactions: % transaction ที่มาจาก chat (target ≥20%)

**Guardrail metrics** (ต้องไม่แย่ลง):
- Hallucination rate <2% (sample-reviewed weekly)
- Response latency p50 <2s, p95 <5s
- Negative sentiment rate <5% (thumbs-down + explicit complaint)
- **"Feel worse after chat"** rate <1% (post-chat 1-question survey, sampling)
- Shame/judgment language incidents = 0 (red-team weekly)

---

## 8. Top 3 Open Questions (ต้อง align ก่อน commit)

1. **Scope MVP: รวม Pattern D (Emotional Support) ไหม?**
   Vision เน้น "แก้ปัญหาอารมณ์ก่อน" แต่ emotional support ทำพลาดได้ง่ายและอันตราย (พูดผิดบริบท → ผู้ใช้รู้สึกแย่กว่าเดิม) — ผมแนะนำเลื่อนไป V2 แต่ **ต้อง align กับ CEO** เพราะขัดกับ vision statement *"แก้ปัญหาอารมณ์ก่อนปัญหาข้อมูล"*

2. **Privacy model: AI เห็น data ทั้งหมดของผู้ใช้ หรือ opt-in per feature?**
   ถ้าเห็นทั้งหมด → ตอบได้แม่น แต่ต้อง consent flow ที่แรง / on-device หรือ cloud LLM? (cost vs privacy trade-off) — **ต้อง align กับ CTO + legal** ก่อนออกแบบ data pipeline

3. **Quick Log เป็น default input method ไหม?**
   ถ้าใช่ → chat กลายเป็น entry point หลักของแอป (แทน FAB+) — เปลี่ยน IA ของแอปเลย ถ้าไม่ → chat เป็น feature เสริม — **ต้อง align กับ CEO** เพราะมีผลต่อ north star ของทั้งแอป ไม่ใช่แค่ chat

---

*เอกสารนี้เป็น starting point — กรุณา challenge และปรับปรุงก่อน commit ลง dev*
