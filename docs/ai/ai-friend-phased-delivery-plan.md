# AI Financial Friend: Phased Delivery Plan

> **Version**: 2.0 (Chat-first revision)
> **Author**: Product Owner
> **Last Updated**: 2026-04-20

---

## Overview

Vision ต้องการให้ AI เป็น "เพื่อน" ที่เข้าใจสถานการณ์ — ไม่ใช่ tool ที่ให้กรอกตัวเลข

**Key Change v2**: Phase 1 เปลี่ยนจาก "AI Record Helper" เป็น **"Chat Q&A"** — AI ตอบคำถามเรื่องข้อมูลการเงินตัวเองก่อน

**เหตุผล**:
- Q&A = read-only, zero mutation risk
- Q&A ทำงานได้ Day 1 แม้ user ยังไม่มี transaction
- validate ว่า "users ยอมพิมพ์ chat" ก่อน invest ใน Quick Log parsing

---

## Phase 1: Chat-First Q&A (เดือน 0-3)

### What to Build

**5 intents หลักที่ chat ตอบได้:**

| Intent | ตัวอย่างคำถาม | สิ่งที่ตอบ |
|--------|----------------|------------|
| 1. เดือนนี้ใช้ไปเท่าไหร่ | "เดือนนี้ใช้ไปเท่าไหร่?" | spent vs budget, days left, daily allowance |
| 2. สรุปรายจ่ายสัปดาห์นี้ | "สรุปรายจ่ายสัปดาห์นี้หน่อย" | 7-day category breakdown |
| 3. หนี้ยังเหลือเท่าไร | "หนี้ยังเหลือเท่าไร?" | obligation balance + months remaining |
| 4. Goal ออมเงินเดือนนี้ | "goal ออมเงินเดือนนี้เป็นยังไง?" | progress + projected completion |
| 5. กี่วันก่อนเงินเดือนเข้า | "กี่วันก่อนเงินเดือนเข้า?" | cash flow positioning |

**Additional Patterns:**
- Pattern E (Onboarding): greeting + 3 suggested chips
- Partial emotional acknowledgment: *"เครีียด"* → acknowledge สั้นๆ + redirect to actionable

### ไม่เอาใน Phase 1

- Quick Log (Pattern A)
- OCR Receipt
- Voice Transaction
- Proactive nudge
- Coaching

### Technical Approach

- **Intent classifier**: rule-based (keyword/regex) + Haiku LLM fallback สำหรับ ambiguous input
- **Data layer**: DB query only — LLM ไม่คำนวณตัวเลข (eliminates hallucination บน numbers)
- **LLM**: structured output (JSON mode) + friend-voice system prompt + anti-shame guardrail
- **Architecture**: Chat screen → Intent classify → DB query → Prompt builder (inject numbers) → LLM → display with follow-up chips
- **p95 latency target**: <5s

### Success Metrics

| Metric | เป้า |
|--------|------|
| Chat WAU / Total WAU | >= 25% by week 4 |
| WMCU (weekly meaningful conversations) | >= 1.5 by week 8 |
| Thumbs-up rate | >= 70% |
| Hallucination rate | < 2% |
| D7 retention delta (chat vs non-chat) | >= +5pp |

### Persona Coverage

| Persona | Use Case ใน Phase 1 |
|---------|---------------------|
| **อนุชา** (32, Account Manager) | ถามหนี้ได้ทันทีโดยไม่ต้องเปิดหน้า Obligation |
| **เอมม่า** (28, Junior Marketer) | ถาม goal progress ได้ทันทีโดยไม่ต้องกด drill-down |
| **สมชาย** (45, Small biz owner) | สรุปสัปดาห์ได้ใน 1 คำถาม — ไม่ต้องกด reports |

---

## Phase 2: + Quick Log + OCR (เดือน 3-6)

### What to Build

- **Pattern A (Quick Log)**: text input → parse to transactions ("กาแฟ 50 ข้าวกลาง 60 bts 34" → 3 transactions)
- **OCR in chat**: ถ่ายใบเสร็จ → อ่านร้าน/ยอด/วันที่ → บันทึก
- **Learning loop**: remember category corrections

### Why Phase 2

- Quick Log requires NLP parser (entity extraction: amount + category + wallet + merchant จากภาษาไทย) — complexity สูงกว่า Q&A มาก
- Phase 1 ทำให้ intent classifier + LLM wrapper + confirm card UI พร้อม — Quick Log สร้างบน foundation นั้น
- Phase 1 ยัง validate ว่า "users ยอมพิมพ์ chat" ก่อนจะ invest ใน parsing layer

### Gate to Ship

- Confirm rate >= 85%
- Mis-categorize < 15%
- Tx share via chat >= 20%

---

## Phase 3: + Insights + Recommendations + Voice (เดือน 6-12)

### What to Build

- **AI Brief (Proactive)**: Nightly push 21:00 — "เดือนนี้เป็นไง summary"
- **AI Recommendations**: 1 recommendation ต่อสัปดาห์ (rule-based + LLM tone)
- **Coaching simulation engine**: debt avalanche/snowball, savings projection
- **Thai Voice input**: speech → transaction

### Why Phase 3

- Proactive requires 30+ days data = post-Phase 2
- Coaching requires deterministic simulation engine — separate subsystem
- Voice requires Thai STT integration

---

## Phase 4: B2B + Open Banking (เดือน 12-24)

### What to Build

- **Open Banking sync**: ดึง statement จากทุกธนาคารอัตโนมัติ
- **B2B Dashboard**: white-label AI insights สำหรับ HR/finance teams
- **Family wallet**: กระเป๋าร่วม + employer benefit plan
- **SEA Expansion**: เวียดนาม + อินโดนีเซีย

---

## Prioritization Rationale

### ลำดับนี้เพราะอะไร

```
Chat Q&A  →  ลด friction ได้เยอะที่สุดสำหรับ query
     ↓
Quick Log + OCR  →  foundation พร้อมจาก Phase 1 สร้างต่อ
     ↓
Insights + Recommendations  →  มี data แล้ว + proactive value
     ↓
Voice  →  natural next step หลัง text + OCR prove แล้ว
     ↓
B2B + Open Banking  →  longest lead time
```

### หลักการตัดสินใจ

1. **Read-first, Write-later** — Q&A ปลอดภัย (read-only, zero mutation risk)
2. **Validate ก่อน invest** — ถ้า Q&A ล้ม (users ไม่ chat), Quick Log ก็ไม่มี point
3. **Foundation reuse** — intent classifier + LLM wrapper จาก Phase 1 ใช้ต่อใน Phase 2+
4. **Rule-based ก่อน LLM** — ship เร็ว, fail fast, ก่อน invest ใน LLM infrastructure

---

## Open Questions (Pre-Phase 1)

ต้อง align ก่อนเริ่ม Phase 1:

1. **Privacy model** — AI เห็น data ทั้งหมด opt-in หรือ per-feature consent?
2. **Onboarding chip placement** — 3 chips แรกอยู่ตรงไหนของ UX flow?
3. **Nightly Brief timing** — 21:00 ตรงไหล่หรือปรับ?

---

## Roadmap Summary

```
Phase 1 (0-3 เดือน)     Phase 2 (3-6 เดือน)      Phase 3 (6-12 เดือน)     Phase 4 (12-24 เดือน)
━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━
Chat Q&A (5 intents)      Quick Log + OCR            AI Brief (proactive)      Open Banking
Onboarding chips          Learning loop              Recommendations +         B2B Dashboard
Emotional ack            OCR enhanced                 Coaching                 SEA Expansion
                                                    Thai Voice (beta)
```

ทุก phase มี "เพื่อน" moment ที่สำคัญ:
- **Phase 1**: "AI รู้เรื่องเงินของเรา" (Q&A)
- **Phase 2**: "AI ช่วยจดให้" (Quick Log + OCR)
- **Phase 3**: "AI รู้ก่อนที่เราจะรู้ตัว" (proactive)
- **Phase 4**: "AI ดูแลเงินให้เราเป็นระบบ" (autonomous)
