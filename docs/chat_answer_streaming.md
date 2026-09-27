# Chat answer streaming (2026-09-27)

## Why

Analyst answers took 6–14 s. The agent spent ~80 ms running the query and
~7 s writing the answer, but the flat ReAct node had token streaming turned
off (`disable_streaming=True`), so the user watched a frozen "กำลังคำนวณ..."
pill for ~7 s and then got the whole answer at once. That silence is what
felt like "ค้าง".

The agent node now streams. Measured through the tunnel on the question
"เดือนนี้ใช้จ่ายหมวดไหนเยอะสุด เทียบกับเดือนก่อน":

| | before | after |
|---|---|---|
| first answer text | 11.2 s | ~6 s |
| longest moment with nothing new on screen | ~7 s | ~2 s |

## The one rule that makes it safe

Gemini writes the narration sentence ("กำลังคำนวณยอดให้นะครับ") **before**
the tool_call of the same message. Routing each chunk on its own would put
that narration into the answer bubble. So `sse_adapter._MessageRouter` holds
a message's opening text until a tool_call shows it is narration, or 160
characters with no tool_call show it is the answer. Narration is capped at
5–12 words by the prompt. If a narration ever runs past 160 characters, the
session log says `leaked into answer`.

## Clients

- Flutter already rendered `answer_token` live. The only change: the status
  pill hides once a narration line exists, because it says the same thing in
  a vaguer form.
- prototype_v2 `chat.js` now builds the answer bubble from `answer_token`
  (with a caret), and the mock path types its answers out the same way.
