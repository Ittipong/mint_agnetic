"""Endpoint pre-processors — short-circuit non-text input before the ReAct loop.

Two surfaces, both wired by Wave 6's `server.py`:
  - `voice_handler.handle_voice_chat` — Gemini STT m4a -> transcript -> ReAct
  - `slip_handler.handle_slip_chat` — Gemini vision -> transaction_proposal_group
    block (NO ReAct, per decision alpha in the v3 spec)

Both emit their own SSE event stream (status_token / block / done) that mobile
decodes via the same `chat_api_datasource.dart` path as the text endpoint —
voice precedes the ReAct stream with a `transcript` block, slip skips ReAct
entirely with a single `transaction_proposal_group` block.

Voice fast-path (the v2 merged STT+classify call) is REMOVED in v3 — pure
ReAct does its own classification implicitly through tool calls (per memory
`project_voice_fast_path`).
"""
