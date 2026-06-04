"""Streaming adapter — translates LangGraph internal events into v2 SSE wire.

The mobile contract is FROZEN (`chat_api_datasource.dart::decodeBlockJson`):
ten block types, snake_case keys, CRLF-framed events, raw-text `answer_token`
+ `status_token` between block events. This package is the only piece of v3
that knows about that wire format — every other module emits / consumes
plain Python dicts and lets the adapter handle translation.

Public surface:
  - `block_emitter.EMITTABLE_BLOCK_TYPES` — list of 9 mobile-facing types.
  - `block_emitter.validate_block(d)` -> bool — A3 Hybrid validation gate.
  - `block_emitter.emit_block(d)` -> str — full SSE wire string for a block.
  - `sse_adapter.stream_chat(graph, init_state, config, *, thread_id)` —
    async generator yielding `{"event": ..., "data": ...}` dicts that
    `sse_starlette.EventSourceResponse` frames as CRLF-delimited events.

See `docs/v3/phase2_streaming_adapter.md` for the design contract and
`mobile/lib/data/datasources/remote/api/chat_api_datasource.dart`
(`_decodeSegment` + `decodeBlockJson`) for the consumer.
"""
