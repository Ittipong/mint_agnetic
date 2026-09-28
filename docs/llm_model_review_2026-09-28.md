# LLM model review — DeepSeek vs current Gemini (2026-09-28)

**Question:** should the chat move to DeepSeek (same size tier) for price and
quality, including the ReAct + CodeAct loop?

**Decision: stay on Gemini 3.1 Flash-Lite and keep prompt caching on.**
Re-check DeepSeek when a non-China provider offers it with prompt caching at a
similar price, or if richer advisor answers become the priority.

## Where the money goes

The REACT model is about 97% of LLM spend. Each call sends ~25K input tokens
and returns ~195. Classify, propose and the codeact resolver together cost about
$0.15/day. **Gemini was not caching at all.** Adding `cache_control` to the
static system prompt, and moving today/user_id to a tail section, cut one call
from $0.00587 to $0.00061 (commit 92eeef4).

## A/B on the eval sets (5 sets, 112 turns, over the tunnel)

| arm | correct | $ per 1k turns (all roles) | turn median / p90 | out tokens/call |
|---|---|---|---|---|
| Gemini 3.1 Flash-Lite, no cache (before) | 69/69 | ~10.5+ (REACT alone) | — | 195 |
| **A: Gemini + cache** | **69/69** | **4.06** | 6.2 s / 9.0 s | 195 |
| B: DeepSeek v4.1-flash (reasoning on, the default) | 69/69 | 6.83 | 9.4 s / 26.4 s | 908 |
| C: DeepSeek v4.1-flash, reasoning off | 69/69 in substance* | 3.95 | 6.1 s / 9.5 s | lower |

\*C's two check misses were correct answers in a different form: a full
6-month income/expense table, and "due 24,111 + new 832" instead of the
24,943 total. DeepSeek's answers were more detailed and table-heavy.

CodeAct: in this app CodeAct is the REACT model writing `run_python`. Every arm
wrote correct code for all checked numbers. The separate `codeact` resolver
role (gemini-2.5-flash-lite) costs about $0.02/day, so there is nothing to gain
there.

## Why not switch

- **Savings are ~3% once caching is on.** Caching was the real lever.
- **PDPA.** DeepSeek is only this cheap, and only caches, on DeepSeek's own
  endpoint (China). The default routing (Parasail/Together) cost more than
  Gemini per call. Many other providers run fp4/fp8.
- **Defaults are risky.** Reasoning is on by default: 4.7× the output tokens
  and a 26 s p90. It must be pinned off.

## Switches left in the code (inactive unless set)

`REACT_PROVIDER_ORDER` (e.g. `deepseek`) and `REACT_REASONING=off`. Changing
`.env` needs `launchctl kickstart -k gui/$(id -u)/uk.minttechdev.chat-agent-v3`;
touching a `.py` does not reload `.env`. Check `model=` on react.llm log lines.
