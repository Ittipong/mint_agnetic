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

## Other OpenRouter models that could replace Gemini (same day)

Screen: OpenRouter `/models` catalog, filtered for tools + response_format,
≥128K context, and a price in this tier. Excluded: China-hosted (DeepSeek,
Qwen, Xiaomi, GLM, InclusionAI; PDPA), `:batch` models (async), and models
far too small. Claude Haiku 4.5 was also dropped at 4× today's price.

Quick screen on advisor + starters (28 turns, 23 checks), reasoning off.
The served model was verified from `model=` in the session logs.

| model | result | why |
|---|---|---|
| **openai/gpt-6-luna** | 21/23 (both misses correct in substance) | finalist |
| google/gemini-2.5-flash-lite | 9/23 | not capable enough |
| upstage/solar-mini4 | 9/23, p90 31 s | not capable enough, single provider |
| mistralai/mistral-small-2603 | Qwen answered 74/84 calls | "rate-limited upstream" on OpenRouter's shared quota |
| openai/gpt-5-nano | every call failed | "Reasoning is mandatory" — always thinks, so slow |

Full 5 sets (112 turns, 69 checks), same harness as the A/B above:

| | Gemini 3.1 Flash-Lite + cache | **GPT-6 Luna + cache** |
|---|---|---|
| checks | 69/69 | 66/69 — every number correct; one weaker answer* |
| $ per 1k turns (all roles) | 4.06 | **2.78 (−32%)** |
| turn median / p90 | 6.2 s / 9.0 s | 6.5 s / 9.7 s |
| ReAct call median / p90 | 2.1 s / 4.3 s | 2.4 s / 4.1 s |
| cache hits | 100% | 100% |
| served by | Google | OpenAI / Azure (7 endpoints) |

\*"ใช้เงินเกินทุกเดือนเลยไหม": Luna pulled a 3-month total, then said it
could not tell per month instead of fetching the months. The other two misses
state correct numbers in another form (220,314 = 6 × 36,719; 151,362 / 126,648).

Luna is the only replacement that clears the bar. It is released 2026-09 (new),
so pricing and behaviour may still shift.
