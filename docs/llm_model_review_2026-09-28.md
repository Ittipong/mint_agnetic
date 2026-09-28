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

## Round 3 (2026-09-28/29): cheap Chinese models, then a Gemini-only tune

### Where the money goes, by role

Session logs now carry `cost=` per call for every role (suggest, preamble,
classify, propose; REACT tokens + cost). 16-turn mixed set, Gemini, before
the chip cache:

| role | model | share | $ / 1k turns |
|---|---|---|---|
| suggestion chips | gemini-2.5-flash | 57% | 1.93 |
| REACT | gemini-3.1-flash-lite | 38% | 1.31 |
| classify / preamble / propose | gemini-2.5-flash-lite | 5% | 0.16 |

Chips were the biggest line because their ~2.3K-token fixed prompt was never
cached (`cached=0` on every call) and 2.5-flash output costs $2.50/1M.
REACT is cheap per call despite a 25K prompt: 97% of it is cached.

### Cheap Chinese models (OpenRouter, pinned to the vendor's own provider)

16-turn mixed set (numbers, add, advice; 14 checks). Gemini 3.1 Flash-Lite
baseline: 14/14, 7.9 s median.

| model | checks | median / p90 | verdict |
|---|---|---|---|
| qwen/qwen3.7-flash | 13/14 | 11.6 / 15.5 s | cheapest (~$0.48/1k REACT at official price), went to full test |
| tencent/hy3 | 14/14 | 11.6 / 16.0 s | 59% cache hit → dearer than Gemini at official price |
| xiaomi/mimo-v2.6-flash | 14/14 | 23.9 / 42.3 s | too slow |
| qwen/qwen3.8-flash | 13/14 | 12.8 / 30.4 s | read the month-end projection as cash left |
| z-ai/glm-5.3-flash | 12/14 | 39 / 114 s | reasoning cannot be disabled |
| stepfun/step-3.7-flash | — | — | reasoning cannot be disabled; stopped |

Qwen 3.7 Flash as REACT + chips on the full 5 sets: **65/69** (two wrong
figures, one failed add, one agent-loop error), turn median 21.3 s vs 6.2 s.
As chips it returned nothing: it thinks by default and spent all 700
max_tokens on thinking (200 OK, empty content), which OpenRouter `models[]`
does not treat as a failure. Rejected; owner chose Gemini only.

### Gemini-only tune (what shipped)

- REACT candidates: 2.5-flash-lite 8/14 (wrong balances/card figures);
  3.5-flash-lite needs reasoning and costs more per token than 3.1. **Keep
  3.1-flash-lite.**
- Chips: the fixed instructions now go in a system message with
  `cache_control`; the per-turn CONTEXT goes in the user message. 2280 tokens
  cached on every call → $0.0017 → $0.00095 per call, same model and quality.
- Chip model candidates (chip_chain, 4 seeds × 6 taps):

  | chips | $ / call | chains | notes |
  |---|---|---|---|
  | **2.5-flash + cache** | 0.00095 | 28/28 turns | varied; 2/80 lookup chips |
  | 2.5-flash-lite + cache | 0.00022 | 26/28 | repeats "เงินจะพอถึงสิ้นเดือนไหม" nearly every hop |
  | 2.5-flash-lite + "already offered" filter | 0.00022 | 22/28 | still repeats after 2 turns; more dead ends — filter dropped |
  | 3.1-flash-lite | 0.00122 | 27/28 | prompt too short to be cached → dearer than 2.5-flash |

- A chip reply that is empty or not JSON now retries once on the first
  fallback model while the 4 s budget lasts (`SUGGESTIONS_REASONING=off` is
  available for reasoning models).
- Fallbacks for every role are Gemini-only (owner's call: one vendor).

Result: about $3.4 → $2.5 per 1k turns (−26%) with the same models where
quality mattered. Next lever is chip output (5 candidates × 4 fields ≈ 265
tokens, 69% of a chip call).

### Chips: exactly 3, no spares (2026-09-29)

The prompt asked for 5 candidates and the app showed the first 3 that passed
the filters. It now asks for exactly 3 (one decide / whatif / ahead) and
drops the `repeats` field. max_tokens is 700 → 400. chip_chain on 2.5-flash:
output 265 → 190 tokens, $0.00095 → $0.00079 per call (−17%, about −7% of
the whole bill). 22/28 turns showed 3 chips, 6 showed 2, none showed 0. The
owner accepts 2.

Product-shopping chips ("แนะนำกองทุนรวม", "แนะนำหุ้น", "ควรลงทุนอะไรดี")
slipped past the prompt ban in several runs, so code now drops them
(`_PRODUCT_CHIP`). How much of their income to invest, and the provident
fund, are still allowed.
