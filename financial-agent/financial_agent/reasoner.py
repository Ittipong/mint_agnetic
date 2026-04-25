import asyncio
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI
from loguru import logger
from financial_agent.config import Settings
from financial_agent.executor import ExecutionResult


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class ReasonerResult:
    code: str
    raw: str


class Reasoner:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://mint-money.app",
                "X-Title": "Financial CodeAct Agent",
            },
        )
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # Application-level cache: system prompt built once, reused every step
        self._cached_system_prompt: str | None = None

    def set_system_prompt(self, prompt: str) -> None:
        self._cached_system_prompt = prompt
        logger.debug("System prompt cached ({} chars)", len(prompt))

    def _render_system_prompt(self, db_schema_summary: str) -> str:
        tmpl = self._jinja.get_template("system_prompt.j2")
        return tmpl.render(
            db_schema_summary=db_schema_summary,
            today=date.today().isoformat(),
        )

    def _render_action_prompt(
        self,
        task: str,
        history: list[dict],
        step: int,
    ) -> str:
        history_ctx = []
        for h in history:
            res: ExecutionResult = h["result"]
            history_ctx.append({
                "step": h["step"],
                "code": h["code"],
                "result": {
                    "status": res.status,
                    "error": res.error,
                    "result": res.result,
                },
            })
        tmpl = self._jinja.get_template("action_program.j2")
        return tmpl.render(task=task, history=history_ctx, step=step)

    async def generate_code(
        self,
        task: str,
        history: list[dict],
        step: int,
        complex: bool = False,
    ) -> ReasonerResult:
        if self._cached_system_prompt is None:
            raise RuntimeError("set_system_prompt() must be called before generate_code()")

        action_prompt = self._render_action_prompt(task, history, step)

        # System prompt as first user message for stable prefix (maximizes provider KV cache)
        messages = [
            {"role": "user", "content": self._cached_system_prompt},
            {"role": "assistant", "content": "Understood. I will generate Python code using the namespace and follow all financial precision rules."},
            {"role": "user", "content": action_prompt},
        ]

        model = self._settings.complex_model if complex else self._settings.model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
        }

        # Qwen3/3.5/coder models: disable thinking by default, enable only for complex tasks
        if "qwen3" in model.lower() or "qwen/qwen3" in model.lower() or "qwen3.5" in model.lower():
            if complex:
                kwargs["extra_body"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self._settings.thinking_budget_tokens,
                    },
                }
            else:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        logger.debug("Calling LLM step={} complex={} model={}", step, complex, model)
        logger.debug("Action prompt:\n{}", action_prompt)

        # Hard timeout per LLM call — prevents runaway thinking models
        response = await asyncio.wait_for(
            self._client.chat.completions.create(**kwargs),
            timeout=90,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("LLM raw response:\n{}", raw)
        return ReasonerResult(code=self._extract_code(raw), raw=raw)

    @staticmethod
    def _extract_code(text: str) -> str:
        matches = _CODE_BLOCK_RE.findall(text)
        if matches:
            return matches[-1].strip()
        # Fallback: return full text (might be raw code without fences)
        return text.strip()
