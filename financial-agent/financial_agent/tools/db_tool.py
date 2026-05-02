import re
from decimal import Decimal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

_MAX_ROWS = 200

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke)\b",
    re.IGNORECASE,
)


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql)


def _to_decimal(value):
    if value is None:
        return None
    if isinstance(value, float):
        return Decimal(str(value))
    try:
        return Decimal(str(value))
    except Exception:
        return value


class DBTools:
    def __init__(self, session_factory: async_sessionmaker, user_id: str):
        self._sf = session_factory
        self._user_id = user_id

    async def raw_sql(self, query: str, params: dict | None = None) -> list[dict]:
        stripped = _strip_comments(query)
        match = _FORBIDDEN_RE.search(stripped)
        if match:
            raise ValueError(f"Only SELECT queries allowed. Found forbidden keyword: '{match.group()}'")

        # Always inject user_id so LLM can use :user_id in WHERE without passing it manually.
        # Caller-supplied params take lower precedence — user_id is always the authenticated value.
        merged_params = {**(params or {}), "user_id": self._user_id}

        async with self._sf() as session:
            result = await session.execute(text(query), merged_params)
            rows = result.mappings().all()

        if len(rows) > _MAX_ROWS:
            raise ValueError(
                f"Query returned {len(rows)} rows, which exceeds the {_MAX_ROWS}-row limit. "
                "Add a more specific WHERE clause or LIMIT to narrow the result set."
            )

        converted = []
        for row in rows:
            converted_row = {}
            for key, val in dict(row).items():
                if isinstance(val, float):
                    converted_row[key] = Decimal(str(val))
                else:
                    converted_row[key] = val
            converted.append(converted_row)
        return converted
