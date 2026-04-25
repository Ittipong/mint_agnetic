"""Unit tests for DBTools.raw_sql security constraints."""
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_agent.tools.db_tool import DBTools, _MAX_ROWS


def _make_tool(user_id: str = "user-abc") -> DBTools:
    sf = MagicMock()
    return DBTools(sf, user_id)


def _mock_session(rows: list[dict]):
    """Return a session_factory whose execute() yields the given rows."""
    mapping_rows = [dict(r) for r in rows]

    result = MagicMock()
    result.mappings.return_value.all.return_value = mapping_rows

    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    sf = MagicMock()
    sf.return_value = session
    return sf


# --- forbidden keyword blocking ---

@pytest.mark.parametrize("sql", [
    "INSERT INTO transactions VALUES (1)",
    "UPDATE wallets SET balance = 0",
    "DELETE FROM transactions",
    "DROP TABLE wallets",
    "TRUNCATE transactions",
    "ALTER TABLE wallets ADD COLUMN x INT",
    "CREATE TABLE evil (id INT)",
    "GRANT ALL ON wallets TO public",
    "REVOKE SELECT ON wallets FROM user1",
])
async def test_forbidden_keywords_raise(sql: str):
    tool = _make_tool()
    with pytest.raises(ValueError, match="Only SELECT queries allowed"):
        await tool.raw_sql(sql)


# --- comment-based evasion is blocked ---

async def test_inline_comment_hides_delete():
    tool = _make_tool()
    with pytest.raises(ValueError, match="Only SELECT queries allowed"):
        await tool.raw_sql("SELECT 1; -- safe\nDELETE FROM transactions")


async def test_block_comment_hides_insert():
    tool = _make_tool()
    with pytest.raises(ValueError, match="Only SELECT queries allowed"):
        await tool.raw_sql("SELECT /* innocent */ 1; INSERT INTO t VALUES (1)")


# --- user_id auto-injection ---

async def test_user_id_injected_into_params():
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.mappings.return_value.all.return_value = []
    session.execute.return_value = result_mock
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    sf = MagicMock()
    sf.return_value = session

    tool = DBTools(sf, "user-xyz")
    await tool.raw_sql("SELECT 1 WHERE user_id = :user_id")

    _, call_params = session.execute.call_args[0]
    assert call_params["user_id"] == "user-xyz"


async def test_user_id_cannot_be_overridden_by_caller():
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.mappings.return_value.all.return_value = []
    session.execute.return_value = result_mock
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    sf = MagicMock()
    sf.return_value = session

    tool = DBTools(sf, "real-user")
    await tool.raw_sql("SELECT 1", params={"user_id": "evil-user"})

    _, call_params = session.execute.call_args[0]
    assert call_params["user_id"] == "real-user"


# --- row cap ---

async def test_row_cap_raises_when_exceeded():
    oversized = [{"id": i} for i in range(_MAX_ROWS + 1)]
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.mappings.return_value.all.return_value = oversized
    session.execute.return_value = result_mock
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    sf = MagicMock()
    sf.return_value = session

    tool = DBTools(sf, "user-1")
    with pytest.raises(ValueError, match=str(_MAX_ROWS)):
        await tool.raw_sql("SELECT * FROM transactions")


async def test_row_cap_ok_at_limit():
    at_limit = [{"id": i} for i in range(_MAX_ROWS)]
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.mappings.return_value.all.return_value = at_limit
    session.execute.return_value = result_mock
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    sf = MagicMock()
    sf.return_value = session

    tool = DBTools(sf, "user-1")
    rows = await tool.raw_sql("SELECT * FROM transactions LIMIT 500")
    assert len(rows) == _MAX_ROWS


# --- float → Decimal conversion ---

async def test_float_columns_converted_to_decimal():
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.mappings.return_value.all.return_value = [{"amount": 1234.56}]
    session.execute.return_value = result_mock
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    sf = MagicMock()
    sf.return_value = session

    tool = DBTools(sf, "user-1")
    rows = await tool.raw_sql("SELECT amount FROM transactions")
    assert isinstance(rows[0]["amount"], Decimal)
    assert rows[0]["amount"] == Decimal("1234.56")
