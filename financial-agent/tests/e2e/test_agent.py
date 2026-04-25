import os
import pytest
from financial_agent.agent import FinancialCodeActAgent


pytestmark = pytest.mark.e2e

_TEST_USER_ID = os.getenv("TEST_USER_ID", "")


@pytest.fixture
def require_env():
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    if not _TEST_USER_ID:
        pytest.skip("TEST_USER_ID not set")


@pytest.mark.asyncio
async def test_agent_wallet_balance(require_env):
    async with FinancialCodeActAgent() as agent:
        result = await agent.solve("แสดง wallet ทั้งหมดและยอดคงเหลือของฉัน", user_id=_TEST_USER_ID)
    assert result.status in ("completed", "partial")


@pytest.mark.asyncio
async def test_agent_budget_status(require_env):
    async with FinancialCodeActAgent() as agent:
        result = await agent.solve("สรุป budget เดือนนี้ ใช้ไปเท่าไรแล้ว", user_id=_TEST_USER_ID)
    assert result.status in ("completed", "partial")


@pytest.mark.asyncio
async def test_agent_net_worth(require_env):
    async with FinancialCodeActAgent() as agent:
        result = await agent.solve("คำนวณ net worth รวม asset และหนี้สิน", user_id=_TEST_USER_ID)
    assert result.status in ("completed", "partial")
