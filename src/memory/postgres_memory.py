"""PostgresSaver checkpointer — persists conversation history per thread."""

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import make_url
from src.config import settings


def get_checkpointer() -> PostgresSaver:
    """Create a synchronous PostgresSaver for LangGraph Studio."""
    url = make_url(settings.database_url)
    return PostgresSaver.from_conn_string(settings.database_url)


async def get_async_checkpointer() -> AsyncPostgresSaver:
    """Create an async PostgresSaver for programmatic use."""
    return AsyncPostgresSaver.from_conn_string(settings.database_url)
