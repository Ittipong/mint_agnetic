"""Shared pytest fixtures for AI Friend agent tests."""

import os
import pytest
from pathlib import Path

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.fixture
def studio_url() -> str:
    """LangGraph Studio URL."""
    return os.getenv("STUDIO_URL", "http://localhost:8080")


@pytest.fixture
def default_user_id() -> str:
    """Default test user ID."""
    return os.getenv("TEST_USER_ID", "ba91d8a5-46b2-46f7-aaf4-189a54e17fe9")


@pytest.fixture
def langsmith_api_key() -> str:
    """LangSmith API key."""
    return os.getenv("LANGSMITH_API_KEY", "")


@pytest.fixture
def dataset_name() -> str:
    """Default dataset name for evaluation."""
    return "ai_friend_phase1"
