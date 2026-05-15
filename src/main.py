"""CLI entry point — run the agent interactively.

Original ReAct agent (uses tools)
"""

import asyncio
import argparse

from src.graph.agent_graph import graph as react_graph
from src.config import settings

async def run_react_mode():
    """Original ReAct agent with tool calling."""
    print("Mint Agentic AI — ReAct Agent")
    print(f"Model: {settings.model}")
    print(f"DB: {settings.database_url}")
    print("Type 'exit' to quit\n")

    thread_id = "local-test"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        result = await react_graph.ainvoke(
            {"messages": [("user", user_input)], "user_id": "test-user-001"},
            config,
        )
        last = result["messages"][-1]
        print(f"AI: {last.content}\n")

async def main():
    parser = argparse.ArgumentParser(description="Mint Agentic AI - Financial Agent")
    args = parser.parse_args()

    await run_react_mode()


if __name__ == "__main__":
    asyncio.run(main())
