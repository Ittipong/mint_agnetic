"""CLI entry point — run the agent interactively.

Supports two modes:
- react: Original ReAct agent (uses tools)
- chat: Phase 1 Chat graph (template-based)
"""

import asyncio
import argparse

from src.graph.agent_graph import graph as react_graph
from src.graph.chat_graph import chat_graph
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


async def run_chat_mode():
    """Phase 1 Chat graph — template-based Q&A."""
    print("Mint Agentic AI — Phase 1 Chat (Template-based)")
    print(f"Writer Model: {settings.model}")
    print(f"Intent Model: {settings.intent_model}")
    print(f"Backend DB: {settings.backend_database_url}")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        result = await chat_graph.ainvoke(
            {
                "user_id": "test-user-001",
                "user_message": user_input,
                "intent": None,
                "entities": {},
                "confidence": 0.0,
                "numbers": {},
                "message": "",
                "chips": [],
                "error": None,
            },
        )

        print(f"Intent: {result.get('intent')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Numbers: {result.get('numbers')}")
        print(f"AI: {result.get('message')}")
        print(f"Chips: {result.get('chips')}")
        print()


async def main():
    parser = argparse.ArgumentParser(description="Mint Agentic AI")
    parser.add_argument(
        "--mode",
        choices=["react", "chat"],
        default="chat",
        help="Agent mode: react (tool-calling) or chat (template-based)",
    )
    args = parser.parse_args()

    if args.mode == "react":
        await run_react_mode()
    else:
        await run_chat_mode()


if __name__ == "__main__":
    asyncio.run(main())
