"""Central tool registry with auto-generated schemas for CodeAct agent."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, date, timedelta, timezone
from typing import Any

from financial_agent.tools.schema_builder import ToolDef, build_async_method_schema, build_sync_function_schema
from financial_agent.tools.wallet_tools import WalletTools
from financial_agent.tools.transaction_tools import TransactionTools
from financial_agent.tools.db_tool import DBTools
from financial_agent.tools.transaction_functions import (
    total_expenses,
    total_income,
    net_change,
    group_by_category,
    group_by_date,
    group_by_currency,
    sum_by_category,
    sum_by_currency,
    filter_by_tags,
    filter_by_type,
    balance_from_transactions,
)


@dataclass
class RegisteredTool:
    instance: Any  # The tool instance or function
    schema: ToolDef


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}
        self._namespace: dict[str, Any] | None = None

    def register(self, name: str, instance: Any, schema: ToolDef) -> None:
        """Register a tool with its instance and schema."""
        self._tools[name] = RegisteredTool(instance=instance, schema=schema)
        self._namespace = None  # invalidate cache

    def get_tool(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def get_tools(self) -> dict[str, RegisteredTool]:
        return dict(self._tools)

    def get_namespace(self) -> dict[str, Any]:
        """Return the execution namespace for the executor.

        Returns dict with tool instances and helper functions/classes.
        Caches result after first call.
        """
        if self._namespace is not None:
            return self._namespace

        namespace: dict[str, Any] = {}

        # Add tool instances (wallet, transaction, db)
        for name, registered in self._tools.items():
            prefix = name.split(".")[0]
            if prefix not in namespace:
                # Get the instance (it's the whole tool object, not individual methods)
                namespace[prefix] = registered.instance

        # Add sync functions
        namespace["total_expenses"] = total_expenses
        namespace["total_income"] = total_income
        namespace["net_change"] = net_change
        namespace["group_by_category"] = group_by_category
        namespace["group_by_date"] = group_by_date
        namespace["group_by_currency"] = group_by_currency
        namespace["sum_by_category"] = sum_by_category
        namespace["sum_by_currency"] = sum_by_currency
        namespace["filter_by_tags"] = filter_by_tags
        namespace["filter_by_type"] = filter_by_type
        namespace["balance_from_transactions"] = balance_from_transactions

        # Add built-ins
        namespace["Decimal"] = Decimal
        namespace["defaultdict"] = defaultdict
        namespace["datetime"] = datetime
        namespace["date"] = date
        namespace["timedelta"] = timedelta
        namespace["timezone"] = timezone
        namespace["context_vars"] = {}

        self._namespace = namespace
        return self._namespace

    def _format_tool_for_prompt(self, name: str, tool: RegisteredTool) -> str:
        """Format a single tool as markdown for system prompt."""
        schema = tool.schema
        lines = [f"### {name}"]

        if schema.description:
            lines.append(f"{schema.description}")

        if schema.parameters.get("properties"):
            lines.append("**Parameters:**")
            for param_name, param_def in schema.parameters["properties"].items():
                param_type = param_def.get("type", "any")
                description = param_def.get("description", "")
                required = param_name in schema.parameters.get("required", [])
                required_str = "(required)" if required else "(optional)"
                if description:
                    lines.append(f"- `{param_name}`: {param_type} {required_str} — {description}")
                else:
                    lines.append(f"- `{param_name}`: {param_type} {required_str}")

        lines.append(f"**Returns:** `{schema.parameters.get('type', 'object')}`")
        return "\n".join(lines)

    def get_schema_for_prompt(self) -> str:
        """Render all tool schemas as markdown for system prompt."""
        # Group by prefix
        groups: dict[str, list[tuple[str, RegisteredTool]]] = defaultdict(list)
        for name, tool in sorted(self._tools.items()):
            prefix = name.split(".")[0]
            groups[prefix].append((name, tool))

        sections = []

        # Async tools section
        async_tools = {n: t for n, t in self._tools.items() if t.schema.is_async}
        if async_tools:
            sections.append("## ASYNC TOOLS (must `await`)")
            for name, tool in sorted(async_tools.items()):
                sections.append(self._format_tool_for_prompt(name, tool))
                sections.append("")

        # Sync functions section
        sync_tools = {n: t for n, t in self._tools.items() if not t.schema.is_async}
        if sync_tools:
            sections.append("## SYNC FUNCTIONS (no `await` needed)")
            for name, tool in sorted(sync_tools.items()):
                sections.append(self._format_tool_for_prompt(name, tool))
                sections.append("")

        return "\n".join(sections)


def build_tool_registry(session_factory, user_id: str) -> ToolRegistry:
    """Build a tool registry with all available tools auto-discovered."""
    registry = ToolRegistry()

    # Instantiate tool classes
    wallet = WalletTools(session_factory, user_id)
    transaction = TransactionTools(session_factory, user_id)
    db = DBTools(session_factory, user_id)

    # Register WalletTools async methods
    for method_name in dir(wallet):
        if method_name.startswith("_"):
            continue
        method = getattr(wallet, method_name)
        if not callable(method):
            continue
        schema = build_async_method_schema(f"wallet.{method_name}", method)
        registry.register(f"wallet.{method_name}", wallet, schema)

    # Register TransactionTools async methods
    for method_name in dir(transaction):
        if method_name.startswith("_"):
            continue
        method = getattr(transaction, method_name)
        if not callable(method):
            continue
        schema = build_async_method_schema(f"transaction.{method_name}", method)
        registry.register(f"transaction.{method_name}", transaction, schema)

    # Register DBTools async methods
    for method_name in dir(db):
        if method_name.startswith("_"):
            continue
        method = getattr(db, method_name)
        if not callable(method):
            continue
        schema = build_async_method_schema(f"db.{method_name}", method)
        registry.register(f"db.{method_name}", db, schema)

    # Register sync functions
    sync_funcs = {
        "total_expenses": total_expenses,
        "total_income": total_income,
        "net_change": net_change,
        "group_by_category": group_by_category,
        "group_by_date": group_by_date,
        "group_by_currency": group_by_currency,
        "sum_by_category": sum_by_category,
        "sum_by_currency": sum_by_currency,
        "filter_by_tags": filter_by_tags,
        "filter_by_type": filter_by_type,
        "balance_from_transactions": balance_from_transactions,
    }

    for name, func in sync_funcs.items():
        schema = build_sync_function_schema(name, func)
        registry.register(name, func, schema)

    return registry
