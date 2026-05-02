"""Auto-generate JSON Schema from Python callables for LLM tool descriptions."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints, Union, List, Dict


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    is_async: bool = False


_PY_TO_JSON_SCHEMA = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "list": {"type": "array"},
    "dict": {"type": "object"},
    "None": {"type": "null"},
    "Decimal": {"type": "string", "description": "Decimal number as string"},
}


def _get_json_type(py_type: Any) -> dict:
    """Convert Python type hint to JSON Schema type."""
    if py_type is None:
        return {"type": "null"}

    # Handle types from typing module
    origin = getattr(py_type, "__origin__", None)
    if origin is Union:
        # Optional[X] becomes union of X and null
        args = [a for a in getattr(py_type, "__args__", []) if a is not type(None)]
        if len(args) == 1:
            result = _get_json_type(args[0])
            result["type"] = [result.get("type", "string"), "null"]
            return result
        return {"type": "string"}

    if origin in (list, List):
        args = getattr(py_type, "__args__", [])
        if args:
            return {"type": "array", "items": _get_json_type(args[0])}
        return {"type": "array"}

    if origin in (dict, Dict):
        return {"type": "object"}

    # Handle str | None syntax
    if hasattr(py_type, "__args__"):
        args = py_type.__args__
        if len(args) == 2 and type(None) in args:
            non_none = next(a for a in args if a is not type(None))
            result = _get_json_type(non_none)
            if isinstance(result.get("type"), str):
                result["type"] = [result["type"], "null"]
            else:
                result["type"] = result.get("type", "string")
            return result

    # Handle types with no __origin__
    name = getattr(py_type, "__name__", str(py_type))
    if name in _PY_TO_JSON_SCHEMA:
        return dict(_PY_TO_JSON_SCHEMA[name])

    return {"type": "string"}


def _extract_use_when(doc: str) -> str | None:
    """Extract 'Use when:' clause from docstring, return None if not present."""
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Use when:"):
            return stripped[len("Use when:"):].strip()
    return None


def _extract_returns(doc: str) -> str | None:
    """Extract 'Returns:' section from docstring, return None if not present."""
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Returns:"):
            return stripped[len("Returns:"):].strip()
    return None


def _build_description_from_docstring(func: Callable, param_names: list[str], param_defaults: dict) -> str:
    """Build description from function docstring + parameter context."""
    doc = inspect.getdoc(func) or ""

    descriptions = []
    if doc:
        first_line = doc.split("\n")[0].strip()
        if first_line:
            descriptions.append(first_line)

        use_when = _extract_use_when(doc)
        if use_when:
            descriptions.append(f"Use when: {use_when}")

        returns = _extract_returns(doc)
        if returns:
            descriptions.append(f"Returns: {returns}")

    return " ".join(descriptions) if descriptions else f"Call {func.__name__}"


def build_tool_schema(name: str, func: Callable, is_async: bool = False) -> ToolDef:
    """Generate ToolDef from a Python callable.

    Extracts:
    - name: fully-qualified name (e.g. "wallet.get_all_balances")
    - description: from docstring or auto-generated
    - parameters: JSON Schema object with properties and required fields
    - is_async: whether the function is async
    """
    sig = inspect.signature(func)
    params = sig.parameters

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in params.items():
        if param_name in ("self", "cls"):
            continue

        # Get type hint
        try:
            hints = get_type_hints(func)
            param_type = hints.get(param_name, str)
        except Exception:
            param_type = str

        json_type = _get_json_type(param_type)

        # Get description from docstring if available
        description = ""
        doc = inspect.getdoc(func) or ""
        if "Args:" in doc:
            # Try to extract parameter description
            for line in doc.split("\n"):
                if param_name in line and ":" in line:
                    desc_part = line.split(":", 1)[1].strip()
                    if desc_part and not desc_part.startswith(param_name):
                        description = desc_part
                    break

        prop = {"description": description} if description else {}
        if isinstance(json_type, dict) and "type" in json_type:
            prop.update({k: v for k, v in json_type.items() if k != "description"})
        else:
            prop["type"] = json_type.get("type", "string")

        properties[param_name] = prop

        # Required if no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    description = _build_description_from_docstring(func, list(params.keys()), {})

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return ToolDef(
        name=name,
        description=description,
        parameters=schema,
        is_async=is_async,
    )


def build_sync_function_schema(name: str, func: Callable) -> ToolDef:
    """Build schema for a sync (non-async) function."""
    return build_tool_schema(name, func, is_async=False)


def build_async_method_schema(name: str, method: Callable) -> ToolDef:
    """Build schema for an async method."""
    return build_tool_schema(name, method, is_async=True)
