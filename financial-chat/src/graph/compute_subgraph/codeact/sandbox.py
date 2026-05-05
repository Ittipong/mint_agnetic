"""Restricted Python execution for Templates-CodeAct.

The validator walks the AST and rejects anything that could escape the
namespace: imports, attribute access on dunders, exec/eval, file/network
ops via builtins, etc. Allowed: assignments, arithmetic, comprehensions,
for/while/if, function defs, calls into the namespace.

Execution runs in the caller thread; the LLM-step node should wrap this
in `asyncio.to_thread(...)` so the main event loop stays responsive.
"""

from __future__ import annotations

import ast
import io
import contextlib
from typing import Any

from src.graph.compute_subgraph.codeact.exceptions import ClarificationNeeded


# ── AST validator ────────────────────────────────────────────────────────────


class _Forbidden(Exception):
    """Raised when the AST contains a banned construct."""


_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
)

# These would let user code reach into the interpreter (e.g. obj.__class__,
# cls.__subclasses__) so we ban any access whose attribute starts/ends with
# double underscores.
def _is_dunder(name: str) -> bool:
    return name.startswith("_") or name.endswith("_")


_BANNED_NAMES = frozenset({
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "breakpoint",
    "memoryview",
    "object",
    "type",
})


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise _Forbidden(f"banned node: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and _is_dunder(node.attr):
            raise _Forbidden(f"banned attribute access: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise _Forbidden(f"banned name: {node.id}")


# ── Restricted builtins ──────────────────────────────────────────────────────


def _denied_import(*_a, **_kw):
    raise RuntimeError("import is not allowed inside the sandbox")


_SAFE_BUILTINS = {
    # math / collection
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "reversed": reversed, "any": any, "all": all,
    "map": map, "filter": filter,
    # types
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset,
    # printing — captured into a string so the LLM can see it
    "print": print,
    # exceptions (allow user code to raise/handle expected ones)
    "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
    "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
    "AttributeError": AttributeError, "IndexError": IndexError,
    # constants
    "True": True, "False": False, "None": None,
    # dunders Python's exec implementation needs even when no class/import
    # appears in user code. `__import__` is stubbed to deny so accidental
    # uses don't crash with a confusing KeyError.
    "__build_class__": __builtins__["__build_class__"]  # type: ignore[index]
        if isinstance(__builtins__, dict) else __builtins__.__build_class__,
    "__import__": _denied_import,
    "__name__": "<sandbox>",
}


# ── Run ──────────────────────────────────────────────────────────────────────


def execute(code: str, namespace: dict[str, Any]) -> tuple[Any, str, str | None]:
    """Validate and execute `code` in `namespace`.

    Returns (result, stdout, error). `result` is taken from `namespace['result']`
    after execution — the LLM sets this when it has the final answer.
    `stdout` captures any print() output so the LLM can self-debug between steps.
    `error` is the formatted exception message if execution failed, else None.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return None, "", f"SyntaxError: {exc.msg} (line {exc.lineno})"

    try:
        _validate(tree)
    except _Forbidden as exc:
        return None, "", f"sandbox rejected code: {exc}"

    namespace.setdefault("__builtins__", _SAFE_BUILTINS)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            compiled = compile(tree, "<codeact>", "exec")
            exec(compiled, namespace)  # noqa: S102 — sandboxed via AST validator above
    except ClarificationNeeded:
        # Propagate up — the codeact_step node turns this into a
        # user-facing question and ends the loop.
        raise
    except Exception as exc:  # pylint: disable=broad-except
        return None, buf.getvalue(), f"{type(exc).__name__}: {exc}"

    return namespace.get("result"), buf.getvalue(), None
