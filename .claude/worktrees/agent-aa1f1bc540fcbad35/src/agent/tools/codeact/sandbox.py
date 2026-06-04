"""Restricted Python execution for Templates-CodeAct.

Ported from v2 codeact_subgraph in Wave 2 — namespace API is FROZEN to
prevent the 1,234.56 hallucination regression (per memory
`project_codeact_dual_impl`). Only import adjustments are allowed.

The validator walks the AST and rejects anything that could escape the
namespace: imports, attribute access on dunders, exec/eval, file/network
ops via builtins, etc. Allowed: assignments, arithmetic, comprehensions,
for/while/if, function defs, calls into the namespace.

Execution runs in the caller thread; the `run_python` tool wrapper wraps
this in `asyncio.to_thread(...)` so the main event loop stays responsive.
"""

from __future__ import annotations

import ast
import io
import contextlib
from typing import Any

from .exceptions import ClarificationNeeded


# ── AST validator ────────────────────────────────────────────────────────────


class _Forbidden(Exception):
    """Raised when the AST contains a banned construct."""


# Imports are handled separately in `execute()` — safe stdlib modules whose
# names are pre-injected in `build_namespace()` get stripped silently (no-op);
# anything else still raises via the `_Forbidden` path below.
_SAFE_IMPORT_MODULES = frozenset({"decimal", "datetime"})

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

    # Strip safe stdlib imports (decimal, datetime) as a no-op — those names
    # are pre-injected in build_namespace, so `from decimal import Decimal`
    # is harmless. The LLM emits these out of Python habit even though the
    # prompt forbids it; silently dropping is more reliable than failing.
    # Imports of OTHER modules (os, sys, subprocess, ...) fall through to the
    # _Forbidden path so the existing security guarantee is preserved.
    def _is_safe_import(node: ast.stmt) -> bool:
        if isinstance(node, ast.ImportFrom):
            return (node.module or "").split(".")[0] in _SAFE_IMPORT_MODULES
        if isinstance(node, ast.Import):
            return all(
                alias.name.split(".")[0] in _SAFE_IMPORT_MODULES
                for alias in node.names
            )
        return False

    tree.body = [n for n in tree.body if not _is_safe_import(n)]

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
        # Propagate up — the run_python tool wrapper turns this into a
        # user-facing question and ends the loop.
        raise
    except Exception as exc:  # pylint: disable=broad-except
        msg = f"{type(exc).__name__}: {exc}"
        # E1 — self-describing errors: a bare `TypeError: foo() missing ...`
        # forces the LLM to GUESS foo's signature on retry. Append the helper's
        # real signature + docstring summary so the next attempt has the fix in
        # hand instead of re-deriving it (raises retry success rate).
        return None, buf.getvalue(), _enrich_signature_error(msg, exc, namespace)

    return namespace.get("result"), buf.getvalue(), None


def _enrich_signature_error(msg: str, exc: Exception, namespace: dict) -> str:
    """If `exc` is a TypeError naming a namespace helper, append its real
    signature + first docstring line. No-op for any other error."""
    if not isinstance(exc, TypeError):
        return msg
    import re as _re

    m = _re.search(r"(\w+)\(\)", str(exc))
    if not m:
        return msg
    name = m.group(1)
    fn = namespace.get(name)
    if fn is None or not callable(fn):
        return msg
    try:
        import inspect

        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        return msg
    doc = (getattr(fn, "__doc__", "") or "").strip().split("\n", 1)[0].strip()
    hint = f"\nCorrect signature: {name}{sig}"
    if doc:
        hint += f"\n{name}: {doc}"
    return msg + hint
