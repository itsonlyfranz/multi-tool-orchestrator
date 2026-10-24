"""``${step_id.path}`` references that wire one step's output into another's arguments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .errors import RefError

_REF = re.compile(r"\$\{([A-Za-z][A-Za-z0-9_]*)((?:\.[A-Za-z0-9_]+)*)\}")


def find_refs(value: Any) -> set[str]:
    """Step ids referenced anywhere inside ``value`` (strings, lists, dicts)."""
    if isinstance(value, str):
        return {m.group(1) for m in _REF.finditer(value)}
    if isinstance(value, Mapping):
        return set().union(*(find_refs(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(find_refs(v) for v in value)) if value else set()
    return set()


def resolve_refs(value: Any, outputs: Mapping[str, Any]) -> Any:
    """Substitute references with step outputs.

    A string that is exactly one reference keeps the referenced value's type;
    a reference embedded in a longer string is rendered with ``str()``.
    """
    if isinstance(value, str):
        whole = _REF.fullmatch(value)
        if whole:
            return _lookup(outputs, whole.group(1), whole.group(2))
        return _REF.sub(lambda m: str(_lookup(outputs, m.group(1), m.group(2))), value)
    if isinstance(value, Mapping):
        return {k: resolve_refs(v, outputs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve_refs(v, outputs) for v in value]
    return value


def _lookup(outputs: Mapping[str, Any], step_id: str, path: str) -> Any:
    if step_id not in outputs:
        raise RefError(f"step {step_id!r} has no output available")
    current = outputs[step_id]
    for part in (p for p in path.split(".") if p):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise RefError(f"cannot resolve {part!r} in reference ${{{step_id}{path}}}")
    return current
