"""Permission scoping: role grants, argument constraints and per-run scope narrowing.

Evaluation order (deny-by-default):

1. Every scope layer on the principal must match the capability (attenuation).
2. Any matching ``deny`` grant wins immediately.
3. Otherwise some ``allow`` grant must match the capability/tool/effect *and*
   satisfy all of its argument constraints.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import ToolSpec


class ArgConstraint(BaseModel):
    """Restricts one argument of a call; an absent argument is not constrained."""

    within: list[str] | None = None  # path must resolve inside one of these directories
    one_of: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None  # regex, full match

    def violation(self, value: Any, base_dir: Path) -> str | None:
        if self.within is not None:
            if not isinstance(value, str):
                return "expected a path string"
            target = (base_dir / value).resolve()
            roots = [(base_dir / r).resolve() for r in self.within]
            if not any(target == r or target.is_relative_to(r) for r in roots):
                return f"path {value!r} is outside {self.within}"
        if self.one_of is not None and value not in self.one_of:
            return f"{value!r} not in {self.one_of}"
        if self.min is not None or self.max is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "expected a number"
            if self.min is not None and value < self.min:
                return f"{value} < minimum {self.min}"
            if self.max is not None and value > self.max:
                return f"{value} > maximum {self.max}"
        if self.pattern is not None and not (isinstance(value, str) and re.fullmatch(self.pattern, value)):
            return f"{value!r} does not match /{self.pattern}/"
        return None


class Grant(BaseModel):
    """One allow/deny rule. Patterns are shell-style globs (``files.*``)."""

    effect: Literal["allow", "deny"] = "allow"
    capabilities: list[str] = Field(default_factory=lambda: ["*"])
    tools: list[str] = Field(default_factory=lambda: ["*"])
    effects: list[str] = Field(default_factory=lambda: ["*"])
    constraints: dict[str, ArgConstraint] = Field(default_factory=dict)

    def matches(self, capability: str, spec: ToolSpec) -> bool:
        return (
            _any_glob(self.capabilities, capability)
            and _any_glob(self.tools, spec.name)
            and _any_glob(self.effects, spec.effect)
        )


class Principal(BaseModel):
    """Who is asking. ``scopes`` are narrowing layers: each must match to be allowed."""

    model_config = ConfigDict(frozen=True)

    name: str
    roles: tuple[str, ...] = ()
    scopes: tuple[tuple[str, ...], ...] = ()

    def narrow(self, *patterns: str) -> Principal:
        """Derive a strictly-less-powerful principal (e.g. for a delegated sub-task)."""
        return self.model_copy(update={"scopes": (*self.scopes, tuple(patterns))})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


@dataclass
class PermissionPolicy:
    """Role -> grants, plus the base directory that relative path constraints resolve against."""

    roles: dict[str, list[Grant]] = field(default_factory=dict)
    base_dir: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str | Path | None = None) -> PermissionPolicy:
        roles = {name: [Grant.model_validate(g) for g in grants] for name, grants in data["roles"].items()}
        return cls(roles=roles, base_dir=Path(base_dir) if base_dir else Path.cwd())

    @classmethod
    def from_file(cls, path: str | Path, base_dir: str | Path | None = None) -> PermissionPolicy:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), base_dir)

    def authorize(
        self,
        principal: Principal,
        spec: ToolSpec,
        capability: str,
        args: dict[str, Any] | None = None,
    ) -> Decision:
        """Decide whether ``principal`` may invoke ``spec`` for ``capability``.

        With ``args=None`` argument constraints are skipped ("could be allowed"),
        which is what catalog filtering needs.
        """
        for layer in principal.scopes:
            if not _any_glob(layer, capability):
                return Decision(False, f"capability {capability!r} is outside delegated scope {list(layer)}")

        grants = [g for role in principal.roles for g in self.roles.get(role, [])]
        for grant in grants:
            if grant.effect == "deny" and grant.matches(capability, spec):
                return Decision(False, f"explicitly denied for {capability!r} via tool {spec.name!r}")

        violations: list[str] = []
        for grant in grants:
            if grant.effect != "allow" or not grant.matches(capability, spec):
                continue
            problems = [] if args is None else self._violations(grant, args)
            if not problems:
                return Decision(True, "allowed by grant")
            violations.extend(problems)
        if violations:
            return Decision(False, "argument constraint failed: " + "; ".join(violations))
        return Decision(False, f"no grant allows {capability!r} (tool {spec.name!r}, effect {spec.effect})")

    def _violations(self, grant: Grant, args: dict[str, Any]) -> list[str]:
        out = []
        for name, constraint in grant.constraints.items():
            if name in args:
                problem = constraint.violation(args[name], self.base_dir)
                if problem:
                    out.append(f"{name}: {problem}")
        return out


def _any_glob(patterns: list[str] | tuple[str, ...], value: str) -> bool:
    return any(fnmatchcase(value, p) for p in patterns)
