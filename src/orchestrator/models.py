"""Pydantic data models: tool metadata, plans, and execution results."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .refs import find_refs

Effect = Literal["read", "write", "external"]
ConsensusStrategy = Literal["priority", "majority", "weighted", "median", "llm"]

_CAPABILITY = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


def version_key(version: str) -> tuple[int, ...]:
    """Sortable key for dotted numeric versions ('1.10.0' > '1.2.0')."""
    return tuple(int(p) for p in re.findall(r"\d+", version))


class ToolSpec(BaseModel):
    """Static description of a tool: what it can do and how it should be ranked."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    capabilities: tuple[str, ...] = Field(min_length=1)
    version: str = "1.0.0"
    effect: Effect = "read"
    priority: int = 0
    cost: float = Field(1.0, ge=0)
    latency_ms: float = Field(100.0, ge=0)
    trust: float = Field(0.5, ge=0, le=1)
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @field_validator("capabilities")
    @classmethod
    def _valid_capabilities(cls, caps: tuple[str, ...]) -> tuple[str, ...]:
        for cap in caps:
            if not _CAPABILITY.match(cap):
                raise ValueError(f"invalid capability name {cap!r} (use dotted lowercase, e.g. 'files.read')")
        return caps

    @property
    def required_args(self) -> tuple[str, ...]:
        return tuple(self.input_schema.get("required", ()))


class Claim(BaseModel):
    """A resource a tool call will touch; used to detect and serialize conflicting calls."""

    model_config = ConfigDict(frozen=True)

    resource: str
    mode: Literal["read", "write"] = "read"

    def conflicts_with(self, other: Claim) -> bool:
        return self.resource == other.resource and "write" in (self.mode, other.mode)


class PlanStep(BaseModel):
    """One unit of work, expressed as a *capability* (never a concrete tool)."""

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    capability: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    description: str = ""
    mode: Literal["single", "consensus"] = "single"
    strategy: ConsensusStrategy = "weighted"
    max_tools: int = Field(3, ge=2, le=8)


class Plan(BaseModel):
    """A DAG of steps. Dependencies come from ``depends_on`` and from ``${id}`` references."""

    rationale: str = ""
    steps: list[PlanStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dag(self) -> Plan:
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate step ids")
        known = set(ids)
        for step in self.steps:
            implied = find_refs(step.args)
            for dep in [*step.depends_on, *implied]:
                if dep not in known:
                    raise ValueError(f"step {step.id!r} depends on unknown step {dep!r}")
                if dep == step.id:
                    raise ValueError(f"step {step.id!r} depends on itself")
            step.depends_on = list(dict.fromkeys([*step.depends_on, *sorted(implied)]))
        self._check_acyclic()
        return self

    def _check_acyclic(self) -> None:
        indegree = {s.id: len(s.depends_on) for s in self.steps}
        children: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for s in self.steps:
            for d in s.depends_on:
                children[d].append(s.id)
        queue = [i for i, n in indegree.items() if n == 0]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if seen != len(self.steps):
            raise ValueError("plan contains a dependency cycle")

    def step(self, step_id: str) -> PlanStep:
        return next(s for s in self.steps if s.id == step_id)


class StepStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    DENIED = "denied"
    SKIPPED = "skipped"


class ConflictReport(BaseModel):
    """Record of a conflict the orchestrator detected and how it resolved it."""

    kind: Literal["resource", "result"]
    step_id: str
    resolution: str
    detail: str = ""
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chosen_tool: str | None = None


class StepResult(BaseModel):
    step_id: str
    capability: str
    status: StepStatus
    tool: str | None = None
    value: Any = None
    error: str | None = None
    attempts: int = 0
    fallbacks: list[str] = Field(default_factory=list)
    consulted: list[str] = Field(default_factory=list)
    conflicts: list[ConflictReport] = Field(default_factory=list)
    started: float = 0.0
    ended: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.ended - self.started)


class RunResult(BaseModel):
    run_id: str
    goal: str
    principal: str
    plan: Plan | None = None
    steps: list[StepResult] = Field(default_factory=list)
    answer: str = ""
    status: Literal["completed", "awaiting_approval", "failed"] = "completed"
    error: str | None = None
    thread_id: str = ""
    pending_approval: list[dict[str, Any]] = Field(default_factory=list)
    wall_seconds: float = 0.0
    peak_concurrency: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "completed" and bool(self.steps) and all(s.status == StepStatus.OK for s in self.steps)

    @property
    def tool_seconds(self) -> float:
        return sum(s.duration for s in self.steps if s.status != StepStatus.SKIPPED)

    def step(self, step_id: str) -> StepResult:
        return next(s for s in self.steps if s.step_id == step_id)
