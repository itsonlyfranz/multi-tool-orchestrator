"""Top-level facade: owns the LangGraph, the checkpointer and the per-run runtimes."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .audit import AuditLog
from .conflicts import ConflictResolver
from .graph import GraphDeps, RunState, build_graph
from .llm import LLMClient
from .models import Plan, PlanStep, RunResult, StepResult, StepStatus
from .permissions import PermissionPolicy, Principal
from .planner import Planner
from .registry import ToolRegistry
from .router import Router, RoutingStrategy
from .step_runner import RetryPolicy, RunRuntime, StepRunner

SYNTH_SYSTEM = """[synthesizer]
You are the final stage of a multi-tool orchestrator. Using ONLY the step results provided,
answer the user's goal concisely. Mention any step that failed, was denied, or was skipped,
and any conflict that was resolved between tools."""

# Reviewer verdict: True = approve all, False = reject all, list[str] = approve exactly these steps.
Approver = Callable[[list[dict[str, Any]]], "bool | list[str]"]


@dataclass
class OrchestratorConfig:
    max_concurrency: int = 8
    step_timeout: float = 30.0
    retry: RetryPolicy = RetryPolicy()
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    synthesize: bool = True
    max_plan_repairs: int = 1
    recursion_limit: int = 60  # LangGraph supersteps; each dependency wave costs two
    approval_effects: tuple[str, ...] = ()  # tool effects that require human approval, e.g. ("write",)


def sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    """File-backed LangGraph checkpointer so runs can be resumed across processes."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


@dataclass
class Orchestrator:
    llm: LLMClient
    registry: ToolRegistry
    policy: PermissionPolicy
    audit: AuditLog = field(default_factory=AuditLog)
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    checkpointer: Any = None

    def __post_init__(self) -> None:
        self.router = Router(self.registry, self.policy, self.config.strategy)
        self.planner = Planner(self.llm, self.router)
        self.runner = StepRunner(
            self.router, ConflictResolver(self.llm), self.audit,
            step_timeout=self.config.step_timeout, retry=self.config.retry,
        )
        self._runtimes: dict[str, RunRuntime] = {}
        self._runtime_lock = threading.Lock()
        self.checkpointer = self.checkpointer or MemorySaver()
        self.graph = build_graph(
            GraphDeps(
                propose=self.planner.propose,
                runner_run=self.runner.run,
                runtime=self._runtime,
                approval_needed=self._approval_needed,
                synthesize=self._synthesize,
                max_plan_repairs=self.config.max_plan_repairs,
                audit=self.audit.record,
            ),
            self.checkpointer,
        )

    # -- public API -----------------------------------------------------------------------------
    def run(
        self, goal: str, principal: Principal, *, thread_id: str | None = None, approver: Approver | None = None
    ) -> RunResult:
        """Plan with the LLM and execute. Pauses for approval if the config requires it and no
        ``approver`` is supplied (see :meth:`resume`)."""
        return self._start({"goal": goal}, principal, thread_id, approver)

    def execute_plan(
        self, plan: Plan, principal: Principal, goal: str = "", *,
        thread_id: str | None = None, approver: Approver | None = None,
    ) -> RunResult:
        """Run a ready-made plan, skipping the planner."""
        return self._start({"goal": goal, "plan": plan.model_dump()}, principal, thread_id, approver)

    def resume(self, thread_id: str, approve: bool | list[str], approver: Approver | None = None) -> RunResult:
        """Continue a run that is waiting for approval (works from a fresh process too)."""
        started = time.perf_counter()
        config = self._graph_config(thread_id)
        state = self.graph.invoke(Command(resume=approve), config)
        return self._finish(thread_id, config, state, approver, started)

    def pending_approval(self, thread_id: str) -> list[dict[str, Any]]:
        snapshot = self.graph.get_state(self._graph_config(thread_id))
        return [i.value for task in snapshot.tasks for i in task.interrupts]

    # -- internals ------------------------------------------------------------------------------
    def _graph_config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.recursion_limit,
            "max_concurrency": self.config.max_concurrency,
        }

    def _start(self, initial: dict[str, Any], principal: Principal, thread_id: str | None, approver: Approver | None) -> RunResult:
        run_id = uuid.uuid4().hex[:8]
        thread_id = thread_id or run_id
        started = time.perf_counter()
        self.audit.record("run.start", run_id, goal=initial["goal"], principal=principal.name)
        config = self._graph_config(thread_id)
        state_in = {"run_id": run_id, "principal": principal.model_dump(), **initial}
        try:
            state = self.graph.invoke(state_in, config)
        except GraphRecursionError as exc:
            return self._failed(run_id, thread_id, initial["goal"], principal.name, f"recursion limit reached: {exc}", started)
        return self._finish(thread_id, config, state, approver, started)

    def _finish(self, thread_id: str, config: dict[str, Any], state: dict[str, Any], approver: Approver | None, started: float) -> RunResult:
        try:
            while state.get("__interrupt__"):
                payload = state["__interrupt__"][0].value
                if approver is None:
                    return self._build(state, thread_id, started, status="awaiting_approval", pending=payload["steps"])
                state = self.graph.invoke(Command(resume=approver(payload["steps"])), config)
        except GraphRecursionError as exc:
            return self._failed(state["run_id"], thread_id, state.get("goal", ""), state["principal"]["name"], f"recursion limit reached: {exc}", started)
        return self._build(state, thread_id, started)

    def _build(self, state: dict[str, Any], thread_id: str, started: float, *, status: str = "completed", pending: list[dict[str, Any]] | None = None) -> RunResult:
        plan = Plan.model_validate(state["plan"]) if state.get("plan") else None
        results = state.get("results", {})
        steps = [StepResult.model_validate(results[s.id]) for s in plan.steps if s.id in results] if plan else []
        failed_planning = plan is None
        runtime = self._runtimes.get(state["run_id"])
        result = RunResult(
            run_id=state["run_id"],
            goal=state.get("goal", ""),
            principal=state["principal"]["name"],
            plan=plan,
            steps=steps,
            answer=state.get("answer", ""),
            status="failed" if failed_planning else status,  # type: ignore[arg-type]
            error=state.get("plan_error") or None if failed_planning else None,
            thread_id=thread_id,
            pending_approval=pending or [],
            wall_seconds=time.perf_counter() - started,
            peak_concurrency=runtime.peak if runtime else 0,
        )
        if status == "completed":
            self.audit.record("run.done", result.run_id, ok=result.ok, wall=round(result.wall_seconds, 3), peak=result.peak_concurrency)
            self._runtimes.pop(result.run_id, None)
        return result

    def _failed(self, run_id: str, thread_id: str, goal: str, principal: str, error: str, started: float) -> RunResult:
        self._runtimes.pop(run_id, None)
        self.audit.record("run.failed", run_id, error=error)
        return RunResult(
            run_id=run_id, goal=goal, principal=principal, status="failed", error=error,
            thread_id=thread_id, wall_seconds=time.perf_counter() - started,
        )

    def _runtime(self, run_id: str) -> RunRuntime:
        with self._runtime_lock:
            return self._runtimes.setdefault(run_id, RunRuntime(run_id))

    def _approval_needed(self, step: PlanStep, principal: Principal) -> dict[str, Any] | None:
        if not self.config.approval_effects:
            return None
        tools = self.router.permitted_capabilities(principal).get(step.capability, [])
        effects = sorted({t.spec.effect for t in tools} & set(self.config.approval_effects))
        if not effects:
            return None
        return {"id": step.id, "capability": step.capability, "args": step.args, "effects": effects}

    def _synthesize(self, state: RunState) -> str:
        plan = Plan.model_validate(state["plan"])
        results = [StepResult.model_validate(state["results"][s.id]) for s in plan.steps if s.id in state.get("results", {})]
        if not self.config.synthesize:
            return fallback_answer(results)
        payload = [
            {
                "step": s.step_id, "capability": s.capability, "status": s.status.value, "tool": s.tool,
                "value": s.value, "error": s.error, "conflicts": [c.model_dump() for c in s.conflicts],
            }
            for s in results
        ]
        prompt = f"Goal: {state.get('goal', '')}\n\nStep results (JSON):\n{json.dumps(payload, default=str)}"
        try:
            return self.llm.complete(SYNTH_SYSTEM, prompt, max_tokens=800).strip()
        except Exception as exc:  # the work is done; a synthesis outage must not lose it
            self.audit.record("run.synthesis_failed", state["run_id"], error=str(exc))
            return fallback_answer(results)


def fallback_answer(results: list[StepResult]) -> str:
    lines = []
    for s in results:
        detail = s.value if s.status == StepStatus.OK else s.error
        lines.append(f"{s.step_id} [{s.status.value}] {s.capability}: {detail}")
    return "\n".join(lines)
