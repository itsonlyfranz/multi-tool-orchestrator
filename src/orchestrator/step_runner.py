"""Runs ONE plan step: route -> (retry, fall back | consensus) -> StepResult.

Scheduling across steps belongs to the LangGraph in :mod:`orchestrator.graph`; this module is
the body of its ``run_step`` node and is safe to call from many threads at once.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .audit import AuditLog
from .conflicts import Answer, ConflictResolver
from .errors import PermanentToolError, RefError
from .models import ConflictReport, PlanStep, StepResult, StepStatus
from .permissions import Principal
from .refs import resolve_refs
from .registry import Tool
from .resources import ResourceManager
from .router import Router


class AbandonedWriteError(TimeoutError):
    """A write timed out but may still be running, so repeating it could apply it twice."""


@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 1  # extra attempts per tool before falling back to the next candidate
    backoff: float = 0.0  # seconds; doubled on each retry


@dataclass
class RunRuntime:
    """Process-local, non-serializable state of one run (lock manager, timings, counters)."""

    run_id: str
    resources: ResourceManager = field(default_factory=ResourceManager)
    t0: float = field(default_factory=time.perf_counter)
    active: int = 0
    peak: int = 0
    conflicts: dict[str, list[ConflictReport]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def now(self) -> float:
        return time.perf_counter() - self.t0


class StepRunner:
    def __init__(
        self,
        router: Router,
        resolver: ConflictResolver,
        audit: AuditLog,
        *,
        step_timeout: float = 30.0,
        retry: RetryPolicy = RetryPolicy(),
    ) -> None:
        self.router = router
        self.registry = router.registry
        self.resolver = resolver
        self.audit = audit
        self.step_timeout = step_timeout
        self.retry = retry
        # Tools run on their own pool so a timeout can abandon a call without blocking the graph.
        self._tool_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="tool")

    # -- entry point --------------------------------------------------------------------------
    def run(self, rt: RunRuntime, principal: Principal, step: PlanStep, rank: int, outputs: dict[str, Any]) -> StepResult:
        rt.resources.on_wait = lambda owner, claims, blockers: self._on_resource_wait(rt, owner, claims, blockers)
        result = StepResult(step_id=step.id, capability=step.capability, status=StepStatus.FAILED, started=rt.now())
        try:
            self._execute(rt, principal, step, rank, outputs, result)
        except Exception as exc:  # a bug in a tool wrapper must not take the whole run down
            result.status, result.error = StepStatus.FAILED, f"{type(exc).__name__}: {exc}"
        finally:
            rt.resources.arrived(step.id)  # never leave siblings waiting on a step that bailed out
        result.ended = rt.now()
        result.conflicts.extend(rt.conflicts.get(step.id, []))
        self.audit.record(
            "step.done", rt.run_id, step=step.id, status=result.status.value, tool=result.tool,
            error=result.error, attempts=result.attempts,
        )
        return result

    def _execute(
        self, rt: RunRuntime, principal: Principal, step: PlanStep, rank: int, outputs: dict[str, Any], result: StepResult
    ) -> None:
        try:
            args = resolve_refs(step.args, outputs)
        except RefError as exc:
            result.error = str(exc)
            return

        decision = self.router.route(step.capability, args, principal)
        self.audit.record("step.routed", rt.run_id, step=step.id, routing=decision.explain())
        for rejection in decision.rejected:
            if rejection.kind == "permission":
                self.audit.record(
                    "permission.denied", rt.run_id, step=step.id, tool=rejection.tool,
                    principal=principal.name, reason=rejection.reason,
                )
        if not decision.candidates:
            result.status = StepStatus.DENIED if decision.denied else StepStatus.FAILED
            result.error = decision.explain() if decision.rejected else f"no tool provides {step.capability!r}"
            return

        tools = [c.tool for c in decision.candidates]
        if step.mode == "consensus" and len(tools) >= 2:
            self._consensus(rt, step, rank, tools[: step.max_tools], args, result)
        else:
            self._first_success(rt, step, rank, tools, args, result)

    # -- strategies ---------------------------------------------------------------------------
    def _first_success(
        self, rt: RunRuntime, step: PlanStep, rank: int, tools: list[Tool], args: dict[str, Any], result: StepResult
    ) -> None:
        errors: list[str] = []
        for tool in tools:
            name = tool.spec.name
            result.consulted.append(name)
            ok, payload, attempts, final = self._try_tool(rt, step, rank, tool, args)
            result.attempts += attempts
            if ok:
                result.status, result.tool, result.value = StepStatus.OK, name, payload
                return
            result.fallbacks.append(name)
            errors.append(f"{name}: {payload}")
            if final:
                break
            self.audit.record("step.fallback", rt.run_id, step=step.id, failed_tool=name, error=str(payload))
        result.error = "all candidate tools failed: " + "; ".join(errors)

    def _consensus(
        self, rt: RunRuntime, step: PlanStep, rank: int, tools: list[Tool], args: dict[str, Any], result: StepResult
    ) -> None:
        with ThreadPoolExecutor(max_workers=len(tools)) as pool:
            outcomes = list(pool.map(lambda t: self._try_tool(rt, step, rank, t, args), tools))
        answers: list[Answer] = []
        errors: list[str] = []
        for tool, (ok, payload, attempts, _) in zip(tools, outcomes):
            result.consulted.append(tool.spec.name)
            result.attempts += attempts
            if ok:
                answers.append(Answer(tool, payload))
            else:
                result.fallbacks.append(tool.spec.name)
                errors.append(f"{tool.spec.name}: {payload}")
        if not answers:
            result.error = "all consulted tools failed: " + "; ".join(errors)
            return
        winner, report = self.resolver.resolve(step.id, answers, step.strategy)
        result.status, result.tool, result.value = StepStatus.OK, winner.tool.spec.name, winner.value
        if report:
            result.conflicts.append(report)
            self.audit.record(
                "conflict.result", rt.run_id, step=step.id, strategy=report.resolution,
                chosen=report.chosen_tool, candidates=report.candidates,
            )

    # -- one tool, with retries -----------------------------------------------------------------
    def _try_tool(
        self, rt: RunRuntime, step: PlanStep, rank: int, tool: Tool, args: dict[str, Any]
    ) -> tuple[bool, Any, int, bool]:
        """Return ``(ok, value or error, attempts, final)``; ``final`` forbids falling back."""
        name = tool.spec.name
        attempts = 0
        last: Exception | None = None
        for attempt in range(self.retry.retries + 1):
            attempts += 1
            try:
                value = self._attempt(rt, step, rank, tool, args)
            except PermanentToolError as exc:  # caller's fault: no retry, tool health untouched
                return False, f"{type(exc).__name__}: {exc}", attempts, False
            except Exception as exc:
                last = exc
                self.registry.record_failure(name)
                self.audit.record(
                    "tool.error", rt.run_id, step=step.id, tool=name, attempt=attempt + 1,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, AbandonedWriteError):
                    return False, f"{type(exc).__name__}: {exc}", attempts, True
                if self.registry.circuit_open(name):
                    break  # stop hammering a tool the breaker just tripped
                if attempt < self.retry.retries and self.retry.backoff:
                    time.sleep(self.retry.backoff * 2**attempt)
            else:
                self.registry.record_success(name)
                return True, value, attempts, False
        return False, f"{type(last).__name__}: {last}", attempts, False

    def _attempt(self, rt: RunRuntime, step: PlanStep, rank: int, tool: Tool, args: dict[str, Any]) -> Any:
        lease = rt.resources.acquire(step.id, rank, f"{step.id}/{tool.spec.name}", tool.claims(args))
        future = None
        try:
            with rt.lock:
                rt.active += 1
                rt.peak = max(rt.peak, rt.active)
            try:
                future = self._tool_pool.submit(self._invoke, tool, args)
                try:
                    return future.result(timeout=self.step_timeout)
                except TimeoutError:
                    future.cancel()
                    writes = any(c.mode == "write" for c in lease.claims)
                    error = AbandonedWriteError if writes else TimeoutError
                    raise error(f"timed out after {self.step_timeout}s") from None
            finally:
                with rt.lock:
                    rt.active -= 1
        finally:
            if future is None:
                rt.resources.release(lease)
            else:  # an abandoned call keeps its claims until it really finishes
                future.add_done_callback(lambda _: rt.resources.release(lease))

    @staticmethod
    def _invoke(tool: Tool, args: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(tool.run):
            return asyncio.run(tool.run(args))
        return tool.run(args)

    def _on_resource_wait(self, rt: RunRuntime, owner: str, claims: Any, blockers: list[str]) -> None:
        step_id = owner.split("/")[0]
        resources = sorted({c.resource for c in claims})
        blocked_by = sorted({b.split("/")[0] for b in blockers})
        with rt.lock:
            rt.conflicts.setdefault(step_id, []).append(
                ConflictReport(
                    kind="resource",
                    step_id=step_id,
                    resolution="serialized",
                    detail=f"waited for {blocked_by} on {resources}",
                )
            )
        self.audit.record("conflict.resource", rt.run_id, step=step_id, resources=resources, blocked_by=blocked_by)
