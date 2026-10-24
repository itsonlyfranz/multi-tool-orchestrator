"""The agent flow as a LangGraph ``StateGraph``.

    START -> plan --(invalid)--> plan            (bounded repair loop, conditional edge)
              |                \\-> plan_failed -> END
              v
            review  (human approval via ``interrupt`` for write-capable steps)
              v
          +-> schedule --(ready steps)--> run_step x N   (parallel branches via ``Send``)
          |     |                            |
          |     +--(nothing left)--> synthesize -> END
          +--------------------------------------+

State is a typed dict of JSON-friendly values so any LangGraph checkpointer can persist it:
a run that is waiting for approval can be resumed later, even from a new process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from .errors import PlanningError
from .models import Plan, PlanStep, StepResult, StepStatus
from .permissions import Principal
from .step_runner import RunRuntime


def merge_results(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer: parallel branches each contribute their own step's result."""
    return {**(left or {}), **(right or {})}


class RunState(TypedDict, total=False):
    run_id: str
    goal: str
    principal: dict[str, Any]
    plan: dict[str, Any]
    plan_feedback: str
    plan_error: str
    plan_attempts: int
    results: Annotated[dict[str, dict[str, Any]], merge_results]
    ready: list[str]
    answer: str


class StepInput(TypedDict):
    run_id: str
    principal: dict[str, Any]
    step: dict[str, Any]
    rank: int
    outputs: dict[str, Any]


class GraphDeps:
    """Callables the graph needs from the orchestrator (kept narrow to avoid a cycle)."""

    def __init__(
        self,
        *,
        propose: Callable[[str, Principal, str], Plan],
        runner_run: Callable[[RunRuntime, Principal, PlanStep, int, dict[str, Any]], StepResult],
        runtime: Callable[[str], RunRuntime],
        approval_needed: Callable[[PlanStep, Principal], dict[str, Any] | None],
        synthesize: Callable[[RunState], str],
        max_plan_repairs: int,
        audit: Callable[..., None],
    ) -> None:
        self.propose = propose
        self.runner_run = runner_run
        self.runtime = runtime
        self.approval_needed = approval_needed
        self.synthesize = synthesize
        self.max_plan_repairs = max_plan_repairs
        self.audit = audit


def build_graph(deps: GraphDeps, checkpointer: Any) -> Any:
    def route_start(state: RunState) -> str:
        return "review" if state.get("plan") else "plan"  # a caller may inject a ready-made plan

    def plan_node(state: RunState) -> dict[str, Any]:
        principal = Principal.model_validate(state["principal"])
        try:
            plan = deps.propose(state["goal"], principal, state.get("plan_feedback", ""))
        except PlanningError as exc:
            deps.audit("plan.rejected", state["run_id"], error=str(exc))
            feedback = (
                f"Your previous plan was rejected:\n{exc}\n\nPrevious reply:\n{exc.reply}\n\n"
                "Return a corrected JSON plan."
            )
            return {
                "plan_error": str(exc),
                "plan_feedback": feedback,
                "plan_attempts": state.get("plan_attempts", 0) + 1,
            }
        deps.audit("run.planned", state["run_id"], plan=plan.model_dump())
        return {"plan": plan.model_dump(), "plan_error": "", "plan_feedback": ""}

    def after_plan(state: RunState) -> str:
        if not state.get("plan_error"):
            return "review"
        return "plan" if state.get("plan_attempts", 0) <= deps.max_plan_repairs else "plan_failed"

    def plan_failed_node(state: RunState) -> dict[str, Any]:
        return {"answer": f"Planning failed: {state.get('plan_error', 'unknown error')}"}

    def review_node(state: RunState) -> dict[str, Any]:
        """Human-in-the-loop gate. Re-runs from the top on resume, so everything before
        ``interrupt`` is pure."""
        plan = Plan.model_validate(state["plan"])
        principal = Principal.model_validate(state["principal"])
        flagged = [info for step in plan.steps if (info := deps.approval_needed(step, principal))]
        if not flagged:
            return {}
        verdict = interrupt({"kind": "approval", "run_id": state["run_id"], "steps": flagged})
        if verdict is True:
            return {}
        approved = set() if verdict is False or verdict is None else set(verdict)
        rejected = [info["id"] for info in flagged if info["id"] not in approved]
        deps.audit("approval.rejected", state["run_id"], steps=rejected)
        runtime = deps.runtime(state["run_id"])
        now = runtime.now()
        return {
            "results": {
                sid: StepResult(
                    step_id=sid,
                    capability=plan.step(sid).capability,
                    status=StepStatus.DENIED,
                    error="rejected by human reviewer",
                    started=now,
                    ended=now,
                ).model_dump(mode="json")
                for sid in rejected
            }
        }

    def schedule_node(state: RunState) -> dict[str, Any]:
        """Mark steps whose dependencies failed as skipped; list the steps ready to run now."""
        plan = Plan.model_validate(state["plan"])
        runtime = deps.runtime(state["run_id"])
        known = dict(state.get("results", {}))
        new: dict[str, Any] = {}
        ready: list[str] = []
        changed = True
        while changed:
            changed = False
            for step in plan.steps:
                if step.id in known or step.id in ready:
                    continue
                if not all(d in known for d in step.depends_on):
                    continue
                broken = [d for d in step.depends_on if known[d]["status"] != StepStatus.OK.value]
                if broken:
                    now = runtime.now()
                    skipped = StepResult(
                        step_id=step.id,
                        capability=step.capability,
                        status=StepStatus.SKIPPED,
                        error=f"dependency {broken} did not succeed",
                        started=now,
                        ended=now,
                    ).model_dump(mode="json")
                    known[step.id] = new[step.id] = skipped
                    deps.audit("step.skipped", state["run_id"], step=step.id, because=broken)
                    changed = True
                else:
                    ready.append(step.id)
        order = {s.id: i for i, s in enumerate(plan.steps)}
        runtime.resources.expect({sid: order[sid] for sid in ready})
        return {"results": new, "ready": ready}

    def dispatch(state: RunState) -> list[Send] | str:
        ready = state.get("ready", [])
        if not ready:
            return "synthesize"
        plan = Plan.model_validate(state["plan"])
        results = state.get("results", {})
        order = {s.id: i for i, s in enumerate(plan.steps)}
        return [
            Send(
                "run_step",
                StepInput(
                    run_id=state["run_id"],
                    principal=state["principal"],
                    step=plan.step(sid).model_dump(),
                    rank=order[sid],
                    outputs={d: results[d]["value"] for d in plan.step(sid).depends_on},
                ),
            )
            for sid in ready
        ]

    def run_step_node(payload: StepInput) -> dict[str, Any]:
        result = deps.runner_run(
            deps.runtime(payload["run_id"]),
            Principal.model_validate(payload["principal"]),
            PlanStep.model_validate(payload["step"]),
            payload["rank"],
            payload["outputs"],
        )
        return {"results": {result.step_id: result.model_dump(mode="json")}}

    def synthesize_node(state: RunState) -> dict[str, Any]:
        return {"answer": deps.synthesize(state)}

    graph = StateGraph(RunState)
    graph.add_node("plan", plan_node)
    graph.add_node("plan_failed", plan_failed_node)
    graph.add_node("review", review_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("run_step", run_step_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_conditional_edges(START, route_start, {"plan": "plan", "review": "review"})
    graph.add_conditional_edges("plan", after_plan, {"review": "review", "plan": "plan", "plan_failed": "plan_failed"})
    graph.add_edge("plan_failed", END)
    graph.add_edge("review", "schedule")
    graph.add_conditional_edges("schedule", dispatch, ["run_step", "synthesize"])
    graph.add_edge("run_step", "schedule")
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=checkpointer)
