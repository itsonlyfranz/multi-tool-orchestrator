"""LLM planner: turns a goal into a validated capability DAG (the graph owns the repair loop)."""

from __future__ import annotations

from pydantic import ValidationError

from .errors import PlanningError
from .llm import LLMClient, extract_json
from .models import Plan
from .permissions import Principal
from .router import Router

PLANNER_SYSTEM = """[planner]
You are the planning component of a multi-tool orchestrator. Decompose the user's goal
into steps. Each step names a CAPABILITY (never a tool); the router picks the tool.

Reply with ONE JSON object and nothing else:
{"rationale": "<one sentence>",
 "steps": [{"id": "s1", "capability": "<capability>", "args": {...},
            "depends_on": [], "description": "<short>",
            "mode": "single" | "consensus", "strategy": "priority|majority|weighted|median|llm"}]}

Rules:
- Independent steps run in parallel, so do NOT add dependencies that are not needed.
- Pass one step's output to another with "${step_id}" or "${step_id.field}" inside args.
- Use mode "consensus" when several tools should answer and disagreements must be reconciled.
- Only use capabilities from the catalog below, and supply every required arg.
"""


class Planner:
    def __init__(self, llm: LLMClient, router: Router) -> None:
        self.llm = llm
        self.router = router

    def catalog(self, principal: Principal) -> str:
        """Capability catalog limited to what ``principal`` may use (least privilege)."""
        lines = []
        for cap, tools in self.router.permitted_capabilities(principal).items():
            lines.append(f"- {cap}")
            for tool in tools:
                spec = tool.spec
                props = ", ".join(spec.input_schema.get("properties", {}))
                req = ",".join(spec.required_args) or "-"
                lines.append(f"    * {spec.name} [{spec.effect}] {spec.description} (args: {props}; required: {req})")
        return "\n".join(lines) or "(no capabilities available)"

    def propose(self, goal: str, principal: Principal, feedback: str = "") -> Plan:
        """One planning attempt. ``feedback`` carries the previous rejection for the repair loop."""
        system = f"{PLANNER_SYSTEM}\nCatalog:\n{self.catalog(principal)}"
        prompt = f"Goal: {goal}" + (f"\n\n{feedback}" if feedback else "")
        reply = self.llm.complete(system, prompt, max_tokens=2048)
        try:
            plan = Plan.model_validate(extract_json(reply))
            self._check_capabilities(plan)
        except (PlanningError, ValidationError) as exc:
            raise PlanningError(str(exc), reply=reply) from exc
        return plan

    def _check_capabilities(self, plan: Plan) -> None:
        known = set(self.router.registry.capabilities(only_available=False))
        for step in plan.steps:
            if step.capability not in known:
                raise PlanningError(
                    f"step {step.id!r} uses unknown capability {step.capability!r}; known: {sorted(known)}"
                )
