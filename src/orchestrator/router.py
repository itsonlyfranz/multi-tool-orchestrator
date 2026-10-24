"""Capability-based routing: pick and rank the tools that may serve a capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from .permissions import PermissionPolicy, Principal
from .registry import Tool, ToolRegistry


class RoutingStrategy(str, Enum):
    """Weight presets for the score that orders tools of equal priority."""

    BALANCED = "balanced"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    MOST_RELIABLE = "most_reliable"


# (reliability, trust, cost, latency)
_WEIGHTS: dict[RoutingStrategy, tuple[float, float, float, float]] = {
    RoutingStrategy.BALANCED: (0.35, 0.25, 0.2, 0.2),
    RoutingStrategy.CHEAPEST: (0.1, 0.1, 0.7, 0.1),
    RoutingStrategy.FASTEST: (0.1, 0.1, 0.1, 0.7),
    RoutingStrategy.MOST_RELIABLE: (0.6, 0.3, 0.05, 0.05),
}


@dataclass
class Candidate:
    tool: Tool
    score: float


@dataclass
class Rejection:
    tool: str
    kind: Literal["unavailable", "permission", "args"]
    reason: str


@dataclass
class RoutingDecision:
    capability: str
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def denied(self) -> bool:
        """True when tools exist for the capability but permissions excluded all of them."""
        return not self.candidates and any(r.kind == "permission" for r in self.rejected) and not any(
            r.kind != "permission" for r in self.rejected
        )

    def explain(self) -> str:
        ranked = ", ".join(f"{c.tool.spec.name}({c.score:.2f})" for c in self.candidates) or "none"
        skipped = "; ".join(f"{r.tool}: {r.kind} - {r.reason}" for r in self.rejected)
        return f"{self.capability} -> [{ranked}]" + (f" rejected [{skipped}]" if skipped else "")


class Router:
    """Filters providers by availability, arguments and permissions, then ranks them.

    Ranking is ``(priority desc, score desc, name)``. ``priority`` is an explicit operator
    override; ``score`` blends smoothed reliability, declared trust, and cost/latency
    according to the :class:`RoutingStrategy`.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.strategy = strategy

    def route(self, capability: str, args: dict[str, Any], principal: Principal) -> RoutingDecision:
        decision = RoutingDecision(capability)
        for tool in self.registry.providers(capability, only_available=False):
            spec = tool.spec
            if not self.registry.is_available(spec.name):
                why = "circuit open" if self.registry.circuit_open(spec.name) else "disabled"
                decision.rejected.append(Rejection(spec.name, "unavailable", why))
                continue
            missing = [a for a in spec.required_args if a not in args]
            if missing:
                decision.rejected.append(Rejection(spec.name, "args", f"missing required {missing}"))
                continue
            verdict = self.policy.authorize(principal, spec, capability, args)
            if not verdict.allowed:
                decision.rejected.append(Rejection(spec.name, "permission", verdict.reason))
                continue
            decision.candidates.append(Candidate(tool, self._score(tool)))
        decision.candidates.sort(key=lambda c: (-c.tool.spec.priority, -c.score, c.tool.spec.name))
        return decision

    def _score(self, tool: Tool) -> float:
        w_rel, w_trust, w_cost, w_lat = _WEIGHTS[self.strategy]
        spec = tool.spec
        reliability = self.registry.health(spec.name).reliability
        return (
            w_rel * reliability
            + w_trust * spec.trust
            + w_cost / (1.0 + spec.cost)
            + w_lat / (1.0 + spec.latency_ms / 1000.0)
        )

    def permitted_capabilities(self, principal: Principal) -> dict[str, list[Tool]]:
        """Capabilities this principal might use (argument constraints not evaluated)."""
        out: dict[str, list[Tool]] = {}
        for cap, names in self.registry.capabilities().items():
            tools = [
                t
                for t in (self.registry.get(n) for n in names)
                if self.policy.authorize(principal, t.spec, cap).allowed
            ]
            if tools:
                out[cap] = tools
        return out
