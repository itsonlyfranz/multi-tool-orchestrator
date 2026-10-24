"""Result-conflict resolution: reconcile disagreeing answers from tools sharing a capability."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, extract_json
from .models import ConflictReport, ConsensusStrategy
from .registry import Tool

ARBITER_SYSTEM = """[arbiter]
Several tools answered the same request with different results. Pick the most trustworthy one.
Reply with ONE JSON object: {"choice": <index>, "reason": "<short>"}"""


@dataclass
class Answer:
    tool: Tool
    value: Any


def values_equal(a: Any, b: Any, rel_tol: float = 1e-9) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k], rel_tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y, rel_tol) for x, y in zip(a, b))
    return bool(a == b)


def _rank(answer: Answer) -> tuple[int, float, str]:
    spec = answer.tool.spec
    return (spec.priority, spec.trust, spec.name)


def _trust(group: list[Answer]) -> float:
    return sum(a.tool.spec.trust for a in group)


class ConflictResolver:
    """Groups equal answers, then picks a winner with the requested strategy.

    ``priority``  highest ``(priority, trust)`` tool wins outright.
    ``majority``  largest group wins (one tool, one vote); ties go to priority.
    ``weighted``  group with the largest summed tool trust wins; ties go to priority.
    ``median``    numeric answers only: the median value wins (robust to one outlier).
    ``llm``       an arbiter model chooses; any invalid reply falls back to ``weighted``.
    """

    def __init__(self, llm: LLMClient | None = None, rel_tol: float = 1e-9) -> None:
        self.llm = llm
        self.rel_tol = rel_tol

    def resolve(
        self, step_id: str, answers: list[Answer], strategy: ConsensusStrategy
    ) -> tuple[Answer, ConflictReport | None]:
        """Return the winning answer and, if the answers disagreed, a report of the conflict."""
        groups = self._group(answers)
        if len(groups) == 1:
            return max(answers, key=_rank), None

        note = ""
        if strategy == "llm":
            winner, note = self._arbitrate(answers)
            if winner is None:
                strategy, note = "weighted", note + " -> fell back to weighted"
        if strategy == "priority":
            winner = max(answers, key=_rank)
        elif strategy == "majority":
            winner = self._best(groups, lambda g: float(len(g)))
        elif strategy == "weighted":
            winner = self._best(groups, _trust)
        elif strategy == "median":
            winner, median_note = self._median(answers, groups)
            note = note or median_note
        report = ConflictReport(
            kind="result",
            step_id=step_id,
            resolution=strategy,
            detail=note or f"{len(groups)} distinct answers from {len(answers)} tools",
            candidates=[
                {"tool": a.tool.spec.name, "value": a.value, "trust": a.tool.spec.trust} for a in answers
            ],
            chosen_tool=winner.tool.spec.name,
        )
        return winner, report

    def _group(self, answers: list[Answer]) -> list[list[Answer]]:
        groups: list[list[Answer]] = []
        for answer in answers:
            for group in groups:
                if values_equal(group[0].value, answer.value, self.rel_tol):
                    group.append(answer)
                    break
            else:
                groups.append([answer])
        return groups

    @staticmethod
    def _best(groups: list[list[Answer]], weight: Any) -> Answer:
        """Highest-ranked answer among the groups that tie for the largest ``weight``."""
        top = max(weight(g) for g in groups)
        return max((a for g in groups if math.isclose(weight(g), top) for a in g), key=_rank)

    def _median(self, answers: list[Answer], groups: list[list[Answer]]) -> tuple[Answer, str]:
        numeric = all(isinstance(a.value, (int, float)) and not isinstance(a.value, bool) for a in answers)
        if not numeric:
            winner = self._best(groups, _trust)
            return winner, "non-numeric answers: used weighted"
        median = statistics.median(a.value for a in answers)
        by_rank = sorted(answers, key=_rank, reverse=True)  # min() keeps the first of equals
        closest = min(by_rank, key=lambda a: abs(a.value - median))
        return closest, f"median of {len(answers)} answers = {median}"

    def _arbitrate(self, answers: list[Answer]) -> tuple[Answer | None, str]:
        if self.llm is None:
            return None, "no arbiter model configured"
        listing = "\n".join(
            f"{i}: tool={a.tool.spec.name} trust={a.tool.spec.trust} value={a.value!r}" for i, a in enumerate(answers)
        )
        try:
            reply = extract_json(self.llm.complete(ARBITER_SYSTEM, listing, max_tokens=300))
            choice = reply["choice"]
            if isinstance(choice, int) and 0 <= choice < len(answers):
                return answers[choice], f"arbiter: {reply.get('reason', '')}".strip()
        except Exception as exc:  # any malformed reply degrades gracefully
            return None, f"arbiter failed ({exc})"
        return None, "arbiter returned an invalid choice"
