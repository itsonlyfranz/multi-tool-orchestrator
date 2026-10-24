import json

import pytest

from orchestrator import Plan, ScriptedLLM
from orchestrator.errors import PlanningError
from orchestrator.llm import extract_json
from orchestrator.planner import Planner
from orchestrator.router import Router


def plan_json(*steps):
    return json.dumps({"rationale": "r", "steps": list(steps)})


def step(id, cap, args=None, deps=None):
    return {"id": id, "capability": cap, "args": args or {}, "depends_on": deps or []}


def test_plan_validates_dag_and_infers_dependencies_from_refs():
    plan = Plan.model_validate(
        {"steps": [step("a", "x.y"), step("b", "x.y", {"v": "got ${a.value} ok"}), step("c", "x.y", deps=["b"])]}
    )
    assert plan.step("b").depends_on == ["a"] and plan.step("c").depends_on == ["b"]


@pytest.mark.parametrize(
    "steps",
    [
        [step("a", "x.y", deps=["b"]), step("b", "x.y", deps=["a"])],
        [step("a", "x.y", deps=["a"])],
        [step("a", "x.y", deps=["zzz"])],
        [step("a", "x.y"), step("a", "x.y")],
    ],
)
def test_invalid_plans_are_rejected(steps):
    with pytest.raises(ValueError):
        Plan.model_validate({"steps": steps})


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nbye') == {"a": 1}
    assert extract_json('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    with pytest.raises(PlanningError):
        extract_json("no json here")


def test_planner_returns_valid_plan(registry, policy, admin):
    llm = ScriptedLLM([plan_json(step("s1", "math.evaluate", {"expression": "1+1"}))])
    plan = Planner(llm, Router(registry, policy)).propose("add", admin)
    assert plan.steps[0].capability == "math.evaluate"


def test_planner_rejects_unknown_capability_and_keeps_the_reply(registry, policy, admin):
    bad = plan_json(step("s1", "teleport.now"))
    with pytest.raises(PlanningError, match="unknown capability") as info:
        Planner(ScriptedLLM([bad]), Router(registry, policy)).propose("goal", admin)
    assert info.value.reply == bad


def test_planner_rejects_cyclic_plan(registry, policy, admin):
    bad = plan_json(step("a", "math.evaluate", deps=["b"]), step("b", "math.evaluate", deps=["a"]))
    with pytest.raises(PlanningError, match="cycle"):
        Planner(ScriptedLLM([bad]), Router(registry, policy)).propose("goal", admin)


def test_catalog_given_to_llm_is_permission_filtered(registry, policy):
    from orchestrator import Principal

    llm = ScriptedLLM([plan_json(step("s1", "files.read", {"path": "a"}))])
    Planner(llm, Router(registry, policy)).propose("g", Principal(name="r", roles=("reader",)))
    system = llm.calls[0][0]
    assert "files.read" in system and "files.delete" not in system and "files.write" not in system
