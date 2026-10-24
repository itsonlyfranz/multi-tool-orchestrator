import pytest

from orchestrator import ConflictResolver, FunctionTool, ScriptedLLM, StepStatus, ToolSpec
from orchestrator.conflicts import Answer, values_equal
from tests.test_executor import custom_orchestrator, plan, s


def ans(name, value, trust=0.5, priority=0):
    spec = ToolSpec(name=name, description="d", capabilities=("c.d",), trust=trust, priority=priority)
    return Answer(FunctionTool(spec, lambda a: value), value)


def test_agreement_is_not_a_conflict():
    winner, report = ConflictResolver().resolve("s", [ans("a", 1.0), ans("b", 1.0)], "majority")
    assert report is None and winner.value == 1.0


def test_numeric_equality_tolerates_float_noise_and_nested_values():
    assert values_equal(0.1 + 0.2, 0.3)
    assert values_equal({"p": [1, 2.0]}, {"p": [1.0, 2]})
    assert not values_equal(True, 1)


def test_majority_and_weighted_can_disagree():
    answers = [ans("hi", 187.42, trust=0.9), ans("lo1", 185.0, trust=0.4), ans("lo2", 185.0, trust=0.4)]
    maj, report = ConflictResolver().resolve("s", answers, "majority")
    wtd, _ = ConflictResolver().resolve("s", answers, "weighted")
    assert maj.value == 185.0 and wtd.value == 187.42
    assert report.kind == "result" and report.resolution == "majority" and len(report.candidates) == 3


def test_priority_strategy_ignores_votes():
    answers = [ans("boss", 1, priority=9), ans("x", 2), ans("y", 2)]
    assert ConflictResolver().resolve("s", answers, "priority")[0].value == 1


def test_median_resists_an_outlier_and_falls_back_for_text():
    nums = [ans("a", 10), ans("b", 11), ans("c", 900)]
    winner, report = ConflictResolver().resolve("s", nums, "median")
    assert winner.value == 11 and "median" in report.detail
    text = [ans("a", "x", trust=0.9), ans("b", "y", trust=0.1)]
    winner, report = ConflictResolver().resolve("s", text, "median")
    assert winner.value == "x" and "non-numeric" in report.detail


def test_llm_arbiter_chooses_and_bad_replies_fall_back_to_weighted():
    answers = [ans("a", 1, trust=0.9), ans("b", 2, trust=0.1)]
    chosen, report = ConflictResolver(ScriptedLLM(['{"choice": 1, "reason": "fresher"}'])).resolve("s", answers, "llm")
    assert chosen.value == 2 and "fresher" in report.detail
    for reply in ["garbage", '{"choice": 7}']:
        chosen, report = ConflictResolver(ScriptedLLM([reply])).resolve("s", answers, "llm")
        assert chosen.value == 1 and "fell back" in report.detail
    chosen, _ = ConflictResolver(None).resolve("s", answers, "llm")
    assert chosen.value == 1


@pytest.mark.parametrize("strategy,expected", [("majority", 185.0), ("weighted", 187.42), ("priority", 187.42)])
def test_consensus_step_end_to_end(registry, policy, admin, make_orchestrator, strategy, expected):
    orch = make_orchestrator()
    p = plan(s("px", "market.price", {"symbol": "ACME"}, mode="consensus", strategy=strategy))
    result = orch.execute_plan(p, admin)
    step = result.step("px")
    assert step.status == StepStatus.OK and step.value == expected
    assert sorted(step.consulted) == ["price_feed_a", "price_feed_b", "price_feed_c"]
    assert step.conflicts[0].kind == "result" and step.conflicts[0].resolution == strategy
    assert orch.audit.events("conflict.result")


def test_consensus_survives_a_failing_source_and_reports_when_all_fail():
    from orchestrator import ToolError

    def down(a):
        raise ToolError("x")

    tools = [
        FunctionTool(ToolSpec(name="up1", description="d", capabilities=("q.a",)), lambda a: 5),
        FunctionTool(ToolSpec(name="up2", description="d", capabilities=("q.a",)), lambda a: 5),
        FunctionTool(ToolSpec(name="dn", description="d", capabilities=("q.a",)), down),
    ]
    orch, who = custom_orchestrator(tools, retries=0)
    r = orch.execute_plan(plan(s("a", "q.a", mode="consensus")), who).step("a")
    assert r.status == StepStatus.OK and r.value == 5 and r.fallbacks == ["dn"] and r.conflicts == []

    orch, who = custom_orchestrator(tools[2:] + [FunctionTool(ToolSpec(name="dn2", description="d", capabilities=("q.a",)), down)], retries=0)
    r = orch.execute_plan(plan(s("a", "q.a", mode="consensus")), who).step("a")
    assert r.status == StepStatus.FAILED and "all consulted tools failed" in r.error
