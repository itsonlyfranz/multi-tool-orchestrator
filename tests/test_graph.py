from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator import (
    AuditLog, ChatModelClient, Orchestrator, OrchestratorConfig, ScriptedLLM, StepStatus,
    sqlite_checkpointer,
)
from tests.test_executor import plan, s
from tests.test_planner import plan_json, step


def make(registry, policy, llm=None, **cfg):
    cfg.setdefault("synthesize", False)
    return Orchestrator(llm or ScriptedLLM([]), registry, policy, AuditLog(), OrchestratorConfig(**cfg))


def test_graph_has_the_expected_nodes_and_edges(registry, policy):
    drawn = make(registry, policy).graph.get_graph()
    assert {"plan", "review", "schedule", "run_step", "synthesize", "plan_failed"} <= set(drawn.nodes)
    assert ("run_step", "schedule") in {(e.source, e.target) for e in drawn.edges}


# -- planning loop ----------------------------------------------------------------------------
def test_planning_repair_loop_feeds_the_error_back_to_the_model(registry, policy, admin):
    llm = ScriptedLLM(["not json", plan_json(step("s1", "math.evaluate", {"expression": "6*7"}))])
    result = make(registry, policy, llm).run("compute", admin)
    assert result.ok and result.step("s1").value == 42
    assert len(llm.calls) == 2 and "rejected" in llm.calls[1][1] and "not json" in llm.calls[1][1]


def test_planning_gives_up_after_the_repair_budget(registry, policy, admin):
    bad = plan_json(step("s1", "teleport.now"))
    llm = ScriptedLLM([bad, bad, bad])
    result = make(registry, policy, llm, max_plan_repairs=1).run("go", admin)
    assert result.status == "failed" and not result.ok and "unknown capability" in result.error
    assert len(llm.calls) == 2 and result.answer.startswith("Planning failed")


def test_recursion_limit_bounds_a_run(registry, policy, admin):
    chain = [s("c0", "math.evaluate", {"expression": "1"})] + [
        s(f"c{i}", "math.evaluate", {"expression": f"${{c{i-1}}} + 1"}) for i in range(1, 12)
    ]
    result = make(registry, policy, recursion_limit=10).execute_plan(plan(*chain), admin)
    assert result.status == "failed" and "recursion" in result.error
    assert make(registry, policy).execute_plan(plan(*chain), admin).step("c11").value == 12


# -- human-in-the-loop interrupt --------------------------------------------------------------
WRITE_PLAN = lambda: plan(  # noqa: E731
    s("calc", "math.evaluate", {"expression": "2+2"}),
    s("save", "files.write", {"path": "out.txt", "content": "${calc}"}),
    s("after", "files.read", {"path": "out.txt"}, depends_on=["save"]),
)


def test_run_pauses_for_approval_then_resumes_and_executes(registry, policy, admin, workspace):
    orch = make(registry, policy, approval_effects=("write",))
    paused = orch.execute_plan(WRITE_PLAN(), admin, thread_id="t1")
    assert paused.status == "awaiting_approval" and [p["id"] for p in paused.pending_approval] == ["save"]
    assert not (workspace / "out.txt").exists()  # nothing risky ran before approval
    assert orch.pending_approval("t1")[0]["steps"][0]["effects"] == ["write"]

    done = orch.resume("t1", approve=True)
    assert done.ok and (workspace / "out.txt").read_text() == "4"
    assert done.step("after").value == "4"


def test_rejection_denies_the_step_and_skips_dependents(registry, policy, admin, workspace):
    orch = make(registry, policy, approval_effects=("write",))
    orch.execute_plan(WRITE_PLAN(), admin, thread_id="t2")
    done = orch.resume("t2", approve=False)
    status = {r.step_id: r.status for r in done.steps}
    assert status == {"calc": StepStatus.OK, "save": StepStatus.DENIED, "after": StepStatus.SKIPPED}
    assert "reviewer" in done.step("save").error and not (workspace / "out.txt").exists()


def test_approver_callback_can_approve_selectively_without_pausing(registry, policy, admin, workspace):
    seen = []

    def approver(steps):
        seen.append([x["id"] for x in steps])
        return []  # approve nothing

    orch = make(registry, policy, approval_effects=("write",))
    done = orch.execute_plan(WRITE_PLAN(), admin, approver=approver)
    assert seen == [["save"]] and done.step("save").status == StepStatus.DENIED


def test_no_approval_needed_means_no_interrupt(registry, policy, admin):
    orch = make(registry, policy, approval_effects=("write",))
    result = orch.execute_plan(plan(s("c", "math.evaluate", {"expression": "1+1"})), admin)
    assert result.status == "completed" and result.ok


def test_paused_run_resumes_from_a_new_process_via_sqlite_checkpoint(registry, policy, admin, tmp_path, workspace):
    db = tmp_path / "ckpt" / "graph.sqlite"
    first = Orchestrator(ScriptedLLM([]), registry, policy, AuditLog(),
                         OrchestratorConfig(synthesize=False, approval_effects=("write",)), sqlite_checkpointer(db))
    assert first.execute_plan(WRITE_PLAN(), admin, thread_id="persist").status == "awaiting_approval"
    del first

    second = Orchestrator(ScriptedLLM([]), registry, policy, AuditLog(),
                          OrchestratorConfig(synthesize=False, approval_effects=("write",)), sqlite_checkpointer(db))
    assert second.pending_approval("persist")
    done = second.resume("persist", approve=["save"])
    assert done.ok and (workspace / "out.txt").exists()


# -- parallel branches / waves ------------------------------------------------------------------
def test_dependency_waves_follow_the_dag(registry, policy, admin):
    p = plan(
        s("a", "math.evaluate", {"expression": "1"}),
        s("b", "math.evaluate", {"expression": "2"}),
        s("c", "math.evaluate", {"expression": "${a} + ${b}"}),
    )
    orch = make(registry, policy)
    result = orch.execute_plan(p, admin)
    assert result.step("c").value == 3
    assert result.step("c").started >= max(result.step("a").ended, result.step("b").ended)


# -- LLM plumbing -------------------------------------------------------------------------------
def test_synthesis_uses_the_llm_and_survives_its_failure(registry, policy, admin):
    p = plan(s("c", "math.evaluate", {"expression": "1+1"}))
    llm = ScriptedLLM(["The answer is 2."])
    result = make(registry, policy, llm, synthesize=True).execute_plan(p, admin, goal="add")
    assert result.answer == "The answer is 2." and "step results" in llm.calls[0][1].lower()

    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("api down")

    orch = make(registry, policy, Broken(), synthesize=True)
    degraded = orch.execute_plan(p, admin)
    assert degraded.ok and "c [ok]" in degraded.answer and orch.audit.events("run.synthesis_failed")


def test_langchain_chat_model_adapter_drives_the_whole_agent(registry, policy, admin):
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content=plan_json(step("s1", "math.evaluate", {"expression": "10/4"}))),
        AIMessage(content=[{"type": "text", "text": "It is 2.5."}]),
    ]))
    result = make(registry, policy, ChatModelClient(model), synthesize=True).run("divide", admin)
    assert result.ok and result.answer == "It is 2.5."
