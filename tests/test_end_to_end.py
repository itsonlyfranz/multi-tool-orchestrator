import json

from orchestrator import AuditLog, Principal, ScriptedLLM, StepStatus
from orchestrator.config import Settings, load_dotenv
from tests.test_executor import plan, s
from tests.test_graph import make
from tests.test_planner import plan_json, step


def test_permission_denial_end_to_end_skips_dependents_and_is_audited(registry, policy, workspace):
    reader = Principal(name="rita", roles=("reader",))
    (workspace / "in.txt").write_text("hello")
    llm_plan = plan_json(
        step("r", "files.read", {"path": "in.txt"}),
        step("w", "files.write", {"path": "out.txt", "content": "${r}"}),  # LLM ignored the catalog
        step("x", "files.read", {"path": "out.txt"}, deps=["w"]),
    )
    orch = make(registry, policy, ScriptedLLM([llm_plan]))
    result = orch.run("copy in.txt to out.txt", reader)
    status = {r.step_id: r.status for r in result.steps}
    assert status == {"r": StepStatus.OK, "w": StepStatus.DENIED, "x": StepStatus.SKIPPED}
    assert not (workspace / "out.txt").exists()
    denial = orch.audit.events("permission.denied")[0]
    assert denial["principal"] == "rita" and denial["tool"] == "write_file"


def test_path_scoping_end_to_end(registry, policy, analyst, workspace):
    p = plan(
        s("ok", "files.write", {"path": "reports/a.md", "content": "fine"}),
        s("bad", "files.write", {"path": "reports/../escape.md", "content": "no"}),
        s("bad2", "files.write", {"path": "other.md", "content": "no"}),
    )
    result = make(registry, policy).execute_plan(p, analyst)
    assert [r.status for r in result.steps] == [StepStatus.OK, StepStatus.DENIED, StepStatus.DENIED]
    assert (workspace / "reports" / "a.md").exists() and not (workspace / "escape.md").exists()


def test_tool_level_sandbox_is_a_second_line_of_defense(registry, policy, admin):
    result = make(registry, policy).execute_plan(plan(s("w", "files.write", {"path": "../evil.txt", "content": "x"})), admin)
    assert result.step("w").status == StepStatus.FAILED and "escapes the workspace" in result.step("w").error


def test_runtime_tool_registration_is_visible_to_the_next_plan(registry, policy, admin):
    from orchestrator import FunctionTool, ToolSpec

    bad = plan_json(step("s1", "geo.distance", {"a": 1}))
    orch = make(registry, policy, ScriptedLLM([bad, bad]))
    assert orch.run("x", admin).status == "failed"  # unknown capability, no such tool yet
    registry.register(FunctionTool(
        ToolSpec(name="geo", description="d", capabilities=("geo.distance",), input_schema={"required": ["a"]}),
        lambda a: a["a"] * 10,
    ))
    llm2 = ScriptedLLM([plan_json(step("s1", "geo.distance", {"a": 4}))])
    orch2 = make(registry, policy, llm2)
    result = orch2.run("x", admin)
    assert result.ok and result.step("s1").value == 40 and "geo.distance" in llm2.calls[0][0]


def test_hot_swap_changes_the_serving_tool(registry, policy, admin):
    from orchestrator import FunctionTool, ToolSpec

    spec = lambda v: ToolSpec(name="calculator", description="d", capabilities=("math.evaluate",), version=v)  # noqa: E731
    registry.register(FunctionTool(spec("2.0.0"), lambda a: "v2"))
    result = make(registry, policy).execute_plan(plan(s("c", "math.evaluate", {"expression": "1"})), admin)
    assert result.step("c").value == "v2"


def test_conflict_order_is_deterministic_across_runs(registry, policy, admin, workspace):
    orch = make(registry, policy)
    for _ in range(15):
        (workspace / "d.txt").unlink(missing_ok=True)
        steps = [s(f"w{i}", "files.write", {"path": "d.txt", "content": str(i), "append": True}) for i in range(6)]
        orch.execute_plan(plan(*steps), admin)
        assert (workspace / "d.txt").read_text().split() == [str(i) for i in range(6)]


def test_audit_log_mirrors_to_jsonl(tmp_path):
    log = AuditLog(tmp_path / "a" / "audit.jsonl")
    log.record("x", "r1", a=1)
    line = json.loads((tmp_path / "a" / "audit.jsonl").read_text())
    assert line["kind"] == "x" and line["run_id"] == "r1" and log.events("x")


def test_settings_and_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nANTHROPIC_API_KEY='k-123'\nORCH_MODEL=claude-test\n")
    for var in ("ANTHROPIC_API_KEY", "ORCH_MODEL"):
        monkeypatch.delenv(var, raising=False)
    load_dotenv(env)
    settings = Settings.from_env()
    assert settings.api_key == "k-123" and settings.model == "claude-test"
    monkeypatch.delenv("ORCH_MODEL")
    assert Settings.from_env().model == "claude-sonnet-5-5"
