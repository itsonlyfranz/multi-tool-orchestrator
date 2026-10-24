import threading
import time

from orchestrator import (
    AuditLog, Claim, FunctionTool, Orchestrator, OrchestratorConfig, Plan, PermissionPolicy, Principal,
    RetryPolicy, ScriptedLLM, StepStatus, ToolError, ToolRegistry, ToolSpec,
)
from orchestrator.permissions import Grant


def plan(*steps):
    return Plan.model_validate({"steps": list(steps)})


def s(id, cap, args=None, **kw):
    return {"id": id, "capability": cap, "args": args or {}, **kw}


def custom_orchestrator(tools, retries=1, **cfg):
    reg = ToolRegistry(failure_threshold=3, cooldown=60)
    reg.register_many(tools)
    policy = PermissionPolicy(roles={"admin": [Grant()]})
    config = OrchestratorConfig(synthesize=False, retry=RetryPolicy(retries=retries), **cfg)
    return Orchestrator(ScriptedLLM([]), reg, policy, AuditLog(), config), Principal(name="t", roles=("admin",))


def fn_tool(name, cap, fn, **kw):
    return FunctionTool(ToolSpec(name=name, description="d", capabilities=(cap,), **kw), fn)


# -- parallel execution ---------------------------------------------------------------------
def test_independent_steps_really_run_concurrently():
    barrier = threading.Barrier(3, timeout=3)  # only passes if 3 calls are in flight at once

    def meet(a):
        barrier.wait()
        return a["n"]

    orch, who = custom_orchestrator([fn_tool("meet", "sync.meet", meet)])
    result = orch.execute_plan(plan(*[s(f"s{i}", "sync.meet", {"n": i}) for i in range(3)]), who)
    assert result.ok and result.peak_concurrency == 3
    assert [r.value for r in result.steps] == [0, 1, 2]


def test_parallel_wall_time_beats_sequential(workspace):
    from orchestrator.tools import build_demo_tools

    reg = ToolRegistry()
    reg.register_many(build_demo_tools(workspace, latency=0.15))
    orch = Orchestrator(
        ScriptedLLM([]), reg, PermissionPolicy(roles={"admin": [Grant()]}), config=OrchestratorConfig(synthesize=False)
    )
    p = plan(*[s(f"q{i}", "search.documents", {"query": "routing"}) for i in range(4)])
    result = orch.execute_plan(p, Principal(name="t", roles=("admin",)))
    assert result.ok and result.tool_seconds > 0.55
    assert result.wall_seconds < result.tool_seconds * 0.6


def test_max_concurrency_is_enforced():
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def work(a):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.03)
        with lock:
            state["now"] -= 1

    orch, who = custom_orchestrator([fn_tool("w", "w.do", work)], max_concurrency=2)
    result = orch.execute_plan(plan(*[s(f"s{i}", "w.do") for i in range(6)]), who)
    assert result.ok and state["peak"] == 2 and result.peak_concurrency == 2


# -- dependencies ---------------------------------------------------------------------------
def test_outputs_flow_through_references_and_order_is_respected():
    order = []
    tools = [
        fn_tool("make", "d.make", lambda a: order.append("make") or {"n": 20}),
        fn_tool("use", "d.use", lambda a: order.append("use") or a["x"] + 1),
    ]
    orch, who = custom_orchestrator(tools)
    result = orch.execute_plan(plan(s("b", "d.use", {"x": "${a.n}"}), s("a", "d.make")), who)
    assert order == ["make", "use"] and result.step("b").value == 21
    assert result.step("b").started >= result.step("a").ended


def test_failure_skips_transitive_dependents_but_not_siblings():
    def boom(a):
        raise ToolError("nope")

    tools = [fn_tool("boom", "f.boom", boom), fn_tool("ok", "f.ok", lambda a: 1)]
    orch, who = custom_orchestrator(tools, retries=0)
    result = orch.execute_plan(
        plan(s("a", "f.boom"), s("b", "f.ok", depends_on=["a"]), s("c", "f.ok", depends_on=["b"]), s("d", "f.ok")), who
    )
    status = {r.step_id: r.status for r in result.steps}
    assert status == {"a": StepStatus.FAILED, "b": StepStatus.SKIPPED, "c": StepStatus.SKIPPED, "d": StepStatus.OK}
    assert not result.ok


def test_bad_reference_fails_the_step():
    orch, who = custom_orchestrator([fn_tool("ok", "f.ok", lambda a: {"x": 1}), fn_tool("t", "f.t", lambda a: 1)])
    result = orch.execute_plan(plan(s("a", "f.ok"), s("b", "f.t", {"v": "${a.missing}"})), who)
    assert result.step("b").status == StepStatus.FAILED and "missing" in result.step("b").error


# -- fallback / retries / breaker -------------------------------------------------------------
def test_retry_succeeds_without_fallback():
    calls = {"n": 0}

    def flaky(a):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ToolError("transient")
        return "fine"

    orch, who = custom_orchestrator([fn_tool("flaky", "r.do", flaky)], retries=1)
    r = orch.execute_plan(plan(s("a", "r.do")), who).step("a")
    assert r.status == StepStatus.OK and r.attempts == 2 and r.fallbacks == []


def test_falls_back_to_next_ranked_tool_and_records_it():
    def down(a):
        raise ToolError("503")

    tools = [fn_tool("primary", "w.get", down, priority=5), fn_tool("backup", "w.get", lambda a: "ok")]
    orch, who = custom_orchestrator(tools, retries=1)
    r = orch.execute_plan(plan(s("a", "w.get")), who).step("a")
    assert r.status == StepStatus.OK and r.tool == "backup" and r.fallbacks == ["primary"]
    assert r.attempts == 3 and orch.audit.events("step.fallback")


def test_circuit_breaker_stops_routing_to_a_dead_tool():
    def down(a):
        raise ToolError("503")

    tools = [fn_tool("primary", "w.get", down, priority=5), fn_tool("backup", "w.get", lambda a: "ok")]
    orch, who = custom_orchestrator(tools, retries=1)
    orch.execute_plan(plan(s("a", "w.get")), who)  # 2 failures
    orch.execute_plan(plan(s("b", "w.get")), who)  # 3rd failure trips the breaker
    assert orch.registry.circuit_open("primary")
    r = orch.execute_plan(plan(s("c", "w.get")), who).step("c")
    assert r.tool == "backup" and r.fallbacks == [] and r.consulted == ["backup"]


def test_permanent_error_is_not_retried_and_does_not_hurt_health():
    from orchestrator import PermanentToolError

    def bad(a):
        raise PermanentToolError("bad args")

    orch, who = custom_orchestrator([fn_tool("bad", "p.do", bad)], retries=3)
    r = orch.execute_plan(plan(s("a", "p.do")), who).step("a")
    assert r.status == StepStatus.FAILED and r.attempts == 1
    assert orch.registry.health("bad").failures == 0


def test_step_timeout_fails_slow_tool_and_falls_back():
    tools = [fn_tool("slow", "t.do", lambda a: time.sleep(0.5), priority=5), fn_tool("fast", "t.do", lambda a: "ok")]
    orch, who = custom_orchestrator(tools, retries=0, step_timeout=0.05)
    r = orch.execute_plan(plan(s("a", "t.do")), who).step("a")
    assert r.tool == "fast" and r.fallbacks == ["slow"]


def test_async_tools_are_supported():
    async def aio(a):
        return "async-ok"

    orch, who = custom_orchestrator([fn_tool("aio", "a.do", aio)])
    assert orch.execute_plan(plan(s("a", "a.do")), who).step("a").value == "async-ok"


def test_unknown_capability_at_execution_fails_cleanly():
    orch, who = custom_orchestrator([fn_tool("ok", "f.ok", lambda a: 1)])
    r = orch.execute_plan(plan(s("a", "nope.nope")), who).step("a")
    assert r.status == StepStatus.FAILED and "no tool provides" in r.error


# -- resource conflicts -----------------------------------------------------------------------
def test_conflicting_writers_are_serialized_in_plan_order_but_readers_overlap():
    lock, state = threading.Lock(), {"writers": 0, "bad": False, "log": []}

    def write(a):
        with lock:
            state["writers"] += 1
            state["bad"] |= state["writers"] > 1
        time.sleep(0.05)
        state["log"].append(a["who"])
        with lock:
            state["writers"] -= 1

    tools = [
        FunctionTool(
            ToolSpec(name="w", description="d", capabilities=("res.write",), effect="write"),
            write,
            lambda a: [Claim(resource="doc", mode="write")],
        ),
        FunctionTool(
            ToolSpec(name="r", description="d", capabilities=("res.read",)),
            lambda a: time.sleep(0.05),
            lambda a: [Claim(resource="other", mode="read")],
        ),
    ]
    orch, who = custom_orchestrator(tools)
    p = plan(*[s(f"w{i}", "res.write", {"who": i}) for i in range(3)], s("r1", "res.read"), s("r2", "res.read"))
    result = orch.execute_plan(p, who)
    assert result.ok and not state["bad"] and state["log"] == [0, 1, 2]  # exclusive + plan order
    assert result.step("w0").conflicts == []
    assert result.step("w2").conflicts[0].kind == "resource"
    assert result.step("r1").conflicts == [] and result.step("r2").conflicts == []
    assert orch.audit.events("conflict.resource")


def test_reader_waits_for_earlier_writer_of_same_resource():
    log = []
    tools = [
        FunctionTool(
            ToolSpec(name="w", description="d", capabilities=("res.write",), effect="write"),
            lambda a: (time.sleep(0.05), log.append("write")),
            lambda a: [Claim(resource="doc", mode="write")],
        ),
        FunctionTool(
            ToolSpec(name="r", description="d", capabilities=("res.read",)),
            lambda a: log.append("read"),
            lambda a: [Claim(resource="doc", mode="read")],
        ),
    ]
    orch, who = custom_orchestrator(tools)
    orch.execute_plan(plan(s("w", "res.write"), s("r", "res.read")), who)
    assert log == ["write", "read"]


def test_real_file_writes_do_not_interleave(registry, policy, admin, make_orchestrator, workspace):
    orch = make_orchestrator()
    steps = [s(f"a{i}", "files.write", {"path": "log.txt", "content": f"line{i}", "append": True}) for i in range(5)]
    result = orch.execute_plan(plan(*steps), admin)
    assert result.ok
    assert (workspace / "log.txt").read_text().splitlines() == [f"line{i}" for i in range(5)]
