"""Offline demo: each scenario exercises one headline behavior and prints what happened.

The planner/synthesizer here is ``DemoLLM`` - a deterministic stand-in that replays a prepared
plan for each goal - so the demo needs no API key. Everything downstream of the plan (registry,
routing, permissions, LangGraph execution, conflict handling) is the real code path.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from .audit import AuditLog
from .models import RunResult, ToolSpec
from .orchestrator import Orchestrator, OrchestratorConfig
from .permissions import ArgConstraint, Grant, PermissionPolicy, Principal
from .registry import FunctionTool, ToolRegistry
from .step_runner import RetryPolicy
from .tools import build_demo_tools


def step(id: str, cap: str, args: dict | None = None, deps: list[str] | None = None, **kw) -> dict:
    return {"id": id, "capability": cap, "args": args or {}, "depends_on": deps or [], **kw}


PLANS: dict[str, list[dict]] = {
    "Research the four design pillars in parallel and save a report": [
        *[step(f"q{i}", "search.documents", {"query": q}) for i, q in enumerate(["routing", "permissions", "parallelism", "conflicts"])],
        step("sum", "text.summarize", {"text": "${q0.0.text} ${q1.0.text} ${q2.0.text} ${q3.0.text}", "max_sentences": 3}),
        step("save", "files.write", {"path": "reports/pillars.md", "content": "${sum}"}),
    ],
    "Weather in Lisbon and Oslo (primary provider is down)": [
        step("w1", "weather.lookup", {"city": "Lisbon"}),
        step("w2", "weather.lookup", {"city": "Oslo"}),
    ],
    "Copy notes.txt to a public file and delete the original": [
        step("read", "files.read", {"path": "notes.txt"}),
        step("copy", "files.write", {"path": "public/notes.txt", "content": "${read}"}),
        step("rm", "files.delete", {"path": "notes.txt"}),
        step("verify", "files.read", {"path": "public/notes.txt"}, deps=["copy"]),
    ],
    "Append four log lines concurrently and quote ACME from every feed": [
        *[step(f"log{i}", "files.write", {"path": "reports/log.txt", "content": f"entry {i}", "append": True}) for i in range(4)],
        step("m", "market.price", {"symbol": "ACME"}, mode="consensus", strategy="majority"),
        step("w", "market.price", {"symbol": "ACME"}, mode="consensus", strategy="weighted"),
        step("med", "market.price", {"symbol": "ACME"}, mode="consensus", strategy="median"),
        step("readback", "files.read", {"path": "reports/log.txt"}, deps=["log0", "log1", "log2", "log3"]),
    ],
    "Compute a budget and write it to disk (needs human approval)": [
        step("calc", "math.evaluate", {"expression": "1200 * 12 + 340"}),
        step("save", "files.write", {"path": "reports/budget.txt", "content": "annual budget: ${calc}"}),
        step("note", "files.write", {"path": "reports/audit-note.txt", "content": "budget generated"}),
    ],
    "Measure the distance from Lisbon to Oslo": [step("d", "geo.distance", {"from": "Lisbon", "to": "Oslo"})],
}


class DemoLLM:
    """Deterministic offline stand-in for the model (planner, synthesizer and arbiter)."""

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        if system.startswith("[planner]"):
            goal = prompt.removeprefix("Goal: ").split("\n\n")[0]
            return json.dumps({"rationale": "replayed demo plan", "steps": PLANS[goal]})
        if system.startswith("[arbiter]"):
            return '{"choice": 0, "reason": "first source"}'
        results = json.loads(prompt.split("Step results (JSON):\n", 1)[1])
        ok = sum(r["status"] == "ok" for r in results)
        bad = [f"{r['step']}={r['status']}" for r in results if r["status"] != "ok"]
        return f"{ok}/{len(results)} steps succeeded" + (f"; not completed: {', '.join(bad)}" if bad else "")


def build_policy(workspace: Path) -> PermissionPolicy:
    return PermissionPolicy(
        base_dir=workspace,
        roles={
            "admin": [Grant()],
            "analyst": [
                Grant(effects=["read", "external"]),
                Grant(capabilities=["files.write"], constraints={"path": ArgConstraint(within=["reports", "public"])}),
            ],
            "reader": [Grant(effects=["read", "external"])],
        },
    )


class Env:
    def __init__(self, *, latency: float = 0.0, primary_down: bool = False, approval: bool = False) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="orch-demo-"))
        self.registry = ToolRegistry(failure_threshold=3, cooldown=60)
        self.registry.register_many(build_demo_tools(self.workspace, latency=latency, weather_primary_down=primary_down))
        self.audit = AuditLog()
        self.orch = Orchestrator(
            DemoLLM(), self.registry, build_policy(self.workspace), self.audit,
            OrchestratorConfig(retry=RetryPolicy(retries=1), approval_effects=("write",) if approval else ()),
        )


def show(result: RunResult) -> None:
    print(f"  status={result.status}  ok={result.ok}  wall={result.wall_seconds:.2f}s  "
          f"tool-time={result.tool_seconds:.2f}s  peak-parallelism={result.peak_concurrency}")
    width = max((s.ended for s in result.steps), default=0.0) or 1.0
    for s in result.steps:
        lo, hi = int(s.started / width * 30), max(int(s.ended / width * 30), int(s.started / width * 30) + 1)
        bar = " " * lo + "#" * (hi - lo)
        detail = s.tool or "-"
        if s.fallbacks:
            detail += f" (fell back from {','.join(s.fallbacks)})"
        print(f"  {s.step_id:<10}{s.status.value:<8}{s.capability:<18}{detail:<44}|{bar:<31}|")
        if s.status.value in ("denied", "failed", "skipped") and s.error:
            print(f"          ↳ {s.error[:150]}")
        for c in s.conflicts:
            print(f"          ↳ {c.kind} conflict -> {c.resolution}: {c.detail}"
                  + (f" => {c.chosen_tool}" if c.chosen_tool else ""))
    if result.answer:
        print(f"  answer: {result.answer.splitlines()[0]}")


def header(title: str, blurb: str) -> None:
    print(f"\n=== {title} ===\n{blurb}")


def parallel() -> None:
    header("1. Parallel execution", "Four searches run as parallel LangGraph branches; summarize/write wait for them.")
    env = Env(latency=0.25)
    result = env.orch.run("Research the four design pillars in parallel and save a report", Principal(name="ana", roles=("analyst",)))
    show(result)
    print(f"  measured speedup: {result.tool_seconds / result.wall_seconds:.1f}x (sum of step times / wall clock)")


def fallback() -> None:
    header("2. Routing, fallback and circuit breaker", "weather_primary (priority 10) is down; the router falls back to weather_backup.")
    env = Env(primary_down=True)
    who = Principal(name="ana", roles=("analyst",))
    show(env.orch.run("Weather in Lisbon and Oslo (primary provider is down)", who))
    show(env.orch.run("Weather in Lisbon and Oslo (primary provider is down)", who))
    print(f"  circuit open for weather_primary: {env.registry.circuit_open('weather_primary')} "
          f"(reliability {env.registry.health('weather_primary').reliability:.2f})")
    show(env.orch.run("Weather in Lisbon and Oslo (primary provider is down)", who))


def permissions() -> None:
    header("3. Permission scoping", "A 'reader' and an 'analyst' ask for the same plan; denials skip dependents.")
    for role in ("reader", "analyst"):
        env = Env()
        (env.workspace / "notes.txt").write_text("quarterly notes")
        print(f"\n  role={role}")
        show(env.orch.run("Copy notes.txt to a public file and delete the original", Principal(name=role, roles=(role,))))
        denied = [e["reason"] for e in env.audit.events("permission.denied")]
        print(f"  audit: {len(denied)} permission denials recorded; notes.txt still exists: {(env.workspace / 'notes.txt').exists()}")


def conflicts() -> None:
    header("4. Conflict resolution", "Concurrent appends are serialized in plan order; disagreeing price feeds are reconciled.")
    env = Env(latency=0.05)
    result = env.orch.run("Append four log lines concurrently and quote ACME from every feed", Principal(name="root", roles=("admin",)))
    show(result)
    print(f"  log file lines: {result.step('readback').value.splitlines()}")
    print("  feeds: a=187.42 (trust .9), b=185.0 (.4), c=185.0 (.4)")
    for sid in ("m", "w", "med"):
        print(f"  {sid}: {result.step(sid).value}")


def approval() -> None:
    header("5. Human-in-the-loop (LangGraph interrupt + checkpoint)", "Write steps pause the graph; the reviewer approves one and rejects one.")
    env = Env(approval=True)
    who = Principal(name="root", roles=("admin",))

    paused = env.orch.run("Compute a budget and write it to disk (needs human approval)", who, thread_id="demo")
    print(f"  paused: status={paused.status}, awaiting approval for {[p['id'] for p in paused.pending_approval]}")
    print(f"  nothing written yet: {not (env.workspace / 'reports').exists()}")
    show(env.orch.resume("demo", approve=["save"]))


def dynamic() -> None:
    header("6. Dynamic registry", "A capability that does not exist is added at runtime and picked up by the next plan.")
    env = Env()
    who = Principal(name="root", roles=("admin",))
    goal = "Measure the distance from Lisbon to Oslo"
    print(f"  before: capabilities = {sorted(env.registry.capabilities())[:4]}... (geo.distance missing)")
    before = env.orch.run(goal, who)
    print(f"  planning outcome: {before.status} ({(before.error or '')[:70]}...)")
    env.registry.register(FunctionTool(
        ToolSpec(name="geo_v1", description="Great-circle distance, rough table", capabilities=("geo.distance",),
                 version="1.0.0", input_schema={"required": ["from", "to"]}),
        lambda a: {"km": 1000, "source": "v1"}))
    first = env.orch.run(goal, who)
    show(first)
    print(f"  value: {first.step('d').value}")
    env.registry.register(FunctionTool(
        ToolSpec(name="geo_v1", description="Great-circle distance, precise", capabilities=("geo.distance",),
                 version="1.1.0", input_schema={"required": ["from", "to"]}),
        lambda a: {"km": 1640, "source": "v1.1 hot-swapped"}))
    print("  after hot-swap to v1.1.0:")
    second = env.orch.run(goal, who)
    show(second)
    print(f"  value: {second.step('d').value}")


SCENARIOS: dict[str, Callable[[], None]] = {
    "parallel": parallel, "fallback": fallback, "permissions": permissions,
    "conflicts": conflicts, "approval": approval, "dynamic": dynamic,
}


def run_demo(name: str = "all") -> None:
    for key, fn in SCENARIOS.items():
        if name in ("all", key):
            fn()
