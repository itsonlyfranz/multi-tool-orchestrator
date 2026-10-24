# Multi-Tool Orchestrator Agent

A LangGraph agent that plans a goal as a capability DAG, routes each step to the best permitted tool from a live registry, runs independent steps in parallel, and reconciles conflicts between tools.

**Stack:** Python 3.11+ · LangGraph · langchain-anthropic (Claude, default `claude-sonnet-5-5`) · Pydantic v2 · SQLite checkpoints · pytest (fully offline)

## Highlights

- **LangGraph `StateGraph` with typed state.** Plan → review → schedule → parallel `run_step` branches → synthesize. Conditional edges drive the planner repair loop and the "more steps ready?" loop; `Send` fans out one branch per ready step; a `recursion_limit` bounds runaway plans.
- **Human-in-the-loop via `interrupt`.** Steps whose tools have a configured effect (e.g. `write`) pause the graph before anything runs. State is checkpointed (in-memory or SQLite), so a run can be resumed by a different process; the test suite does exactly that.
- **Dynamic tool registry + capability routing.** Plans name *capabilities* (`weather.lookup`), never tools. Tools can be registered, hot-swapped by version, disabled, or loaded from plugins at runtime; the router ranks providers by priority, observed reliability, trust, cost and latency, and trips a circuit breaker on repeated failures.
- **Deny-by-default permission scoping.** Role grants with glob matching, explicit denies, argument constraints (path containment that survives `..` and symlinks, numeric ranges, enums, regexes) and attenuating per-run scopes. The planner is only shown capabilities the principal may use, and the router re-checks every call with its real arguments.
- **Measured parallelism.** In the demo environment (8 independent searches with a simulated 0.25 s I/O latency each) the run took 0.26 s wall-clock against 2.02 s of summed step time, a 7.8x speedup with peak concurrency 8 (numbers from a local run; reproduce with `orchestrator demo parallel`). Concurrency itself is proven deterministically in the tests with a `threading.Barrier` that only releases if three calls are truly in flight at once.
- **Two kinds of conflict resolution.** *Resource* conflicts (two steps writing one file) are serialized in plan order by a lock manager with read/write claims; the order was checked over 15 repeated runs in the tests. *Result* conflicts (tools disagree) are reconciled by `priority`, `majority`, trust-`weighted` vote, `median`, or an LLM arbiter that falls back to weighted on a bad reply.
- **Offline test suite.** 90 tests, no network and no API key: a scripted fake LLM, a LangChain fake chat model for the Claude adapter, and deterministic local tools.

## Architecture

```
                         ┌────────────────────────── LangGraph StateGraph (graph.py) ─────────────────────────┐
 goal ──► plan ──ok──► review ──► schedule ──ready steps──► run_step ×N (parallel, via Send) ──┐
           │ ▲  (interrupt:        │  ▲                                                          │
  invalid  │ │   human approval)   │  └──────────────────────────────────────────────────────────┘
           ▼ │ repair loop         └─ nothing left ──► synthesize ──► answer
        plan_failed                                         
```

| Component | File | Responsibility |
|---|---|---|
| **Agent graph** | `graph.py` | The `StateGraph`: typed `RunState`, nodes `plan`, `review`, `schedule`, `run_step`, `synthesize`, conditional edges, `Send` fan-out, `interrupt`. |
| **Orchestrator** | `orchestrator.py` | Facade: builds the graph with a checkpointer (`MemorySaver` or `sqlite_checkpointer`), `run` / `execute_plan` / `resume`, recursion limit, result assembly. |
| **Planner** | `planner.py` | Asks the LLM for a JSON plan over a permission-filtered capability catalog; validates it (Pydantic, DAG, known capabilities). The graph owns the repair loop. |
| **Tool registry** | `registry.py` | Thread-safe name/capability index; register, version hot-swap, disable, plugin loading, change events, health stats and circuit breaker. |
| **Router** | `router.py` | Filters providers by availability, required args and permissions, then ranks them (`balanced`, `cheapest`, `fastest`, `most_reliable` strategies); returns an explainable decision. |
| **Permission policy** | `permissions.py` | Roles → allow/deny grants, argument constraints, `Principal.narrow()` scope attenuation, JSON policy files. |
| **Step runner** | `step_runner.py` | Executes one step: route, retry, fall back to the next ranked tool, or fan out for consensus; enforces timeouts and records timings. |
| **Resource manager** | `resources.py` | Read/write claims granted atomically in plan-rank order; readers share, writers are exclusive. |
| **Conflict resolver** | `conflicts.py` | Groups equal answers (float-tolerant) and picks a winner by the chosen strategy, producing a `ConflictReport`. |
| **LLM interface** | `llm.py` | Small `LLMClient` protocol; `claude_client()` wraps `ChatAnthropic`; `ScriptedLLM` is the offline test double. |
| **Plan references** | `refs.py` | `${step.field}` wiring between steps; also derives implicit dependencies. |
| **Audit log** | `audit.py` | Thread-safe event trail (routing, denials, fallbacks, conflicts), optionally mirrored to JSONL. |
| **Demo tools & CLI** | `tools/builtin.py`, `demo.py`, `cli.py` | Local deterministic tools, six runnable scenarios, and the `orchestrator` command. |

## Quickstart

```bash
uv sync --extra dev            # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env           # only needed for live runs; add ANTHROPIC_API_KEY
pytest                         # offline, no key required
orchestrator demo              # offline demo of every headline behavior
```

Live run with Claude (model from `ORCH_MODEL`, default `claude-sonnet-5-5`):

```bash
orchestrator tools
orchestrator run "Compare the weather in Lisbon and Oslo, then write a summary to reports/weather.md" \
    --role analyst --approve-writes --checkpoint-db .checkpoints.sqlite
```

Use it as a library:

```python
from orchestrator import (Orchestrator, OrchestratorConfig, PermissionPolicy, Principal,
                          ToolRegistry, claude_client)
from orchestrator.tools import build_demo_tools

registry = ToolRegistry()
registry.register_many(build_demo_tools(workspace))      # or your own Tool / FunctionTool objects
policy = PermissionPolicy.from_file("examples/policy.json", base_dir=workspace)
agent = Orchestrator(claude_client("claude-sonnet-5-5"), registry, policy,
                     config=OrchestratorConfig(approval_effects=("write",)))

result = agent.run("...", Principal(name="ana", roles=("analyst",)))
if result.status == "awaiting_approval":                  # graph paused at the interrupt
    result = agent.resume(result.thread_id, approve=["save"])
```

## Demo

`orchestrator demo [parallel|fallback|permissions|conflicts|approval|dynamic]` replays prepared plans through a deterministic stand-in for the model (`DemoLLM`), so it needs no key; everything after planning is the real code path. Each scenario prints a per-step table with a timeline:

| Scenario | What you see |
|---|---|
| `parallel` | Four searches overlap on the timeline; summarize and write wait for them; speedup computed from measured times. |
| `fallback` | `weather_primary` is down, steps fall back to `weather_backup`; after three failures the circuit opens and the router stops trying it. |
| `permissions` | A `reader` and an `analyst` run the same plan: writes/deletes are denied, dependents are skipped, and the audit trail records why. |
| `conflicts` | Four concurrent appends to one file land in plan order; three disagreeing price feeds are resolved by majority, weighted vote and median (185.0 / 187.42 / 185.0). |
| `approval` | The graph pauses at `interrupt`; nothing is written; the reviewer approves one write and the other is denied. |
| `dynamic` | An unknown capability fails planning, a tool registered at runtime fixes it, and a version bump hot-swaps the implementation. |

## Design decisions and trade-offs

- **Capabilities, not tool names, in plans.** The LLM stays decoupled from the tool set, so tools can change at runtime and fallbacks are the router's job. Cost: a plan cannot pin a specific tool.
- **LangGraph supersteps mean dependency waves.** All branches of a wave finish before the next `schedule` runs, so a fast step whose dependent is ready still waits for the slowest sibling. A free-running DAG scheduler would be tighter; the graph gives checkpoints, interrupts and an inspectable control flow in exchange.
- **Sync nodes on threads, not asyncio.** Tools are ordinary blocking functions and the SQLite checkpointer is synchronous; LangGraph runs `Send` branches on a thread pool. Async tools are supported (each runs in its own event loop on a worker thread).
- **Timeouts abandon, they don't kill.** Python cannot kill a thread, so a timed-out tool call keeps running in the background after the step fails over; mutual exclusion on a resource is released at that point. Tools with side effects should be idempotent or enforce their own timeouts.
- **Conflict order is plan order within a wave.** The scheduler announces each wave to the lock manager, which makes claim-holding steps wait for lower-ranked siblings that have not yet declared claims. Conservative (slight serialization) but deterministic. Claims come from the first tool tried; a fallback tool that touches different resources is not re-arbitrated against siblings already started.
- **Deny by default, enforced twice.** The policy filters the planner's catalog (least privilege) and the router re-authorizes each call with its resolved arguments; tools additionally sandbox paths themselves. Permissions are role-based and in-process; there is no authentication layer.
- **Reliability is a Laplace-smoothed success rate** so untried tools are neither favored nor shunned; explicit `priority` still overrides the score.
- **Local infrastructure only.** Audit log is JSONL, checkpoints are SQLite, tools are local. Plugins execute arbitrary code, so only load trusted ones.
- **Consensus needs numbers to be numbers.** `median` degrades to `weighted` for non-numeric answers and says so in the report.

## How the tests prove each capability

Run `pytest` (90 tests, ~3 s, offline).

| Capability | Tests | What they prove |
|---|---|---|
| Dynamic tool registry | `test_registry.py`, `test_end_to_end.py` | Runtime register/unregister, numeric version compare and hot-swap, disable/enable, circuit breaker open → half-open → recovered with an injected clock, plugin loading; a tool registered after the orchestrator exists is used by the next plan. |
| Capability routing | `test_router.py`, `test_planner.py` | Priority then score ordering; strategy presets reorder tools; failures demote a tool; unavailable/arg-mismatch/permission rejections carry reasons; planner validates DAG, cycles, unknown capabilities. |
| Permission scoping | `test_permissions.py`, `test_end_to_end.py` | Default deny, explicit deny beats allow, `..` traversal and symlink escapes blocked, numeric/enum/regex constraints, scope narrowing only removes power, planner catalog hides forbidden capabilities, denied steps skip dependents and appear in the audit log, tool-level sandbox as second line. |
| Parallel execution | `test_executor.py`, `test_graph.py` | Barrier-based proof of real concurrency, wall time well under summed step time, `max_concurrency` honored, references flow between steps, failures skip only dependents, retries, fallback, timeouts, async tools, recursion limit, dependency waves. |
| Conflict resolution | `test_conflicts.py`, `test_executor.py`, `test_end_to_end.py` | Writers never overlap and run in plan order (repeated 15×), readers share, reader waits for earlier writer; every consensus strategy including the LLM arbiter and its fallbacks; majority vs weighted disagree on the same data; partial and total source failure. |
| LangGraph flow | `test_graph.py` | Graph topology, planner repair loop with error feedback and its budget, `interrupt` pause/approve/reject/selective approve, resume from a new process through the SQLite checkpoint, LangChain chat-model adapter driving a full run. |
| Demo and config | `test_cli_demo.py`, `test_end_to_end.py` | Every demo scenario runs offline and prints its headline evidence; `.env` and settings loading; CLI refuses to run live without a key. |

## Project layout

```
src/orchestrator/   graph, orchestrator, planner, registry, router, permissions, step_runner,
                    resources, conflicts, llm, refs, audit, config, demo, cli, tools/
tests/              90 offline tests          examples/policy.json   role/grant file format
.env.example        environment variables     pyproject.toml         uv/pip compatible
```
