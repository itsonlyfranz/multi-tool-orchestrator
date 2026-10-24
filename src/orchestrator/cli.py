"""Command line: ``orchestrator demo | tools | run``."""

from __future__ import annotations

import argparse
import sys

from .audit import AuditLog
from .config import Settings, load_dotenv
from .demo import SCENARIOS, build_policy, run_demo, show
from .llm import claude_client
from .orchestrator import Orchestrator, OrchestratorConfig, sqlite_checkpointer
from .permissions import PermissionPolicy, Principal
from .registry import ToolRegistry
from .step_runner import RetryPolicy
from .tools import build_demo_tools


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="orchestrator", description="Multi-tool orchestrator agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="offline demo of the headline behaviors (no API key needed)")
    demo.add_argument("scenario", nargs="?", default="all", choices=["all", *SCENARIOS])

    sub.add_parser("tools", help="list the demo tool catalog and its capabilities")

    run = sub.add_parser("run", help="plan and execute a goal with Claude (needs ANTHROPIC_API_KEY)")
    run.add_argument("goal")
    run.add_argument("--role", default="analyst", help="role to act as (default: analyst)")
    run.add_argument("--policy", help="JSON policy file (default: built-in demo roles)")
    run.add_argument("--plugin", action="append", default=[], help="plugin module or .py file defining register(registry)")
    run.add_argument("--approve-writes", action="store_true", help="ask before running write-capable steps")
    run.add_argument("--checkpoint-db", help="SQLite file for LangGraph checkpoints")

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        run_demo(args.scenario)
        return 0

    settings = Settings.from_env()
    registry = ToolRegistry()
    registry.register_many(build_demo_tools(settings.workspace))
    if args.cmd == "tools":
        for cap, names in registry.capabilities().items():
            print(f"{cap:<20} {', '.join(names)}")
        return 0

    if not settings.api_key:
        print("ANTHROPIC_API_KEY is not set (put it in the environment or a .env file). "
              "Try `orchestrator demo` for the offline demo.", file=sys.stderr)
        return 2
    for plugin in args.plugin:
        registry.load_plugin(plugin)
    policy = PermissionPolicy.from_file(args.policy, settings.workspace) if args.policy else build_policy(settings.workspace)
    orch = Orchestrator(
        claude_client(settings.model, settings.api_key),
        registry,
        policy,
        AuditLog(settings.audit_path),
        OrchestratorConfig(
            max_concurrency=settings.max_concurrency,
            step_timeout=settings.step_timeout,
            retry=RetryPolicy(retries=1),
            approval_effects=("write",) if args.approve_writes else (),
        ),
        sqlite_checkpointer(args.checkpoint_db) if args.checkpoint_db else None,
    )

    def ask(steps: list[dict]) -> list[str]:
        approved = []
        for s in steps:
            reply = input(f"approve {s['id']} {s['capability']} {s['args']}? [y/N] ").strip().lower()
            if reply == "y":
                approved.append(s["id"])
        return approved

    result = orch.run(args.goal, Principal(name=args.role, roles=(args.role,)), approver=ask if args.approve_writes else None)
    show(result)
    print(f"\n{result.answer}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
