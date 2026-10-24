from __future__ import annotations

import pytest

from orchestrator import (
    AuditLog, Orchestrator, OrchestratorConfig, PermissionPolicy, Principal, RetryPolicy, ToolRegistry,
)
from orchestrator.permissions import ArgConstraint, Grant
from orchestrator.tools import build_demo_tools


def make_policy(workspace) -> PermissionPolicy:
    return PermissionPolicy(
        base_dir=workspace,
        roles={
            "admin": [Grant()],
            "analyst": [
                Grant(effects=["read", "external"]),
                Grant(capabilities=["files.write"], constraints={"path": ArgConstraint(within=["reports"])}),
            ],
            "reader": [Grant(effects=["read"]), Grant(effect="deny", capabilities=["files.*"], effects=["write"])],
        },
    )


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def registry(workspace):
    reg = ToolRegistry(failure_threshold=2, cooldown=60)
    reg.register_many(build_demo_tools(workspace))
    return reg


@pytest.fixture
def policy(workspace):
    return make_policy(workspace)


@pytest.fixture
def admin():
    return Principal(name="root", roles=("admin",))


@pytest.fixture
def analyst():
    return Principal(name="ana", roles=("analyst",))


@pytest.fixture
def make_orchestrator(registry, policy):
    def build(llm=None, **cfg):
        from orchestrator import ScriptedLLM

        config = OrchestratorConfig(synthesize=False, retry=RetryPolicy(retries=1), **cfg)
        return Orchestrator(llm or ScriptedLLM([]), registry, policy, AuditLog(), config)

    return build
