import pytest

from orchestrator.cli import main
from orchestrator.demo import SCENARIOS
from orchestrator.permissions import PermissionPolicy


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_demo_scenario_runs_offline(name, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["demo", name]) == 0
    assert "===" in capsys.readouterr().out


def test_demo_output_proves_headline_behaviors(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "fell back from weather_primary" in out
    assert "circuit open for weather_primary: True" in out
    assert "denied" in out and "rejected by human reviewer" in out
    assert "resource conflict -> serialized" in out and "result conflict -> majority" in out


def test_tools_command_lists_capabilities(capsys):
    assert main(["tools"]) == 0
    assert "market.price" in capsys.readouterr().out


def test_run_requires_an_api_key(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["run", "anything"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_example_policy_file_is_valid():
    policy = PermissionPolicy.from_file("examples/policy.json")
    assert set(policy.roles) == {"admin", "analyst", "reader"}
