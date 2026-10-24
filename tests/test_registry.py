import pytest

from orchestrator import FunctionTool, ToolRegistry, ToolSpec
from orchestrator.errors import DuplicateToolError, UnknownToolError


def tool(name="t", caps=("a.b",), version="1.0.0", **kw):
    return FunctionTool(ToolSpec(name=name, description="d", capabilities=caps, version=version, **kw), lambda a: 1)


def test_register_and_discover_by_capability():
    reg = ToolRegistry()
    reg.register(tool("one", ("x.y", "x.z")))
    reg.register(tool("two", ("x.y",)))
    assert [t.spec.name for t in reg.providers("x.y")] == ["one", "two"]
    assert reg.capabilities() == {"x.y": ["one", "two"], "x.z": ["one"]}


def test_tools_appear_and_disappear_at_runtime():
    reg = ToolRegistry()
    events = []
    reg.subscribe(lambda e, n: events.append((e, n)))
    reg.register(tool("one"))
    reg.unregister("one")
    assert reg.providers("a.b") == []
    assert events == [("registered", "one"), ("unregistered", "one")]
    with pytest.raises(UnknownToolError):
        reg.unregister("one")


def test_newer_version_hot_swaps_but_same_or_older_needs_replace():
    reg = ToolRegistry()
    reg.register(tool("one", version="1.2.0"))
    reg.register(tool("one", version="1.10.0"))  # numeric compare: 1.10 > 1.2
    assert reg.get("one").spec.version == "1.10.0"
    with pytest.raises(DuplicateToolError):
        reg.register(tool("one", version="1.10.0"))
    reg.register(tool("one", version="1.0.0"), replace=True)
    assert reg.get("one").spec.version == "1.0.0"


def test_disable_hides_tool_without_forgetting_it():
    reg = ToolRegistry()
    reg.register(tool("one"))
    reg.disable("one")
    assert reg.providers("a.b") == [] and reg.providers("a.b", only_available=False)
    reg.enable("one")
    assert reg.providers("a.b")


def test_circuit_breaker_opens_then_recovers_after_cooldown():
    now = [0.0]
    reg = ToolRegistry(failure_threshold=2, cooldown=10, clock=lambda: now[0])
    reg.register(tool("one"))
    reg.record_failure("one")
    assert reg.is_available("one")
    reg.record_failure("one")
    assert not reg.is_available("one") and reg.circuit_open("one")
    now[0] = 11
    assert reg.is_available("one")  # half-open: probe allowed
    reg.record_failure("one")  # failing the probe re-opens immediately
    assert not reg.is_available("one")
    now[0] = 30
    reg.record_success("one")
    assert reg.health("one").consecutive_failures == 0


def test_invalid_capability_name_rejected():
    with pytest.raises(ValueError):
        ToolSpec(name="x", description="d", capabilities=("Bad Name",))


def test_plugin_loading_from_file(tmp_path):
    plugin = tmp_path / "myplugin.py"
    plugin.write_text(
        "from orchestrator import FunctionTool, ToolSpec\n"
        "def register(registry):\n"
        "    registry.register(FunctionTool(ToolSpec(name='plug', description='d', capabilities=('p.q',)), lambda a: 7))\n"
    )
    reg = ToolRegistry()
    reg.load_plugin(str(plugin))
    assert reg.get("plug").run({}) == 7
    (tmp_path / "bad.py").write_text("x = 1\n")
    with pytest.raises(ImportError):
        reg.load_plugin(str(tmp_path / "bad.py"))
