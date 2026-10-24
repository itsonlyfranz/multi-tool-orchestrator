from orchestrator import FunctionTool, Principal, Router, RoutingStrategy, ToolRegistry, ToolSpec
from orchestrator.permissions import Grant, PermissionPolicy


def names(decision):
    return [c.tool.spec.name for c in decision.candidates]


def test_priority_then_score_ordering(registry, policy, admin):
    d = Router(registry, policy).route("weather.lookup", {"city": "Oslo"}, admin)
    assert names(d) == ["weather_primary", "weather_backup"]


def test_strategies_reorder_equal_priority_tools(registry, policy, admin):
    cheapest = Router(registry, policy, RoutingStrategy.CHEAPEST).route("market.price", {"symbol": "ACME"}, admin)
    fastest = Router(registry, policy, RoutingStrategy.FASTEST).route("market.price", {"symbol": "ACME"}, admin)
    assert names(cheapest)[0] == "price_feed_c" and names(fastest)[0] == "price_feed_a"


def test_observed_failures_demote_a_tool(registry, policy, admin):
    router = Router(registry, policy, RoutingStrategy.MOST_RELIABLE)
    before = names(router.route("market.price", {"symbol": "ACME"}, admin))[0]
    assert before == "price_feed_a"
    registry._threshold = 100  # keep the breaker closed; only the score should change
    for _ in range(10):
        registry.record_failure("price_feed_a")
    assert names(router.route("market.price", {"symbol": "ACME"}, admin))[0] != "price_feed_a"


def test_open_circuit_and_disabled_tools_are_skipped_with_reason(registry, policy, admin):
    registry.disable("weather_primary")
    d = Router(registry, policy).route("weather.lookup", {"city": "Oslo"}, admin)
    assert names(d) == ["weather_backup"]
    assert d.rejected[0].kind == "unavailable" and "disabled" in d.rejected[0].reason
    registry.record_failure("weather_backup")
    registry.record_failure("weather_backup")
    d = Router(registry, policy).route("weather.lookup", {"city": "Oslo"}, admin)
    assert not d.candidates and not d.denied


def test_tools_missing_required_args_are_rejected(registry, policy, admin):
    d = Router(registry, policy).route("weather.lookup", {}, admin)
    assert not d.candidates and {r.kind for r in d.rejected} == {"args"}


def test_permission_filter_marks_decision_denied(registry, policy):
    reader = Principal(name="r", roles=("reader",))
    d = Router(registry, policy).route("files.write", {"path": "a", "content": "x"}, reader)
    assert d.denied and "files.write" in d.explain()


def test_router_picks_up_tools_registered_after_construction(policy, admin):
    reg = ToolRegistry()
    router = Router(reg, PermissionPolicy(roles={"admin": [Grant()]}))
    assert not router.route("new.cap", {}, admin).candidates
    reg.register(FunctionTool(ToolSpec(name="late", description="d", capabilities=("new.cap",)), lambda a: 1))
    assert names(router.route("new.cap", {}, admin)) == ["late"]
