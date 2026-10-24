from orchestrator import Principal
from orchestrator.permissions import ArgConstraint, PermissionPolicy
from orchestrator.router import Router


def spec(registry, name):
    return registry.get(name).spec


def test_default_deny_for_unknown_role(registry, policy):
    p = Principal(name="x", roles=("nobody",))
    d = policy.authorize(p, spec(registry, "read_file"), "files.read", {"path": "a"})
    assert not d.allowed and "no grant" in d.reason


def test_effect_scoped_role(registry, policy):
    reader = Principal(name="r", roles=("reader",))
    assert policy.authorize(reader, spec(registry, "read_file"), "files.read", {"path": "a"}).allowed
    assert not policy.authorize(reader, spec(registry, "write_file"), "files.write", {"path": "a", "content": ""}).allowed


def test_explicit_deny_beats_allow(registry, policy):
    p = Principal(name="r", roles=("admin", "reader"))
    d = policy.authorize(p, spec(registry, "write_file"), "files.write", {"path": "a", "content": ""})
    assert not d.allowed and "explicitly denied" in d.reason


def test_path_constraint_allows_inside_blocks_outside_and_traversal(registry, policy, analyst):
    write = spec(registry, "write_file")
    assert policy.authorize(analyst, write, "files.write", {"path": "reports/q1.md", "content": "x"}).allowed
    assert not policy.authorize(analyst, write, "files.write", {"path": "secrets.txt", "content": "x"}).allowed
    traversal = policy.authorize(analyst, write, "files.write", {"path": "reports/../secrets.txt", "content": "x"})
    assert not traversal.allowed and "outside" in traversal.reason


def test_symlink_escape_is_blocked(registry, policy, analyst, workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "reports").mkdir()
    (workspace / "reports" / "link").symlink_to(outside)
    d = policy.authorize(analyst, spec(registry, "write_file"), "files.write", {"path": "reports/link/x.txt", "content": ""})
    assert not d.allowed


def test_numeric_and_choice_constraints():
    c = ArgConstraint(min=1, max=10)
    assert c.violation(5, None) is None and c.violation(11, None) and c.violation("5", None)
    assert ArgConstraint(one_of=["a"]).violation("b", None)
    assert ArgConstraint(pattern=r"[A-Z]{4}").violation("ACME", None) is None


def test_scope_narrowing_only_removes_power(registry, policy):
    admin = Principal(name="root", roles=("admin",))
    read_file = spec(registry, "read_file")
    narrowed = admin.narrow("files.*").narrow("files.read")
    assert policy.authorize(narrowed, read_file, "files.read", {"path": "a"}).allowed
    d = policy.authorize(narrowed, spec(registry, "calculator"), "math.evaluate", {"expression": "1"})
    assert not d.allowed and "delegated scope" in d.reason
    assert not policy.authorize(admin.narrow("files.*").narrow("math.*"), read_file, "files.read", {"path": "a"}).allowed


def test_policy_loads_from_json(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"roles": {"r": [{"effect": "allow", "capabilities": ["a.*"], "constraints": {"n": {"max": 3}}}]}}')
    policy = PermissionPolicy.from_file(f)
    assert policy.roles["r"][0].constraints["n"].max == 3


def test_planner_catalog_shows_only_permitted_capabilities(registry, policy):
    router = Router(registry, policy)
    reader_caps = router.permitted_capabilities(Principal(name="r", roles=("reader",)))
    assert "files.read" in reader_caps and "files.write" not in reader_caps and "files.delete" not in reader_caps


def test_unconstrained_grant_without_args_means_could_be_allowed(registry, policy, analyst):
    assert policy.authorize(analyst, spec(registry, "write_file"), "files.write").allowed
