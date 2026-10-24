"""The README's agent-flow diagram must match the compiled LangGraph graph (offline, no key)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def readme_flow() -> tuple[set[str], set[tuple[str, str]]]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return render_graph.parse_mermaid(render_graph.readme_flow_block(readme))


def test_readme_flow_diagram_matches_the_compiled_graph():
    drawn = render_graph.compiled_graph()
    nodes = set(drawn.nodes)
    edges = {(e.source, e.target) for e in drawn.edges}
    assert nodes and edges
    readme_nodes, readme_edges = readme_flow()
    hint = "run `python scripts/render_graph.py` to regenerate the README flow diagram"
    assert readme_nodes == nodes, f"node drift: {sorted(readme_nodes ^ nodes)}; {hint}"
    assert readme_edges == edges, f"edge drift: {sorted(readme_edges ^ edges)}; {hint}"


def test_parser_reads_labels_and_ignores_layout_lines():
    nodes, edges = render_graph.parse_mermaid(
        "graph TD;\n\ta([<p>a</p>]):::first\n\tb(b)\n\tc(c)\n\ta -.-> b;\n\tb --> b;\n\tclassDef x fill:#fff\n"
        "\tb -- &nbsp;done&nbsp; --> a;\n\ta -. &nbsp;retry&nbsp; .-> c;\n"
    )
    assert nodes == {"a", "b", "c"}
    assert edges == {("a", "b"), ("b", "b"), ("b", "a"), ("a", "c")}
