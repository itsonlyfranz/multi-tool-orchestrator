"""Regenerate the agent flow diagram in README.md from the real compiled LangGraph graph.

    python scripts/render_graph.py

The Mermaid block between the ``flow-diagram`` markers is replaced in place; everything else in
the README is left alone. ``tests/test_readme_diagrams.py`` fails when the block drifts.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

README = ROOT / "README.md"
START_MARK = "<!-- flow-diagram:start -->"
END_MARK = "<!-- flow-diagram:end -->"
BLOCK_RE = re.compile(
    re.escape(START_MARK) + r"\s*```mermaid\n(?P<body>.*?)```\s*" + re.escape(END_MARK), re.DOTALL
)
NODE_RE = re.compile(r"^\s*(\w+)\s*[(\[]")
EDGE_RE = re.compile(r"^\s*(\w+)\s*(?:-->|-\.->|--\s.*?\s-->|-\.\s.*?\s\.->)\s*(\w+)\s*;?\s*$")


def compiled_graph():
    """The orchestrator's real compiled graph, built offline with a scripted LLM."""
    from orchestrator import Orchestrator, PermissionPolicy, ScriptedLLM, ToolRegistry

    workspace = Path(tempfile.mkdtemp())
    return Orchestrator(ScriptedLLM([]), ToolRegistry(), PermissionPolicy(base_dir=workspace)).graph.get_graph()


def render_flow() -> str:
    return compiled_graph().draw_mermaid().strip() + "\n"


def parse_mermaid(source: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Node ids and (source, target) edges of a LangGraph-style ``graph TD`` Mermaid block."""
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for line in source.splitlines():
        if edge := EDGE_RE.match(line):
            edges.add((edge[1], edge[2]))
        elif (node := NODE_RE.match(line)) and node[1] not in {"graph", "classDef"}:
            nodes.add(node[1])
    return nodes, edges


def readme_flow_block(readme: str) -> str:
    match = BLOCK_RE.search(readme)
    if not match:
        raise SystemExit(f"README.md has no mermaid block between {START_MARK} and {END_MARK}")
    return match["body"]


def main() -> None:
    readme = README.read_text(encoding="utf-8")
    readme_flow_block(readme)  # fail early if the markers are missing
    block = f"{START_MARK}\n```mermaid\n{render_flow()}```\n{END_MARK}"
    README.write_text(BLOCK_RE.sub(lambda _: block, readme, count=1), encoding="utf-8")
    print(f"updated flow diagram in {README}")


if __name__ == "__main__":
    main()
