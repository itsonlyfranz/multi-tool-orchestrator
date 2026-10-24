"""Local, deterministic tool pack used by the demo, the CLI and the tests (no network)."""

from __future__ import annotations

import ast
import operator
import re
import time
from pathlib import Path
from typing import Any

from ..errors import PermanentToolError, ToolError
from ..models import Claim, ToolSpec
from ..registry import FunctionTool, Tool

_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

CORPUS = {
    "routing": "Capability routing selects a tool by what it can do instead of by name. Rankings blend priority, reliability, trust, cost and latency.",
    "permissions": "Permission scoping is deny-by-default. Roles hold allow and deny grants, and argument constraints confine paths and numeric ranges.",
    "parallelism": "Independent plan steps run concurrently. Dependencies are expressed as references to earlier step outputs.",
    "conflicts": "Conflicting writes are serialized in plan order. Disagreeing tool answers are reconciled by priority, majority, trust-weighted vote or median.",
}
WEATHER = {"lisbon": (22.0, "sunny"), "oslo": (9.0, "rain"), "cairo": (33.0, "clear")}
PRICES = {"ACME": (187.42, 185.0, 185.0)}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic only (no names, calls or attributes)."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise PermanentToolError(f"unsupported expression element: {type(node).__name__}")

    try:
        return walk(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
        raise PermanentToolError(f"cannot evaluate {expression!r}: {exc}") from exc


def build_demo_tools(workspace: Path, *, latency: float = 0.0, weather_primary_down: bool = False) -> list[Tool]:
    """Create the demo tool pack rooted at ``workspace``. ``latency`` simulates slow I/O (seconds)."""
    root = workspace.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def nap() -> None:
        if latency:
            time.sleep(latency)

    def inside(path: str) -> Path:
        target = (root / path).resolve()
        if target != root and not target.is_relative_to(root):
            raise PermanentToolError(f"path {path!r} escapes the workspace")
        return target

    def file_claim(mode: str):
        return lambda a: [Claim(resource=f"file:{inside(a['path']).relative_to(root)}", mode=mode)]  # type: ignore[arg-type]

    def read_file(a: dict[str, Any]) -> str:
        nap()
        target = inside(a["path"])
        if not target.is_file():
            raise PermanentToolError(f"no such file: {a['path']}")
        return target.read_text(encoding="utf-8")

    def write_file(a: dict[str, Any]) -> dict[str, Any]:
        nap()
        target = inside(a["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        text = str(a["content"])
        if a.get("append"):
            with target.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        else:
            target.write_text(text, encoding="utf-8")
        return {"path": a["path"], "bytes": len(text.encode())}

    def delete_file(a: dict[str, Any]) -> dict[str, Any]:
        target = inside(a["path"])
        if not target.is_file():
            raise PermanentToolError(f"no such file: {a['path']}")
        target.unlink()
        return {"deleted": a["path"]}

    def summarize(a: dict[str, Any]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", str(a["text"]).strip())
        return " ".join(sentences[: int(a.get("max_sentences", 2))])

    def analyze(a: dict[str, Any]) -> dict[str, Any]:
        words = re.findall(r"\w+", str(a["text"]).lower())
        return {"words": len(words), "unique": len(set(words))}

    def search(a: dict[str, Any]) -> list[dict[str, Any]]:
        nap()
        terms = set(re.findall(r"\w+", str(a["query"]).lower()))
        hits = []
        for key, text in CORPUS.items():
            score = len(terms & set(re.findall(r"\w+", text.lower()))) + (2 if key in terms else 0)
            if score:
                hits.append({"doc": key, "score": score, "text": text})
        return sorted(hits, key=lambda h: (-h["score"], h["doc"]))[:3]

    def weather(down: bool):
        def fn(a: dict[str, Any]) -> dict[str, Any]:
            nap()
            if down:
                raise ToolError("upstream 503")
            city = str(a["city"]).lower()
            if city not in WEATHER:
                raise PermanentToolError(f"unknown city {a['city']!r}")
            temp, sky = WEATHER[city]
            return {"city": a["city"], "temp_c": temp, "sky": sky}

        return fn

    def price(index: int):
        def fn(a: dict[str, Any]) -> float:
            nap()
            symbol = str(a["symbol"]).upper()
            if symbol not in PRICES:
                raise PermanentToolError(f"unknown symbol {a['symbol']!r}")
            return PRICES[symbol][index]

        return fn

    path_schema = {"type": "object", "properties": {"path": {}}, "required": ["path"]}

    def spec(name: str, desc: str, caps: tuple[str, ...], props: dict[str, Any], required: list[str], **kw: Any) -> ToolSpec:
        schema = {"type": "object", "properties": props, "required": required}
        return ToolSpec(name=name, description=desc, capabilities=caps, input_schema=schema, **kw)

    return [
        FunctionTool(spec("calculator", "Evaluate an arithmetic expression", ("math.evaluate",),
                          {"expression": {}}, ["expression"], cost=0.1, latency_ms=1, trust=0.95),
                     lambda a: safe_eval(str(a["expression"]))),
        FunctionTool(spec("read_file", "Read a UTF-8 file from the workspace", ("files.read",),
                          {"path": {}}, ["path"], cost=0.1, latency_ms=5, trust=0.9),
                     read_file, file_claim("read")),
        FunctionTool(spec("write_file", "Write or append text to a workspace file", ("files.write",),
                          {"path": {}, "content": {}, "append": {}}, ["path", "content"], effect="write",
                          cost=0.2, latency_ms=10, trust=0.9),
                     write_file, file_claim("write")),
        FunctionTool(spec("delete_file", "Delete a workspace file", ("files.delete",),
                          path_schema["properties"], ["path"], effect="write", cost=0.2, latency_ms=5, trust=0.9),
                     delete_file, file_claim("write")),
        FunctionTool(spec("summarize_text", "Extractive summary: first N sentences", ("text.summarize",),
                          {"text": {}, "max_sentences": {}}, ["text"], cost=0.1, latency_ms=2, trust=0.7),
                     summarize),
        FunctionTool(spec("word_stats", "Word and unique-word counts", ("text.analyze",),
                          {"text": {}}, ["text"], cost=0.1, latency_ms=1, trust=0.9),
                     analyze),
        FunctionTool(spec("doc_search", "Keyword search over the built-in knowledge base", ("search.documents",),
                          {"query": {}}, ["query"], cost=0.5, latency_ms=200, trust=0.8),
                     search),
        FunctionTool(spec("weather_primary", "Primary weather API (fast, preferred)", ("weather.lookup",),
                          {"city": {}}, ["city"], priority=10, cost=1.0, latency_ms=80, trust=0.9, effect="external"),
                     weather(weather_primary_down)),
        FunctionTool(spec("weather_backup", "Backup weather API (slower, cheaper)", ("weather.lookup",),
                          {"city": {}}, ["city"], priority=0, cost=0.3, latency_ms=300, trust=0.6, effect="external"),
                     weather(False)),
        FunctionTool(spec("price_feed_a", "Premium market data feed", ("market.price",),
                          {"symbol": {}}, ["symbol"], cost=2.0, latency_ms=50, trust=0.9, effect="external"), price(0)),
        FunctionTool(spec("price_feed_b", "Aggregator market data feed", ("market.price",),
                          {"symbol": {}}, ["symbol"], cost=0.5, latency_ms=120, trust=0.4, effect="external"), price(1)),
        FunctionTool(spec("price_feed_c", "Delayed public market data feed", ("market.price",),
                          {"symbol": {}}, ["symbol"], cost=0.1, latency_ms=300, trust=0.4, effect="external"), price(2)),
    ]
