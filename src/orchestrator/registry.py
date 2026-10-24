"""Dynamic tool registry: runtime register / hot-swap / disable, with circuit breakers."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DuplicateToolError, UnknownToolError
from .models import Claim, ToolSpec, version_key


class Tool(ABC):
    """A callable capability provider. ``run`` may be sync or ``async``."""

    spec: ToolSpec

    @abstractmethod
    def run(self, args: dict[str, Any]) -> Any: ...

    def claims(self, args: dict[str, Any]) -> list[Claim]:
        """Resources this call touches. Conflicting calls are serialized by the executor."""
        return []


class FunctionTool(Tool):
    """Adapter that turns a plain function into a :class:`Tool`."""

    def __init__(
        self,
        spec: ToolSpec,
        fn: Callable[[dict[str, Any]], Any],
        claims: Callable[[dict[str, Any]], list[Claim]] | None = None,
    ) -> None:
        self.spec = spec
        self._fn = fn
        self._claims = claims
        if inspect.iscoroutinefunction(fn):
            self.run = fn  # type: ignore[method-assign,assignment]

    def run(self, args: dict[str, Any]) -> Any:  # noqa: D102 - replaced for coroutine functions
        return self._fn(args)

    def claims(self, args: dict[str, Any]) -> list[Claim]:
        return self._claims(args) if self._claims else []


@dataclass
class ToolHealth:
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    open_until: float | None = None

    @property
    def reliability(self) -> float:
        """Laplace-smoothed success rate: untried tools start at 0.5, not 0 or 1."""
        return (self.successes + 1) / (self.successes + self.failures + 2)


RegistryListener = Callable[[str, str], None]


class ToolRegistry:
    """Thread-safe registry indexed by name and by capability.

    * A higher ``version`` of an existing name hot-swaps it; same/lower versions need ``replace=True``.
    * Tools can be disabled/enabled without losing their health history.
    * A circuit breaker opens after ``failure_threshold`` consecutive failures and
      lets the tool be probed again after ``cooldown`` seconds.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._health: dict[str, ToolHealth] = {}
        self._disabled: set[str] = set()
        self._listeners: list[RegistryListener] = []
        self._lock = threading.RLock()
        self._threshold = failure_threshold
        self._cooldown = cooldown
        self._clock = clock

    # -- membership ---------------------------------------------------------------------
    def register(self, tool: Tool, *, replace: bool = False) -> None:
        name = tool.spec.name
        with self._lock:
            existing = self._tools.get(name)
            if existing is not None:
                upgrade = version_key(tool.spec.version) > version_key(existing.spec.version)
                if not (replace or upgrade):
                    raise DuplicateToolError(
                        f"{name!r} v{existing.spec.version} is registered; "
                        f"v{tool.spec.version} is not newer (pass replace=True to force)"
                    )
            self._tools[name] = tool
            self._health[name] = ToolHealth()
            event = "replaced" if existing is not None else "registered"
        self._emit(event, name)

    def register_many(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        with self._lock:
            if name not in self._tools:
                raise UnknownToolError(name)
            del self._tools[name]
            self._health.pop(name, None)
            self._disabled.discard(name)
        self._emit("unregistered", name)

    def get(self, name: str) -> Tool:
        with self._lock:
            try:
                return self._tools[name]
            except KeyError:
                raise UnknownToolError(name) from None

    def tools(self) -> list[Tool]:
        with self._lock:
            return sorted(self._tools.values(), key=lambda t: t.spec.name)

    def disable(self, name: str) -> None:
        self.get(name)
        with self._lock:
            self._disabled.add(name)
        self._emit("disabled", name)

    def enable(self, name: str) -> None:
        self.get(name)
        with self._lock:
            self._disabled.discard(name)
        self._emit("enabled", name)

    # -- discovery ----------------------------------------------------------------------
    def providers(self, capability: str, *, only_available: bool = True) -> list[Tool]:
        with self._lock:
            found = [t for t in self._tools.values() if capability in t.spec.capabilities]
            if only_available:
                found = [t for t in found if self._available(t.spec.name)]
            return sorted(found, key=lambda t: t.spec.name)

    def capabilities(self, *, only_available: bool = True) -> dict[str, list[str]]:
        with self._lock:
            index: dict[str, list[str]] = {}
            for tool in self.tools():
                if only_available and not self._available(tool.spec.name):
                    continue
                for cap in tool.spec.capabilities:
                    index.setdefault(cap, []).append(tool.spec.name)
            return dict(sorted(index.items()))

    # -- health -------------------------------------------------------------------------
    def is_available(self, name: str) -> bool:
        with self._lock:
            return name in self._tools and self._available(name)

    def _available(self, name: str) -> bool:
        if name in self._disabled:
            return False
        until = self._health[name].open_until
        return until is None or self._clock() >= until

    def health(self, name: str) -> ToolHealth:
        with self._lock:
            return self._health[name]

    def circuit_open(self, name: str) -> bool:
        with self._lock:
            until = self._health[name].open_until
            return until is not None and self._clock() < until

    def record_success(self, name: str) -> None:
        with self._lock:
            h = self._health.get(name)
            if h:
                h.successes += 1
                h.consecutive_failures = 0
                h.open_until = None

    def record_failure(self, name: str) -> None:
        opened = False
        with self._lock:
            h = self._health.get(name)
            if not h:
                return
            h.failures += 1
            h.consecutive_failures += 1
            if h.consecutive_failures >= self._threshold:
                h.open_until = self._clock() + self._cooldown
                opened = True
        if opened:
            self._emit("circuit_open", name)

    # -- plugins & events ---------------------------------------------------------------
    def load_plugin(self, ref: str) -> None:
        """Import a module (dotted name or ``.py`` path) and call its ``register(registry)``.

        Plugins are arbitrary code: only load modules you trust.
        """
        if ref.endswith(".py"):
            path = Path(ref).resolve()
            spec = importlib.util.spec_from_file_location(f"_orch_plugin_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load plugin from {ref}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(ref)
        register = getattr(module, "register", None)
        if not callable(register):
            raise ImportError(f"plugin {ref} defines no register(registry) function")
        register(self)

    def subscribe(self, listener: RegistryListener) -> None:
        self._listeners.append(listener)

    def _emit(self, event: str, name: str) -> None:
        for listener in list(self._listeners):
            listener(event, name)
