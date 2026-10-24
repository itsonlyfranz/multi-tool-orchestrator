"""Append-only audit trail of routing, permission, execution and conflict events."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class AuditLog:
    """Thread-safe event log, optionally mirrored to a JSONL file."""

    def __init__(self, path: str | Path | None = None, max_events: int = 10_000) -> None:
        self._path = Path(path) if path else None
        self._events: list[dict[str, Any]] = []
        self._max = max_events
        self._lock = threading.Lock()

    def record(self, kind: str, run_id: str = "", **data: Any) -> None:
        event = {"ts": round(time.time(), 3), "run_id": run_id, "kind": kind, **data}
        with self._lock:
            self._events.append(event)
            del self._events[: -self._max]
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")

    def events(self, kind: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                e
                for e in self._events
                if (kind is None or e["kind"] == kind) and (run_id is None or e["run_id"] == run_id)
            ]
