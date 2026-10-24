"""Environment-driven settings and a tiny ``.env`` loader (no extra dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"


def load_dotenv(path: str | Path = ".env") -> None:
    """Populate ``os.environ`` from a KEY=VALUE file without overriding existing variables."""
    file = Path(path)
    if not file.is_file():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    max_concurrency: int
    step_timeout: float
    audit_path: str | None
    workspace: Path

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            api_key=env.get("ANTHROPIC_API_KEY") or None,
            model=env.get("ORCH_MODEL", DEFAULT_MODEL),
            max_concurrency=int(env.get("ORCH_MAX_CONCURRENCY", "8")),
            step_timeout=float(env.get("ORCH_STEP_TIMEOUT", "30")),
            audit_path=env.get("ORCH_AUDIT_PATH") or None,
            workspace=Path(env.get("ORCH_WORKSPACE", "workspace")),
        )
