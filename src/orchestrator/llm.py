"""Minimal LLM interface so the orchestrator never depends on a specific provider.

The system prompts the orchestrator sends start with a purpose tag
(``[planner]``, ``[synthesizer]``, ``[arbiter]``) which offline fakes can dispatch on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import PlanningError


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str: ...


class ChatModelClient:
    """Adapts any LangChain chat model (``BaseChatModel``) to :class:`LLMClient`."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        reply = self.model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        content = reply.content
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def claude_client(model: str, api_key: str | None = None, max_tokens: int = 2048) -> ChatModelClient:
    """Claude through ``langchain-anthropic``. The key defaults to ANTHROPIC_API_KEY."""
    from langchain_anthropic import ChatAnthropic

    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatModelClient(ChatAnthropic(**kwargs))


Responder = str | Callable[[str, str], str]


@dataclass
class ScriptedLLM:
    """Test double: returns queued responses in order and records every call.

    A response may be a string or a ``(system, prompt) -> str`` callable.
    """

    responses: Iterable[Responder]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue = list(self.responses)

    def complete(self, system: str, prompt: str, max_tokens: int = 2048) -> str:
        self.calls.append((system, prompt))
        if not self._queue:
            raise AssertionError("ScriptedLLM has no responses left")
        item = self._queue.pop(0)
        return item(system, prompt) if callable(item) else item


def extract_json(text: str) -> Any:
    """Parse the first JSON object in an LLM reply, tolerating code fences and chatter."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise PlanningError("no JSON object found in model reply")
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise PlanningError(f"invalid JSON in model reply: {exc}") from exc
    return value
