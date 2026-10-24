"""Exception hierarchy shared across the orchestrator."""


class OrchestratorError(Exception):
    """Base class for every error raised by this package."""


class DuplicateToolError(OrchestratorError):
    """A tool with this name is already registered and replacement was not requested."""


class UnknownToolError(OrchestratorError):
    """No tool with this name is registered."""


class PlanningError(OrchestratorError):
    """The planner could not produce a valid plan; ``reply`` is the rejected model output."""

    def __init__(self, message: str, reply: str = "") -> None:
        super().__init__(message)
        self.reply = reply


class RefError(OrchestratorError):
    """A ``${step.path}`` reference could not be resolved."""


class ToolError(Exception):
    """Raised by a tool to signal a failure that may succeed on retry."""


class PermanentToolError(ToolError):
    """Raised by a tool when retrying is pointless (bad arguments, sandbox violation)."""
