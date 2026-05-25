"""Exceptions raised when control plugin evaluation blocks a GenAI call."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_trace.plugins.control import ControlResult


class ControlEvaluationBlocked(Exception):
    """Raised when control registry ``evaluate_agent_span`` returns ``block=True``."""

    def __init__(self, result: "ControlResult") -> None:
        self.result = result
        msg = getattr(result, "response_message", None) or "Forbidden"
        super().__init__(msg)
