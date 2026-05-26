"""Control plugin protocol and registry for request/span policy evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

from opentelemetry.trace import Span


@dataclass
class ControlResult:
    """Outcome of a control plugin evaluation."""

    block: bool = False
    propagate: bool = False
    response_status_code: int = 403
    response_message: str = "Forbidden"
    attributes: List[tuple] = field(default_factory=list)
    header_injections: List[tuple] = field(default_factory=list)
    span_type: Optional[Any] = None

    def add_attributes_to_span(self, span: Span) -> None:
        for key, value in self.attributes:
            span.set_attribute(key, value)
        if self.span_type is not None:
            span.set_attribute("span_type", str(self.span_type))


@runtime_checkable
class ControlPlugin(Protocol):
    """Policy plugin invoked on ingress and GenAI spans before downstream work."""

    name: str

    def on_init(self, config: Any) -> None:
        ...

    def evaluate(
        self,
        span: Span,
        url: str,
        headers: dict,
        body,
        is_grpc: bool,
    ) -> ControlResult:
        ...

    def evaluate_agent_span(self, span: Span, body: str = "") -> ControlResult:
        ...

    def shutdown(self) -> None:
        ...


class ControlRegistry:
    """Singleton registry that chains control plugins (short-circuit on block)."""

    _instance: Optional["ControlRegistry"] = None

    def __new__(cls) -> "ControlRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._plugins: List[ControlPlugin] = []
        self._last_result = ControlResult()
        self._has_blocking_capability = False

    def register(self, plugin: ControlPlugin) -> None:
        self._plugins.append(plugin)
        if getattr(plugin, "provides_blocking", False):
            self._has_blocking_capability = True

    def clear(self) -> None:
        self._plugins.clear()
        self._has_blocking_capability = False
        self._last_result = ControlResult()

    def has_blocking_capability(self) -> bool:
        return self._has_blocking_capability

    def evaluate(
        self,
        span: Span,
        url: str,
        headers: dict,
        body,
        is_grpc: bool,
    ) -> ControlResult:
        result = ControlResult()
        for plugin in self._plugins:
            result = plugin.evaluate(span, url, headers, body, is_grpc)
            self._last_result = result
            if result.block:
                return result
        self._last_result = result
        return result

    def last_evaluate_result(self) -> ControlResult:
        return self._last_result

    def evaluate_agent_span(self, span: Span, body: str = "") -> ControlResult:
        from harness_sdk.config.config import Config  # pylint: disable=import-outside-toplevel

        if not Config().config.gen_ai.payload_evaluation_enabled.value:
            return ControlResult()
        result = ControlResult()
        for plugin in self._plugins:
            result = plugin.evaluate_agent_span(span, body)
            if result.block:
                return result
        return result


def get_control_registry() -> ControlRegistry:
    return ControlRegistry()
