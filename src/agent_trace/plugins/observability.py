"""Observability plugin protocol for span processors and exporters."""
from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from opentelemetry.sdk.trace import SpanProcessor


@runtime_checkable
class ObservabilityPlugin(Protocol):
    """Plugin that contributes span processors to the tracer pipeline."""

    name: str
    priority: int

    def on_init(self, config: Any) -> None:
        ...

    def create_span_processors(self, config: Any) -> List[SpanProcessor]:
        ...

    def shutdown(self) -> None:
        ...
