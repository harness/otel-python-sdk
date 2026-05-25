"""Span attributes processor observability plugin."""
from typing import Any, List

from opentelemetry.sdk.trace import SpanProcessor

from agent_trace.span_attributes_processor import SpanAttributesProcessor


class BuiltinSpanAttributesPlugin:
    name = "builtin_span_attributes"
    priority = 200

    def on_init(self, config: Any) -> None:
        self._config = config

    def create_span_processors(self, config: Any) -> List[SpanProcessor]:
        span_attrs = {"service.name": config.config.service_name.value}
        span_attrs.update(config.config.span_attributes)
        return [SpanAttributesProcessor(span_attrs)]

    def shutdown(self) -> None:
        pass


def factory(config: Any) -> BuiltinSpanAttributesPlugin:
    return BuiltinSpanAttributesPlugin()
