"""Public helpers for enriching the current OpenTelemetry span."""

from typing import Mapping

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue


def set_span_attribute(key: str, value: AttributeValue) -> None:
    """Set one attribute on the current recording span."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, value)


def set_span_attributes(attributes: Mapping[str, AttributeValue]) -> None:
    """Set attributes on the current recording span."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)
