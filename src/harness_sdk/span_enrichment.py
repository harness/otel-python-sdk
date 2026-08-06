"""Public helpers for enriching the current OpenTelemetry span."""

from typing import Any, Mapping, Union

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from harness_sdk.flatten_dict_registry import get_registry, is_flatten_enabled

EnrichmentValue = Union[AttributeValue, Mapping[str, Any]]


def set_span_attribute(key: str, value: EnrichmentValue) -> None:
    """Set one attribute on the current recording span."""
    span = trace.get_current_span()
    if span.is_recording():
        _set(span, key, value)


def set_span_attributes(attributes: Mapping[str, EnrichmentValue]) -> None:
    """Set attributes on the current recording span."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for key, value in attributes.items():
        _set(span, key, value)


def _set(span, key: str, value: EnrichmentValue) -> None:
    # Dict values are parked for FlattenDictSpanProcessor to expand into
    # dot-notation keys at span end; nothing is serialized on this thread.
    if isinstance(value, Mapping) and is_flatten_enabled():
        get_registry().register(span, key, value)
        return
    span.set_attribute(key, value)
