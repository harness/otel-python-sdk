from opentelemetry.sdk.trace import SpanProcessor


def span_kind_name(span) -> str | None:
    """Return OTel span kind as a lowercase string (client, server, internal, ...)."""
    kind = span.kind
    if kind is None:
        return None
    if hasattr(kind, "name"):
        return kind.name.lower()
    kind_str = str(kind)
    if kind_str.startswith("SpanKind."):
        return kind_str[len("SpanKind."):].lower()
    return kind_str.lower()


class SpanAttributesProcessor(SpanProcessor):
    def __init__(self, attributes: dict):
        self._attributes = attributes

    def on_start(self, span, parent_context=None):
        for key, value in self._attributes.items():
            span.set_attribute(key, value)
        span_kind = span_kind_name(span)
        if span_kind:
            span.set_attribute("span.kind", span_kind)

    def on_end(self, span):
        pass

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        pass
