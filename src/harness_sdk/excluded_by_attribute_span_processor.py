from opentelemetry.sdk.trace import SpanProcessor


class ExcludeByAttributeSpanProcessor(SpanProcessor):
    """
    A span processor that excludes spans with a specific attribute value.

    Spans are processed (passed through) if:
    - They don't have the specified attribute, OR
    - The attribute value doesn't match the excluded value

    Spans are excluded (filtered out) if:
    - They have the specified attribute AND its value matches the excluded value
    """

    def __init__(self, processor, attribute_name, excluded_value):
        """
        Args:
            processor: The span processor to wrap
            attribute_name: Name of the attribute to check
            excluded_value: Value that will cause a span to be excluded if it matches the attribute's value
        """
        self._processor = processor
        self._attribute_name = attribute_name
        self._excluded_value = excluded_value

    def on_start(self, span, parent_context=None):
        self._processor.on_start(span, parent_context)

    def on_end(self, span):
        # Only process spans that don't have the excluded attribute value
        span_attributes = span.attributes or {}
        if span_attributes.get(self._attribute_name) != self._excluded_value:
            self._processor.on_end(span)

    def force_flush(self, timeout_millis=30000):
        return self._processor.force_flush(timeout_millis)

    def shutdown(self):
        return self._processor.shutdown()
