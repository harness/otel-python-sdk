from opentelemetry.sdk.trace import SpanProcessor


class SpanAttributesProcessor(SpanProcessor):
    def __init__(self, attributes: dict):
        self._attributes = attributes

    def on_start(self, span, parent_context=None):
        for key, value in self._attributes.items():
            span.set_attribute(key, value)

    def on_end(self, span):
        pass

    def force_flush(self, timeout_millis=30000):
        return True

    def shutdown(self):
        pass
