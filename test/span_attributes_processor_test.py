import unittest

from opentelemetry.sdk.trace import TracerProvider
from harness_sdk.span_attributes_processor import SpanAttributesProcessor


class TestSpanAttributesProcessor(unittest.TestCase):
    def setUp(self):
        self.tracer_provider = TracerProvider()
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def create_test_span(self):
        return self.tracer.start_span("test-span")

    def test_attributes_set_on_start(self):
        processor = SpanAttributesProcessor({"env": "prod", "team": "platform"})
        span = self.create_test_span()

        processor.on_start(span)

        assert span.attributes.get("env") == "prod"
        assert span.attributes.get("team") == "platform"

    def test_empty_attributes(self):
        processor = SpanAttributesProcessor({})
        span = self.create_test_span()

        processor.on_start(span)

        assert "env" not in (span.attributes or {})

    def test_service_name_set(self):
        processor = SpanAttributesProcessor({"service.name": "my-service", "env": "prod"})
        span = self.create_test_span()

        processor.on_start(span)

        assert span.attributes.get("service.name") == "my-service"
        assert span.attributes.get("env") == "prod"

    def test_on_end_does_not_modify_span(self):
        processor = SpanAttributesProcessor({"env": "prod"})
        span = self.create_test_span()
        processor.on_start(span)

        processor.on_end(span)

        assert span.attributes.get("env") == "prod"

    def test_force_flush_returns_true(self):
        processor = SpanAttributesProcessor({})
        assert processor.force_flush() is True
        assert processor.force_flush(5000) is True

    def test_shutdown_does_not_raise(self):
        processor = SpanAttributesProcessor({"env": "prod"})
        processor.shutdown()


if __name__ == "__main__":
    unittest.main()
