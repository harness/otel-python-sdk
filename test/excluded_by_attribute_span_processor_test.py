import unittest
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from harness_sdk.excluded_by_attribute_span_processor import ExcludeByAttributeSpanProcessor


class TestFilteringSpanProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_processor = MagicMock()
        self.filtering_processor = ExcludeByAttributeSpanProcessor(
            processor=self.mock_processor,
            attribute_name="traceableai.span_kind",
            excluded_value="no_span"
        )
        self.tracer_provider = TracerProvider()
        self.tracer = self.tracer_provider.get_tracer(__name__)

    def create_test_span(self, attributes=None):
        """Helper method to create a test span with the given attributes."""
        with self.tracer.start_as_current_span("test-span") as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span

    def test_span_without_span_kind_is_passed_through(self):
        """Test that spans without traceableai.span_kind are passed through."""
        span = self.create_test_span({"other_attr": "value"})

        self.filtering_processor.on_start(span)
        self.filtering_processor.on_end(span)

        # Verify the span was passed through to the underlying processor
        self.mock_processor.on_start.assert_called_once()
        self.mock_processor.on_end.assert_called_once()

    def test_span_with_different_span_kind_is_passed_through(self):
        """Test that spans with traceableai.span_kind not equal to 'no_span' are passed through."""
        span = self.create_test_span({"traceableai.span_kind": "some_other_kind"})

        self.filtering_processor.on_start(span)
        self.filtering_processor.on_end(span)

        # Verify the span was passed through to the underlying processor
        self.mock_processor.on_start.assert_called_once()
        self.mock_processor.on_end.assert_called_once()

    def test_span_with_no_span_is_filtered_out(self):
        """Test that spans with traceableai.span_kind='no_span' are filtered out."""
        span = self.create_test_span({"traceableai.span_kind": "no_span"})

        self.filtering_processor.on_start(span)
        self.filtering_processor.on_end(span)

        # on_start should be called (it doesn't have access to attributes yet)
        self.mock_processor.on_start.assert_called_once()
        # But on_end should not be called for the filtered span
        self.mock_processor.on_end.assert_not_called()

    def test_force_flush_delegates_to_processor(self):
        """Test that force_flush is properly delegated to the underlying processor."""
        self.filtering_processor.force_flush(5000)
        self.mock_processor.force_flush.assert_called_once_with(5000)

    def test_shutdown_delegates_to_processor(self):
        """Test that shutdown is properly delegated to the underlying processor."""
        self.filtering_processor.shutdown()
        self.mock_processor.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()