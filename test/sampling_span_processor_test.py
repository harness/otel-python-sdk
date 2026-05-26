import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from harness_sdk.sampling_span_processor import SamplingSpanProcessor


class TestSamplingSpanProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_processor = MagicMock()
        self.sampling_processor = SamplingSpanProcessor(processor=self.mock_processor)
        self.tracer_provider = TracerProvider()
        self.tracer = self.tracer_provider.get_tracer(__name__)

        self.mock_registry = MagicMock()
        self.registry_patcher = patch(
            'harness_sdk.sampling_span_processor.get_control_registry',
            return_value=self.mock_registry,
        )
        self.registry_patcher.start()

    def tearDown(self):
        self.registry_patcher.stop()

    def create_test_span(self, attributes=None):
        """Helper method to create a test span with the given attributes."""
        with self.tracer.start_as_current_span("test-span") as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span

    def test_on_start_delegates_to_processor(self):
        span = self.create_test_span()
        parent_context = MagicMock()

        self.sampling_processor.on_start(span, parent_context)

        self.mock_processor.on_start.assert_called_once_with(span, parent_context)

    def test_non_db_span_is_passed_through(self):
        span = self.create_test_span({"other_attr": "value"})

        self.sampling_processor.on_end(span)

        self.mock_processor.on_end.assert_called_once_with(span)
        self.mock_registry.evaluate.assert_not_called()

    def test_mysql_span_passes_registry_filter(self):
        span = self.create_test_span({
            "db.system": "mysql",
            "db.statement": "SELECT * FROM users"
        })

        mock_filter_result = MagicMock()
        mock_filter_result.block = False
        self.mock_registry.evaluate.return_value = mock_filter_result

        self.sampling_processor.on_end(span)

        self.mock_registry.evaluate.assert_called_once_with(
            span, "", {}, "SELECT * FROM users", False
        )
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_postgresql_span_passes_registry_filter(self):
        span = self.create_test_span({
            "db.system": "postgresql",
            "db.statement": "INSERT INTO users VALUES (1, 'test')"
        })

        mock_filter_result = MagicMock()
        mock_filter_result.block = False
        self.mock_registry.evaluate.return_value = mock_filter_result

        self.sampling_processor.on_end(span)

        self.mock_registry.evaluate.assert_called_once_with(
            span, "", {}, "INSERT INTO users VALUES (1, 'test')", False
        )
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_db_span_blocked_by_registry_filter(self):
        span = self.create_test_span({
            "db.system": "mysql",
            "db.statement": "SELECT * FROM sensitive_data"
        })

        mock_filter_result = MagicMock()
        mock_filter_result.block = True
        self.mock_registry.evaluate.return_value = mock_filter_result

        self.sampling_processor.on_end(span)

        self.mock_registry.evaluate.assert_called_once()
        self.mock_processor.on_end.assert_not_called()

    def test_db_span_without_statement(self):
        span = self.create_test_span({
            "db.system": "mysql"
        })

        mock_filter_result = MagicMock()
        mock_filter_result.block = False
        self.mock_registry.evaluate.return_value = mock_filter_result

        self.sampling_processor.on_end(span)

        self.mock_registry.evaluate.assert_called_once_with(
            span, "", {}, "", False
        )
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_registry_filter_exception_handling(self):
        span = self.create_test_span({
            "db.system": "mysql",
            "db.statement": "SELECT * FROM users"
        })

        self.mock_registry.evaluate.side_effect = Exception("Test exception")

        self.sampling_processor.on_end(span)

        self.mock_processor.on_end.assert_called_once_with(span)

    def test_force_flush_delegates_to_processor(self):
        timeout = 5000
        self.sampling_processor.force_flush(timeout)
        self.mock_processor.force_flush.assert_called_once_with(timeout)

    def test_shutdown_delegates_to_processor(self):
        self.sampling_processor.shutdown()
        self.mock_processor.shutdown.assert_called_once()
