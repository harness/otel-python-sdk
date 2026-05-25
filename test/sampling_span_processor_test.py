import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from agent_trace.sampling_span_processor import SamplingSpanProcessor
from agent_trace.filter.registry import Registry


class TestSamplingSpanProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_processor = MagicMock()
        self.sampling_processor = SamplingSpanProcessor(processor=self.mock_processor)
        self.tracer_provider = TracerProvider()
        self.tracer = self.tracer_provider.get_tracer(__name__)
        
        # Mock Registry instance
        self.registry_patcher = patch('agent_trace.sampling_span_processor.Registry')
        self.mock_registry_class = self.registry_patcher.start()
        self.mock_registry = MagicMock()
        self.mock_registry_class.return_value = self.mock_registry

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
        """Test that on_start is properly delegated to the underlying processor."""
        span = self.create_test_span()
        parent_context = MagicMock()
        
        self.sampling_processor.on_start(span, parent_context)
        
        self.mock_processor.on_start.assert_called_once_with(span, parent_context)

    def test_non_db_span_is_passed_through(self):
        """Test that non-database spans are passed through without filtering."""
        span = self.create_test_span({"other_attr": "value"})
        
        self.sampling_processor.on_end(span)
        
        # Verify the span was passed through to the underlying processor
        self.mock_processor.on_end.assert_called_once_with(span)
        # Registry filter should not be called for non-DB spans
        self.mock_registry.apply_filter.assert_not_called()

    def test_mysql_span_passes_registry_filter(self):
        """Test that MySQL spans are passed through when they pass the Registry filter."""
        # Create a MySQL span with a SQL statement
        span = self.create_test_span({
            "db.system": "mysql",
            "db.statement": "SELECT * FROM users"
        })
        
        # Configure mock registry to allow the span (not block)
        mock_filter_result = MagicMock()
        mock_filter_result.block = False
        self.mock_registry.apply_filter.return_value = mock_filter_result
        
        self.sampling_processor.on_end(span)
        
        # Verify Registry filter was called with correct parameters
        self.mock_registry.apply_filter.assert_called_once_with(
            span, "", {}, "SELECT * FROM users", False
        )
        
        # Verify the span was passed through to the underlying processor
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_postgresql_span_passes_registry_filter(self):
        """Test that PostgreSQL spans are passed through when they pass the Registry filter."""
        # Create a PostgreSQL span with a SQL statement
        span = self.create_test_span({
            "db.system": "postgresql",
            "db.statement": "INSERT INTO users VALUES (1, 'test')"
        })
        
        # Configure mock registry to allow the span (not block)
        mock_filter_result = MagicMock()
        mock_filter_result.block = False
        self.mock_registry.apply_filter.return_value = mock_filter_result
        
        self.sampling_processor.on_end(span)
        
        # Verify Registry filter was called with correct parameters
        self.mock_registry.apply_filter.assert_called_once_with(
            span, "", {}, "INSERT INTO users VALUES (1, 'test')", False
        )
        
        # Verify the span was passed through to the underlying processor
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_db_span_blocked_by_registry_filter(self):
        """Test that database spans are filtered out when blocked by Registry filter."""
        # Create a MySQL span with a SQL statement
        span = self.create_test_span({
            "db.system": "mysql",
            "db.statement": "SELECT * FROM sensitive_data"
        })
        
        # Configure mock registry to block the span
        mock_filter_result = MagicMock()
        mock_filter_result.block = True
        self.mock_registry.apply_filter.return_value = mock_filter_result
        
        self.sampling_processor.on_end(span)
        
        # Verify Registry filter was called
        self.mock_registry.apply_filter.assert_called_once()
        
        # Verify the span was NOT passed through to the underlying processor
        self.mock_processor.on_end.assert_not_called()

    def test_db_span_without_statement(self):
        """Test that database spans without a statement are still processed."""
        # Create a MySQL span without a SQL statement
        span = self.create_test_span({
            "db.system": "mysql"
        })
        
        # Configure mock registry to allow the span (not block)
        mock_filter_result = MagicMock()
        mock_filter_result.block = False
        self.mock_registry.apply_filter.return_value = mock_filter_result
        
        self.sampling_processor.on_end(span)
        
        # Verify Registry filter was called with empty statement
        self.mock_registry.apply_filter.assert_called_once_with(
            span, "", {}, "", False
        )
        
        # Verify the span was passed through to the underlying processor
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_registry_filter_exception_handling(self):
        """Test that exceptions in Registry filter are properly handled."""
        # Create a MySQL span
        span = self.create_test_span({
            "db.system": "mysql",
            "db.statement": "SELECT * FROM users"
        })
        
        # Configure mock registry to raise an exception
        self.mock_registry.apply_filter.side_effect = Exception("Test exception")
        
        # Should not raise an exception
        self.sampling_processor.on_end(span)
        
        # Verify the span was still passed through to the underlying processor
        self.mock_processor.on_end.assert_called_once_with(span)

    def test_force_flush_delegates_to_processor(self):
        """Test that force_flush is properly delegated to the underlying processor."""
        timeout = 5000
        self.sampling_processor.force_flush(timeout)
        self.mock_processor.force_flush.assert_called_once_with(timeout)

    def test_shutdown_delegates_to_processor(self):
        """Test that shutdown is properly delegated to the underlying processor."""
        self.sampling_processor.shutdown()
        self.mock_processor.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
