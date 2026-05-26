"""Span processor that applies control plugins to database spans."""
from opentelemetry.sdk.trace import SpanProcessor
from harness_sdk.plugins.control import get_control_registry
from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)


class SamplingSpanProcessor(SpanProcessor):
    """Applies control plugins on database spans before export."""

    DB_SYSTEM_ATTR = "db.system"
    DB_MYSQL = "mysql"
    DB_POSTGRESQL = "postgresql"

    def __init__(self, processor):
        self._processor = processor

    def on_start(self, span, parent_context=None):
        self._processor.on_start(span, parent_context)

    def on_end(self, span):
        span_attributes = span.attributes or {}
        db_system = span_attributes.get(self.DB_SYSTEM_ATTR)

        if db_system in [self.DB_MYSQL, self.DB_POSTGRESQL]:
            try:
                db_statement = span_attributes.get("db.statement", "")
                filter_result = get_control_registry().evaluate(span, "", {}, db_statement, False)
                if filter_result and filter_result.block:
                    logger.debug("Filtering out %s span based on control plugin", db_system)
                    return
            except Exception as e:  # pylint:disable=W0718
                logger.error("Error applying control plugin to %s span: %s", db_system, str(e))

        self._processor.on_end(span)

    def force_flush(self, timeout_millis=30000):
        return self._processor.force_flush(timeout_millis)

    def shutdown(self):
        return self._processor.shutdown()
