"""Default OTLP export pipeline with db-filter and attribute processors."""
from typing import Any, List

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from harness_sdk.agent_init import AgentInit
from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.env import is_env_var_present
from harness_sdk.excluded_by_attribute_span_processor import ExcludeByAttributeSpanProcessor
from harness_sdk.db_control_span_processor import DbControlSpanProcessor

logger = get_custom_logger(__name__)


class BuiltinPipelinePlugin:
    """Observability plugin that wires exporter + db-filter + exclusion processors."""

    name = "builtin_pipeline"
    priority = 100

    def on_init(self, config: Any) -> None:
        self._config = config
        self._agent_init = AgentInit(config)

    def create_span_processors(self, config: Any) -> List[SpanProcessor]:
        if is_env_var_present("ENABLE_CONSOLE_SPAN_EXPORTER"):
            self._agent_init.set_console_span_processor()
            return []

        exporter = self._agent_init.init_exporter(
            config.config.reporting.trace_reporter_type
        )
        if exporter is None:
            logger.warning("Unable to initialize exporter for builtin pipeline")
            return []

        span_processor = BatchSpanProcessor(exporter)
        filter_processor = ExcludeByAttributeSpanProcessor(
            processor=span_processor,
            attribute_name="traceableai.span_type",
            excluded_value="nospan",
        )
        db_control_processor = DbControlSpanProcessor(filter_processor)
        return [db_control_processor]

    def shutdown(self) -> None:
        pass


def factory(_config: Any) -> BuiltinPipelinePlugin:
    return BuiltinPipelinePlugin()
