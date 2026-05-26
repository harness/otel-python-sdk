"""Default OTLP export pipeline with sampling and attribute processors."""
import os
from typing import Any, List

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from harness_sdk.agent_init import AgentInit
from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.excluded_by_attribute_span_processor import ExcludeByAttributeSpanProcessor
from harness_sdk.sampling_span_processor import SamplingSpanProcessor

logger = get_custom_logger(__name__)


class BuiltinPipelinePlugin:
    """Observability plugin that wires exporter + sampling + exclusion processors."""

    name = "builtin_pipeline"
    priority = 100

    def on_init(self, config: Any) -> None:
        self._config = config
        self._agent_init = AgentInit(config)

    def create_span_processors(self, config: Any) -> List[SpanProcessor]:
        if (
            "HA_ENABLE_CONSOLE_SPAN_EXPORTER" in os.environ
            or "AT_ENABLE_CONSOLE_SPAN_EXPORTER" in os.environ
            or "TA_ENABLE_CONSOLE_SPAN_EXPORTER" in os.environ
        ):
            self._agent_init.set_console_span_processor()
            return []

        exporter = self._agent_init._init_exporter(  # pylint: disable=protected-access
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
        sampling_processor = SamplingSpanProcessor(filter_processor)
        return [sampling_processor]

    def shutdown(self) -> None:
        pass


def factory(_config: Any) -> BuiltinPipelinePlugin:
    return BuiltinPipelinePlugin()
