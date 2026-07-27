"""Tests for the default observability pipeline wiring (builtin_pipeline plugin)."""
from harness_sdk.config.config import Config
from harness_sdk.db_control_span_processor import DbControlSpanProcessor
from harness_sdk.excluded_by_attribute_span_processor import ExcludeByAttributeSpanProcessor
from harness_sdk.gen_ai_payload_scrub_span_processor import GenAiPayloadScrubSpanProcessor
from harness_sdk.plugins.builtin.pipeline import BuiltinPipelinePlugin


def test_scrub_processor_wraps_chain_as_outermost_layer(monkeypatch):
    # Force the real OTLP-export branch (skip the console-exporter early-return)
    # so the full processor chain gets assembled.
    monkeypatch.delenv("HA_ENABLE_CONSOLE_SPAN_EXPORTER", raising=False)

    config = Config()
    plugin = BuiltinPipelinePlugin()
    plugin.on_init(config)

    processors = plugin.create_span_processors(config)

    assert len(processors) == 1
    scrub_processor = processors[0]
    assert isinstance(scrub_processor, GenAiPayloadScrubSpanProcessor)
    # pylint: disable=protected-access
    db_control_processor = scrub_processor._processor
    assert isinstance(db_control_processor, DbControlSpanProcessor)
    filter_processor = db_control_processor._processor
    assert isinstance(filter_processor, ExcludeByAttributeSpanProcessor)
