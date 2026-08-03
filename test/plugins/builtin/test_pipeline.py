"""Tests for the default observability pipeline wiring (builtin_pipeline plugin)."""
from harness_sdk.config.config import Config
from harness_sdk.db_control_span_processor import DbControlSpanProcessor
from harness_sdk.excluded_by_attribute_span_processor import ExcludeByAttributeSpanProcessor
from harness_sdk.flatten_dict_registry import FLATTEN_ENABLED_ENV
from harness_sdk.flatten_dict_span_processor import FlattenDictSpanProcessor
from harness_sdk.gen_ai_payload_scrub_span_processor import GenAiPayloadScrubSpanProcessor
from harness_sdk.plugins.builtin.pipeline import BuiltinPipelinePlugin


def _build_processors(monkeypatch):
    # Force the real OTLP-export branch (skip the console-exporter early-return)
    # so the full processor chain gets assembled.
    monkeypatch.delenv("HA_ENABLE_CONSOLE_SPAN_EXPORTER", raising=False)

    config = Config()
    plugin = BuiltinPipelinePlugin()
    plugin.on_init(config)
    return plugin.create_span_processors(config)


def test_flatten_processor_wraps_chain_as_outermost_layer(monkeypatch):
    processors = _build_processors(monkeypatch)

    assert len(processors) == 1
    flatten_processor = processors[0]
    assert isinstance(flatten_processor, FlattenDictSpanProcessor)
    # pylint: disable=protected-access
    assert isinstance(flatten_processor._processor, GenAiPayloadScrubSpanProcessor)


def test_scrub_processor_wraps_chain_as_outermost_layer(monkeypatch):
    monkeypatch.setenv(f"HARNESS_{FLATTEN_ENABLED_ENV}", "false")

    processors = _build_processors(monkeypatch)

    assert len(processors) == 1
    scrub_processor = processors[0]
    assert isinstance(scrub_processor, GenAiPayloadScrubSpanProcessor)
    # pylint: disable=protected-access
    db_control_processor = scrub_processor._processor
    assert isinstance(db_control_processor, DbControlSpanProcessor)
    filter_processor = db_control_processor._processor
    assert isinstance(filter_processor, ExcludeByAttributeSpanProcessor)
