"""Tests for FlattenDictSpanProcessor and the dict attribute registry."""
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import INVALID_SPAN

from harness_sdk.flatten_dict_registry import (
    FLATTEN_ENABLED_ENV,
    FLATTEN_RAW_JSON_ENV,
    FlattenDictRegistry,
    get_registry,
    is_flatten_enabled,
    is_raw_json_enabled,
)
from harness_sdk.flatten_dict_span_processor import (
    MAX_LEAF_ATTRIBUTES,
    FlattenDictSpanProcessor,
)


class RecordingProcessor:
    """Innermost processor capturing what reaches export."""

    def __init__(self):
        self.started = []
        self.ended = []
        self.flushed = False
        self.shutdown_called = False

    def on_start(self, span, parent_context=None):
        self.started.append(span)

    def on_end(self, span):
        self.ended.append(span)

    def force_flush(self, timeout_millis=30000):
        self.flushed = True
        return True

    def shutdown(self):
        self.shutdown_called = True


@pytest.fixture(autouse=True)
def _clean_registry():
    get_registry().clear()
    yield
    get_registry().clear()


@pytest.fixture(name="downstream")
def _downstream():
    return RecordingProcessor()


def _ended_attributes(downstream, attributes):
    """Run a span through the processor and return its exported attributes."""
    processor = FlattenDictSpanProcessor(downstream)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    with provider.get_tracer(__name__).start_as_current_span("span") as span:
        for key, value in attributes.items():
            get_registry().register(span, key, value)
    assert len(downstream.ended) == 1
    return dict(downstream.ended[0].attributes or {})


def test_flat_dict_becomes_dot_notation_keys(downstream):
    attributes = _ended_attributes(
        downstream, {"agent": {"action": "generate", "name": "devops"}}
    )

    assert attributes["agent.action"] == "generate"
    assert attributes["agent.name"] == "devops"
    assert "agent" not in attributes


def test_native_types_are_preserved(downstream):
    attributes = _ended_attributes(
        downstream,
        {"agent": {"name": "devops", "active": True, "retries": 3, "score": 1.5}},
    )

    assert attributes["agent.name"] == "devops"
    assert attributes["agent.active"] is True
    assert attributes["agent.retries"] == 3
    assert attributes["agent.score"] == 1.5


def test_nesting_up_to_max_depth_is_flattened(downstream):
    attributes = _ended_attributes(
        downstream, {"agent": {"model": {"provider": {"name": "vertex"}}}}
    )

    assert attributes["agent.model.provider.name"] == "vertex"


def test_dict_beyond_max_depth_is_json_encoded(downstream):
    attributes = _ended_attributes(
        downstream, {"agent": {"a": {"b": {"c": {"d": 1}}}}}
    )

    assert json.loads(attributes["agent.a.b.c"]) == {"d": 1}


def test_none_leaf_is_skipped(downstream):
    attributes = _ended_attributes(
        downstream, {"agent": {"name": "devops", "parent": None}}
    )

    assert attributes["agent.name"] == "devops"
    assert "agent.parent" not in attributes


def test_non_scalar_leaf_falls_back_to_string(downstream):
    class Model:
        def __str__(self):
            return "gemini-2.0"

    attributes = _ended_attributes(downstream, {"agent": {"model": Model()}})

    assert attributes["agent.model"] == "gemini-2.0"


def test_scalar_list_becomes_array_attribute(downstream):
    attributes = _ended_attributes(
        downstream, {"agent": {"tools": ["search", "shell"]}}
    )

    assert attributes["agent.tools"] == ("search", "shell")


def test_mixed_and_dict_lists_are_json_encoded(downstream):
    attributes = _ended_attributes(
        downstream,
        {
            "agent": {
                "mixed": ["search", 2],
                "steps": [{"name": "plan"}, {"name": "act"}],
            }
        },
    )

    assert json.loads(attributes["agent.mixed"]) == ["search", 2]
    assert json.loads(attributes["agent.steps"]) == [{"name": "plan"}, {"name": "act"}]


def test_explicit_attribute_wins_over_flattened_key(downstream):
    processor = FlattenDictSpanProcessor(downstream)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    with provider.get_tracer(__name__).start_as_current_span("span") as span:
        span.set_attribute("agent.name", "explicit")
        get_registry().register(span, "agent", {"name": "flattened"})

    assert downstream.ended[0].attributes["agent.name"] == "explicit"


def test_leaf_cap_stops_flattening(downstream):
    oversized = {f"key{index}": index for index in range(MAX_LEAF_ATTRIBUTES + 10)}

    attributes = _ended_attributes(downstream, {"agent": oversized})

    flattened = [key for key in attributes if key.startswith("agent.")]
    assert len(flattened) == MAX_LEAF_ATTRIBUTES


def test_raw_json_flag_also_sets_original_key(downstream, monkeypatch):
    monkeypatch.setenv(f"HARNESS_{FLATTEN_RAW_JSON_ENV}", "true")

    attributes = _ended_attributes(downstream, {"agent": {"name": "devops"}})

    assert attributes["agent.name"] == "devops"
    assert json.loads(attributes["agent"]) == {"name": "devops"}


def test_span_without_pending_dicts_passes_through(downstream):
    processor = FlattenDictSpanProcessor(downstream)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    with provider.get_tracer(__name__).start_as_current_span("span") as span:
        span.set_attribute("agent.name", "devops")

    assert len(downstream.started) == 1
    assert dict(downstream.ended[0].attributes) == {"agent.name": "devops"}


def test_lifecycle_calls_delegate_to_wrapped_processor(downstream):
    processor = FlattenDictSpanProcessor(downstream)

    assert processor.force_flush() is True
    processor.shutdown()

    assert downstream.flushed
    assert downstream.shutdown_called


def test_registry_pop_is_one_shot():
    registry = FlattenDictRegistry()
    with TracerProvider().get_tracer(__name__).start_as_current_span("span") as span:
        registry.register(span, "agent", {"name": "devops"})

        assert registry.pop(span) == {"agent": {"name": "devops"}}
        assert registry.pop(span) == {}


def test_registry_last_write_wins_per_key():
    registry = FlattenDictRegistry()
    with TracerProvider().get_tracer(__name__).start_as_current_span("span") as span:
        registry.register(span, "agent", {"name": "first"})
        registry.register(span, "agent", {"name": "second"})

        assert registry.pop(span) == {"agent": {"name": "second"}}


def test_registry_evicts_oldest_spans_over_limit():
    registry = FlattenDictRegistry(max_tracked_spans=1)
    tracer = TracerProvider().get_tracer(__name__)
    with tracer.start_as_current_span("first") as first:
        registry.register(first, "agent", {"name": "first"})
        with tracer.start_as_current_span("second") as second:
            registry.register(second, "agent", {"name": "second"})

            assert registry.pop(first) == {}
            assert registry.pop(second) == {"agent": {"name": "second"}}


def test_registry_ignores_non_recording_span():
    registry = FlattenDictRegistry()

    registry.register(INVALID_SPAN, "agent", {"name": "devops"})

    assert registry.pop(INVALID_SPAN) == {}


def test_flatten_enabled_defaults_to_true(monkeypatch):
    for prefix in ("HARNESS_", "HA_", "AT_", "TA_"):
        monkeypatch.delenv(f"{prefix}{FLATTEN_ENABLED_ENV}", raising=False)
        monkeypatch.delenv(f"{prefix}{FLATTEN_RAW_JSON_ENV}", raising=False)

    assert is_flatten_enabled() is True
    assert is_raw_json_enabled() is False


def test_flatten_disabled_only_when_explicitly_false(monkeypatch):
    monkeypatch.setenv(f"HARNESS_{FLATTEN_ENABLED_ENV}", "false")
    assert is_flatten_enabled() is False

    monkeypatch.setenv(f"HARNESS_{FLATTEN_ENABLED_ENV}", "FALSE")
    assert is_flatten_enabled() is False

    monkeypatch.setenv(f"HARNESS_{FLATTEN_ENABLED_ENV}", "true")
    assert is_flatten_enabled() is True


def test_flatten_stays_enabled_for_non_false_values(monkeypatch):
    for value in ("", "1", "yes", "on", "garbage"):
        monkeypatch.setenv(f"HARNESS_{FLATTEN_ENABLED_ENV}", value)
        assert is_flatten_enabled() is True
