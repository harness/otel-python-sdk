import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider

from harness_sdk import set_span_attribute, set_span_attributes
from harness_sdk.flatten_dict_registry import FLATTEN_ENABLED_ENV, get_registry
from harness_sdk.flatten_dict_span_processor import FlattenDictSpanProcessor


def _tracer():
    return TracerProvider().get_tracer(__name__)


@pytest.fixture(autouse=True)
def _clean_registry():
    get_registry().clear()
    yield
    get_registry().clear()


def test_set_span_attribute_updates_current_recording_span():
    with _tracer().start_as_current_span("parent") as span:
        set_span_attribute("request.client.name", "acme")

        assert span.attributes["request.client.name"] == "acme"


def test_set_span_attributes_preserves_supported_types():
    with _tracer().start_as_current_span("parent") as span:
        set_span_attributes({
            "string": "value",
            "boolean": True,
            "integer": 7,
            "float": 1.5,
            "sequence": ("a", "b"),
        })

        assert dict(span.attributes) == {
            "string": "value",
            "boolean": True,
            "integer": 7,
            "float": 1.5,
            "sequence": ("a", "b"),
        }


def test_last_write_wins():
    with _tracer().start_as_current_span("parent") as span:
        set_span_attribute("custom.key", "first")
        set_span_attribute("custom.key", "second")

        assert span.attributes["custom.key"] == "second"


def test_no_active_recording_span_is_noop():
    assert set_span_attribute("custom.key", "value") is None
    assert set_span_attributes({"custom.key": "value"}) is None


def test_empty_mapping_is_noop():
    with _tracer().start_as_current_span("parent") as span:
        assert set_span_attributes({}) is None
        assert not span.attributes


def test_attributes_do_not_inherit_to_child_span():
    tracer = _tracer()
    with tracer.start_as_current_span("parent") as parent:
        set_span_attribute("custom.key", "parent")

        with tracer.start_as_current_span("child") as child:
            assert "custom.key" not in child.attributes

        assert parent.attributes["custom.key"] == "parent"


def test_async_code_updates_active_span():
    async def enrich():
        set_span_attribute("agent.action.type", "search")

    with _tracer().start_as_current_span("parent") as span:
        asyncio.run(enrich())

        assert span.attributes["agent.action.type"] == "search"


def test_dict_value_is_deferred_not_set_on_span():
    with _tracer().start_as_current_span("parent") as span:
        set_span_attribute("agent", {"action": "generate"})

        assert not span.attributes
        assert get_registry().pop(span) == {"agent": {"action": "generate"}}


def test_dict_value_is_flattened_at_span_end():
    class Capture:
        def __init__(self):
            self.ended = []

        def on_start(self, span, parent_context=None):
            pass

        def on_end(self, span):
            self.ended.append(span)

        def force_flush(self, timeout_millis=30000):
            return True

        def shutdown(self):
            pass

    capture = Capture()
    provider = TracerProvider()
    provider.add_span_processor(FlattenDictSpanProcessor(capture))
    with provider.get_tracer(__name__).start_as_current_span("parent"):
        set_span_attributes({
            "agent": {"action": "generate", "name": "devops"},
            "custom.retry.count": 2,
        })

    attributes = dict(capture.ended[0].attributes)
    assert attributes == {
        "custom.retry.count": 2,
        "agent.action": "generate",
        "agent.name": "devops",
    }


def test_dict_value_is_rejected_when_flatten_disabled(monkeypatch):
    monkeypatch.setenv(f"HARNESS_{FLATTEN_ENABLED_ENV}", "false")

    with _tracer().start_as_current_span("parent") as span:
        set_span_attribute("agent", {"action": "generate"})

        # OTel drops the unsupported mapping value itself; nothing is deferred.
        assert not span.attributes
        assert get_registry().pop(span) == {}
