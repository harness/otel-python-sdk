import asyncio

from opentelemetry.sdk.trace import TracerProvider

from harness_sdk import set_span_attribute, set_span_attributes


def _tracer():
    return TracerProvider().get_tracer(__name__)


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
