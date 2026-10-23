"""Tests for httpx client blocking via control plugins."""
import httpx
import pytest

from harness_sdk.gen_ai.exceptions import ControlRequestBlocked
from harness_sdk.instrumentation.httpx import HTTPXClientInstrumentorWrapper
from harness_sdk.plugins.control import ControlResult, get_control_registry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


class _BlockingPlugin:
    name = "test_blocking"
    provides_blocking = True

    def on_init(self, config):  # pylint: disable=unused-argument
        pass

    def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
        return ControlResult(
            block=True,
            response_status_code=403,
            response_message="Blocked by policy",
        )

    def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
        return ControlResult()

    def shutdown(self):
        pass


@pytest.fixture
def httpx_wrapper():
    wrapper = HTTPXClientInstrumentorWrapper()
    yield wrapper
    if wrapper.is_instrumented_by_opentelemetry:
        wrapper.uninstrument()


def test_httpx_request_hook_raises_when_control_blocks(httpx_wrapper):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    get_control_registry().register(_BlockingPlugin())
    httpx_wrapper.instrument(tracer_provider=provider)

    request_info = httpx.Request("GET", "https://example.com/get", headers={"x-test": "1"})
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("httpx") as span:
        with pytest.raises(ControlRequestBlocked) as exc_info:
            httpx_wrapper._process_request(span, request_info)  # pylint: disable=protected-access

    assert exc_info.value.result.response_status_code == 403
    get_control_registry().clear()
