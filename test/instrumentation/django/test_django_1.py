import pytest
from pytest_django.lazy_django import skip_if_no_django

from harness_sdk.plugins.control import get_control_registry
from harness_sdk.instrumentation.instrumentation_definitions import (
    DJANGO_KEY,
    SUPPORTED_LIBRARIES,
    _INSTRUMENTATION_STATE,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
memoryExporter = InMemorySpanExporter()
simpleExportSpanProcessor = SimpleSpanProcessor(memoryExporter)
_processor_registered = False

_DJANGO_ONLY_SKIP_LIBRARIES = [key for key in SUPPORTED_LIBRARIES if key != DJANGO_KEY]


@pytest.fixture(autouse=True)
def django_agent_setup(agent):
    global _processor_registered  # pylint: disable=global-statement
    if not agent.is_initialized():
        agent._init.init_trace_provider()  # pylint: disable=protected-access
    if not _processor_registered:
        agent.register_processor(simpleExportSpanProcessor)
        _processor_registered = True
    get_control_registry().clear()
    agent.instrument(None, skip_libraries=_DJANGO_ONLY_SKIP_LIBRARIES)
    yield
    memoryExporter.clear()


@pytest.fixture()
def django_client():
    """A Django test client instance."""
    skip_if_no_django()

    from django.test.client import Client

    return Client()


@pytest.fixture(autouse=True)
def clear_instance():
    memoryExporter.clear()
    yield
    memoryExporter.clear()


def _django_spans(span_list):
    seen = set()
    unique = []
    for span in span_list:
        if span.name not in {"GET test/<int:id>", "POST test/<int:id>"}:
            continue
        ctx = span.get_span_context()
        key = (ctx.trace_id, ctx.span_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(span)
    return unique


def test_basic_span_data(django_client, agent):
    response = django_client.get('/test/123')
    assert response.status_code == 200

    span_list = _django_spans(memoryExporter.get_finished_spans())
    memoryExporter.clear()
    assert len(span_list) == 1
    django_span = span_list[0]
    assert django_span.name == 'GET test/<int:id>'
    attrs = django_span.attributes
    assert attrs["http.method"] == "GET"
    assert attrs["http.server_name"] == "testserver"
    assert attrs["http.url"] == "http://testserver/test/123"
    assert attrs["http.target"] == "/test/123"
    assert attrs.get("http.route", attrs.get("http.target")) == "test/<int:id>"


def test_collects_body_data(django_client, agent):
    response = django_client.post(
        '/test/123',
        data={"some_client_data": "123"},
        content_type="application/json",
    )
    assert response.status_code == 200

    span_list = _django_spans(memoryExporter.get_finished_spans())
    memoryExporter.clear()
    assert len(span_list) == 1
    django_span = span_list[0]
    assert django_span.name == 'POST test/<int:id>'
    attrs = django_span.attributes
    assert attrs["http.request.header.content-type"] == 'application/json'
    assert attrs["http.request.body"] == '{"some_client_data": "123"}'
    assert attrs["http.response.header.content-type"] == 'application/json'
    assert attrs["http.response.body"] == '{"data": 123}'


def test_can_block(django_client, agent_with_filter, exporter):
    response = django_client.post(
        '/test/123',
        data={"some_client_data": "123"},
        content_type="application/json",
    )
    assert response.status_code == 403

    django_spans = _django_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(django_spans) >= 1
    span = django_spans[0]
    assert span.attributes['http.method'] == 'POST'
    assert span.attributes['http.url'] == 'http://testserver/test/123'
    assert span.attributes['http.target'] == '/test/123'
    assert span.attributes['http.status_code'] == 403


def _reset_django_app_getters():
    """Restore Django's real getters after prior tests may have left wrapper lambdas."""
    from django.core.asgi import get_asgi_application as django_get_asgi
    from django.core.wsgi import get_wsgi_application as django_get_wsgi
    from django.core import asgi, wsgi

    asgi.get_asgi_application = django_get_asgi
    wsgi.get_wsgi_application = django_get_wsgi


def test_asgi_wrappers(django_client, agent):
    _reset_django_app_getters()
    if _INSTRUMENTATION_STATE.get(DJANGO_KEY) is not None:
        del _INSTRUMENTATION_STATE[DJANGO_KEY]
    from django.core import asgi

    original_asgi = asgi.get_asgi_application
    agent._instrument(DJANGO_KEY, auto_instrument=True)  # pylint: disable=protected-access
    assert asgi.get_asgi_application is not original_asgi
    asgi.get_asgi_application()
    assert asgi.get_asgi_application is original_asgi


def test_wsgi_wrappers(django_client, agent):
    _reset_django_app_getters()
    if _INSTRUMENTATION_STATE.get(DJANGO_KEY) is not None:
        del _INSTRUMENTATION_STATE[DJANGO_KEY]
    from django.core import wsgi

    original_wsgi = wsgi.get_wsgi_application
    agent._instrument(DJANGO_KEY, auto_instrument=True)  # pylint: disable=protected-access
    assert wsgi.get_wsgi_application is not original_wsgi
    wsgi.get_wsgi_application()
    assert wsgi.get_wsgi_application is original_wsgi
