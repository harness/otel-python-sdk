'''multipart/form-data bodies are captured as base64 of the raw bytes.

File parts are binary, so they are not stored as text under ``http.*.body``. They go
under ``http.*.body.base64`` and the Traceable agent decodes them after export.
'''
import base64
import io

import aiohttp
import flask
import httpx
import pytest
import requests
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from flask import Flask
from opentelemetry.trace import SpanKind

from harness_sdk.config.config import Config
from harness_sdk.instrumentation import BaseInstrumentorWrapper
from harness_sdk.instrumentation.instrumentation_definitions import (
    AIOHTTP_CLIENT_KEY,
    FAST_API_KEY,
    FLASK_KEY,
    SUPPORTED_LIBRARIES,
    _INSTRUMENTATION_STATE,
    _uninstrument_all,
)
from test.instrumentation.flask.app import FlaskServer

BOUNDARY = 'xBoundaryx'
CONTENT_TYPE = f'multipart/form-data; boundary={BOUNDARY}'
# A text field and a file part whose bytes are not valid UTF-8 and contain NUL.
MULTIPART_BODY = (
    f'--{BOUNDARY}\r\n'
    'Content-Disposition: form-data; name="comment"\r\n\r\n'
    'hello\r\n'
    f'--{BOUNDARY}\r\n'
    'Content-Disposition: form-data; name="file"; filename="a.bin"\r\n'
    'Content-Type: application/octet-stream\r\n\r\n'
).encode() + b'\x00\x01\xff\xfe\x80binary' + f'\r\n--{BOUNDARY}--\r\n'.encode()


def _decode(value: str) -> bytes:
    return base64.b64decode(value + '=' * (-len(value) % 4))


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, key, value):
        self.attributes[key] = value

    @property
    def body_attributes(self):
        return {key: value for key, value in self.attributes.items() if '.body' in key}


@pytest.fixture(autouse=True)
def _clear_exported_spans(exporter):
    # the in-memory exporter is shared across tests
    exporter.clear()
    yield
    exporter.clear()


@pytest.fixture(name='wrapper')
def wrapper_fixture(agent):  # pylint: disable=unused-argument
    return BaseInstrumentorWrapper()


def test_multipart_request_is_base64_of_raw_bytes(wrapper):
    span = _FakeSpan()
    wrapper.generic_request_handler({'content-type': CONTENT_TYPE}, MULTIPART_BODY, span)

    assert 'http.request.body' not in span.attributes
    encoded = span.attributes['http.request.body.base64']
    assert '=' not in encoded
    assert _decode(encoded) == MULTIPART_BODY


def test_multipart_response_is_base64_of_raw_bytes(wrapper):
    span = _FakeSpan()
    wrapper.generic_response_handler({'content-type': CONTENT_TYPE}, MULTIPART_BODY, span)

    assert 'http.response.body' not in span.attributes
    assert _decode(span.attributes['http.response.body.base64']) == MULTIPART_BODY


def test_multipart_str_body_is_base64_of_utf8(wrapper):
    span = _FakeSpan()
    wrapper.generic_request_handler({'content-type': CONTENT_TYPE}, 'a=b', span)

    assert _decode(span.attributes['http.request.body.base64']) == b'a=b'


def test_multipart_body_cut_to_max_size_before_encoding(wrapper):
    wrapper._max_body_size = 10  # pylint: disable=protected-access
    span = _FakeSpan()
    wrapper.generic_request_handler({'content-type': CONTENT_TYPE}, MULTIPART_BODY, span)

    encoded = span.attributes['http.request.body.base64']
    assert _decode(encoded) == MULTIPART_BODY[:10]
    assert len(encoded) % 4 != 1  # always valid base64


def test_empty_multipart_body_is_not_captured(wrapper):
    span = _FakeSpan()
    wrapper.generic_request_handler({'content-type': CONTENT_TYPE}, b'', span)

    assert not span.body_attributes


@pytest.mark.parametrize('content_type,body,expected', [
    ('application/json', b'{"a": 1}', '{"a": 1}'),
    ('application/x-www-form-urlencoded', b'\x80abc', '\\x80abc'),
])
def test_other_content_types_are_still_captured_as_text(wrapper, content_type, body, expected):
    span = _FakeSpan()
    wrapper.generic_request_handler({'content-type': content_type}, body, span)

    assert span.body_attributes == {'http.request.body': expected}


def test_multipart_is_allowed_by_default(wrapper):
    allowed = [item.value for item in Config().config.data_capture.allowed_content_types]
    assert 'multipart/form-data' in allowed
    assert wrapper.eligible_based_on_content_type({'content-type': CONTENT_TYPE})


def test_multipart_not_captured_when_not_in_configured_allowlist(agent, monkeypatch):
    monkeypatch.setenv('HA_DATA_CAPTURE_ALLOWED_CONTENT_TYPES', 'json')
    Config._instance = None  # pylint: disable=protected-access
    wrapper = BaseInstrumentorWrapper()
    span = _FakeSpan()
    wrapper.generic_request_handler({'content-type': CONTENT_TYPE}, MULTIPART_BODY, span)

    assert not span.body_attributes


def _flask_app():
    app = Flask(__name__)
    app.use_reloader = False

    @app.route('/upload', methods=['POST'])
    def upload():
        # capturing the body must not stop the app from parsing the form and files
        assert flask.request.form['comment'] == 'hello'
        assert flask.request.files['file'].read() == b'\x00\x01\xff\xfe\x80binary'
        return flask.Response(MULTIPART_BODY, content_type=CONTENT_TYPE)

    return app


def _spans_by_kind(exporter, kind):
    return [span for span in exporter.get_finished_spans() if span.kind == kind]


def _client_span_for(exporter, port):
    # other tests in the process leave extra instrumentors behind, so pick our request by URL
    spans = [span for span in _spans_by_kind(exporter, SpanKind.CLIENT)
             if str(span.attributes.get('http.url', '')).endswith(f':{port}/upload')]
    assert spans
    return spans[-1]


def test_flask_server_and_requests_client_capture_multipart(agent, exporter):
    app = _flask_app()
    agent.instrument(app)
    server = FlaskServer(app)
    server.start()
    try:
        response = requests.post(f'http://localhost:{server.port}/upload',
                                 data=MULTIPART_BODY, headers={'Content-Type': CONTENT_TYPE},
                                 timeout=10)
        assert response.content == MULTIPART_BODY
    finally:
        server.shutdown()

    server_span, = _spans_by_kind(exporter, SpanKind.SERVER)
    assert 'http.request.body' not in server_span.attributes
    assert 'http.response.body' not in server_span.attributes
    assert _decode(server_span.attributes['http.request.body.base64']) == MULTIPART_BODY
    assert _decode(server_span.attributes['http.response.body.base64']) == MULTIPART_BODY

    client_span = _client_span_for(exporter, server.port)
    assert 'http.request.body' not in client_span.attributes
    assert _decode(client_span.attributes['http.request.body.base64']) == MULTIPART_BODY


def test_httpx_client_captures_multipart(agent, exporter):
    app = _flask_app()
    agent.instrument(app)
    server = FlaskServer(app)
    server.start()
    try:
        with httpx.Client() as client:
            client.post(f'http://localhost:{server.port}/upload',
                        content=MULTIPART_BODY, headers={'Content-Type': CONTENT_TYPE})
    finally:
        server.shutdown()

    client_span = _client_span_for(exporter, server.port)
    assert 'http.request.body' not in client_span.attributes
    assert _decode(client_span.attributes['http.request.body.base64']) == MULTIPART_BODY
    assert _decode(client_span.attributes['http.response.body.base64']) == MULTIPART_BODY


@pytest.mark.asyncio
async def test_aiohttp_client_captures_multipart(agent, exporter):
    skip_libraries = [key for key in SUPPORTED_LIBRARIES if key not in (FLASK_KEY, AIOHTTP_CLIENT_KEY)]
    app = _flask_app()
    server = FlaskServer(app)
    server.start()
    try:
        _uninstrument_all()
        exporter.clear()
        agent.instrument(app, skip_libraries=skip_libraries)
        exporter.clear()

        async with aiohttp.ClientSession() as session:
            async with session.post(f'http://localhost:{server.port}/upload',
                                    data=MULTIPART_BODY, headers={'Content-Type': CONTENT_TYPE}) as response:
                assert await response.read() == MULTIPART_BODY
    finally:
        server.shutdown()

    client_span = _client_span_for(exporter, server.port)
    assert 'http.request.body' not in client_span.attributes
    assert _decode(client_span.attributes['http.request.body.base64']) == MULTIPART_BODY
    assert _decode(client_span.attributes['http.response.body.base64']) == MULTIPART_BODY


def test_fastapi_server_captures_multipart_request(agent, exporter):
    if FAST_API_KEY in _INSTRUMENTATION_STATE:
        del _INSTRUMENTATION_STATE[FAST_API_KEY]
    agent._instrument(FAST_API_KEY, auto_instrument=True)  # pylint: disable=protected-access
    app = FastAPI()

    @app.post('/upload')
    async def upload(request: Request):
        assert await request.body() == MULTIPART_BODY
        return {'ok': True}

    client = TestClient(app)
    response = client.post('/upload', content=MULTIPART_BODY, headers={'Content-Type': CONTENT_TYPE})
    assert response.status_code == 200

    span = [s for s in exporter.get_finished_spans() if s.name == 'POST /upload'][-1]
    assert 'http.request.body' not in span.attributes
    assert _decode(span.attributes['http.request.body.base64']) == MULTIPART_BODY


def _flask_form_app():
    app = Flask(__name__)
    app.use_reloader = False

    @app.route('/upload', methods=['POST'])
    def upload():
        return {'comment': flask.request.form['comment'], 'size': len(flask.request.files['file'].read())}

    return app


def test_flask_multipart_larger_than_max_size_is_not_buffered(agent, exporter, monkeypatch):
    monkeypatch.setenv('HA_DATA_CAPTURE_BODY_MAX_SIZE_BYTES', '64')
    Config._instance = None  # pylint: disable=protected-access
    app = _flask_form_app()
    agent.instrument(app)

    big_file = b'x' * 1024
    response = app.test_client().post('/upload', data={'comment': 'hi', 'file': (io.BytesIO(big_file), 'big.bin')},
                                      content_type='multipart/form-data')
    assert response.get_json() == {'comment': 'hi', 'size': len(big_file)}  # the app is unaffected

    span, = [s for s in exporter.get_finished_spans() if s.name.startswith('POST')]
    assert not [key for key in span.attributes if key.startswith('http.request.body')]


def test_flask_multipart_within_max_size_is_captured_and_form_still_parses(agent, exporter):
    app = _flask_form_app()
    agent.instrument(app)

    response = app.test_client().post('/upload', data={'comment': 'hi', 'file': (io.BytesIO(b'\x00\xff'), 'a.bin')},
                                      content_type='multipart/form-data')
    assert response.get_json() == {'comment': 'hi', 'size': 2}

    span, = [s for s in exporter.get_finished_spans() if s.name.startswith('POST')]
    assert 'http.request.body' not in span.attributes
    decoded = _decode(span.attributes['http.request.body.base64'])
    assert b'name="comment"' in decoded and b'\x00\xff' in decoded
