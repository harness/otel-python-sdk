import json

import aiohttp
import flask
import pytest
from flask import Flask
from opentelemetry.trace import SpanKind

from harness_sdk.instrumentation.instrumentation_definitions import (
    AIOHTTP_CLIENT_KEY,
    FLASK_KEY,
    SUPPORTED_LIBRARIES,
    _uninstrument_all,
)
from test import setup_custom_logger
from test.instrumentation.flask.app import FlaskServer

_SKIP_LIBRARIES = [key for key in SUPPORTED_LIBRARIES if key not in (FLASK_KEY, AIOHTTP_CLIENT_KEY)]


def _aiohttp_client_span(spans, port):
    for span in spans:
        if span.kind != SpanKind.CLIENT:
            continue
        attrs = span.attributes or {}
        url = attrs.get("http.url", "")
        if (
            attrs.get("http.method") == "POST"
            and f"localhost:{port}/route1" in url
            and "http.request.body" in attrs
        ):
            return span
    return None


@pytest.mark.asyncio
async def test_aiohttp_post(agent, exporter):
    logger = setup_custom_logger(__name__)
    app = Flask(__name__)
    app.use_reloader = False

    @app.route("/route1", methods=["POST"])
    def api_example():
        response = flask.Response(mimetype='application/json')
        response.headers['tester3'] = 'tester3'
        response.data = str('{ "a": "a", "xyz": "xyz" }')
        return response

    server = FlaskServer(app)
    server.start()
    try:
        _uninstrument_all()
        exporter.clear()
        agent.instrument(app, skip_libraries=_SKIP_LIBRARIES)
        exporter.clear()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'http://localhost:{server.port}/route1',
                data='{ "a":"b", "c": "d" }',
                headers={
                    'tester1': 'tester1',
                    'tester2': 'tester2',
                    'content-type': 'application/json',
                },
            ) as response:
                response_body = await response.json()
                logger.info('Received: %s', str(response_body))
                assert response_body['a'] == 'a'

                span_list = exporter.get_finished_spans()
                assert span_list

                client_span = _aiohttp_client_span(span_list, server.port)
                assert client_span is not None, (
                    f"no aiohttp client span in {[s.name for s in span_list]}"
                )
                aiohttp_span = json.loads(client_span.to_json())

                assert aiohttp_span['attributes']['http.method'] == 'POST'
                assert aiohttp_span['attributes']['http.url'] == (
                    f'http://localhost:{server.port}/route1'
                )
                assert aiohttp_span['attributes']['http.request.header.tester1'] == 'tester1'
                assert aiohttp_span['attributes']['http.request.header.tester2'] == 'tester2'
                assert aiohttp_span['attributes']['http.request.body'] == '{ "a":"b", "c": "d" }'
                assert aiohttp_span['attributes']['http.response.header.content-type'] == 'application/json'
                assert aiohttp_span['attributes']['http.response.body'] == '{ "a": "a", "xyz": "xyz" }'
                assert aiohttp_span['attributes']['http.status_code'] == 200
    finally:
        server.shutdown()
