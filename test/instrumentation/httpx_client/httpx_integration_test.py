import json

import flask
import httpx
import pytest
from flask import Flask
from opentelemetry.trace import SpanKind

from test.instrumentation.flask.app import FlaskServer


def _client_span(spans):
    for span in spans:
        if span.kind == SpanKind.CLIENT:
            return json.loads(span.to_json())
    raise AssertionError('No client span found')


def test_httpx_client_get(agent, exporter):
    try:
        app = Flask(__name__)
        app.use_reloader = False

        @app.route("/route1")
        def api_example():
            response = flask.Response(mimetype='application/json')
            response.headers['tester3'] = 'tester3'
            response.data = str('{ "a": "a", "xyz": "xyz" }')
            return response

        agent.instrument(app)
        server = FlaskServer(app)
        server.start()

        url = f'http://localhost:{server.port}/route1'
        with httpx.Client() as client:
            client.get(url)

        spans = exporter.get_finished_spans()
        assert spans
        client_span = _client_span(spans)

        assert client_span['attributes']['http.method'] == 'GET'
        assert client_span['attributes']['http.url'] == url
        assert client_span['attributes']['http.response.body'] == '{ "a": "a", "xyz": "xyz" }'
        assert client_span['attributes']['http.status_code'] == 200
        assert client_span['attributes']['http.response.header.tester3'] == 'tester3'
    finally:
        server.shutdown()


def test_httpx_client_post(agent, exporter):
    try:
        app = Flask(__name__)
        app.use_reloader = False

        @app.route("/route1", methods=["POST"])
        def api_example():
            response = flask.Response(mimetype='application/json')
            response.headers['tester3'] = 'tester3'
            response.data = str('{ "a": "a", "xyz": "xyz" }')
            return response

        agent.instrument(app)
        server = FlaskServer(app)
        server.start()

        url = f'http://localhost:{server.port}/route1'
        with httpx.Client() as client:
            client.post(url, json={"test": "body"})

        spans = exporter.get_finished_spans()
        assert spans
        client_span = _client_span(spans)

        assert client_span['kind'] == "SpanKind.CLIENT"
        assert client_span['attributes']['http.method'] == 'POST'
        assert client_span['attributes']['http.url'] == url
        assert client_span['attributes']['http.request.header.content-type'] == 'application/json'
        assert client_span['attributes']['http.request.body'] == '{"test":"body"}'
        assert client_span['attributes']['http.response.body'] == '{ "a": "a", "xyz": "xyz" }'
        assert client_span['attributes']['http.status_code'] == 200
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_httpx_async_client_post(agent, exporter):
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
    agent.instrument(app)

    url = f'http://localhost:{server.port}/route1'
    async with httpx.AsyncClient() as client:
        await client.post(
            url,
            content=b'{ "a":"b", "c": "d" }',
            headers={
                'tester1': 'tester1',
                'tester2': 'tester2',
                'content-type': 'application/json',
            },
        )

    span_list = exporter.get_finished_spans()
    assert span_list
    httpx_span = _client_span(span_list)

    assert httpx_span['attributes']['http.method'] == 'POST'
    assert httpx_span['attributes']['http.url'] == url
    assert httpx_span['attributes']['http.request.header.tester1'] == 'tester1'
    assert httpx_span['attributes']['http.request.header.tester2'] == 'tester2'
    assert httpx_span['attributes']['http.request.body'] == '{ "a":"b", "c": "d" }'
    assert httpx_span['attributes']['http.response.header.content-type'] == 'application/json'
    assert httpx_span['attributes']['http.response.body'] == '{ "a": "a", "xyz": "xyz" }'
    assert httpx_span['attributes']['http.status_code'] == 200
    server.shutdown()
