"""Verify rpc_body / rpc_metadata config gates gRPC span body capture."""
import json
import os
from concurrent import futures

import grpc
import pytest
from opentelemetry.trace import SpanKind

from harness_sdk.agent import Agent
from harness_sdk.config.config import Config
from harness_sdk.instrumentation.instrumentation_definitions import _uninstrument_all
from test import configure_inmemory_span_exporter
from test.instrumentation.grpc import helloworld_pb2, helloworld_pb2_grpc


@pytest.fixture
def agent_with_rpc_body_disabled():
    config_dir = os.path.dirname(__file__)
    os.environ["HA_CONFIG_FILE"] = os.path.join(config_dir, "rpc_body_disabled_config.yaml")
    os.environ["HA_ENABLE_CONSOLE_SPAN_EXPORTER"] = "true"
    os.environ["HARNESS_ENABLE_API"] = "true"
    for key in (
        "HARNESS_ENABLE_AI_LITELLM",
        "HARNESS_ENABLE_AI_OPENAI",
        "HARNESS_ENABLE_AI_ANTHROPIC",
        "HARNESS_ENABLE_AI_GOOGLE_GENAI",
        "HARNESS_ENABLE_AI_MCP",
    ):
        os.environ.pop(key, None)
    _uninstrument_all()
    Config._instance = None
    Agent._instance = None
    agent = Agent()
    agent._init.init_trace_provider()  # pylint: disable=protected-access
    yield agent
    _uninstrument_all()
    Config._instance = None
    Agent._instance = None
    os.environ.pop("HA_CONFIG_FILE", None)


@pytest.fixture
def exporter(agent_with_rpc_body_disabled):
    exporter = configure_inmemory_span_exporter(agent_with_rpc_body_disabled)
    yield exporter
    exporter.clear()


def _server_span(spans):
    for span in spans:
        attrs = span.attributes or {}
        if attrs.get("rpc.system") == "grpc" and span.kind == SpanKind.SERVER:
            return json.loads(span.to_json())
    raise AssertionError("No gRPC server span found")


def test_grpc_respects_rpc_body_disabled(agent_with_rpc_body_disabled, exporter):
    agent_with_rpc_body_disabled.instrument()

    class Greeter(helloworld_pb2_grpc.GreeterServicer):
        def SayHello(self, request, context):  # pylint: disable=unused-argument
            return helloworld_pb2.HelloReply(message=f"Hello, {request.name}!")

    executor = futures.ThreadPoolExecutor(max_workers=10)
    server = grpc.server(executor)
    try:
        helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
        port = server.add_insecure_port("[::]:0")
        server.start()

        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = helloworld_pb2_grpc.GreeterStub(channel)
            response = stub.SayHello(helloworld_pb2.HelloRequest(name="world"))
            assert response.message == "Hello, world!"

        span_object = _server_span(exporter.get_finished_spans())
        attrs = span_object["attributes"]

        assert attrs["rpc.method"] == "SayHello"
        assert attrs["rpc.grpc.status_code"] == 0
        assert "rpc.request.body" not in attrs
        assert "rpc.response.body" not in attrs
        assert not any(key.startswith("rpc.request.metadata.") for key in attrs)
        assert not any(key.startswith("rpc.response.metadata.") for key in attrs)
        exporter.clear()
    finally:
        server.stop(grace=0)
        executor.shutdown(wait=False, cancel_futures=True)
