import os

os.environ["HA_ENABLE_CONSOLE_SPAN_EXPORTER"] = "true"

import pytest
from opentelemetry.trace import Span

from harness_sdk.agent import Agent
from harness_sdk.config.config import Config
from harness_sdk.plugins.control import ControlResult, get_control_registry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from harness_sdk.instrumentation.instrumentation_definitions import _uninstrument_all
from harness_sdk.instrumentation.genai_env import maybe_set_genai_payload_capture_env_vars
from test import configure_inmemory_span_exporter


def _shutdown_trace_provider():
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        try:
            provider.shutdown()
        except Exception:  # pylint: disable=broad-except
            pass


def _cleanup_test_runtime():
    _uninstrument_all()


class SampleBlockingControlPlugin:
    name = "test_blocking"
    provides_blocking = True

    def on_init(self, config):  # pylint: disable=unused-argument
        pass

    def evaluate(self, span: Span, url: str, headers: dict, body, is_grpc):  # pylint: disable=unused-argument
        res = ControlResult()
        res.block = True
        res.response_status_code = 403
        return res

    def evaluate_agent_span(self, span: Span, body: str = ""):  # pylint: disable=unused-argument
        return ControlResult()

    def shutdown(self):
        pass


@pytest.fixture(autouse=True)
def reset_singletons():
    if "_HANDLER" in os.environ:
        del os.environ["_HANDLER"]
    keys_to_delete = [
        key for key in os.environ
        if (key.startswith("HA_") or key.startswith("HARNESS_"))
        and key != "HA_ENABLE_CONSOLE_SPAN_EXPORTER"
    ]
    for key in keys_to_delete:
        del os.environ[key]
    os.environ["HA_ENABLE_CONSOLE_SPAN_EXPORTER"] = "true"
    Config._instance = None
    Agent._instance = None
    get_control_registry().clear()
    yield
    _cleanup_test_runtime()


def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    _cleanup_test_runtime()
    _shutdown_trace_provider()


@pytest.fixture
def agent():
    os.environ['HA_ENABLE_CONSOLE_SPAN_EXPORTER'] = 'true'
    os.environ['HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED'] = 'true'
    os.environ['HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED'] = 'true'
    os.environ['HA_CONTROL_PLUGINS'] = ''
    # Instrumentation is opt-in; enable every category so shared fixtures cover
    # both API and AI instrumentation suites.
    os.environ['HARNESS_ENABLE_API'] = 'true'
    os.environ['HARNESS_ENABLE_AI_OPENAI'] = 'true'
    os.environ['HARNESS_ENABLE_AI_ANTHROPIC'] = 'true'
    os.environ['HARNESS_ENABLE_AI_LITELLM'] = 'true'
    os.environ['HARNESS_ENABLE_AI_GOOGLE_GENAI'] = 'true'
    os.environ['HARNESS_ENABLE_AI_MCP'] = 'true'

    _uninstrument_all()
    maybe_set_genai_payload_capture_env_vars()
    agent = Agent()
    agent._init.init_trace_provider()
    get_control_registry().clear()

    return agent


@pytest.fixture(scope="function", autouse=True)
def _dj_autoclear_mailbox() -> None:
    pass


@pytest.fixture
def agent_with_filter(agent):
    get_control_registry().register(SampleBlockingControlPlugin())
    yield agent
    get_control_registry().clear()


@pytest.fixture
def exporter(agent):
    exporter = configure_inmemory_span_exporter(agent)

    yield exporter

    exporter.clear()
