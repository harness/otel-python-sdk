'''Unittest for control plugin registry'''
from opentelemetry.trace import NonRecordingSpan

from harness_sdk.plugins.control import ControlResult, get_control_registry


class _NoopControlPlugin:
    name = "noop"

    def on_init(self, config):  # pylint: disable=unused-argument
        pass

    def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
        return ControlResult()

    def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
        return ControlResult()

    def shutdown(self):
        pass


def test_register():
    registry = get_control_registry()
    registry.clear()
    registry.register(_NoopControlPlugin())
    assert len(registry._plugins) == 1  # pylint: disable=protected-access
    registry.clear()


def test_evaluate_returns_control_result():
    registry = get_control_registry()
    registry.clear()
    registry.register(_NoopControlPlugin())
    res = registry.evaluate(NonRecordingSpan(None), 'a_url', {'key': 'v'}, 'body_data', True)
    assert res.block is False
    registry.clear()
