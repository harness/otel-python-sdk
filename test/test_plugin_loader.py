"""Tests for config-ordered plugin loading."""
from unittest.mock import MagicMock, patch

from agent_trace.config.config import Config
from agent_trace.plugins.control import get_control_registry
from agent_trace.plugins.loader import load_control_plugins, load_observability_plugins


class _PluginA:
    name = "plugin_a"
    provides_blocking = False

    def on_init(self, config):  # pylint: disable=unused-argument
        pass

    def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
        from agent_trace.plugins.control import ControlResult
        result = ControlResult()
        result.response_message = "a"
        return result

    def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
        from agent_trace.plugins.control import ControlResult
        return ControlResult()

    def shutdown(self):
        pass


class _PluginB:
    name = "plugin_b"
    provides_blocking = False

    def on_init(self, config):  # pylint: disable=unused-argument
        pass

    def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
        from agent_trace.plugins.control import ControlResult
        result = ControlResult()
        result.response_message = "b"
        return result

    def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
        from agent_trace.plugins.control import ControlResult
        return ControlResult()

    def shutdown(self):
        pass


def _fake_entry_point(name, plugin_factory):
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = plugin_factory
    return ep


def test_control_plugins_load_in_config_order():
    get_control_registry().clear()
    config = Config()
    config.enabled_control_plugins = ["plugin_b", "plugin_a"]

    with patch(
        "agent_trace.plugins.loader._entry_points_by_name",
        return_value={
            "plugin_a": _fake_entry_point("plugin_a", lambda c: _PluginA()),
            "plugin_b": _fake_entry_point("plugin_b", lambda c: _PluginB()),
        },
    ):
        load_control_plugins(config)

    registry = get_control_registry()
    assert [p.name for p in registry._plugins] == ["plugin_b", "plugin_a"]  # pylint: disable=protected-access

    from opentelemetry.trace import NonRecordingSpan
    result = registry.evaluate(NonRecordingSpan(None), "", {}, None, False)
    assert result.response_message == "b"
    registry.clear()


def test_skips_uninstalled_plugin_name():
    get_control_registry().clear()
    config = Config()
    config.enabled_control_plugins = ["missing_plugin"]

    with patch("agent_trace.plugins.loader._entry_points_by_name", return_value={}):
        load_control_plugins(config)

    assert len(get_control_registry()._plugins) == 0  # pylint: disable=protected-access
    get_control_registry().clear()


def test_observability_plugins_register_in_config_order():
    processors = []

    class _ObsPlugin:
        def __init__(self, label):
            self.name = label
            self.priority = 0

        def on_init(self, config):  # pylint: disable=unused-argument
            pass

        def create_span_processors(self, config):  # pylint: disable=unused-argument
            p = MagicMock()
            p.label = self.name
            processors.append(p)
            return [p]

        def shutdown(self):
            pass

    config = Config()
    config.enabled_observability_plugins = ["second", "first"]

    provider = MagicMock()

    with patch(
        "agent_trace.plugins.loader._entry_points_by_name",
        return_value={
            "first": _fake_entry_point("first", lambda c: _ObsPlugin("first")),
            "second": _fake_entry_point("second", lambda c: _ObsPlugin("second")),
        },
    ), patch("agent_trace.plugins.loader.trace.get_tracer_provider", return_value=provider):
        load_observability_plugins(config)

    assert [p.label for p in processors] == ["second", "first"]
    assert provider.add_span_processor.call_count == 2
