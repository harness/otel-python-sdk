"""Shared control plugin stubs for unit tests."""
from harness_sdk.plugins.control import ControlResult, get_control_registry


class AlwaysBlockControlPlugin:
    name = "test_always_block"
    provides_blocking = True

    def on_init(self, config):  # pylint: disable=unused-argument
        pass

    def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
        return ControlResult()

    def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
        return ControlResult(block=True, response_message="blocked")

    def shutdown(self):
        pass


def register_always_block_plugin():
    get_control_registry().clear()
    get_control_registry().register(AlwaysBlockControlPlugin())


def clear_control_plugins():
    get_control_registry().clear()
