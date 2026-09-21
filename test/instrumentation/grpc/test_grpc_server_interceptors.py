"""grpc.server wrapper must preserve user-supplied interceptors."""
from unittest.mock import MagicMock, patch

import harness_sdk.instrumentation.grpc as grpc_module
from harness_sdk.instrumentation.grpc import GrpcInstrumentorServerWrapper


def test_server_wrapper_prepends_harness_interceptor():
    wrapper = GrpcInstrumentorServerWrapper()
    existing_interceptor = object()
    original_server = MagicMock(return_value="server-instance")

    with patch(
        "harness_sdk.instrumentation.grpc.GrpcInstrumentorServer._instrument",
        return_value=None,
    ):
        with patch(
            "harness_sdk.instrumentation.grpc.server_interceptor_wrapper",
            return_value="harness-interceptor",
        ):
            with patch("harness_sdk.instrumentation.grpc.grpc.server", original_server):
                wrapper._instrument()  # pylint: disable=protected-access

                grpc_module.grpc.server(
                    futures=MagicMock(),
                    interceptors=[existing_interceptor],
                )

    _, kwargs = original_server.call_args
    assert kwargs["interceptors"] == ["harness-interceptor", existing_interceptor]
