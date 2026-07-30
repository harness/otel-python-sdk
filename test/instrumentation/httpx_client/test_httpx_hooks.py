from types import SimpleNamespace

import pytest

from harness_sdk.instrumentation.httpx import HTTPXClientInstrumentorWrapper


class _SyncStream:
    def __init__(self):
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield b"response"


class _AsyncStream:
    def __init__(self):
        self.iterations = 0

    async def __aiter__(self):
        self.iterations += 1
        yield b"response"


def _wrapper_with_response_capture():
    wrapper = HTTPXClientInstrumentorWrapper()
    calls = []
    wrapper.generic_response_handler = (
        lambda headers, body, span: calls.append((headers, body, span))
    )
    return wrapper, calls


def test_response_hook_does_not_consume_or_replace_stream():
    wrapper, calls = _wrapper_with_response_capture()
    span = object()
    transport_stream = _SyncStream()
    response_stream = SimpleNamespace(_httpcore_stream=transport_stream)
    response_info = SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/plain"},
        stream=response_stream,
        extensions={},
    )

    wrapper.response_hook(span, None, response_info)

    assert calls == [({"content-type": "text/plain"}, None, span)]
    assert transport_stream.iterations == 0
    assert response_stream._httpcore_stream is transport_stream


@pytest.mark.asyncio
async def test_async_response_hook_does_not_consume_or_replace_stream():
    wrapper, calls = _wrapper_with_response_capture()
    span = object()
    transport_stream = _AsyncStream()
    response_stream = SimpleNamespace(_httpcore_stream=transport_stream)
    response_info = SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/plain"},
        stream=response_stream,
        extensions={},
    )

    await wrapper.async_response_hook(span, None, response_info)

    assert calls == [({"content-type": "text/plain"}, None, span)]
    assert transport_stream.iterations == 0
    assert response_stream._httpcore_stream is transport_stream
