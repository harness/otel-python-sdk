from test.control_test_helpers import AlwaysBlockControlPlugin
"""Tests for Anthropic SDK instrumentation (gen_ai spans + evaluate_agent_span)."""

import os
from types import SimpleNamespace
from unittest.mock import patch

from anthropic.types import Message, TextBlock, Usage

import pytest

pytest.importorskip("anthropic")

from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.messages import AsyncMessages, Messages

from harness_sdk.plugins.control import ControlResult, get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation.anthropic import AnthropicInstrumentorWrapper


@pytest.fixture
def anthropic_instrumentor():
    wrapper = AnthropicInstrumentorWrapper()
    yield wrapper
    if AnthropicInstrumentorWrapper._applied:
        wrapper.uninstrument()
    get_control_registry().clear()


class _FakeUsage:
    input_tokens = 2
    output_tokens = 4
    cache_creation_input_tokens = 1
    cache_read_input_tokens = 3
    reasoning_output_tokens = 2


class _FakeMessage:
    id = "msg_test"
    model = "claude-3-haiku-20240307"
    stop_reason = "end_turn"
    role = "assistant"
    content = [{"type": "text", "text": "ok"}]
    usage = _FakeUsage()


def _fake_messages_create(_self, *_args, **_kwargs):
    return _FakeMessage()


def test_anthropic_span_has_gen_ai_attributes(agent, exporter, anthropic_instrumentor):
    with patch.object(Messages, "create", new=_fake_messages_create):
        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.provider.name") == "anthropic"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.request.model") == "claude-3-haiku-20240307"
    assert attrs.get("gen_ai.response.id") == "msg_test"
    assert attrs.get("gen_ai.usage.input_tokens") == 2
    assert attrs.get("gen_ai.usage.output_tokens") == 4
    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 3
    assert attrs.get("gen_ai.usage.cache_creation.input_tokens") == 1
    assert attrs.get("gen_ai.usage.reasoning.output_tokens") == 2


async def test_async_messages_create_span_has_gen_ai_attributes(agent, exporter, anthropic_instrumentor):
    async def _fake(_self, *_args, **_kwargs):
        return _FakeMessage()

    with patch.object(AsyncMessages, "create", new=_fake):
        anthropic_instrumentor.instrument()
        client = AsyncAnthropic(api_key="sk-test")
        await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.provider.name") == "anthropic"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.request.model") == "claude-3-haiku-20240307"
    assert attrs.get("gen_ai.response.id") == "msg_test"
    assert attrs.get("gen_ai.usage.input_tokens") == 2
    assert attrs.get("gen_ai.usage.output_tokens") == 4
    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 3
    assert attrs.get("gen_ai.usage.cache_creation.input_tokens") == 1
    assert attrs.get("gen_ai.usage.reasoning.output_tokens") == 2



def test_anthropic_evaluate_blocks_before_fake_create(agent, exporter, anthropic_instrumentor):
    calls = {"n": 0}

    def counting_fake(self, *a, **k):  # pylint: disable=unused-argument
        calls["n"] += 1
        return _FakeMessage()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(Messages, "create", new=counting_fake):
        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            )

    assert calls["n"] == 0

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.request.model") == "claude-3-haiku-20240307"


async def test_async_messages_create_evaluate_blocks(agent, exporter, anthropic_instrumentor):
    calls = {"n": 0}

    async def counting_fake(self, *a, **k):  # pylint: disable=unused-argument
        calls["n"] += 1
        return _FakeMessage()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(AsyncMessages, "create", new=counting_fake):
        anthropic_instrumentor.instrument()
        client = AsyncAnthropic(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            )

    assert calls["n"] == 0
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def _stream_events():
    usage_start = SimpleNamespace(input_tokens=3, output_tokens=0)
    msg = SimpleNamespace(
        id="msg_stream",
        model="claude-3-haiku-20240307",
        usage=usage_start,
    )
    yield SimpleNamespace(type="message_start", message=msg)
    delta = SimpleNamespace(stop_reason="end_turn")
    out_usage = SimpleNamespace(output_tokens=5)
    yield SimpleNamespace(type="message_delta", delta=delta, usage=out_usage)


_FINAL_SNAPSHOT = Message(
    id="msg_stream",
    model="claude-3-haiku-20240307",
    role="assistant",
    stop_reason="end_turn",
    type="message",
    content=[TextBlock(type="text", text="hi")],
    usage=Usage(input_tokens=3, output_tokens=5),
)


class _FakeSSEStream:
    """Minimal sync stream for create(stream=True) tests."""

    def __init__(self) -> None:
        self._items = list(_stream_events())
        self._i = 0
        self.response = SimpleNamespace(close=lambda: None)
        self.current_message_snapshot = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._items):
            raise StopIteration
        item = self._items[self._i]
        self._i += 1
        if self._i == len(self._items):
            self.current_message_snapshot = _FINAL_SNAPSHOT
        return item

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def close(self) -> None:
        pass


def _fake_messages_create_streaming(_self, *_args, **_kwargs):
    assert _kwargs.get("stream") is True
    return _FakeSSEStream()


def test_anthropic_streaming_span_after_consume(agent, exporter, anthropic_instrumentor):
    with patch.object(Messages, "create", new=_fake_messages_create_streaming):
        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        stream = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        list(stream)

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.streaming") is True
    assert attrs.get("gen_ai.usage.output_tokens") == 5


class _FakeAsyncSSEStream:
    """Minimal async stream for create(stream=True) tests."""

    def __init__(self) -> None:
        self._items = list(_stream_events())
        self._i = 0
        self.response = SimpleNamespace(aclose=lambda: None)
        self.current_message_snapshot = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        if self._i == len(self._items):
            self.current_message_snapshot = _FINAL_SNAPSHOT
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def close(self) -> None:
        pass


async def _fake_async_messages_create_streaming(_self, *_args, **_kwargs):
    assert _kwargs.get("stream") is True
    return _FakeAsyncSSEStream()


async def test_anthropic_async_streaming_span_after_consume(agent, exporter, anthropic_instrumentor):
    with patch.object(AsyncMessages, "create", new=_fake_async_messages_create_streaming):
        anthropic_instrumentor.instrument()
        client = AsyncAnthropic(api_key="sk-test")
        stream = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        async for _ in stream:
            pass

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.usage.output_tokens") == 5


# ── Messages.stream (sync context-manager API) ─────────────────────────────

class _FakeMessageStreamManager:
    """Mimics anthropic.lib.streaming.MessageStreamManager for Messages.stream."""

    def __init__(self, stream: _FakeSSEStream) -> None:
        self._stream = stream

    def __enter__(self) -> _FakeSSEStream:
        return self._stream

    def __exit__(self, *args) -> None:
        pass


def _fake_messages_stream(_self, *_args, **_kwargs):
    return _FakeMessageStreamManager(_FakeSSEStream())


def test_messages_stream_sync_span_after_consume(agent, exporter, anthropic_instrumentor):
    with patch.object(Messages, "stream", new=_fake_messages_stream):
        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        with client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        ) as stream:
            list(stream)

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.provider.name") == "anthropic"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.usage.output_tokens") == 5


def test_messages_stream_sync_evaluate_blocks(agent, exporter, anthropic_instrumentor):
    calls = {"n": 0}

    def counting_stream(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeMessageStreamManager(_FakeSSEStream())

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(Messages, "stream", new=counting_stream):
        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            with client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            ):
                pass

    assert calls["n"] == 0


# ── AsyncMessages.stream (async context-manager API) ───────────────────────

class _FakeAsyncMessageStreamManager:
    """Mimics anthropic.lib.streaming.AsyncMessageStreamManager."""

    def __init__(self, stream: _FakeAsyncSSEStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> _FakeAsyncSSEStream:
        return self._stream

    async def __aexit__(self, *args) -> None:
        pass


def _fake_async_messages_stream(_self, *_args, **_kwargs):
    return _FakeAsyncMessageStreamManager(_FakeAsyncSSEStream())


async def test_messages_stream_async_span_after_consume(agent, exporter, anthropic_instrumentor):
    with patch.object(AsyncMessages, "stream", new=_fake_async_messages_stream):
        anthropic_instrumentor.instrument()
        client = AsyncAnthropic(api_key="sk-test")
        async with client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        ) as stream:
            async for _ in stream:
                pass

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.usage.output_tokens") == 5


async def test_messages_stream_async_evaluate_blocks(agent, exporter, anthropic_instrumentor):
    calls = {"n": 0}

    def counting_stream(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeAsyncMessageStreamManager(_FakeAsyncSSEStream())

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(AsyncMessages, "stream", new=counting_stream):
        anthropic_instrumentor.instrument()
        client = AsyncAnthropic(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            async with client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            ):
                pass

    assert calls["n"] == 0


# ── Double-wrap guard ───────────────────────────────────────────────────────

def test_double_instrument_does_not_double_wrap(agent, exporter, anthropic_instrumentor):
    call_count = {"n": 0}

    def counting_fake(_self, *_a, **_k):
        call_count["n"] += 1
        return _FakeMessage()

    with patch.object(Messages, "create", new=counting_fake):
        anthropic_instrumentor.instrument()
        AnthropicInstrumentorWrapper().instrument()  # second instance — should be a no-op

        client = Anthropic(api_key="sk-test")
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1  # only one span, not two


# ── gen_ai disabled passthrough ─────────────────────────────────────────────

def test_gen_ai_disabled_passthrough(agent, exporter, anthropic_instrumentor):
    call_count = {"n": 0}

    def counting_fake(_self, *_a, **_k):
        call_count["n"] += 1
        return _FakeMessage()

    with patch.object(Messages, "create", new=counting_fake):
        os.environ["HA_GEN_AI_ENABLED"] = "false"
        from harness_sdk.config.config import Config
        Config._instance = None  # force re-read of env

        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        result = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 0
    assert call_count["n"] == 1  # reached the real (fake) implementation


# ── Error path: API exception records a failed span ─────────────────────────

def test_messages_create_exception_records_error_span(agent, exporter, anthropic_instrumentor):
    def exploding_fake(_self, *_a, **_k):
        raise RuntimeError("upstream API error")

    with patch.object(Messages, "create", new=exploding_fake):
        anthropic_instrumentor.instrument()
        client = Anthropic(api_key="sk-test")
        with pytest.raises(RuntimeError, match="upstream API error"):
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR


async def test_async_messages_create_exception_records_error_span(agent, exporter, anthropic_instrumentor):
    async def exploding_fake(_self, *_a, **_k):
        raise RuntimeError("upstream async API error")

    with patch.object(AsyncMessages, "create", new=exploding_fake):
        anthropic_instrumentor.instrument()
        client = AsyncAnthropic(api_key="sk-test")
        with pytest.raises(RuntimeError, match="upstream async API error"):
            await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR
