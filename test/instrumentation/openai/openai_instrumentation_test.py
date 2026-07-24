from test.control_test_helpers import AlwaysBlockControlPlugin
"""Tests for OpenAI SDK instrumentation (gen_ai spans + evaluate_agent_span)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from openai import AsyncOpenAI, OpenAI
from openai.resources.chat.completions.completions import AsyncCompletions, Completions
from openai.resources.embeddings import AsyncEmbeddings, Embeddings

from harness_sdk.plugins.control import ControlResult, get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation.openai import OpenAIInstrumentorWrapper


@pytest.fixture
def openai_instrumentor():
    wrapper = OpenAIInstrumentorWrapper()
    yield wrapper
    if getattr(wrapper, "_applied", False):
        wrapper.uninstrument()
    get_control_registry().clear()


class _FakeMessage:
    role = "assistant"
    content = "hello from model"
    tool_calls = None


class _FakeChoice:
    finish_reason = "stop"
    message = _FakeMessage()


class _FakeUsage:
    prompt_tokens = 3
    completion_tokens = 7


class _FakeChatCompletion:
    id = "chatcmpl-test"
    model = "gpt-4o-mini"
    choices = [_FakeChoice()]
    usage = _FakeUsage()


def _fake_completions_create(_self, *_args, **_kwargs):
    return _FakeChatCompletion()


def test_openai_span_has_gen_ai_attributes(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    with patch.object(Completions, "create", new=_fake_completions_create):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.response.id") == "chatcmpl-test"
    assert attrs.get("gen_ai.usage.input_tokens") == 3
    assert attrs.get("gen_ai.usage.output_tokens") == 7



def test_openai_evaluate_blocks_before_fake_create(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    calls = {"n": 0}

    def counting_fake(self, *a, **k):  # pylint: disable=unused-argument
        calls["n"] += 1
        return _FakeChatCompletion()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(Completions, "create", new=counting_fake):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

    assert calls["n"] == 0

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.request.model") == "gpt-4o-mini"


def _stream_chunks():
    yield SimpleNamespace(
        id="chatcmpl-stream",
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content="Hel", tool_calls=None),
                finish_reason=None,
            )
        ],
        usage=None,
    )
    yield SimpleNamespace(
        id="chatcmpl-stream",
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content="lo", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


class _FakeOpenAIStream:
    def __init__(self) -> None:
        self._it = iter(list(_stream_chunks()))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def close(self) -> None:
        pass


def _fake_completions_create_streaming(_self, *_args, **_kwargs):
    assert _kwargs.get("stream") is True
    return _FakeOpenAIStream()


def test_openai_streaming_span_after_consume(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    with patch.object(Completions, "create", new=_fake_completions_create_streaming):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        list(stream)

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert attrs.get("gen_ai.request.streaming") is True
    assert attrs.get("gen_ai.response.id") == "chatcmpl-stream"
    assert attrs.get("gen_ai.usage.output_tokens") == 2


class _FakeAsyncOpenAIStream:
    def __init__(self) -> None:
        self._items = list(_stream_chunks())
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def close(self) -> None:
        pass


async def _fake_async_completions_create_streaming(_self, *_args, **_kwargs):
    assert _kwargs.get("stream") is True
    return _FakeAsyncOpenAIStream()


@pytest.mark.asyncio
async def test_openai_async_streaming_span_after_consume(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    with patch.object(AsyncCompletions, "create", new=_fake_async_completions_create_streaming):
        openai_instrumentor.instrument()
        client = AsyncOpenAI(api_key="sk-test")
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in stream:
            pass

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.response.id") == "chatcmpl-stream"


class _FakeEmbeddingObj:
    embedding = [0.1, 0.2, 0.3, 0.4]


class _FakeEmbeddingUsage:
    prompt_tokens = 9
    total_tokens = 9


class _FakeEmbeddingResponse:
    model = "text-embedding-3-small"
    data = [_FakeEmbeddingObj()]
    usage = _FakeEmbeddingUsage()


def _fake_embeddings_create(_self, *_args, **_kwargs):
    return _FakeEmbeddingResponse()


def test_openai_embeddings_span_has_gen_ai_attributes(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    with patch.object(Embeddings, "create", new=_fake_embeddings_create):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        client.embeddings.create(
            model="text-embedding-3-small",
            input="trace this",
            encoding_format="float",
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert attrs.get("gen_ai.operation.name") == "embeddings"
    assert attrs.get("gen_ai.request.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.response.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.usage.input_tokens") == 9
    assert attrs.get("gen_ai.embeddings.dimension.count") == 4
    assert "float" in (attrs.get("gen_ai.request.encoding_formats") or ())


def test_openai_embeddings_evaluate_blocks_before_wrapped(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    calls = {"n": 0}

    def counting_fake(self, *a, **k):  # pylint: disable=unused-argument
        calls["n"] += 1
        return _FakeEmbeddingResponse()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(Embeddings, "create", new=counting_fake):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            client.embeddings.create(model="text-embedding-3-small", input="x")

    assert calls["n"] == 0
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.operation.name") == "embeddings"


async def _fake_async_embeddings_create(_self, *_args, **_kwargs):
    return _FakeEmbeddingResponse()


@pytest.mark.asyncio
async def test_openai_async_embeddings_span(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    with patch.object(AsyncEmbeddings, "create", new=_fake_async_embeddings_create):
        openai_instrumentor.instrument()
        client = AsyncOpenAI(api_key="sk-test")
        await client.embeddings.create(model="text-embedding-3-small", input="async")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.operation.name") == "embeddings"
    assert spans[0].attributes.get("gen_ai.usage.input_tokens") == 9


def test_double_instrument_is_noop(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    with patch.object(Completions, "create", new=_fake_completions_create):
        openai_instrumentor.instrument()
        openai_instrumentor.instrument()  # second call must not double-wrap
        client = OpenAI(api_key="sk-test")
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def test_legacy_gen_ai_master_ignored_chat(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    import os
    # Legacy master env must not disable instrumentation once the provider is opted in.
    os.environ["HA_GEN_AI_ENABLED"] = "false"
    from harness_sdk.config.config import Config
    Config._instance = None

    calls = {"n": 0}

    def counting_fake(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeChatCompletion()

    with patch.object(Completions, "create", new=counting_fake):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


@pytest.mark.asyncio
async def test_legacy_gen_ai_master_ignored_async_chat(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    import os
    os.environ["HA_GEN_AI_ENABLED"] = "false"
    from harness_sdk.config.config import Config
    Config._instance = None

    calls = {"n": 0}

    async def counting_fake(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeChatCompletion()

    with patch.object(AsyncCompletions, "create", new=counting_fake):
        openai_instrumentor.instrument()
        client = AsyncOpenAI(api_key="sk-test")
        await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert calls["n"] == 1
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def test_legacy_gen_ai_master_ignored_embeddings(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    import os
    os.environ["HA_GEN_AI_ENABLED"] = "false"
    from harness_sdk.config.config import Config
    Config._instance = None

    calls = {"n": 0}

    def counting_fake(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeEmbeddingResponse()

    with patch.object(Embeddings, "create", new=counting_fake):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        client.embeddings.create(model="text-embedding-3-small", input="x")

    assert calls["n"] == 1
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


@pytest.mark.asyncio
async def test_legacy_gen_ai_master_ignored_async_embeddings(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    import os
    os.environ["HA_GEN_AI_ENABLED"] = "false"
    from harness_sdk.config.config import Config
    Config._instance = None

    calls = {"n": 0}

    async def counting_fake(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeEmbeddingResponse()

    with patch.object(AsyncEmbeddings, "create", new=counting_fake):
        openai_instrumentor.instrument()
        client = AsyncOpenAI(api_key="sk-test")
        await client.embeddings.create(model="text-embedding-3-small", input="x")

    assert calls["n"] == 1
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def test_completions_exception_records_error(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    def raising_fake(_self, *_a, **_k):
        raise RuntimeError("upstream error")

    with patch.object(Completions, "create", new=raising_fake):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        with pytest.raises(RuntimeError, match="upstream error"):
            client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_completions_exception_records_error(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    async def raising_fake(_self, *_a, **_k):
        raise RuntimeError("async upstream error")

    with patch.object(AsyncCompletions, "create", new=raising_fake):
        openai_instrumentor.instrument()
        client = AsyncOpenAI(api_key="sk-test")
        with pytest.raises(RuntimeError, match="async upstream error"):
            await client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR


def test_embeddings_exception_records_error(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    def raising_fake(_self, *_a, **_k):
        raise RuntimeError("embed error")

    with patch.object(Embeddings, "create", new=raising_fake):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        with pytest.raises(RuntimeError, match="embed error"):
            client.embeddings.create(model="text-embedding-3-small", input="x")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_embeddings_exception_records_error(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    async def raising_fake(_self, *_a, **_k):
        raise RuntimeError("async embed error")

    with patch.object(AsyncEmbeddings, "create", new=raising_fake):
        openai_instrumentor.instrument()
        client = AsyncOpenAI(api_key="sk-test")
        with pytest.raises(RuntimeError, match="async embed error"):
            await client.embeddings.create(model="text-embedding-3-small", input="x")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    from opentelemetry.trace import StatusCode
    assert spans[0].status.status_code == StatusCode.ERROR


def test_evaluate_invocation_skips_when_disabled(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    import os
    os.environ["HA_GEN_AI_PAYLOAD_EVALUATION_ENABLED"] = "false"
    from harness_sdk.config.config import Config
    Config._instance = None

    calls = {"n": 0}

    class _CountingPlugin:
        name = "counting"
        def on_init(self, config): pass  # pylint: disable=unused-argument
        def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
            return ControlResult()
        def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
            calls["n"] += 1
            return ControlResult()
        def shutdown(self): pass

    get_control_registry().register(_CountingPlugin())

    with patch.object(Completions, "create", new=_fake_completions_create):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    assert calls["n"] == 0
    exporter.clear()


def test_evaluate_invocation_swallows_non_block_errors(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    class _ErrorPlugin:
        name = "error"
        def on_init(self, config): pass  # pylint: disable=unused-argument
        def evaluate(self, span, url, headers, body, is_grpc):  # pylint: disable=unused-argument
            return ControlResult()
        def evaluate_agent_span(self, span, body=""):  # pylint: disable=unused-argument
            raise ValueError("filter exploded")
        def shutdown(self): pass

    get_control_registry().register(_ErrorPlugin())

    with patch.object(Completions, "create", new=_fake_completions_create):
        openai_instrumentor.instrument()
        client = OpenAI(api_key="sk-test")
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def test_embeddings_evaluate_blocks_async(agent, exporter, openai_instrumentor):  # pylint: disable=unused-argument
    calls = {"n": 0}

    async def counting_fake(_self, *_a, **_k):
        calls["n"] += 1
        return _FakeEmbeddingResponse()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(AsyncEmbeddings, "create", new=counting_fake):
        openai_instrumentor.instrument()
        import asyncio
        client = AsyncOpenAI(api_key="sk-test")
        with pytest.raises(ControlEvaluationBlocked):
            asyncio.get_event_loop().run_until_complete(
                client.embeddings.create(model="text-embedding-3-small", input="x")
            )

    assert calls["n"] == 0
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.operation.name") == "embeddings"
