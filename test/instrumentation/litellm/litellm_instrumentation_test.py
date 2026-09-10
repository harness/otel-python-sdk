from test.control_test_helpers import AlwaysBlockControlPlugin
"""Tests for LiteLLM instrumentation (gen_ai spans + evaluate_agent_span)."""

from unittest.mock import patch

import pytest

pytest.importorskip("litellm")

import litellm
from litellm.types.utils import EmbeddingResponse, ModelResponse

from harness_sdk.plugins.control import ControlResult, get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation.litellm import LiteLLMInstrumentorWrapper
import harness_sdk.instrumentation.litellm as harness_litellm


@pytest.fixture
def litellm_instrumentor():
    wrapper = LiteLLMInstrumentorWrapper()
    yield wrapper
    if getattr(wrapper, "_applied", False):
        wrapper.uninstrument()
    get_control_registry().clear()


def _fake_model_response(*_args, **_kwargs):
    return ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        model="gpt-4o-mini",
        usage={
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
            "prompt_tokens_details": {
                "cached_tokens": 1,
                "cache_creation_tokens": 2,
            },
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    )


def _fake_embedding_response(*_args, **_kwargs):
    return EmbeddingResponse(
        model="text-embedding-3-small",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        usage={"prompt_tokens": 4, "total_tokens": 4},
    )


def _request_span(spans):
    for span in spans:
        attrs = span.attributes or {}
        if attrs.get("gen_ai.request.model"):
            return span
    return spans[0]


def _litellm_spans(spans):
    return [span for span in spans if span.name == "litellm_request"]


def test_litellm_completion_span_has_gen_ai_attributes(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) >= 1
    attrs = _request_span(spans).attributes
    assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.framework") == "litellm"
    assert attrs.get("gen_ai.response.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.response.id") == "chatcmpl-test"
    assert attrs.get("gen_ai.response.finish_reasons") == "['stop']"
    assert attrs.get("gen_ai.usage.input_tokens") == 3
    assert attrs.get("gen_ai.usage.output_tokens") == 5
    assert attrs.get("gen_ai.usage.total_tokens") == 8
    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 1
    assert attrs.get("gen_ai.usage.cache_creation.input_tokens") == 2
    assert attrs.get("gen_ai.usage.reasoning.output_tokens") == 1


def test_litellm_input_messages_captured_when_enabled(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert "gen_ai.input.messages" in attrs
    assert "hi" in attrs.get("gen_ai.input.messages")


def test_litellm_input_messages_omitted_when_capture_disabled(agent, exporter, litellm_instrumentor, monkeypatch):  # pylint: disable=unused-argument
    monkeypatch.setenv("HA_GEN_AI_PAYLOAD_CAPTURE_ENABLED", "false")
    from harness_sdk.config.config import Config  # pylint: disable=import-outside-toplevel

    Config._instance = None

    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert "gen_ai.input.messages" not in attrs
    # Non-payload request attributes are still recorded.
    assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.operation.name") == "chat"


def test_litellm_evaluate_blocks_before_wrapped(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    calls = {"n": 0}

    def counting_fake(*_a, **_k):
        calls["n"] += 1
        return _fake_model_response()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch("litellm.main.completion", new=counting_fake):
        litellm_instrumentor.instrument()
        with pytest.raises(ControlEvaluationBlocked):
            litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

    assert calls["n"] == 0
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.request.model") == "gpt-4o-mini"


def test_litellm_embedding_span_has_gen_ai_attributes(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    with patch("litellm.main.embedding", new=_fake_embedding_response):
        litellm_instrumentor.instrument()
        litellm.embedding(model="text-embedding-3-small", input="trace this")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) >= 1
    attrs = _request_span(spans).attributes
    assert attrs.get("gen_ai.request.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.operation.name") == "embeddings"
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.framework") == "litellm"
    assert attrs.get("gen_ai.response.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.usage.input_tokens") == 4
    assert attrs.get("gen_ai.usage.total_tokens") == 4


@pytest.mark.asyncio
async def test_litellm_async_completion_span(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    async def _fake_async(*_args, **_kwargs):
        return _fake_model_response()

    with patch("litellm.main.acompletion", new=_fake_async):
        litellm_instrumentor.instrument()
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) >= 1
    attrs = _request_span(spans).attributes
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.response.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.response.id") == "chatcmpl-test"
    assert attrs.get("gen_ai.response.finish_reasons") == "['stop']"
    assert attrs.get("gen_ai.usage.input_tokens") == 3
    assert attrs.get("gen_ai.usage.output_tokens") == 5
    assert attrs.get("gen_ai.usage.total_tokens") == 8


def test_litellm_streaming_span_defers_until_consumed(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    litellm_instrumentor.instrument()
    stream = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
        mock_response="hello world",
    )

    # The span must NOT be finished before the stream is consumed: this is the
    # core bug this fix addresses.
    assert len(_litellm_spans(exporter.get_finished_spans())) == 0

    chunks = list(stream)
    assert len(chunks) > 0

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.request.streaming") == "True"
    # Response-side metadata is aggregated from the streamed chunks.
    assert attrs.get("gen_ai.response.finish_reasons") == "['stop']"


@pytest.mark.asyncio
async def test_litellm_async_streaming_span_defers_until_consumed(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    litellm_instrumentor.instrument()
    stream = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
        mock_response="hello world",
    )

    assert len(_litellm_spans(exporter.get_finished_spans())) == 0

    chunks = [chunk async for chunk in stream]
    assert len(chunks) > 0

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.streaming") == "True"
    assert attrs.get("gen_ai.response.finish_reasons") == "['stop']"


def test_litellm_double_instrument_is_noop(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        litellm_instrumentor.instrument()
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) >= 1


@pytest.mark.asyncio
async def test_litellm_async_embedding_emits_single_span(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    # Real litellm.aembedding re-dispatches to the sync embedding via an executor.
    # Only one litellm_request span must be produced for that single provider call.
    with patch("litellm.main.embedding", new=_fake_embedding_response):
        litellm_instrumentor.instrument()
        await litellm.aembedding(model="text-embedding-3-small", input="trace this")

    spans = exporter.get_finished_spans()
    exporter.clear()
    llm_spans = _litellm_spans(spans)
    assert len(llm_spans) == 1
    attrs = llm_spans[0].attributes
    assert attrs.get("gen_ai.operation.name") == "embeddings"
    assert attrs.get("gen_ai.response.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.usage.input_tokens") == 4


@pytest.mark.asyncio
async def test_litellm_async_completion_emits_single_span(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    llm_spans = _litellm_spans(spans)
    assert len(llm_spans) == 1
    attrs = llm_spans[0].attributes
    assert attrs.get("gen_ai.usage.input_tokens") == 3
    assert attrs.get("gen_ai.usage.output_tokens") == 5


def test_litellm_embedding_dict_response(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    def _dict_embedding(*_args, **_kwargs):
        return {
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.1], "index": 0, "object": "embedding"}],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }

    with patch("litellm.main.embedding", new=_dict_embedding):
        litellm_instrumentor.instrument()
        litellm.embedding(model="text-embedding-3-small", input="trace this")

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert attrs.get("gen_ai.response.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.usage.input_tokens") == 7
    assert attrs.get("gen_ai.usage.total_tokens") == 7


def test_litellm_response_model_fallback_to_request_model(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    def _no_model_embedding(*_args, **_kwargs):
        resp = EmbeddingResponse(
            model="text-embedding-3-small",
            data=[{"embedding": [0.1], "index": 0, "object": "embedding"}],
            usage={"prompt_tokens": 4, "total_tokens": 4},
        )
        resp.model = None
        return resp

    with patch("litellm.main.embedding", new=_no_model_embedding):
        litellm_instrumentor.instrument()
        litellm.embedding(model="text-embedding-3-small", input="trace this")

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert attrs.get("gen_ai.response.model") == "text-embedding-3-small"


def test_litellm_bedrock_execution_model_from_hidden_params(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    def _bedrock_response(*_args, **_kwargs):
        resp = _fake_model_response()
        resp._hidden_params = {
            "additional_headers": {
                "x-amzn-bedrock-model-id": "anthropic.claude-haiku-4-5-v1:0"
            }
        }
        return resp

    with patch("litellm.main.completion", new=_bedrock_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="bedrock/converse/arn:aws:bedrock:us-east-1:1234:application-inference-profile/abc",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert attrs.get("aws.bedrock.execution_model_id") == "anthropic.claude-haiku-4-5-v1:0"


def test_litellm_bedrock_execution_model_from_normalized_header(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    def _bedrock_response(*_args, **_kwargs):
        resp = _fake_model_response()
        resp._hidden_params = {
            "additional_headers": {
                "llm_provider-x-amzn-bedrock-model-id": "anthropic.claude-haiku-4-5-v1:0"
            }
        }
        return resp

    with patch("litellm.main.completion", new=_bedrock_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="bedrock/converse/arn:aws:bedrock:us-east-1:1234:application-inference-profile/abc",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert attrs.get("aws.bedrock.execution_model_id") == "anthropic.claude-haiku-4-5-v1:0"


def test_litellm_raw_usage_capture_opt_in(agent, exporter, litellm_instrumentor, monkeypatch):  # pylint: disable=unused-argument
    monkeypatch.setenv("HA_GEN_AI_RAW_CAPTURE_ENABLED", "true")
    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    raw = attrs.get("gen_ai.response.usage.raw")
    assert raw is not None
    assert "prompt_tokens" in raw


def test_litellm_raw_usage_capture_disabled_by_default(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    with patch("litellm.main.completion", new=_fake_model_response):
        litellm_instrumentor.instrument()
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    spans = exporter.get_finished_spans()
    exporter.clear()
    attrs = _request_span(spans).attributes
    assert "gen_ai.response.usage.raw" not in attrs


def test_litellm_legacy_gen_ai_master_ignored(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    import os

    # Legacy master env must not disable instrumentation once opted in.
    os.environ["HA_GEN_AI_ENABLED"] = "false"
    from harness_sdk.config.config import Config

    Config._instance = None

    calls = {"n": 0}

    def counting_fake(*_a, **_k):
        calls["n"] += 1
        return _fake_model_response()

    with patch("litellm.main.completion", new=counting_fake):
        litellm_instrumentor.instrument()
        import litellm.main as litellm_main

        litellm_main.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert calls["n"] == 1
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def test_litellm_mock_response_with_wrapper_enrichment(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    litellm_instrumentor.instrument()
    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="ok",
    )

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) >= 1
    attrs = _request_span(spans).attributes
    assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "openai"
    assert "gen_ai.system" not in attrs
    assert attrs.get("gen_ai.framework") == "litellm"


def _fake_anthropic_message_response(*_args, **_kwargs):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class _FakeAnthropicStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _anthropic_messages_kwargs():
    return {
        "model": "bedrock/anthropic.claude-sonnet-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
        "temperature": 0.2,
        "custom_llm_provider": "bedrock",
    }


@pytest.mark.asyncio
async def test_acreate_not_captured_when_only_acompletion_is_wrapped(  # pylint: disable=unused-argument
    agent, exporter, litellm_instrumentor, monkeypatch
):
    # Pre-fix behavior: wrapping acompletion alone misses native Anthropic
    # pass-through (acreate does not call acompletion).
    monkeypatch.setattr(harness_litellm, "_ANTHROPIC_MESSAGES_FUNCTIONS", ())

    async def _native_acreate(*_args, **_kwargs):
        return _fake_anthropic_message_response()

    with patch("litellm.anthropic_interface.messages.acreate", new=_native_acreate):
        litellm_instrumentor.instrument()
        await litellm.anthropic.messages.acreate(**_anthropic_messages_kwargs())

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert spans == []


@pytest.mark.asyncio
async def test_litellm_anthropic_acreate_span_has_gen_ai_attributes(  # pylint: disable=unused-argument
    agent, exporter, litellm_instrumentor
):
    async def _native_acreate(*_args, **_kwargs):
        return _fake_anthropic_message_response()

    with patch("litellm.anthropic_interface.messages.acreate", new=_native_acreate):
        litellm_instrumentor.instrument()
        await litellm.anthropic.messages.acreate(**_anthropic_messages_kwargs())

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.model") == "bedrock/anthropic.claude-sonnet-4"
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "aws.bedrock"
    assert attrs.get("gen_ai.framework") == "litellm"
    assert attrs.get("gen_ai.request.max_tokens") == 32
    assert attrs.get("gen_ai.request.temperature") == 0.2
    assert attrs.get("gen_ai.response.id") == "msg_test"
    assert attrs.get("gen_ai.response.model") == "claude-sonnet-4"
    assert attrs.get("gen_ai.response.finish_reasons") == "['end_turn']"
    assert attrs.get("gen_ai.usage.input_tokens") == 10
    assert attrs.get("gen_ai.usage.output_tokens") == 5


def test_litellm_anthropic_create_span_has_gen_ai_attributes(  # pylint: disable=unused-argument
    agent, exporter, litellm_instrumentor
):
    with patch(
        "litellm.anthropic_interface.messages.create",
        new=_fake_anthropic_message_response,
    ):
        litellm_instrumentor.instrument()
        litellm.anthropic.messages.create(**_anthropic_messages_kwargs())

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.model") == "bedrock/anthropic.claude-sonnet-4"
    assert attrs.get("gen_ai.response.finish_reasons") == "['end_turn']"
    assert attrs.get("gen_ai.usage.input_tokens") == 10


@pytest.mark.asyncio
async def test_litellm_anthropic_acreate_delegating_to_acompletion_emits_single_span(  # pylint: disable=unused-argument
    agent, exporter, litellm_instrumentor
):
    async def _fake_async_completion(*_args, **_kwargs):
        return _fake_model_response()

    async def _delegating_acreate(*_args, **kwargs):
        return await litellm.acompletion(
            model=kwargs["model"],
            messages=kwargs["messages"],
        )

    with patch("litellm.main.acompletion", new=_fake_async_completion), patch(
        "litellm.anthropic_interface.messages.acreate", new=_delegating_acreate
    ):
        litellm_instrumentor.instrument()
        await litellm.anthropic.messages.acreate(**_anthropic_messages_kwargs())

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.model") == "bedrock/anthropic.claude-sonnet-4"
    assert attrs.get("gen_ai.usage.input_tokens") == 3
    assert attrs.get("gen_ai.usage.output_tokens") == 5


@pytest.mark.asyncio
async def test_litellm_anthropic_acreate_streaming_defers_until_consumed(  # pylint: disable=unused-argument
    agent, exporter, litellm_instrumentor
):
    async def _streaming_acreate(*_args, **_kwargs):
        return _FakeAnthropicStream(
            [
                {"type": "content_block_delta", "delta": {"text": "hello"}},
                {
                    "id": "msg_stream",
                    "model": "claude-sonnet-4",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 6, "output_tokens": 2},
                },
            ]
        )

    with patch("litellm.anthropic_interface.messages.acreate", new=_streaming_acreate):
        litellm_instrumentor.instrument()
        stream = await litellm.anthropic.messages.acreate(
            **_anthropic_messages_kwargs(), stream=True
        )

        assert len(_litellm_spans(exporter.get_finished_spans())) == 0

        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 2

    spans = _litellm_spans(exporter.get_finished_spans())
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.streaming") == "True"
    assert attrs.get("gen_ai.response.finish_reasons") == "['end_turn']"
    assert attrs.get("gen_ai.usage.input_tokens") == 6
    assert attrs.get("gen_ai.usage.output_tokens") == 2
