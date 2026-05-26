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
        usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
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
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.model") == "gpt-4o-mini"
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.system") == "openai"
    assert attrs.get("gen_ai.framework") == "litellm"


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
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.model") == "text-embedding-3-small"
    assert attrs.get("gen_ai.operation.name") == "embeddings"
    assert attrs.get("gen_ai.framework") == "litellm"


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
    assert spans[0].attributes.get("gen_ai.operation.name") == "chat"


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


def test_litellm_gen_ai_disabled_passthrough(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
    import os

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
    assert len(spans) == 0


def test_litellm_mock_response_with_otel_callback(agent, exporter, litellm_instrumentor):  # pylint: disable=unused-argument
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
    assert attrs.get("gen_ai.system") == "openai"
    assert attrs.get("gen_ai.framework") == "litellm"
