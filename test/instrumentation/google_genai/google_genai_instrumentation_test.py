"""End-to-end google-genai instrumentation tests (skipped if SDK not installed)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness_sdk.plugins.control import get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation import google_genai as gg

pytest.importorskip("google.genai")

from google.genai.models import Models  # noqa: E402


def _fake_usage(prompt=5, candidates=8, total=13):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
    )


def _fake_response(text="hello from gemini", finish_reason="STOP"):
    candidate = SimpleNamespace(
        finish_reason=finish_reason,
        content=SimpleNamespace(
            role="model",
            parts=[SimpleNamespace(text=text, function_call=None, function_response=None)],
        ),
    )
    return SimpleNamespace(
        response_id="resp-123",
        model_version="gemini-2.0-flash",
        usage_metadata=_fake_usage(),
        candidates=[candidate],
        text=text,
    )


def _client():
    from google import genai  # pylint: disable=import-outside-toplevel

    return genai.Client(api_key="test-key")


def _vertex_client():
    from google import genai  # pylint: disable=import-outside-toplevel

    return genai.Client(vertexai=True, project="p", location="us-central1")


@pytest.fixture
def google_genai_instrumentor():
    wrapper = gg.GoogleGenAIInstrumentorWrapper()
    yield wrapper
    if getattr(wrapper, "_applied", False):
        wrapper.uninstrument()
    get_control_registry().clear()


def test_generate_content_span_has_gen_ai_attributes(agent, exporter, google_genai_instrumentor):  # pylint: disable=unused-argument
    def fake_generate(_self, *_args, **_kwargs):
        return _fake_response()

    with patch.object(Models, "generate_content", new=fake_generate):
        google_genai_instrumentor.instrument()
        client = _client()
        client.models.generate_content(model="gemini-2.0-flash", contents="hi")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == "gcp.gemini"
    assert attrs.get("gen_ai.request.model") == "gemini-2.0-flash"
    assert attrs.get("gen_ai.response.id") == "resp-123"
    assert attrs.get("gen_ai.response.model") == "gemini-2.0-flash"
    assert attrs.get("gen_ai.usage.input_tokens") == 5
    assert attrs.get("gen_ai.usage.output_tokens") == 8


def test_vertex_generate_content_span_has_vertex_provider(
    agent, exporter, google_genai_instrumentor
):  # pylint: disable=unused-argument
    def fake_generate(_self, *_args, **_kwargs):
        return _fake_response()

    with patch.object(Models, "generate_content", new=fake_generate):
        google_genai_instrumentor.instrument()
        client = _vertex_client()
        client.models.generate_content(model="gemini-2.0-flash", contents="hi")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    assert spans[0].attributes.get("gen_ai.provider.name") == "gcp.vertex_ai"


def test_embed_content_span_has_embedding_attributes(
    agent, exporter, google_genai_instrumentor
):  # pylint: disable=unused-argument
    response = SimpleNamespace(
        model_version="text-embedding-004",
        embeddings=[
            SimpleNamespace(
                values=[0.1, 0.2, 0.3],
                statistics=SimpleNamespace(token_count=7),
            )
        ],
    )

    def fake_embed(_self, *_args, **_kwargs):
        return response

    with patch.object(Models, "embed_content", new=fake_embed):
        google_genai_instrumentor.instrument()
        client = _client()
        client.models.embed_content(model="text-embedding-004", contents="hi")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.operation.name") == "embeddings"
    assert attrs.get("gen_ai.usage.input_tokens") == 7


def test_uninstrument_roundtrip(
    agent, exporter, google_genai_instrumentor
):  # pylint: disable=unused-argument
    def fake_generate(_self, *_args, **_kwargs):
        return _fake_response()

    with patch.object(Models, "generate_content", new=fake_generate):
        client = _client()
        google_genai_instrumentor.instrument()
        client.models.generate_content(model="gemini-2.0-flash", contents="first")
        google_genai_instrumentor.uninstrument()
        client.models.generate_content(model="gemini-2.0-flash", contents="untraced")
        google_genai_instrumentor.instrument()
        client.models.generate_content(model="gemini-2.0-flash", contents="second")

    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 2


def test_generate_content_evaluate_blocks_before_call(agent, exporter, google_genai_instrumentor):  # pylint: disable=unused-argument
    from test.control_test_helpers import AlwaysBlockControlPlugin  # pylint: disable=import-outside-toplevel

    calls = {"n": 0}

    def counting_fake(_self, *_a, **_k):
        calls["n"] += 1
        return _fake_response()

    get_control_registry().register(AlwaysBlockControlPlugin())

    with patch.object(Models, "generate_content", new=counting_fake):
        google_genai_instrumentor.instrument()
        client = _client()
        with pytest.raises(ControlEvaluationBlocked):
            client.models.generate_content(model="gemini-2.0-flash", contents="hi")

    assert calls["n"] == 0
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1


def test_generate_content_stream_accumulates(agent, exporter, google_genai_instrumentor):  # pylint: disable=unused-argument
    def fake_stream(_self, *_args, **_kwargs):
        yield SimpleNamespace(
            response_id="resp-s", model_version="gemini-2.0-flash",
            usage_metadata=None, candidates=[], text="Hel",
        )
        yield SimpleNamespace(
            response_id=None, model_version=None,
            usage_metadata=_fake_usage(prompt=1, candidates=2),
            candidates=[SimpleNamespace(finish_reason="STOP")], text="lo",
        )

    with patch.object(Models, "generate_content_stream", new=fake_stream):
        google_genai_instrumentor.instrument()
        client = _client()
        chunks = list(client.models.generate_content_stream(model="gemini-2.0-flash", contents="hi"))

    assert len(chunks) == 2
    spans = exporter.get_finished_spans()
    exporter.clear()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.request.streaming") is True
    assert attrs.get("gen_ai.usage.input_tokens") == 1
    assert attrs.get("gen_ai.usage.output_tokens") == 2
