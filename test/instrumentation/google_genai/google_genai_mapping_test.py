"""Pure request/response -> gen_ai attribute mapping tests for google-genai.

These do not require the ``google-genai`` package to be installed; they only
exercise the mapping helpers against fake response objects.
"""

from types import SimpleNamespace

from harness_sdk.instrumentation import google_genai as gg


def _fake_usage(prompt=5, candidates=8, total=13):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=total,
    )


def _fake_text_part(text):
    return SimpleNamespace(text=text, function_call=None, function_response=None)


def _fake_function_call_part(name, args, call_id="call-1"):
    return SimpleNamespace(
        text=None,
        function_call=SimpleNamespace(name=name, args=args, id=call_id),
        function_response=None,
    )


def _fake_response(text="hello from gemini", finish_reason="STOP", with_tool=False):
    parts = [_fake_text_part(text)]
    if with_tool:
        parts.append(_fake_function_call_part("get_weather", {"city": "SF"}))
    candidate = SimpleNamespace(
        finish_reason=finish_reason,
        content=SimpleNamespace(role="model", parts=parts),
    )
    return SimpleNamespace(
        response_id="resp-123",
        model_version="gemini-2.0-flash",
        usage_metadata=_fake_usage(),
        candidates=[candidate],
        text=text,
    )


def test_apply_response_sets_metadata_fields():
    invocation = gg.LLMInvocation(request_model="gemini-2.0-flash", provider="gcp.gemini")
    gg._apply_response(invocation, _fake_response(), capture_content=False)

    assert invocation.response_id == "resp-123"
    assert invocation.response_model_name == "gemini-2.0-flash"
    assert invocation.input_tokens == 5
    assert invocation.output_tokens == 8
    assert invocation.finish_reasons == ["STOP"]
    assert invocation.output_messages[0].parts == []


def test_apply_response_captures_content_and_tool_calls():
    invocation = gg.LLMInvocation(request_model="gemini-2.0-flash", provider="gcp.gemini")
    gg._apply_response(invocation, _fake_response(with_tool=True), capture_content=True)

    parts = invocation.output_messages[0].parts
    kinds = {p.type for p in parts}
    assert "text" in kinds
    assert "tool_call" in kinds
    tool_part = next(p for p in parts if p.type == "tool_call")
    assert tool_part.name == "get_weather"
    assert tool_part.arguments == {"city": "SF"}


def test_to_input_messages_from_string():
    messages = gg._to_input_messages("what is the weather?", capture_content=True)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].parts[0].content == "what is the weather?"


def test_to_input_messages_empty_when_capture_off():
    assert gg._to_input_messages("hi", capture_content=False) == []


def test_stream_accumulator_merges_chunks():
    acc = gg._StreamAccumulator(capture_content=True)
    acc.add(SimpleNamespace(
        response_id="r1", model_version="gemini-2.0-flash",
        usage_metadata=None, candidates=[], text="Hel",
    ))
    acc.add(SimpleNamespace(
        response_id=None, model_version=None,
        usage_metadata=_fake_usage(prompt=2, candidates=4),
        candidates=[SimpleNamespace(finish_reason="STOP")], text="lo",
    ))
    invocation = gg.LLMInvocation(request_model="gemini-2.0-flash", provider="gcp.gemini")
    acc.apply(invocation)

    assert invocation.response_id == "r1"
    assert invocation.input_tokens == 2
    assert invocation.output_tokens == 4
    assert invocation.finish_reasons == ["STOP"]
    assert invocation.output_messages[0].parts[0].content == "Hello"


def test_resolve_provider_detects_backend():
    vertex_instance = SimpleNamespace(_api_client=SimpleNamespace(vertexai=True))
    gemini_instance = SimpleNamespace(_api_client=SimpleNamespace(vertexai=False))
    unknown_instance = SimpleNamespace()
    assert gg._resolve_provider(vertex_instance) == "gcp.vertex_ai"
    assert gg._resolve_provider(gemini_instance) == "gcp.gemini"
    assert gg._resolve_provider(unknown_instance) == "gcp.gen_ai"
