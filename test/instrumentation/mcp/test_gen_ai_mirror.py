import os
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.semconv_ai import SpanAttributes as AiSpanAttributes
from opentelemetry.semconv_ai import TraceloopSpanKindValues

from agent_trace.instrumentation.mcp import gen_ai_mirror as gen_ai_mirror_mod
from agent_trace.instrumentation.mcp.gen_ai_mirror import (
    GenAiMirroringTracer,
    apply_gen_ai_env_for_mcp,
    mirror_traceloop_to_gen_ai,
    mcp_instrumentation_get_tracer_patched,
    patch_get_tracer_for_mcp,
)


@pytest.fixture
def mock_gen_ai_config():
    gen = MagicMock()
    gen.enabled.value = True
    gen.payload_capture_enabled.value = True
    gen.payload_evaluation_enabled.value = True
    cfg = MagicMock()
    cfg.gen_ai = gen
    with patch("agent_trace.instrumentation.mcp.gen_ai_mirror.Config") as mock_cfg:
        root = MagicMock()
        root.config = cfg
        mock_cfg.return_value = root
        yield gen


def test_mirror_sets_execute_tool_and_system(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_SPAN_KIND, kind, kind
    )
    span.set_attribute.assert_any_call("gen_ai.operation.name", "execute_tool")
    span.set_attribute.assert_any_call("gen_ai.system", "mcp")


def test_mirror_tool_name_and_mcp_method(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(span, AiSpanAttributes.TRACELOOP_ENTITY_NAME, "weather", kind)
    span.set_attribute.assert_any_call("gen_ai.tool.name", "weather")
    span.set_attribute.assert_any_call(AiSpanAttributes.MCP_METHOD_NAME, "tools/call")


def test_mirror_skips_when_gen_ai_disabled(mock_gen_ai_config):
    mock_gen_ai_config.enabled.value = False
    span = MagicMock()
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value, None
    )
    span.set_attribute.assert_not_called()


def test_mirror_arguments_when_capture_and_eval(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    payload = '{"tool_name": "t", "arguments": {"city": "NYC"}}'
    mirror_traceloop_to_gen_ai(span, AiSpanAttributes.TRACELOOP_ENTITY_INPUT, payload, kind)
    span.set_attribute.assert_any_call(
        "gen_ai.tool.call.arguments", '{"city": "NYC"}'
    )


def test_apply_gen_ai_env_sets_trace_loop_from_capture(mock_gen_ai_config):
    mock_gen_ai_config.payload_capture_enabled.value = False
    with patch.dict(os.environ):
        os.environ.pop("TRACELOOP_TRACE_CONTENT", None)
        apply_gen_ai_env_for_mcp()
        assert os.environ.get("TRACELOOP_TRACE_CONTENT") == "false"


def test_apply_gen_ai_env_sets_true_when_capture_on(mock_gen_ai_config):
    mock_gen_ai_config.payload_capture_enabled.value = True
    with patch.dict(os.environ):
        os.environ.pop("TRACELOOP_TRACE_CONTENT", None)
        apply_gen_ai_env_for_mcp()
        assert os.environ.get("TRACELOOP_TRACE_CONTENT") == "true"


def test_apply_gen_ai_env_respects_existing_env(mock_gen_ai_config):
    with patch.dict(os.environ, {"TRACELOOP_TRACE_CONTENT": "true"}):
        apply_gen_ai_env_for_mcp()
        assert os.environ.get("TRACELOOP_TRACE_CONTENT") == "true"


def test_mirror_entity_name_skipped_when_not_tool_kind(mock_gen_ai_config):
    span = MagicMock()
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_ENTITY_NAME, "x", "session"
    )
    span.set_attribute.assert_not_called()


def test_mirror_output_dict_with_result(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span,
        AiSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        '{"result": {"ok": true}}',
        kind,
    )
    span.set_attribute.assert_any_call('gen_ai.tool.call.result', '{"ok": true}')


def test_mirror_output_list_payload(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    raw = '[{"type": "text", "text": "hi"}]'
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_ENTITY_OUTPUT, raw, kind
    )
    span.set_attribute.assert_any_call("gen_ai.tool.call.result", raw)


def test_mirror_output_invalid_json_skips_result(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_ENTITY_OUTPUT, "not-json", kind
    )
    for call in span.set_attribute.call_args_list:
        assert call[0][0] != "gen_ai.tool.call.result"


def test_mirror_output_skips_when_evaluation_off(mock_gen_ai_config):
    mock_gen_ai_config.payload_evaluation_enabled.value = False
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span,
        AiSpanAttributes.TRACELOOP_ENTITY_OUTPUT,
        '{"result": 1}',
        kind,
    )
    for call in span.set_attribute.call_args_list:
        assert call[0][0] != "gen_ai.tool.call.result"


def test_mirror_input_invalid_json_no_arguments(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_ENTITY_INPUT, "not-json", kind
    )
    for call in span.set_attribute.call_args_list:
        assert call[0][0] != "gen_ai.tool.call.arguments"


def test_mirror_input_skips_when_capture_off(mock_gen_ai_config):
    mock_gen_ai_config.payload_capture_enabled.value = False
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span,
        AiSpanAttributes.TRACELOOP_ENTITY_INPUT,
        '{"arguments": {}}',
        kind,
    )
    for call in span.set_attribute.call_args_list:
        assert call[0][0] != "gen_ai.tool.call.arguments"


def test_mirror_input_skips_when_evaluation_off(mock_gen_ai_config):
    mock_gen_ai_config.payload_evaluation_enabled.value = False
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    mirror_traceloop_to_gen_ai(
        span,
        AiSpanAttributes.TRACELOOP_ENTITY_INPUT,
        '{"arguments": {}}',
        kind,
    )
    for call in span.set_attribute.call_args_list:
        assert call[0][0] != "gen_ai.tool.call.arguments"


def test_mirror_input_arguments_string_uses_as_is(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    payload = '{"arguments": "plain-arg"}'
    mirror_traceloop_to_gen_ai(
        span, AiSpanAttributes.TRACELOOP_ENTITY_INPUT, payload, kind
    )
    span.set_attribute.assert_any_call("gen_ai.tool.call.arguments", "plain-arg")


def test_mirror_input_json_dump_fallback_uses_str_args(mock_gen_ai_config):
    span = MagicMock()
    kind = TraceloopSpanKindValues.TOOL.value
    payload = '{"tool_name": "t", "arguments": {"x": 1}}'

    real_dumps = gen_ai_mirror_mod.json.dumps

    def dumps_side_effect(obj, **kwargs):
        if obj == {"x": 1}:
            raise TypeError("not serializable")
        return real_dumps(obj, **kwargs)

    with patch.object(gen_ai_mirror_mod.json, "dumps", side_effect=dumps_side_effect):
        mirror_traceloop_to_gen_ai(
            span, AiSpanAttributes.TRACELOOP_ENTITY_INPUT, payload, kind
        )
    span.set_attribute.assert_any_call("gen_ai.tool.call.arguments", str({"x": 1}))


def test_mcp_instrumentation_get_tracer_patch_wraps_tracer():
    import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst

    prev = mcp_inst.get_tracer
    with mcp_instrumentation_get_tracer_patched():
        t = mcp_inst.get_tracer("opentelemetry.instrumentation.mcp.instrumentation", "1.0")
        assert isinstance(t, GenAiMirroringTracer)
    assert mcp_inst.get_tracer is prev


def test_patch_get_tracer_wraps_mcp_module_only():
    def prev(name, version=None, tracer_provider=None, schema_url=None, attributes=None):
        t = MagicMock()
        t.module_name = name
        return t

    factory = patch_get_tracer_for_mcp(prev)
    mcp_tracer = factory("opentelemetry.instrumentation.mcp.instrumentation", "1.0")
    other = factory("some.other.module", "1.0")
    assert isinstance(mcp_tracer, GenAiMirroringTracer)
    assert mcp_tracer._inner.module_name == "opentelemetry.instrumentation.mcp.instrumentation"
    assert other.module_name == "some.other.module"


def test_gen_ai_mirroring_tracer_delegates_unknown_attr():
    inner = MagicMock()
    inner.other_method.return_value = 42
    tracer = GenAiMirroringTracer(inner)
    assert tracer.other_method() == 42


def test_gen_ai_mirroring_span_set_attribute_swallows_mirror_errors(mock_gen_ai_config):
    inner_cm = MagicMock()
    inner_span = MagicMock()
    inner_cm.__enter__.return_value = inner_span
    inner_cm.__exit__.return_value = False
    inner = MagicMock()
    inner.start_as_current_span.return_value = inner_cm
    tracer = GenAiMirroringTracer(inner)
    with patch(
        "agent_trace.instrumentation.mcp.gen_ai_mirror.mirror_traceloop_to_gen_ai",
        side_effect=ValueError("boom"),
    ):
        with tracer.start_as_current_span("s") as span:
            span.set_attribute("any", "v")
    inner_span.set_attribute.assert_called_once_with("any", "v")
