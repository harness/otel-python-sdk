"""Map MCP / Traceloop span attributes to OpenTelemetry GenAI + MCP semantic conventions."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from opentelemetry.semconv_ai import SpanAttributes as AiSpanAttributes
from opentelemetry.semconv_ai import TraceloopSpanKindValues

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger

logger = get_custom_logger(__name__)

_GEN_AI_OPERATION_EXECUTE_TOOL = "execute_tool"
_GEN_AI_SYSTEM_MCP = "mcp"
_MCP_METHOD_TOOLS_CALL = "tools/call"


def apply_gen_ai_env_for_mcp() -> None:
    """
    Align FastMCP content capture with Traceable gen_ai payload_capture.

    The MCP contrib package gates body capture on TRACELOOP_TRACE_CONTENT; we
    mirror TA_GEN_AI_* via Config (see environment.default / TA_GEN_AI_*).
    """
    gen = Config().config.gen_ai
    if "TRACELOOP_TRACE_CONTENT" in os.environ:
        return
    os.environ["TRACELOOP_TRACE_CONTENT"] = (
        "true" if gen.payload_capture_enabled.value else "false"
    )


def _mirror_span_kind(span, key: str, value: Any, tool_kind: str) -> None:
    if key == AiSpanAttributes.TRACELOOP_SPAN_KIND and value == tool_kind:
        span.set_attribute("gen_ai.operation.name", _GEN_AI_OPERATION_EXECUTE_TOOL)
        span.set_attribute("gen_ai.system", _GEN_AI_SYSTEM_MCP)


def _mirror_entity_name(span, key: str, value: Any, tool_kind: str, current_kind: Optional[str]) -> None:
    if current_kind != tool_kind or key != AiSpanAttributes.TRACELOOP_ENTITY_NAME:
        return
    span.set_attribute("gen_ai.tool.name", value)
    span.set_attribute(AiSpanAttributes.MCP_METHOD_NAME, _MCP_METHOD_TOOLS_CALL)


def _mirror_entity_input(span, key: str, value: Any, current_kind: Optional[str], gen) -> None:
    tool_kind = TraceloopSpanKindValues.TOOL.value
    if (
        current_kind != tool_kind
        or key != AiSpanAttributes.TRACELOOP_ENTITY_INPUT
        or not gen.payload_capture_enabled.value
        or not isinstance(value, str)
    ):
        return
    if not gen.payload_evaluation_enabled.value:
        return
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return
    if not payload or not isinstance(payload, dict):
        return
    args = payload.get("arguments")
    if args is None:
        return
    try:
        span.set_attribute(
            "gen_ai.tool.call.arguments",
            json.dumps(args) if not isinstance(args, str) else args,
        )
    except (TypeError, ValueError):
        span.set_attribute("gen_ai.tool.call.arguments", str(args))


def _mirror_entity_output(span, key: str, value: Any, current_kind: Optional[str], gen) -> None:
    tool_kind = TraceloopSpanKindValues.TOOL.value
    if (
        current_kind != tool_kind
        or key != AiSpanAttributes.TRACELOOP_ENTITY_OUTPUT
        or not gen.payload_capture_enabled.value
        or not isinstance(value, str)
        or not gen.payload_evaluation_enabled.value
    ):
        return
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return
    if isinstance(payload, dict) and "result" in payload:
        span.set_attribute("gen_ai.tool.call.result", json.dumps(payload["result"]))
    elif isinstance(payload, list):
        span.set_attribute("gen_ai.tool.call.result", value)


def mirror_traceloop_to_gen_ai(span, key: str, value: Any, current_kind: Optional[str]) -> None:
    """Copy select Traceloop attributes to gen_ai.* / mcp.* on the same recording span."""
    gen = Config().config.gen_ai
    tool_kind = TraceloopSpanKindValues.TOOL.value
    _mirror_span_kind(span, key, value, tool_kind)
    _mirror_entity_name(span, key, value, tool_kind, current_kind)
    _mirror_entity_input(span, key, value, current_kind, gen)
    _mirror_entity_output(span, key, value, current_kind, gen)


class _GenAiSpanContext:
    """Context manager that wraps the active span for GenAI attribute mirroring."""

    def __init__(self, inner):
        self._inner = inner
        self.span_kind: Optional[str] = None

    def __enter__(self):
        span = self._inner.__enter__()
        return _MirroringSpan(span, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._inner.__exit__(exc_type, exc_val, exc_tb)


class _MirroringSpan:
    """Delegates to the SDK span and mirrors Traceloop attributes to gen_ai / mcp."""

    def __init__(self, inner, ctx: _GenAiSpanContext):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_ctx", ctx)

    def set_attribute(self, key: str, value: Any) -> None:
        if key == AiSpanAttributes.TRACELOOP_SPAN_KIND:
            self._ctx.span_kind = value
        self._inner.set_attribute(key, value)
        try:
            mirror_traceloop_to_gen_ai(self._inner, key, value, self._ctx.span_kind)
        except Exception as err:  # pylint: disable=broad-except
            logger.debug("gen_ai MCP mirror skipped: %s", err, exc_info=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class GenAiMirroringTracer:
    """Tracer that wraps start_as_current_span so span.set_attribute can mirror to GenAI."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def start_as_current_span(self, *args: Any, **kwargs: Any):
        return _GenAiSpanContext(self._inner.start_as_current_span(*args, **kwargs))


@contextmanager
def mcp_instrumentation_get_tracer_patched() -> Iterator[None]:
    """
    Temporarily wrap ``get_tracer`` inside ``opentelemetry.instrumentation.mcp.instrumentation``.

    The MCP package does ``from opentelemetry.trace import get_tracer``, so patching
    ``opentelemetry.trace.get_tracer`` does not change the function MCP uses. We must
    replace the name bound in the MCP instrumentation module.
    """
    import opentelemetry.instrumentation.mcp.instrumentation as mcp_inst  # pylint: disable=import-outside-toplevel

    prev = mcp_inst.get_tracer

    def _wrapped(*args: Any, **kwargs: Any):
        return GenAiMirroringTracer(prev(*args, **kwargs))

    mcp_inst.get_tracer = _wrapped
    try:
        yield
    finally:
        mcp_inst.get_tracer = prev


def patch_get_tracer_for_mcp(prev_get_tracer):
    """Return a get_tracer replacement (legacy helper; prefer :func:`mcp_instrumentation_get_tracer_patched`)."""

    def _wrapped(
        instrumenting_module_name: str,
        instrumenting_library_version=None,
        tracer_provider=None,
        schema_url=None,
        attributes=None,
    ):
        tracer = prev_get_tracer(
            instrumenting_module_name,
            instrumenting_library_version,
            tracer_provider=tracer_provider,
            schema_url=schema_url,
            attributes=attributes,
        )
        if instrumenting_module_name == "opentelemetry.instrumentation.mcp.instrumentation":
            return GenAiMirroringTracer(tracer)
        return tracer

    return _wrapped
