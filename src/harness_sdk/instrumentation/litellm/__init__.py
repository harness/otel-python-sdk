"""
LiteLLM instrumentation using the LiteLLM OpenTelemetry integration for span
telemetry, plus Traceable pre-call policy evaluation via libtraceable.

Coverage:
  - litellm.completion / litellm.acompletion
  - litellm.embedding / litellm.aembedding

Wraps the public entry points so evaluation runs on an active span before the
provider call. The wrapper enriches that span with response metadata before it
ends.

Streaming (``stream=True``) is handled by deferring span completion: the
returned LiteLLM ``CustomStreamWrapper`` is wrapped in a transparent proxy that
forwards every chunk to the caller, accumulates them, and only when the stream
is fully consumed (or errors) rebuilds the aggregated response, copies response
metadata (usage, id, finish reasons) onto the span, and ends it. Without this
the span would close as soon as ``completion()`` returned the not-yet-consumed
stream object, losing all response-side telemetry.

Optional: ``pip install harness-sdk[litellm]``
"""

from __future__ import annotations

import contextvars
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import wrapt
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Status, StatusCode

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.plugins.control import get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation import BaseInstrumentorWrapper

logger = get_custom_logger(__name__)

_LITELLM_MAIN = "litellm.main"
_LITELLM_REQUEST_SPAN_NAME = "litellm_request"

_LITELLM_SPAN_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "harness_litellm_span_active", default=False
)

# Env var to opt into bounded raw response/usage capture as a resilience fallback
# when LiteLLM changes response object shape or field names.
_RAW_CAPTURE_ENV = "HA_GEN_AI_RAW_CAPTURE_ENABLED"
_RAW_CAPTURE_MAX_BYTES_ENV = "HA_GEN_AI_RAW_CAPTURE_MAX_BYTES"
_RAW_CAPTURE_DEFAULT_MAX_BYTES = 8192

# Fields that must never be serialized into the raw usage/response fallback.
_RAW_SENSITIVE_KEYS = frozenset(
    {
        "messages",
        "input",
        "prompt",
        "content",
        "choices",
        "data",
        "embedding",
        "api_key",
        "authorization",
        "headers",
        "aws_secret_access_key",
        "aws_session_token",
    }
)

# Bedrock returns the actual executing model id in a response header, which is
# useful when the request targets an inference profile ARN.
_BEDROCK_MODEL_ID_HEADER = "x-amzn-bedrock-model-id"
_LITELLM_PROVIDER_HEADER_PREFIX = "llm_provider-"

_WRAPPED_FUNCTIONS = (
    ("completion", False),
    ("acompletion", True),
    ("embedding", False),
    ("aembedding", True),
)

_PROVIDER_NAME_MAP = {
    "azure": "azure.ai.openai",
    "azure_ai": "azure.ai.openai",
    "azure_ai_openai": "azure.ai.openai",
    "azureopenai": "azure.ai.openai",
    "bedrock": "aws.bedrock",
    "bedrock_converse": "aws.bedrock",
    "gemini": "gcp.gemini",
    "google": "gcp.gemini",
    "vertex_ai": "gcp.vertex_ai",
    "vertexai": "gcp.vertex_ai",
}


def _evaluate_span(span: Any) -> None:
    """Run Traceable policy evaluation against the live span; raise if blocked."""
    if not Config().config.gen_ai.payload_evaluation_enabled.value:
        logger.debug("LiteLLM: evaluate_body disabled, skipping policy evaluation")
        return
    if span is None or not span.is_recording():
        logger.debug("LiteLLM: no active span, skipping evaluation")
        return
    try:
        logger.debug(
            "LiteLLM: running policy evaluation on span %s",
            span.get_span_context().span_id,
        )
        result = get_control_registry().evaluate_agent_span(span, body="")
        if result.block:
            logger.debug("Traceable policy blocked LiteLLM request")
            raise ControlEvaluationBlocked(result)
        logger.debug("LiteLLM: policy evaluation passed")
    except ControlEvaluationBlocked:
        raise
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM span evaluation error: %s", err)


def _extract_model_and_input(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Optional[str], Any]:
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    payload = kwargs.get("messages")
    if payload is None:
        payload = kwargs.get("input")
    if payload is None and len(args) > 1:
        payload = args[1]
    return model, payload


def _resolve_provider(model: Optional[str], kwargs: dict[str, Any]) -> str:
    provider = kwargs.get("custom_llm_provider")
    if provider:
        return str(provider)
    litellm_params = kwargs.get("litellm_params") or {}
    provider = litellm_params.get("custom_llm_provider")
    if provider:
        return str(provider)
    if not model:
        return "Unknown"
    try:
        from litellm.litellm_core_utils.get_llm_provider_logic import (  # pylint: disable=import-outside-toplevel
            get_llm_provider,
        )

        _model, inferred, _, _ = get_llm_provider(
            model=model,
            custom_llm_provider=kwargs.get("custom_llm_provider"),
        )
        if inferred:
            return str(inferred)
        if _model:
            return str(_model)
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM: could not infer provider for model %s: %s", model, err)
    return "Unknown"


def _canonical_provider_name(provider: str) -> str:
    normalized = (provider or "unknown").strip().lower().replace("-", "_")
    return _PROVIDER_NAME_MAP.get(normalized, normalized)


def _operation_name(func_name: str) -> str:
    if func_name in ("embedding", "aembedding"):
        return "embeddings"
    return "chat"


@dataclass(frozen=True)
class _PreCallSpanContext:
    model: Optional[str]
    payload: Any
    kwargs: dict[str, Any]
    call_type: str


def _set_pre_call_request_attributes(
    otel_logger: Any,
    span: Any,
    pre_call: _PreCallSpanContext,
) -> None:
    """Populate request-side gen_ai attributes for libtraceable evaluation."""
    provider = _resolve_provider(pre_call.model, pre_call.kwargs)
    optional_params = pre_call.kwargs.get("optional_params") or pre_call.kwargs

    otel_logger.safe_set_attribute(span, "gen_ai.request.model", pre_call.model or "")
    otel_logger.safe_set_attribute(
        span, "gen_ai.operation.name", _operation_name(pre_call.call_type)
    )
    otel_logger.safe_set_attribute(
        span, "gen_ai.provider.name", _canonical_provider_name(provider)
    )
    otel_logger.safe_set_attribute(span, "gen_ai.framework", "litellm")
    otel_logger.safe_set_attribute(
        span,
        "gen_ai.request.streaming",
        str(optional_params.get("stream", False)),
    )

    if optional_params.get("max_tokens") is not None:
        otel_logger.safe_set_attribute(
            span, "gen_ai.request.max_tokens", optional_params.get("max_tokens")
        )
    if optional_params.get("temperature") is not None:
        otel_logger.safe_set_attribute(
            span, "gen_ai.request.temperature", optional_params.get("temperature")
        )
    if optional_params.get("top_p") is not None:
        otel_logger.safe_set_attribute(
            span, "gen_ai.request.top_p", optional_params.get("top_p")
        )

    if pre_call.payload is None:
        return
    try:
        import litellm  # pylint: disable=import-outside-toplevel
        from litellm.litellm_core_utils.safe_json_dumps import (  # pylint: disable=import-outside-toplevel
            safe_dumps,
        )

        if litellm.turn_off_message_logging or not otel_logger.message_logging:
            return
        if pre_call.call_type in ("embedding", "aembedding"):
            otel_logger.safe_set_attribute(
                span, "gen_ai.input.messages", safe_dumps(pre_call.payload)
            )
            return
        transformed = otel_logger._transform_messages_to_otel_semantic_conventions(  # pylint: disable=protected-access
            pre_call.payload
        )
        otel_logger.safe_set_attribute(span, "gen_ai.input.messages", safe_dumps(transformed))
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM: failed to set input attributes on span: %s", err)


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _set_if_present(otel_logger: Any, span: Any, key: str, value: Any) -> None:
    if value is not None:
        otel_logger.safe_set_attribute(span, key, value)


@dataclass(frozen=True)
class _LiteLLMResponseMetadata:
    """Version-tolerant accessor for LiteLLM response dicts/objects."""

    response: Any

    def value(self, key: str) -> Any:
        return _get_value(self.response, key)

    def hidden_params(self) -> Any:
        return self.value("_hidden_params")

    def usage(self) -> Any:
        return self.value("usage")

    def choices(self) -> list[Any]:
        choices = self.value("choices")
        if choices is None:
            return []
        return list(choices)

    def finish_reasons(self) -> list[str]:
        finish_reasons = []
        for choice in self.choices():
            finish_reason = _get_value(choice, "finish_reason")
            if finish_reason:
                finish_reasons.append(str(finish_reason))
        return finish_reasons

    def header_maps(self) -> list[dict[Any, Any]]:
        hidden = self.hidden_params()
        if hidden is None:
            return []

        headers = []
        additional = _get_value(hidden, "additional_headers")
        if isinstance(additional, dict):
            headers.append(additional)
        if isinstance(hidden, dict):
            headers.append(hidden)

        response_metadata = _get_value(hidden, "response_metadata") or _get_value(
            hidden, "ResponseMetadata"
        )
        http_headers = _get_value(response_metadata, "HTTPHeaders") or _get_value(
            response_metadata, "http_headers"
        )
        if isinstance(http_headers, dict):
            headers.append(http_headers)
        return headers

    def bedrock_execution_model(self) -> Optional[str]:
        for headers in self.header_maps():
            for key, value in headers.items():
                if _is_bedrock_model_id_header(key) and value:
                    return str(value)
        return None


def _is_bedrock_model_id_header(key: Any) -> bool:
    normalized = str(key).lower()
    return normalized in {
        _BEDROCK_MODEL_ID_HEADER,
        f"{_LITELLM_PROVIDER_HEADER_PREFIX}{_BEDROCK_MODEL_ID_HEADER}",
    }


def _raw_capture_enabled() -> bool:
    return os.getenv(_RAW_CAPTURE_ENV, "").strip().lower() == "true"


def _raw_capture_max_bytes() -> int:
    raw = os.getenv(_RAW_CAPTURE_MAX_BYTES_ENV, "").strip()
    if not raw:
        return _RAW_CAPTURE_DEFAULT_MAX_BYTES
    try:
        value = int(raw)
        return value if value > 0 else _RAW_CAPTURE_DEFAULT_MAX_BYTES
    except ValueError:
        return _RAW_CAPTURE_DEFAULT_MAX_BYTES


def _to_plain(obj: Any) -> Any:
    """Convert dict/object/Pydantic usage payloads into a JSON-safe dict."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {
            key: _to_plain(value)
            for key, value in obj.items()
            if str(key).lower() not in _RAW_SENSITIVE_KEYS
        }
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return _to_plain(method())
            except Exception:  # pylint: disable=broad-except
                pass
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {
            key: _to_plain(value)
            for key, value in data.items()
            if not key.startswith("_")
            and str(key).lower() not in _RAW_SENSITIVE_KEYS
        }
    return str(obj)


def _serialize_bounded(obj: Any, max_bytes: int) -> tuple[str, bool]:
    try:
        serialized = json.dumps(_to_plain(obj), default=str, ensure_ascii=False)
    except Exception:  # pylint: disable=broad-except
        serialized = str(obj)
    if len(serialized) > max_bytes:
        return serialized[:max_bytes], True
    return serialized, False


def _set_raw_usage_capture(otel_logger: Any, span: Any, usage: Any) -> None:
    if usage is None or not _raw_capture_enabled():
        return
    serialized, truncated = _serialize_bounded(usage, _raw_capture_max_bytes())
    otel_logger.safe_set_attribute(span, "gen_ai.response.usage.raw", serialized)
    if truncated:
        otel_logger.safe_set_attribute(
            span, "gen_ai.response.usage.raw_truncated", True
        )


def _set_response_attributes(
    otel_logger: Any,
    span: Any,
    response: Any,
    request_model: Optional[str] = None,
) -> None:
    """Copy LiteLLM response metadata before the wrapper-owned span ends."""
    metadata = _LiteLLMResponseMetadata(response)
    response_model = metadata.value("model")
    bedrock_execution_model = metadata.bedrock_execution_model()
    if bedrock_execution_model:
        otel_logger.safe_set_attribute(
            span, "aws.bedrock.execution_model_id", bedrock_execution_model
        )

    _set_if_present(otel_logger, span, "gen_ai.response.id", metadata.value("id"))

    finish_reasons = metadata.finish_reasons()
    if finish_reasons:
        otel_logger.safe_set_attribute(
            span, "gen_ai.response.finish_reasons", finish_reasons
        )

    usage = metadata.usage()

    # Guarantee gen_ai.response.model on any span that carries token usage so it
    # can join rate cards by provider + model. Prefer the provider response
    # model, fall back to the requested model when the provider omits it.
    effective_response_model = response_model or request_model
    if effective_response_model is not None and (
        response_model is not None or usage is not None
    ):
        otel_logger.safe_set_attribute(
            span, "gen_ai.response.model", effective_response_model
        )

    if usage is None:
        return

    _set_raw_usage_capture(otel_logger, span, usage)

    prompt_details = _get_value(usage, "prompt_tokens_details")
    completion_details = _get_value(usage, "completion_tokens_details")

    input_tokens = _get_value(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _get_value(usage, "input_tokens")

    output_tokens = _get_value(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _get_value(usage, "output_tokens")

    _set_if_present(otel_logger, span, "gen_ai.usage.input_tokens", input_tokens)
    _set_if_present(otel_logger, span, "gen_ai.usage.output_tokens", output_tokens)
    _set_if_present(
        otel_logger,
        span,
        "gen_ai.usage.total_tokens",
        _get_value(usage, "total_tokens"),
    )
    _set_if_present(
        otel_logger,
        span,
        "gen_ai.usage.cache_read.input_tokens",
        _get_value(usage, "cache_read_input_tokens")
        or _get_value(prompt_details, "cached_tokens"),
    )
    _set_if_present(
        otel_logger,
        span,
        "gen_ai.usage.cache_creation.input_tokens",
        _get_value(usage, "cache_creation_input_tokens")
        or _get_value(prompt_details, "cache_creation_tokens"),
    )
    _set_if_present(
        otel_logger,
        span,
        "gen_ai.usage.reasoning.output_tokens",
        _get_value(completion_details, "reasoning_tokens"),
    )


def _is_stream_response(response: Any) -> bool:
    """Detect a LiteLLM streaming response (``CustomStreamWrapper``).

    Falls back to duck-typing on the async/sync iterator protocol so that the
    check still works if LiteLLM renames or relocates the wrapper class.
    """
    try:
        from litellm import CustomStreamWrapper  # pylint: disable=import-outside-toplevel

        if isinstance(response, CustomStreamWrapper):
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    # ModelResponse / EmbeddingResponse are not iterators; a streaming response
    # exposes __anext__ (async) or __next__ (sync).
    return hasattr(response, "__anext__") or hasattr(response, "__next__")


def _aggregate_stream_response(chunks: list[Any], messages: Any) -> Any:
    """Rebuild a complete ``ModelResponse`` from streamed chunks.

    Uses ``litellm.stream_chunk_builder`` (the same helper LiteLLM uses
    internally) so usage, choices, finish_reason and content are aggregated
    exactly as they would be for a non-streaming call. Falls back to the last
    chunk that carries usage if the builder is unavailable or fails.
    """
    if not chunks:
        return None
    try:
        import litellm  # pylint: disable=import-outside-toplevel

        builder_messages = messages if isinstance(messages, list) else None
        aggregated = litellm.stream_chunk_builder(chunks, messages=builder_messages)
        if aggregated is not None:
            return aggregated
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("LiteLLM: stream_chunk_builder failed: %s", err)
    for chunk in reversed(chunks):
        if _get_value(chunk, "usage") is not None:
            return chunk
    return chunks[-1]


class _StreamSpanWrapper(wrapt.ObjectProxy):
    """Transparent proxy over a LiteLLM stream that defers span completion.

    Forwards every chunk to the caller unchanged while collecting them. When the
    stream is exhausted (or raises) it aggregates the chunks, enriches the span
    with response metadata, and ends the span. Attribute access and any other
    protocol falls through to the wrapped ``CustomStreamWrapper`` via
    ``wrapt.ObjectProxy``.
    """

    def __init__(
        self,
        wrapped: Any,
        otel_logger: Any,
        span: Any,
        request_model: Optional[str],
        messages: Any,
    ) -> None:
        super().__init__(wrapped)
        self._self_otel_logger = otel_logger
        self._self_span = span
        self._self_request_model = request_model
        self._self_messages = messages
        self._self_chunks: list[Any] = []
        self._self_finished = False

    def __iter__(self) -> "_StreamSpanWrapper":
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._finalize(None)
            raise
        except BaseException as exc:  # pylint: disable=broad-except
            self._finalize(exc)
            raise
        self._self_chunks.append(chunk)
        return chunk

    def __aiter__(self) -> "_StreamSpanWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finalize(None)
            raise
        except BaseException as exc:  # pylint: disable=broad-except
            self._finalize(exc)
            raise
        self._self_chunks.append(chunk)
        return chunk

    def _finalize(self, exc: Optional[BaseException]) -> None:
        if self._self_finished:
            return
        self._self_finished = True
        try:
            if exc is not None:
                self._self_span.record_exception(exc)
                self._self_span.set_status(Status(StatusCode.ERROR, str(exc)))
            else:
                aggregated = _aggregate_stream_response(
                    self._self_chunks, self._self_messages
                )
                if aggregated is not None:
                    _set_response_attributes(
                        self._self_otel_logger,
                        self._self_span,
                        aggregated,
                        request_model=self._self_request_model,
                    )
        except Exception as err:  # pylint: disable=broad-except
            logger.debug("LiteLLM: failed to finalize stream span: %s", err)
        finally:
            self._self_span.end()

    def __del__(self) -> None:
        # Safety net: if the consumer abandoned the stream before exhausting it,
        # still end the span (with whatever chunks were collected) so it is not
        # leaked/never exported.
        try:
            if not self._self_finished:
                self._finalize(None)
        except Exception:  # pylint: disable=broad-except
            pass


def _build_traceable_otel_class() -> type:
    from litellm.integrations.opentelemetry import (  # pylint: disable=import-outside-toplevel
        OpenTelemetry,
        OpenTelemetryConfig,
    )

    class TraceableLiteLLMOpenTelemetry(OpenTelemetry):  # pylint: disable=abstract-method
        """LiteLLM ``OpenTelemetry`` callback using Traceable's global tracer provider."""

        def __init__(self) -> None:
            super().__init__(config=OpenTelemetryConfig.from_env())

    return TraceableLiteLLMOpenTelemetry


_TraceableLiteLLMOpenTelemetry: Optional[type] = None
_otel_logger: Any = None


def _get_otel_logger() -> Any:
    global _otel_logger, _TraceableLiteLLMOpenTelemetry  # pylint: disable=global-statement
    if _otel_logger is None:
        if _TraceableLiteLLMOpenTelemetry is None:
            _TraceableLiteLLMOpenTelemetry = _build_traceable_otel_class()
        _otel_logger = _TraceableLiteLLMOpenTelemetry()
    return _otel_logger


def _register_otel_callback(otel_logger: Any) -> None:
    import litellm  # pylint: disable=import-outside-toplevel
    from litellm.integrations.opentelemetry import OpenTelemetry  # pylint: disable=import-outside-toplevel

    updated: list[Any] = []
    replaced = False
    for callback in litellm.callbacks:
        if callback == "otel":
            replaced = True
            continue
        if type(callback) is OpenTelemetry:  # pylint: disable=unidiomatic-typecheck
            replaced = True
            continue
        updated.append(callback)
    traceable_cls = type(otel_logger)
    if not any(isinstance(cb, traceable_cls) for cb in updated):
        updated.append(otel_logger)
    elif not replaced:
        for index, callback in enumerate(updated):
            if isinstance(callback, traceable_cls):
                updated[index] = otel_logger
                break
    litellm.callbacks = updated
    logger.debug("LiteLLM OpenTelemetry callback registered for Traceable instrumentation.")


def _unregister_otel_callback() -> None:
    import litellm  # pylint: disable=import-outside-toplevel

    if _TraceableLiteLLMOpenTelemetry is None:
        return
    litellm.callbacks = [
        callback
        for callback in litellm.callbacks
        if not isinstance(callback, _TraceableLiteLLMOpenTelemetry)
    ]


def _activate_span(span: Any) -> object:
    return otel_context.attach(trace.set_span_in_context(span))


def _deactivate_span(token: object, span: Any) -> None:
    otel_context.detach(token)
    span.end()


def _fail_pre_call_span(span: Any, exc: BaseException, *, blocked: bool = False) -> None:
    if blocked:
        span.set_status(Status(StatusCode.ERROR, "blocked"))
    else:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.end()


def _start_evaluated_span(
    otel_logger: Any,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    model, payload = _extract_model_and_input(args, kwargs)
    pre_call = _PreCallSpanContext(
        model=model, payload=payload, kwargs=kwargs, call_type=func_name
    )
    span = otel_logger.tracer.start_span(_LITELLM_REQUEST_SPAN_NAME)
    try:
        _set_pre_call_request_attributes(otel_logger, span, pre_call)
        _evaluate_span(span)
    except ControlEvaluationBlocked as exc:
        _fail_pre_call_span(span, exc, blocked=True)
        raise
    except Exception as exc:  # pylint: disable=broad-except
        _fail_pre_call_span(span, exc)
        raise
    return span


@dataclass
class _LiteLLMSpanRun:
    otel_logger: Any
    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    request_model: Optional[str] = None
    span: Any = None
    token: Optional[object] = None
    guard: Optional[object] = None

    def __enter__(self) -> "_LiteLLMSpanRun":
        self.request_model, _ = _extract_model_and_input(self.args, self.kwargs)
        self.span = _start_evaluated_span(
            self.otel_logger, self.func_name, self.args, self.kwargs
        )
        self.token = _activate_span(self.span)
        self.guard = _LITELLM_SPAN_ACTIVE.set(True)
        return self

    def set_response_attributes(self, response: Any) -> None:
        _set_response_attributes(
            self.otel_logger, self.span, response, request_model=self.request_model
        )

    def wrap_stream(self, response: Any) -> Any:
        """Hand span ownership to a stream proxy and stop managing it here.

        Detaches the active context and clears the re-dispatch guard without
        ending the span, then returns a proxy that ends the span once the
        stream is consumed. The context manager ``__exit__`` becomes a no-op for
        the span because ``token``/``guard`` are cleared.
        """
        if self.guard is not None:
            _LITELLM_SPAN_ACTIVE.reset(self.guard)
            self.guard = None
        if self.token is not None:
            otel_context.detach(self.token)
            self.token = None
        _, payload = _extract_model_and_input(self.args, self.kwargs)
        return _StreamSpanWrapper(
            response,
            self.otel_logger,
            self.span,
            self.request_model,
            payload,
        )

    def __exit__(
        self,
        _exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        _traceback: Any,
    ) -> bool:
        try:
            if exc is not None:
                self.span.record_exception(exc)
                self.span.set_status(Status(StatusCode.ERROR, str(exc)))
        finally:
            if self.guard is not None:
                _LITELLM_SPAN_ACTIVE.reset(self.guard)
            if self.token is not None and self.span is not None:
                _deactivate_span(self.token, self.span)
        return False


def _make_wrapper(func_name: str, is_async: bool) -> Callable[..., Any]:
    otel_logger = _get_otel_logger()

    def _sync_wrapper(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            return wrapped(*args, **kwargs)

        # Skip nested re-dispatch (e.g. aembedding -> embedding) so each provider
        # call yields exactly one litellm_request span.
        if _LITELLM_SPAN_ACTIVE.get():
            return wrapped(*args, **kwargs)

        with _LiteLLMSpanRun(otel_logger, func_name, args, kwargs) as span_run:
            response = wrapped(*args, **kwargs)
            if _is_stream_response(response):
                return span_run.wrap_stream(response)
            span_run.set_response_attributes(response)
            return response

    async def _async_wrapper(
        wrapped: Callable[..., Any],
        _instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if not Config().config.gen_ai.enabled.value:
            return await wrapped(*args, **kwargs)

        if _LITELLM_SPAN_ACTIVE.get():
            return await wrapped(*args, **kwargs)

        with _LiteLLMSpanRun(otel_logger, func_name, args, kwargs) as span_run:
            response = await wrapped(*args, **kwargs)
            if _is_stream_response(response):
                return span_run.wrap_stream(response)
            span_run.set_response_attributes(response)
            return response

    return _async_wrapper if is_async else _sync_wrapper


class LiteLLMInstrumentorWrapper(BaseInstrumentorWrapper):
    """Instrument LiteLLM with its OpenTelemetry SDK and Traceable policy evaluation."""

    _applied: bool = False

    def __init__(self) -> None:
        BaseInstrumentorWrapper.__init__(self)

    def instrument(self, **_kwargs: Any) -> None:
        if self._applied:
            logger.debug("LiteLLM instrumentation already applied.")
            return
        if not Config().config.gen_ai.enabled.value:
            logger.debug("Gen AI instrumentation disabled; skip LiteLLM wraps.")
            return
        try:
            import litellm  # pylint: disable=import-outside-toplevel

            main_mod = __import__(_LITELLM_MAIN, fromlist=["*"])
            for func_name, is_async in _WRAPPED_FUNCTIONS:
                wrapt.wrap_function_wrapper(
                    _LITELLM_MAIN,
                    func_name,
                    _make_wrapper(func_name, is_async),
                )
                if hasattr(litellm, func_name):
                    setattr(litellm, func_name, getattr(main_mod, func_name))
            LiteLLMInstrumentorWrapper._applied = True
            logger.debug("Traceable LiteLLM instrumentation applied.")
        except ImportError as err:
            logger.error("LiteLLM not available: %s", err)
            logger.error("Install with: pip install 'harness-sdk[litellm]'")
            raise

    def uninstrument(self, **_kwargs: Any) -> None:
        if not self._applied:
            return
        from importlib import import_module  # pylint: disable=import-outside-toplevel

        import litellm  # pylint: disable=import-outside-toplevel

        errors: list[Exception] = []
        mod = import_module(_LITELLM_MAIN)
        for func_name, _ in _WRAPPED_FUNCTIONS:
            try:
                unwrap(mod, func_name)
                if hasattr(litellm, func_name):
                    setattr(litellm, func_name, getattr(mod, func_name))
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to uninstrument LiteLLM %s: %s", func_name, err)
                errors.append(err)

        _unregister_otel_callback()
        global _otel_logger  # pylint: disable=global-statement
        _otel_logger = None
        LiteLLMInstrumentorWrapper._applied = False
        logger.debug("LiteLLM instrumentation removed.")
        if errors:
            raise errors[0]


__all__ = ["LiteLLMInstrumentorWrapper"]
