"""
Google Gen AI SDK (``google-genai``) instrumentation using
``opentelemetry-util-genai`` TelemetryHandler for span/metric telemetry, plus
Traceable pre-call policy evaluation.

Covers the unified Google Gen AI SDK (``from google import genai``) against both
the Gemini Developer API and the Gemini API on Vertex AI
(``genai.Client(vertexai=True, ...)``).

Coverage:
  - Models.generate_content / AsyncModels.generate_content        (non-streaming)
  - Models.generate_content_stream / AsyncModels.generate_content_stream (streaming)
  - Models.embed_content / AsyncModels.embed_content

The classic ``vertexai.generative_models`` / ``google-cloud-aiplatform``
generative modules are deprecated by Google (removal 2026-06-24) and are NOT
instrumented here; ``google-genai`` is the supported client.

Optional: ``pip install harness-sdk[google-genai]``
"""
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional

import wrapt
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)
from opentelemetry.util.genai.utils import should_capture_content_on_spans_in_experimental_mode

from harness_sdk.config.config import Config
from harness_sdk.custom_logger import get_custom_logger
from harness_sdk.plugins.control import get_control_registry
from harness_sdk.gen_ai.exceptions import ControlEvaluationBlocked
from harness_sdk.instrumentation import BaseInstrumentorWrapper

logger = get_custom_logger(__name__)

_MODELS_MODULE = "google.genai.models"

_PROVIDER_VERTEX = "gcp.vertex_ai"
_PROVIDER_GEMINI = "gcp.gemini"
_PROVIDER_FALLBACK = "gcp.gen_ai"


def _get_handler() -> TelemetryHandler:
    return TelemetryHandler()


def _get(source: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an object, returning ``default`` if absent."""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _resolve_provider(instance: Any) -> str:
    """Return the gen_ai.provider.name based on the client backend (Vertex vs Gemini).

    ``BaseApiClient.vertexai`` is truthy only for the Vertex AI backend; it is
    falsy (``None``/``False``) for the Gemini Developer API. When the client
    cannot be located we fall back to the generic GCP provider.
    """
    api_client = _get(instance, "_api_client")
    if api_client is None:
        return _PROVIDER_FALLBACK
    return _PROVIDER_VERTEX if _get(api_client, "vertexai") else _PROVIDER_GEMINI


# --------------------------------------------------------------------------- #
# Request / response mapping
# --------------------------------------------------------------------------- #
def _extract_model(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    model = kwargs.get("model")
    if model is None and args:
        model = args[0]
    return str(model) if model is not None else None


def _extract_contents(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    contents = kwargs.get("contents")
    if contents is None and len(args) > 1:
        contents = args[1]
    return contents


def _part_to_message_part(part: Any) -> Optional[Any]:
    """Map a google-genai Part to a util-genai MessagePart (best-effort)."""
    text = _get(part, "text")
    if isinstance(text, str) and text:
        return Text(content=text)

    function_call = _get(part, "function_call")
    if function_call is not None:
        return ToolCallRequest(
            arguments=_get(function_call, "args"),
            name=_get(function_call, "name") or "",
            id=_get(function_call, "id"),
        )

    function_response = _get(part, "function_response")
    if function_response is not None:
        return ToolCallResponse(
            response=_get(function_response, "response"),
            id=_get(function_response, "id"),
        )
    return None


def _content_to_parts(content: Any) -> List[Any]:
    if isinstance(content, str):
        return [Text(content=content)]
    parts_out: List[Any] = []
    raw_parts = _get(content, "parts")
    if raw_parts:
        for raw in raw_parts:
            mapped = _part_to_message_part(raw)
            if mapped is not None:
                parts_out.append(mapped)
        return parts_out
    # A bare Part (has text/function_call directly)
    mapped = _part_to_message_part(content)
    if mapped is not None:
        parts_out.append(mapped)
    return parts_out


def _to_input_messages(contents: Any, capture_content: bool) -> List[InputMessage]:
    if not capture_content or contents is None:
        return []
    items = contents if isinstance(contents, list) else [contents]
    messages: List[InputMessage] = []
    for item in items:
        role = _get(item, "role") or "user"
        parts = _content_to_parts(item)
        if parts:
            messages.append(InputMessage(role=str(role), parts=parts))
    return messages


def _to_system_instruction(config: Any, capture_content: bool) -> List[Any]:
    if not capture_content or config is None:
        return []
    system = _get(config, "system_instruction")
    if system is None:
        return []
    return _content_to_parts(system)


def _build_invocation(
    handler: TelemetryHandler,
    provider: str,
    model: Optional[str],
    contents: Any,
    config: Any,
    capture_content: bool,
    *,
    streaming: bool,
) -> LLMInvocation:
    attributes: dict[str, Any] = {"gen_ai.framework": "google-genai"}
    if streaming:
        attributes["gen_ai.request.streaming"] = True
    invocation = LLMInvocation(
        request_model=model,
        provider=provider,
        input_messages=_to_input_messages(contents, capture_content),
        system_instruction=_to_system_instruction(config, capture_content),
        temperature=_get(config, "temperature"),
        top_p=_get(config, "top_p"),
        max_tokens=_get(config, "max_output_tokens"),
        stop_sequences=_get(config, "stop_sequences"),
        seed=_get(config, "seed"),
        attributes=attributes,
    )
    handler.start_llm(invocation)
    return invocation


def _candidate_to_output_message(candidate: Any, capture_content: bool) -> Optional[OutputMessage]:
    finish_reason = _get(candidate, "finish_reason")
    finish_reason_str = str(finish_reason) if finish_reason is not None else "stop"
    parts: List[Any] = []
    if capture_content:
        content = _get(candidate, "content")
        parts = _content_to_parts(content) if content is not None else []
    role = _get(_get(candidate, "content"), "role") or "assistant"
    return OutputMessage(role=str(role), parts=parts, finish_reason=finish_reason_str)


def _apply_response(invocation: LLMInvocation, response: Any, capture_content: bool) -> None:
    """Populate the invocation from a google-genai GenerateContentResponse."""
    if response is None:
        return

    invocation.response_id = _get(response, "response_id")
    invocation.response_model_name = _get(response, "model_version")

    usage = _get(response, "usage_metadata")
    if usage is not None:
        invocation.input_tokens = _get(usage, "prompt_token_count")
        invocation.output_tokens = _get(usage, "candidates_token_count")

    candidates = _get(response, "candidates") or []
    output_messages: List[OutputMessage] = []
    finish_reasons: List[str] = []
    for candidate in candidates:
        message = _candidate_to_output_message(candidate, capture_content)
        if message is not None:
            output_messages.append(message)
            finish_reasons.append(message.finish_reason)
    if output_messages:
        invocation.output_messages = output_messages
    if finish_reasons:
        invocation.finish_reasons = finish_reasons


class _StreamAccumulator:
    """Accumulates streamed chunks so the span can be finalized once complete."""

    def __init__(self, capture_content: bool) -> None:
        self._capture_content = capture_content
        self._last_response: Any = None
        self._text_parts: List[str] = []
        self._finish_reasons: List[str] = []
        self._response_id: Optional[str] = None
        self._model_version: Optional[str] = None
        self._input_tokens: Optional[int] = None
        self._output_tokens: Optional[int] = None

    def add(self, chunk: Any) -> None:
        self._last_response = chunk
        response_id = _get(chunk, "response_id")
        if response_id is not None:
            self._response_id = response_id
        model_version = _get(chunk, "model_version")
        if model_version is not None:
            self._model_version = model_version
        usage = _get(chunk, "usage_metadata")
        if usage is not None:
            input_tokens = _get(usage, "prompt_token_count")
            output_tokens = _get(usage, "candidates_token_count")
            if input_tokens is not None:
                self._input_tokens = input_tokens
            if output_tokens is not None:
                self._output_tokens = output_tokens
        if self._capture_content:
            text = _get(chunk, "text")
            if isinstance(text, str) and text:
                self._text_parts.append(text)
        for candidate in _get(chunk, "candidates") or []:
            finish_reason = _get(candidate, "finish_reason")
            if finish_reason is not None:
                self._finish_reasons.append(str(finish_reason))

    def apply(self, invocation: LLMInvocation) -> None:
        invocation.response_id = self._response_id
        invocation.response_model_name = self._model_version
        invocation.input_tokens = self._input_tokens
        invocation.output_tokens = self._output_tokens
        if self._finish_reasons:
            invocation.finish_reasons = self._finish_reasons
        if self._capture_content and self._text_parts:
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content="".join(self._text_parts))],
                    finish_reason=self._finish_reasons[-1] if self._finish_reasons else "stop",
                )
            ]


def _evaluate_invocation(invocation: LLMInvocation) -> None:
    """Run Traceable policy evaluation against the live span; raise if blocked."""
    if not Config().config.gen_ai.payload_evaluation_enabled.value:
        logger.debug("google-genai: evaluate_body disabled, skipping policy evaluation")
        return
    span = getattr(invocation, "span", None)
    if span is None:
        inference_invocation = getattr(invocation, "_inference_invocation", None)
        if inference_invocation is not None:
            span = getattr(inference_invocation, "span", None)
    if span is None or not span.is_recording():
        logger.debug("google-genai: no active span, skipping evaluation")
        return
    try:
        result = get_control_registry().evaluate_agent_span(span, body="")
        if result.block:
            logger.debug("Traceable policy blocked google-genai request")
            raise ControlEvaluationBlocked(result)
    except ControlEvaluationBlocked:
        raise
    except Exception as err:  # pylint: disable=broad-except
        logger.debug("google-genai span evaluation error: %s", err)


# --------------------------------------------------------------------------- #
# generate_content (non-streaming)
# --------------------------------------------------------------------------- #
def _make_generate_sync(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()

    def _wrapper(wrapped, instance, args, kwargs):
        invocation = _build_invocation(
            handler, _resolve_provider(instance),
            _extract_model(args, kwargs), _extract_contents(args, kwargs),
            kwargs.get("config"), capture_content, streaming=False,
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = wrapped(*args, **kwargs)
            _apply_response(invocation, result, capture_content)
            handler.stop_llm(invocation)
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_generate_async(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()

    async def _wrapper(wrapped, instance, args, kwargs):
        invocation = _build_invocation(
            handler, _resolve_provider(instance),
            _extract_model(args, kwargs), _extract_contents(args, kwargs),
            kwargs.get("config"), capture_content, streaming=False,
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = await wrapped(*args, **kwargs)
            _apply_response(invocation, result, capture_content)
            handler.stop_llm(invocation)
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


# --------------------------------------------------------------------------- #
# generate_content_stream (streaming)
# --------------------------------------------------------------------------- #
def _finalize_stream(handler, invocation, accumulator, error: Optional[BaseException]) -> None:
    if error is not None:
        handler.fail_llm(invocation, Error(message=str(error), type=type(error)))
        return
    accumulator.apply(invocation)
    handler.stop_llm(invocation)


def _iterate_sync(result: Iterator[Any], handler, invocation, accumulator) -> Iterator[Any]:
    finalized = False
    try:
        for chunk in result:
            accumulator.add(chunk)
            yield chunk
        _finalize_stream(handler, invocation, accumulator, None)
        finalized = True
    except Exception as exc:  # pylint: disable=broad-except
        _finalize_stream(handler, invocation, accumulator, exc)
        finalized = True
        raise
    finally:
        if not finalized:
            _finalize_stream(handler, invocation, accumulator, None)


async def _iterate_async(result, handler, invocation, accumulator):
    finalized = False
    try:
        async for chunk in result:
            accumulator.add(chunk)
            yield chunk
        _finalize_stream(handler, invocation, accumulator, None)
        finalized = True
    except Exception as exc:  # pylint: disable=broad-except
        _finalize_stream(handler, invocation, accumulator, exc)
        finalized = True
        raise
    finally:
        if not finalized:
            _finalize_stream(handler, invocation, accumulator, None)


def _make_generate_stream_sync(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()

    def _wrapper(wrapped, instance, args, kwargs):
        invocation = _build_invocation(
            handler, _resolve_provider(instance),
            _extract_model(args, kwargs), _extract_contents(args, kwargs),
            kwargs.get("config"), capture_content, streaming=True,
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise
        return _iterate_sync(result, handler, invocation, _StreamAccumulator(capture_content))

    return _wrapper


def _make_generate_stream_async(handler: TelemetryHandler) -> Callable[..., Any]:
    capture_content = should_capture_content_on_spans_in_experimental_mode()

    async def _wrapper(wrapped, instance, args, kwargs):
        invocation = _build_invocation(
            handler, _resolve_provider(instance),
            _extract_model(args, kwargs), _extract_contents(args, kwargs),
            kwargs.get("config"), capture_content, streaming=True,
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            handler.fail_llm(invocation, Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            handler.fail_llm(invocation, Error(message=str(exc), type=type(exc)))
            raise
        return _iterate_async(result, handler, invocation, _StreamAccumulator(capture_content))

    return _wrapper


# --------------------------------------------------------------------------- #
# embed_content
# --------------------------------------------------------------------------- #
def _apply_embedding_response(invocation: Any, result: Any) -> None:
    invocation.response_model_name = _get(result, "model_version")
    embeddings = _get(result, "embeddings") or []
    if embeddings:
        values = _get(embeddings[0], "values")
        if isinstance(values, list):
            invocation.dimension_count = len(values)
        stats = _get(embeddings[0], "statistics")
        token_count = _get(stats, "token_count")
        if token_count is not None:
            invocation.input_tokens = int(token_count)


def _make_embed_sync(handler: TelemetryHandler) -> Callable[..., Any]:
    def _wrapper(wrapped, instance, args, kwargs):
        invocation = handler.start_embedding(
            _resolve_provider(instance),
            request_model=_extract_model(args, kwargs),
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            invocation.fail(Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = wrapped(*args, **kwargs)
            _apply_embedding_response(invocation, result)
            invocation.stop()
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            invocation.fail(Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


def _make_embed_async(handler: TelemetryHandler) -> Callable[..., Any]:
    async def _wrapper(wrapped, instance, args, kwargs):
        invocation = handler.start_embedding(
            _resolve_provider(instance),
            request_model=_extract_model(args, kwargs),
        )
        try:
            _evaluate_invocation(invocation)
        except ControlEvaluationBlocked:
            invocation.fail(Error(message="blocked", type=ControlEvaluationBlocked))
            raise
        try:
            result = await wrapped(*args, **kwargs)
            _apply_embedding_response(invocation, result)
            invocation.stop()
            return result
        except ControlEvaluationBlocked:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            invocation.fail(Error(message=str(exc), type=type(exc)))
            raise

    return _wrapper


# --------------------------------------------------------------------------- #
# Instrumentor
# --------------------------------------------------------------------------- #
_WRAPPED_METHODS = [
    ("Models", "generate_content", _make_generate_sync),
    ("Models", "generate_content_stream", _make_generate_stream_sync),
    ("Models", "embed_content", _make_embed_sync),
    ("AsyncModels", "generate_content", _make_generate_async),
    ("AsyncModels", "generate_content_stream", _make_generate_stream_async),
    ("AsyncModels", "embed_content", _make_embed_async),
]


class GoogleGenAIInstrumentorWrapper(BaseInstrumentorWrapper):
    """Instrument the Google Gen AI SDK (google-genai) with OTEL GenAI telemetry and policy evaluation."""  # pylint: disable=line-too-long

    _applied: bool = False

    def __init__(self) -> None:
        BaseInstrumentorWrapper.__init__(self)

    def instrument(self, **_kwargs: Any) -> None:
        if self._applied:
            logger.debug("google-genai instrumentation already applied.")
            return
        try:
            handler = _get_handler()
            for cls_name, method, factory in _WRAPPED_METHODS:
                wrapped = factory(handler)
                wrapt.wrap_function_wrapper(_MODELS_MODULE, f"{cls_name}.{method}", wrapped)
            GoogleGenAIInstrumentorWrapper._applied = True
            logger.debug("Traceable google-genai instrumentation applied.")
        except ImportError as err:
            logger.error("google-genai SDK not available: %s", err)
            logger.error("Install with: pip install 'harness-sdk[google-genai]'")
            raise

    def uninstrument(self, **_kwargs: Any) -> None:
        if not self._applied:
            return
        from importlib import import_module  # pylint: disable=import-outside-toplevel

        errors: List[Exception] = []
        try:
            mod = import_module(_MODELS_MODULE)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("google-genai SDK not available: %s", err)
            raise
        for cls_name, method, _ in _WRAPPED_METHODS:
            try:
                unwrap(getattr(mod, cls_name), method)
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to uninstrument google-genai %s.%s: %s", cls_name, method, err)
                errors.append(err)

        GoogleGenAIInstrumentorWrapper._applied = False
        logger.debug("google-genai instrumentation removed.")
        if errors:
            raise errors[0]


__all__ = ["GoogleGenAIInstrumentorWrapper"]
